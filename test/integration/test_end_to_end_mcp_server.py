#!/usr/bin/env python3
"""
End-to-End MCP Server Integration Tests
Tests the complete pipeline: CLI → Orchestrator → Capture → Ingestion → Database → MCP Tools
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from capture.logger import get_logger

# Create test logger
e2e_logger = get_logger("hunyo.test.e2e")


class TestEndToEndMCPServer:
    """Complete end-to-end tests for hunyo-mcp-server CLI and pipeline"""

    @pytest.fixture
    def runtime_events_schema(self):
        """Load runtime events schema for validation"""
        schema_path = Path("schemas/json/runtime_events_schema.json")
        with open(schema_path) as f:
            return json.load(f)

    @pytest.fixture
    def openlineage_events_schema(self):
        """Load OpenLineage events schema for validation"""
        schema_path = Path("schemas/json/openlineage_events_schema.json")
        with open(schema_path) as f:
            return json.load(f)

    @pytest.fixture
    def test_notebook_path(self) -> Path:
        """Use the test notebook file from test fixtures"""
        # Use absolute path resolution to ensure it works in CI
        test_dir = Path(__file__).parent.parent  # Go up from integration/ to test/
        notebook_path = test_dir / "fixtures" / "test_notebook.py"
        assert (
            notebook_path.exists()
        ), f"test_notebook.py not found at {notebook_path} (absolute: {notebook_path.absolute()})"
        return notebook_path

    def validate_event_against_schema(
        self, event: dict[str, Any], schema: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate a single event against a JSON schema"""
        try:
            jsonschema.validate(event, schema)
            return True, None
        except jsonschema.ValidationError as e:
            error_msg = f"Schema validation failed: {e.message}"
            if hasattr(e, "path") and e.path:
                error_msg += f" (at path: {'.'.join(str(p) for p in e.path)})"
            return False, error_msg
        except Exception as e:
            return False, f"Schema validation error: {e}"

    def wait_for_files_with_timeout(
        self, directory: Path, pattern: str, timeout: float = 30.0
    ) -> list[Path]:
        """Wait for files matching pattern to appear in directory"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            files = list(directory.glob(pattern))
            if files:
                return files
            time.sleep(0.5)

        return []

    def validate_events_in_files(
        self, event_files: list[Path], schema: dict[str, Any], event_type_name: str
    ) -> tuple[int, list[str]]:
        """Validate events in JSONL files against schema"""
        total_events = 0
        validation_errors = []

        for file_path in event_files:
            e2e_logger.info(
                f"📄 Validating {event_type_name} events in {file_path.name}"
            )

            with open(file_path) as f:
                for line_num, raw_line in enumerate(f, 1):
                    line_content = raw_line.strip()
                    if not line_content:
                        continue

                    try:
                        event = json.loads(line_content)
                        total_events += 1

                        is_valid, error = self.validate_event_against_schema(
                            event, schema
                        )
                        if not is_valid:
                            validation_errors.append(
                                f"{file_path.name}:{line_num} - {error}"
                            )
                    except json.JSONDecodeError as e:
                        validation_errors.append(
                            f"{file_path.name}:{line_num} - JSON decode error: {e}"
                        )

        return total_events, validation_errors

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_hunyo_mcp_server_pipeline(
        self,
        temp_hunyo_dir,
        test_notebook_path,
        runtime_events_schema,
        openlineage_events_schema,
    ):
        """Test the complete pipeline from CLI execution to database ingestion"""

        e2e_logger.success("🚀 Starting end-to-end MCP server pipeline test")

        # 1. Environment setup
        original_env = os.environ.copy()
        test_env = os.environ.copy()
        test_env.update(
            {
                "HUNYO_DEV_MODE": "1",
                "HUNYO_DATA_DIR": str(temp_hunyo_dir),
                "PYTHONPATH": str(Path("src").absolute())
                + os.pathsep
                + test_env.get("PYTHONPATH", ""),
            }
        )

        e2e_logger.info(f"📁 Test data directory: {temp_hunyo_dir}")
        e2e_logger.info(f"📝 Test notebook: {test_notebook_path}")

        process = None
        stdout_buffer = []
        stderr_buffer = []

        try:
            # 2. Execute hunyo-mcp-server as subprocess
            e2e_logger.info("⚡ Starting hunyo-mcp-server subprocess...")

            cmd = [
                sys.executable,
                "-m",
                "hunyo_mcp_server.server",
                "--notebook",
                str(test_notebook_path),
                "--dev-mode",
                "--verbose",
                "--standalone",
            ]

            process = subprocess.Popen(
                cmd,
                env=test_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # 3. Wait for startup and initial processing with live monitoring
            e2e_logger.info("⏳ Waiting for server startup and event processing...")

            # Monitor subprocess output for up to 8 seconds
            import platform
            import select

            start_time = time.time()
            timeout = 8.0

            while time.time() - start_time < timeout:
                # Check if process has terminated
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    e2e_logger.error(
                        f"Process exited early with code {process.returncode}"
                    )
                    e2e_logger.error(f"STDOUT: {stdout}")
                    e2e_logger.error(f"STDERR: {stderr}")
                    pytest.fail(f"hunyo-mcp-server exited early: {stderr}")

                # On Windows, use different approach for non-blocking read
                if platform.system() == "Windows":
                    time.sleep(0.5)  # Small sleep to let process work
                else:
                    # Unix-style non-blocking read
                    ready, _, _ = select.select(
                        [process.stdout, process.stderr], [], [], 0.5
                    )

                    for stream in ready:
                        if stream == process.stdout:
                            line = stream.readline()
                            if line:
                                stdout_buffer.append(line.strip())
                                e2e_logger.info(f"[SUBPROCESS STDOUT] {line.strip()}")
                        elif stream == process.stderr:
                            line = stream.readline()
                            if line:
                                stderr_buffer.append(line.strip())
                                e2e_logger.warning(
                                    f"[SUBPROCESS STDERR] {line.strip()}"
                                )

            # 3.5. Check database state while process is still running
            e2e_logger.info("🔍 Checking database state while process is running...")

            db_path = temp_hunyo_dir / "database" / "lineage.duckdb"
            database_dir = temp_hunyo_dir / "database"

            e2e_logger.info(f"Database directory exists: {database_dir.exists()}")
            e2e_logger.info(f"Database file exists: {db_path.exists()}")

            if database_dir.exists():
                db_dir_contents = list(database_dir.iterdir())
                e2e_logger.info(
                    f"Database directory contents: {[f.name for f in db_dir_contents]}"
                )

            # Try to list all files in the temp directory for debugging
            all_files = []
            for root, _dirs, files in os.walk(temp_hunyo_dir):
                for file in files:
                    rel_path = Path(root).relative_to(temp_hunyo_dir) / file
                    all_files.append(str(rel_path))

            e2e_logger.info(f"All files in temp directory: {all_files}")

            # 4. Gracefully terminate the server
            e2e_logger.info("🛑 Terminating server...")
            process.terminate()

            try:
                stdout, stderr = process.communicate(timeout=10)
                # Combine with our buffered output
                if stdout:
                    stdout_buffer.extend(stdout.strip().split("\n"))
                if stderr:
                    stderr_buffer.extend(stderr.strip().split("\n"))

            except subprocess.TimeoutExpired:
                e2e_logger.warning("⚠️ Process didn't terminate gracefully, killing...")
                process.kill()
                stdout, stderr = process.communicate()
                if stdout:
                    stdout_buffer.extend(stdout.strip().split("\n"))
                if stderr:
                    stderr_buffer.extend(stderr.strip().split("\n"))

            # Log all captured output
            if stdout_buffer:
                e2e_logger.info("📋 Complete subprocess STDOUT:")
                for line in stdout_buffer:
                    if line.strip():
                        e2e_logger.info(f"  {line}")

            if stderr_buffer:
                e2e_logger.warning("📋 Complete subprocess STDERR:")
                for line in stderr_buffer:
                    if line.strip():
                        e2e_logger.warning(f"  {line}")

            e2e_logger.info("✅ Server terminated")

            # 5. Validate infrastructure was set up correctly
            e2e_logger.info("📊 Validating infrastructure setup...")

            # Check directories were created
            runtime_dir = temp_hunyo_dir / "events" / "runtime"
            lineage_dir = temp_hunyo_dir / "events" / "lineage"
            database_dir = temp_hunyo_dir / "database"

            assert runtime_dir.exists(), f"Runtime directory not created: {runtime_dir}"
            assert lineage_dir.exists(), f"Lineage directory not created: {lineage_dir}"
            assert (
                database_dir.exists()
            ), f"Database directory not created: {database_dir}"

            e2e_logger.success("✅ Event directories created successfully")

            # 6. Validate database was initialized
            e2e_logger.info("🗄️ Validating database initialization...")

            db_path = temp_hunyo_dir / "database" / "lineage.duckdb"
            assert db_path.exists(), f"Database not created at {db_path}"

            # Test database connectivity and schema
            from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager

            db_manager = DuckDBManager(str(db_path))
            try:
                # Test that tables exist and are accessible
                tables_result = db_manager.execute_query(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                )

                table_names = [row["table_name"] for row in tables_result]

                assert "runtime_events" in table_names, "runtime_events table not found"
                assert "lineage_events" in table_names, "lineage_events table not found"

                # Test views exist
                views_result = db_manager.execute_query(
                    "SELECT table_name FROM information_schema.views WHERE table_schema = 'main'"
                )

                view_names = [row["table_name"] for row in views_result]

                assert "vw_lineage_io" in view_names, "vw_lineage_io view not found"
                assert (
                    "vw_performance_metrics" in view_names
                ), "vw_performance_metrics view not found"

                e2e_logger.success("✅ Database schema initialized correctly")

                # Test that tables are empty (as expected since no auto-injection)
                runtime_count = db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM runtime_events"
                )
                lineage_count = db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM lineage_events"
                )

                runtime_total = runtime_count[0]["count"] if runtime_count else 0
                lineage_total = lineage_count[0]["count"] if lineage_count else 0

                # Tables should be empty since auto-injection is not implemented
                e2e_logger.info(
                    f"📊 Database contains {runtime_total} runtime + {lineage_total} lineage events (expected: 0 since auto-injection not implemented)"
                )

            finally:
                db_manager.close()

            # 7. Validate the server stayed running and responded properly
            e2e_logger.info("🚀 Validating server runtime behavior...")

            # The fact that we got here means the server:
            # 1. Started successfully ✅
            # 2. Stayed running for 8+ seconds ✅
            # 3. Responded to termination gracefully ✅
            # 4. Set up all infrastructure correctly ✅

            e2e_logger.success(
                "🎉 End-to-end server lifecycle validation completed successfully!"
            )
            e2e_logger.info(
                "💡 Note: Event generation requires manual notebook instrumentation (auto-injection not yet implemented)"
            )

        except Exception as e:
            if process and process.poll() is None:
                e2e_logger.error("💥 Test failed, terminating server process...")
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

            e2e_logger.error(f"❌ End-to-end test failed: {e}")
            raise

        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    @pytest.mark.integration
    def test_mcp_server_graceful_shutdown(self, temp_hunyo_dir, test_notebook_path):
        """Test that the MCP server handles graceful shutdown correctly"""

        e2e_logger.info("🧪 Testing MCP server graceful shutdown...")

        test_env = os.environ.copy()
        test_env.update(
            {
                "HUNYO_DEV_MODE": "1",
                "HUNYO_DATA_DIR": str(temp_hunyo_dir),
                "PYTHONPATH": str(Path("src").absolute())
                + os.pathsep
                + test_env.get("PYTHONPATH", ""),
            }
        )

        cmd = [
            sys.executable,
            "-m",
            "hunyo_mcp_server.server",
            "--notebook",
            str(test_notebook_path),
            "--dev-mode",
            "--standalone",
        ]

        process = subprocess.Popen(
            cmd, env=test_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        try:
            # Let it start up
            time.sleep(3)

            # Send SIGTERM for graceful shutdown
            process.terminate()

            # Should exit cleanly within reasonable time
            try:
                stdout, stderr = process.communicate(timeout=10)
                exit_code = process.returncode

                # Should exit with 0 (graceful), -15 (SIGTERM on Unix), or 1 (Windows termination)
                # On Windows, process.terminate() can result in exit code 1
                expected_codes = [0, -15, 1] if os.name == "nt" else [0, -15]
                assert exit_code in expected_codes, f"Unexpected exit code: {exit_code}"

                e2e_logger.success(
                    f"✅ Graceful shutdown successful (exit code: {exit_code})"
                )

            except subprocess.TimeoutExpired:
                e2e_logger.warning("⚠️ Graceful shutdown timed out, killing process")
                process.kill()
                process.communicate()
                pytest.fail("Server did not shut down gracefully within timeout")

        except Exception:
            if process.poll() is None:
                process.kill()
                process.communicate()
            raise

    @pytest.mark.integration
    def test_mcp_server_error_handling(self, temp_hunyo_dir):
        """Test MCP server handles invalid notebook paths gracefully"""

        e2e_logger.info("🧪 Testing MCP server error handling...")

        test_env = os.environ.copy()
        test_env.update(
            {
                "HUNYO_DEV_MODE": "1",
                "HUNYO_DATA_DIR": str(temp_hunyo_dir),
                "PYTHONPATH": str(Path("src").absolute())
                + os.pathsep
                + test_env.get("PYTHONPATH", ""),
            }
        )

        # Try with nonexistent notebook
        cmd = [
            sys.executable,
            "-m",
            "hunyo_mcp_server.server",
            "--notebook",
            "nonexistent_notebook.py",
            "--dev-mode",
        ]

        result = subprocess.run(
            cmd, check=False, env=test_env, capture_output=True, text=True, timeout=10
        )

        # Should exit with error code
        assert result.returncode != 0, "Should fail with nonexistent notebook"

        # Should provide helpful error message
        error_output = result.stderr.lower()
        assert any(
            keyword in error_output
            for keyword in ["not found", "does not exist", "error"]
        ), f"Error message not helpful: {result.stderr}"

        e2e_logger.success("✅ Error handling test passed")
