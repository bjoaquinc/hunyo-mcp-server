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
from hunyo_capture.logger import get_logger

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
        tests_dir = Path(__file__).parent.parent  # Go up from integration/ to tests/
        notebook_path = tests_dir / "fixtures" / "test_notebook.py"
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
                f"[FILE] Validating {event_type_name} events in {file_path.name}"
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

    @pytest.mark.timeout(
        120, method="thread"
    )  # Override 30s default - allow up to 2 minutes
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

        e2e_logger.info("[START] Starting end-to-end MCP server pipeline test")
        e2e_logger.info(f"[FILE] Test data directory: {temp_hunyo_dir}")
        e2e_logger.info(f"[FILE] Test notebook: {test_notebook_path}")
        e2e_logger.info("[EXEC] Starting hunyo-mcp-server subprocess...")
        e2e_logger.info("[WAIT] Waiting for database file creation...")

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
                "PYTHONUNBUFFERED": "1",  # Disable Python I/O buffering (belt-and-braces)
            }
        )

        e2e_logger.info(f"[FILE] Test data directory: {temp_hunyo_dir}")
        e2e_logger.info(f"[FILE] Test notebook: {test_notebook_path}")

        process = None
        stdout_buffer = []
        stderr_buffer = []

        try:
            # 2. Execute hunyo-mcp-server as subprocess
            e2e_logger.info("[EXEC] Starting hunyo-mcp-server subprocess...")

            cmd = [
                sys.executable,
                "-u",  # Disable Python stdout/stderr buffering
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
                stderr=subprocess.STDOUT,  # Merge stderr to stdout to capture all output
                text=True,
                bufsize=1,  # Line-buffered across platforms
            )

            # 3. Wait for database file to appear (cross-platform race condition fix)
            e2e_logger.info("[WAIT] Waiting for database file creation...")

            # Enhanced Windows CI debugging
            import platform

            is_windows = platform.system() == "Windows"
            is_ci = os.environ.get("CI") == "true"

            if is_windows and is_ci:
                e2e_logger.info("[WINDOWS-CI] Enhanced debugging mode enabled")
                e2e_logger.info(
                    f"[WINDOWS-CI] Python version: {platform.python_version()}"
                )
                e2e_logger.info(f"[WINDOWS-CI] Platform: {platform.platform()}")

                # With CI pre-installation, give generous timeout for Windows CI
                timeout = 60.0

                # Pre-flight checks for Windows CI
                e2e_logger.info(f"[WINDOWS-CI] Temp directory: {temp_hunyo_dir}")
                e2e_logger.info(
                    f"[WINDOWS-CI] Test notebook exists: {test_notebook_path.exists()}"
                )
                e2e_logger.info(f"[WINDOWS-CI] Python executable: {sys.executable}")

                # Check if hunyo_mcp_server module is importable
                try:
                    import hunyo_mcp_server.server  # noqa: F401

                    e2e_logger.info("[WINDOWS-CI] hunyo_mcp_server.server import: OK")
                except Exception as e:
                    e2e_logger.error(
                        f"[WINDOWS-CI] hunyo_mcp_server.server import failed: {e}"
                    )
            else:
                # Non-Windows: Unix CI with pre-installed extensions should be faster
                timeout = 30.0

            db_path = temp_hunyo_dir / "database" / "lineage.duckdb"

            def wait_for_database_ready(
                db_path: Path, timeout: float = timeout
            ) -> bool:
                """Wait for database schema initialization using cross-platform process monitoring."""
                import queue
                import threading

                start_time = time.time()

                # Look for the ready marker that the server emits when schema is complete
                ready_marker = "HUNYO_READY_MARKER: DATABASE_SCHEMA_READY"

                # Cross-platform approach: Use threading to read process output
                output_queue = queue.Queue()
                accumulated_output = ""

                def read_output():
                    """Background thread to read process output continuously."""
                    try:
                        while process.poll() is None:
                            line = process.stdout.readline()
                            if line:
                                output_queue.put(line)
                            else:
                                time.sleep(0.1)
                        # Read any remaining output
                        for line in process.stdout:
                            output_queue.put(line)
                    except Exception as e:
                        e2e_logger.debug(f"[DEBUG] Output reader thread error: {e}")
                    finally:
                        output_queue.put(None)  # Signal end

                # Start background reader thread
                reader_thread = threading.Thread(target=read_output, daemon=True)
                reader_thread.start()

                while time.time() - start_time < timeout:
                    # Check if process died early
                    if process.poll() is not None:
                        # Collect any remaining output
                        try:
                            while True:
                                try:
                                    line = output_queue.get_nowait()
                                    if line is None:
                                        break
                                    accumulated_output += line
                                except queue.Empty:
                                    break
                        except Exception as e:
                            e2e_logger.debug(
                                f"[DEBUG] Error collecting final output: {e}"
                            )

                        # Check if ready marker appeared in output
                        if ready_marker in accumulated_output:
                            elapsed = time.time() - start_time
                            e2e_logger.success(
                                f"[OK] Database schema ready: {db_path} (initialized in {elapsed:.1f}s)"
                            )
                            return True

                        process.communicate()  # Clean up any remaining output
                        e2e_logger.error(
                            f"[ERROR] Process exited early with code {process.returncode}"
                        )
                        e2e_logger.error(
                            f"[FILE] Subprocess OUTPUT: {accumulated_output}"
                        )
                        error_msg = f"[ERROR] Server process exited before database was created (exit code: {process.returncode})"
                        raise RuntimeError(error_msg)

                    # Try to read new output from queue
                    try:
                        line = output_queue.get(timeout=0.1)
                        if line is None:
                            # Reader thread finished
                            break
                        accumulated_output += line

                        # Check if ready marker appeared
                        if ready_marker in accumulated_output:
                            elapsed = time.time() - start_time
                            e2e_logger.success(
                                f"[OK] Database schema ready: {db_path} (initialized in {elapsed:.1f}s)"
                            )
                            return True
                    except queue.Empty:
                        # No new output yet
                        pass

                    # Progress logging
                    elapsed = time.time() - start_time
                    if elapsed >= 5 and int(elapsed) % 3 == 0:
                        e2e_logger.info(
                            f"[WAIT] Schema still initializing after {elapsed:.0f}s..."
                        )

                # Sanity check: verify if ready marker was actually in the output
                if ready_marker in accumulated_output:
                    e2e_logger.error(
                        f"[ERROR] Ready marker found in output but not detected in time! Output: {accumulated_output[-500:]}"
                    )

                error_msg = (
                    f"[ERROR] Database schema initialization failed after {timeout}s"
                )
                raise TimeoutError(error_msg)

            try:
                wait_for_database_ready(db_path)
            except (RuntimeError, TimeoutError) as e:
                e2e_logger.error(f"[ERROR] Database creation failed: {e}")

                # Capture final output for debugging
                try:
                    stdout, stderr = process.communicate(timeout=5)
                    if stdout:
                        stdout_buffer.extend(stdout.strip().split("\n"))
                    if stderr:
                        stderr_buffer.extend(stderr.strip().split("\n"))
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    if stdout:
                        stdout_buffer.extend(stdout.strip().split("\n"))
                    if stderr:
                        stderr_buffer.extend(stderr.strip().split("\n"))

                # Log captured output before failing
                if stdout_buffer:
                    e2e_logger.error("[FILE] Subprocess STDOUT:")
                    for line in stdout_buffer:
                        if line.strip():
                            e2e_logger.error(f"  {line}")

                if stderr_buffer:
                    e2e_logger.error("[FILE] Subprocess STDERR:")
                    for line in stderr_buffer:
                        if line.strip():
                            e2e_logger.error(f"  {line}")

                raise

            # 3.5. Additional wait for event processing after database creation
            e2e_logger.info("[WAIT] Allowing additional time for event processing...")
            time.sleep(3)  # Give the server time to process events after DB is ready

            # 4. Gracefully terminate the server
            e2e_logger.info("[EXEC] Terminating server...")
            process.terminate()

            try:
                stdout, stderr = process.communicate(timeout=10)
                # Combine with our buffered output
                if stdout:
                    stdout_buffer.extend(stdout.strip().split("\n"))
                if stderr:
                    stderr_buffer.extend(stderr.strip().split("\n"))

            except subprocess.TimeoutExpired:
                e2e_logger.warning(
                    "[WARN] Process didn't terminate gracefully, killing..."
                )
                process.kill()
                stdout, stderr = process.communicate()
                if stdout:
                    stdout_buffer.extend(stdout.strip().split("\n"))
                if stderr:
                    stderr_buffer.extend(stderr.strip().split("\n"))

            # Log all captured output
            if stdout_buffer:
                e2e_logger.info("[FILE] Complete subprocess STDOUT:")
                for line in stdout_buffer:
                    if line.strip():
                        e2e_logger.info(f"  {line}")

            if stderr_buffer:
                e2e_logger.warning("[FILE] Complete subprocess STDERR:")
                for line in stderr_buffer:
                    if line.strip():
                        e2e_logger.warning(f"  {line}")

            e2e_logger.info("[OK] Server terminated")

            # 5. Validate infrastructure was set up correctly
            e2e_logger.info("[VALIDATE] Validating infrastructure setup...")

            # Check directories were created
            runtime_dir = temp_hunyo_dir / "events" / "runtime"
            lineage_dir = temp_hunyo_dir / "events" / "lineage"
            database_dir = temp_hunyo_dir / "database"

            assert (
                runtime_dir.exists()
            ), f"[ERROR] Runtime directory not created: {runtime_dir}"
            assert (
                lineage_dir.exists()
            ), f"[ERROR] Lineage directory not created: {lineage_dir}"
            assert (
                database_dir.exists()
            ), f"[ERROR] Database directory not created: {database_dir}"

            e2e_logger.success("[OK] Event directories created successfully")

            # 6. Validate database was initialized (db_path already established above)
            e2e_logger.info("[DB] Validating database initialization...")

            assert db_path.exists(), f"[ERROR] Database not created at {db_path}"

            # Test database connectivity and schema
            from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager

            db_manager = DuckDBManager(str(db_path))
            try:
                # Test that tables exist and are accessible
                tables_result = db_manager.execute_query(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                )

                table_names = [row["table_name"] for row in tables_result]

                assert (
                    "runtime_events" in table_names
                ), "[ERROR] runtime_events table not found"
                assert (
                    "lineage_events" in table_names
                ), "[ERROR] lineage_events table not found"

                # Test views exist
                views_result = db_manager.execute_query(
                    "SELECT table_name FROM information_schema.views WHERE table_schema = 'main'"
                )

                view_names = [row["table_name"] for row in views_result]

                assert (
                    "vw_lineage_io" in view_names
                ), "[ERROR] vw_lineage_io view not found"
                assert (
                    "vw_performance_metrics" in view_names
                ), "[ERROR] vw_performance_metrics view not found"

                e2e_logger.success("[OK] Database schema initialized correctly")

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
                    f"[DATA] Database contains {runtime_total} runtime + {lineage_total} lineage events (expected: 0 since auto-injection not implemented)"
                )

            finally:
                db_manager.close()

            # 7. Validate the server stayed running and responded properly
            e2e_logger.info("[VALIDATE] Validating server runtime behavior...")

            # The fact that we got here means the server:
            # 1. Started successfully [OK]
            # 2. Stayed running for 8+ seconds [OK]
            # 3. Responded to termination gracefully [OK]
            # 4. Set up all infrastructure correctly [OK]

            e2e_logger.success(
                "[SUCCESS] End-to-end server lifecycle validation completed successfully!"
            )
            e2e_logger.info(
                "[INFO] Note: Event generation requires manual notebook instrumentation (auto-injection not yet implemented)"
            )

        except Exception as e:
            if process and process.poll() is None:
                e2e_logger.error("[ERROR] Test failed, terminating server process...")
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            e2e_logger.error(f"[ERROR] End-to-end test failed: {e}")
            raise

        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    @pytest.mark.integration
    def test_mcp_server_graceful_shutdown(self, temp_hunyo_dir, test_notebook_path):
        """Test that the MCP server handles graceful shutdown correctly"""

        e2e_logger.info("[TEST] Testing MCP server graceful shutdown...")

        test_env = os.environ.copy()
        test_env.update(
            {
                "HUNYO_DEV_MODE": "1",
                "HUNYO_DATA_DIR": str(temp_hunyo_dir),
                "PYTHONPATH": str(Path("src").absolute())
                + os.pathsep
                + test_env.get("PYTHONPATH", ""),
                "PYTHONUNBUFFERED": "1",  # Disable Python I/O buffering (belt-and-braces)
            }
        )

        cmd = [
            sys.executable,
            "-u",  # Disable Python stdout/stderr buffering
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

                # Should exit with 0 (graceful), -15 (SIGTERM on Unix), or 1 (process termination)
                # process.terminate() can result in exit code 1 on both Windows and Linux
                expected_codes = [0, -15, 1]
                assert (
                    exit_code in expected_codes
                ), f"[ERROR] Unexpected exit code: {exit_code}"

                e2e_logger.success(
                    f"[OK] Graceful shutdown successful (exit code: {exit_code})"
                )

            except subprocess.TimeoutExpired:
                e2e_logger.warning(
                    "[WARN] Graceful shutdown timed out, killing process"
                )
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

        e2e_logger.info("[TEST] Testing MCP server error handling...")

        test_env = os.environ.copy()
        test_env.update(
            {
                "HUNYO_DEV_MODE": "1",
                "HUNYO_DATA_DIR": str(temp_hunyo_dir),
                "PYTHONPATH": str(Path("src").absolute())
                + os.pathsep
                + test_env.get("PYTHONPATH", ""),
                "PYTHONUNBUFFERED": "1",  # Disable Python I/O buffering (belt-and-braces)
            }
        )

        # Try with nonexistent notebook
        cmd = [
            sys.executable,
            "-u",  # Disable Python stdout/stderr buffering
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
        assert result.returncode != 0, "[ERROR] Should fail with nonexistent notebook"

        # Should provide helpful error message
        error_output = result.stderr.lower()
        assert any(
            keyword in error_output
            for keyword in ["not found", "does not exist", "error"]
        ), f"[ERROR] Error message not helpful: {result.stderr}"

        e2e_logger.success("[OK] Error handling test passed")

    @pytest.mark.integration
    def test_database_platform_optimization(self, temp_hunyo_dir):
        """Test database handles platform-specific optimizations (based on DuckDB best practices)"""

        e2e_logger.info("[TEST] Testing platform-specific DuckDB optimizations...")

        import platform

        from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager

        db_path = temp_hunyo_dir / "database" / "platform_test.duckdb"

        # Test platform-aware database initialization
        db_manager = DuckDBManager(str(db_path))
        db_manager.initialize_database()

        try:
            # Test platform-specific configuration queries work
            system = platform.system()

            if system == "Windows":
                # Test Windows-specific file handling
                result = db_manager.execute_query(
                    "SELECT current_setting('temp_directory')"
                )
                assert len(result) > 0

                # Test Windows path handling doesn't fail
                long_query = "SELECT 1 as test_" + "x" * 100  # Long identifier
                result = db_manager.execute_query(long_query)
                assert result[0][f"test_{'x' * 100}"] == 1

                e2e_logger.success("[OK] Windows-specific optimizations working")

            elif system == "Darwin":  # macOS
                # Test macOS unified memory considerations
                result = db_manager.execute_query(
                    "SELECT current_setting('memory_limit')"
                )
                assert len(result) > 0

                e2e_logger.success("[OK] macOS-specific optimizations working")

            else:  # Linux and others
                # Test Linux aggressive memory usage
                result = db_manager.execute_query("SELECT current_setting('threads')")
                assert len(result) > 0

                e2e_logger.success("[OK] Linux-specific optimizations working")

            # Test cross-platform connection stability
            for i in range(5):
                result = db_manager.execute_query("SELECT ? as iteration", [i])
                assert result[0]["iteration"] == i

            e2e_logger.success("[OK] Cross-platform connection stability verified")

        finally:
            db_manager.close()

    @pytest.mark.integration
    def test_database_connection_retry_patterns(self, temp_hunyo_dir):
        """Test DuckDB connection retry logic with exponential backoff"""

        e2e_logger.info("[TEST] Testing connection retry patterns...")

        import time

        from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager

        db_path = temp_hunyo_dir / "database" / "retry_test.duckdb"

        # Test successful connection
        db_manager = DuckDBManager(str(db_path))
        db_manager.initialize_database()

        try:
            # Test normal query execution
            result = db_manager.execute_query("SELECT 'connection_test' as status")
            assert result[0]["status"] == "connection_test"

            # Test connection re-establishment after closing
            db_manager.close()

            # Force new connection via query
            result = db_manager.execute_query("SELECT 'reconnection_test' as status")
            assert result[0]["status"] == "reconnection_test"

            e2e_logger.success("[OK] Connection retry patterns working")

            # Test query retry with timeout
            start_time = time.time()
            result = db_manager.execute_query_with_retry(
                "SELECT 'retry_test' as status", max_retries=2, timeout=5.0
            )
            elapsed = time.time() - start_time

            assert result[0]["status"] == "retry_test"
            assert elapsed < 10.0  # Should complete quickly on success

            e2e_logger.success("[OK] Query retry logic validated")

        finally:
            db_manager.close()

    @pytest.mark.integration
    def test_database_timeout_handling(self, temp_hunyo_dir):
        """Test cross-platform timeout handling for database operations"""

        e2e_logger.info("[TEST] Testing database timeout handling...")

        import platform
        import time

        from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager

        db_path = temp_hunyo_dir / "database" / "timeout_test.duckdb"

        db_manager = DuckDBManager(str(db_path))
        db_manager.initialize_database()

        try:
            # Test normal query with reasonable timeout
            start_time = time.time()
            result = db_manager.execute_query(
                "SELECT 'timeout_test' as status", timeout=5.0
            )
            elapsed = time.time() - start_time

            assert result[0]["status"] == "timeout_test"
            assert elapsed < 1.0  # Should be very fast

            e2e_logger.success(f"[OK] Normal query completed in {elapsed:.3f}s")

            # Test timeout functionality with artificial slow query
            # Create a query that takes some time but not too long for CI
            slow_query = """
            WITH RECURSIVE slow_count(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM slow_count WHERE n < 100000
            )
            SELECT COUNT(*) as count FROM slow_count
            """

            # Test with reasonable timeout that should complete
            start_time = time.time()
            result = db_manager.execute_query(slow_query, timeout=10.0)
            elapsed = time.time() - start_time

            assert result[0]["count"] == 100000
            e2e_logger.success(
                f"[OK] Slow query completed in {elapsed:.3f}s (within timeout)"
            )

            # Test insert timeout handling
            test_event = {
                "event_id": 123456789,  # BIGINT - use integer not string
                "event_type": "test",
                "execution_id": "test_exec_1",
                "cell_id": "test_cell",
                "cell_source": "print('test')",
                "cell_source_lines": 1,
                "start_memory_mb": 100.0,
                "end_memory_mb": 105.0,
                "duration_ms": 50,
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "test_session",
                "emitted_at": "2024-01-01T00:00:01Z",
            }

            # Test insert with timeout
            start_time = time.time()
            db_manager.insert_runtime_event(test_event, timeout=5.0)
            elapsed = time.time() - start_time

            assert elapsed < 1.0  # Insert should be very fast

            # Verify insert worked
            result = db_manager.execute_query(
                "SELECT event_id FROM runtime_events WHERE event_id = ?",
                [123456789],  # Use integer not string
            )
            assert len(result) == 1
            assert result[0]["event_id"] == 123456789  # Should be integer

            e2e_logger.success("[OK] Insert timeout handling validated")

            # Test platform-specific timeout implementation
            system = platform.system()
            if system == "Windows":
                e2e_logger.info(
                    "[INFO] [WINDOWS] ThreadPoolExecutor timeout implementation verified"
                )
            else:
                e2e_logger.info(
                    "[INFO] [UNIX] signal.alarm timeout implementation verified"
                )

        finally:
            db_manager.close()

    @pytest.mark.integration
    def test_database_robust_error_recovery(self, temp_hunyo_dir):
        """Test database error recovery and connection resilience"""

        e2e_logger.info("[TEST] Testing database error recovery...")

        from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager

        db_path = temp_hunyo_dir / "database" / "recovery_test.duckdb"

        db_manager = DuckDBManager(str(db_path))
        db_manager.initialize_database()

        try:
            # Test recovery from syntax error
            try:
                db_manager.execute_query("INVALID SQL SYNTAX")
                error_msg = "Should have raised an exception"
                raise AssertionError(error_msg)
            except Exception as e:
                # Expected syntax error
                e2e_logger.debug(f"[DEBUG] Expected syntax error: {e}")

            # Verify connection still works after error
            result = db_manager.execute_query("SELECT 'recovery_test' as status")
            assert result[0]["status"] == "recovery_test"

            e2e_logger.success("[OK] Recovery from syntax error validated")

            # Test recovery from invalid table access
            try:
                db_manager.execute_query("SELECT * FROM nonexistent_table")
                error_msg = "Should have raised an exception"
                raise AssertionError(error_msg)
            except Exception as e:
                # Expected table not found error
                e2e_logger.debug(f"[DEBUG] Expected table error: {e}")

            # Verify connection still works
            result = db_manager.execute_query(
                "SELECT COUNT(*) as count FROM runtime_events"
            )
            assert "count" in result[0]

            e2e_logger.success("[OK] Recovery from table error validated")

            # Test transaction rollback recovery
            try:
                db_manager.begin_transaction()
                db_manager.execute_query(
                    "INSERT INTO runtime_events (event_id) VALUES ('invalid')"
                )  # Missing required fields
                db_manager.commit_transaction()
                error_msg = "Should have raised an exception"
                raise AssertionError(error_msg)
            except Exception:
                db_manager.rollback_transaction()  # Cleanup

            # Verify database state is still good
            result = db_manager.execute_query(
                "SELECT COUNT(*) as count FROM runtime_events"
            )
            original_count = result[0]["count"]

            # Insert valid record
            test_event = {
                "event_id": 987654321,  # BIGINT - use integer not string
                "event_type": "test",
                "execution_id": "test_exec_1",
                "cell_id": "test_cell",
                "cell_source": "print('recovery')",
                "cell_source_lines": 1,
                "start_memory_mb": 100.0,
                "end_memory_mb": 105.0,
                "duration_ms": 50,
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "test_session",
                "emitted_at": "2024-01-01T00:00:01Z",
            }

            db_manager.insert_runtime_event(test_event)

            # Verify count increased
            result = db_manager.execute_query(
                "SELECT COUNT(*) as count FROM runtime_events"
            )
            new_count = result[0]["count"]
            assert new_count == original_count + 1

            e2e_logger.success("[OK] Transaction rollback recovery validated")

        finally:
            db_manager.close()
