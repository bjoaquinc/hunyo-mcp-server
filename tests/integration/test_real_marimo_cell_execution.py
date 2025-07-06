#!/usr/bin/env python3
"""
Real Marimo Cell Execution Integration Tests

This module tests ACTUAL marimo cell execution by using Playwright to interact
with the marimo edit interface and properly intercept WebSocket JSON-RPC messages.
This validates that our unified tracker hooks fire correctly during real execution.

Based on research into marimo's actual WebSocket protocol and architecture.
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from hunyo_capture.logger import get_logger

# Playwright for browser automation (marimo's official test framework)
try:
    import os
    import subprocess
    import sys

    from playwright.async_api import async_playwright

    # Check if browsers are actually installed
    def check_playwright_browsers():
        try:
            # Try to check if chromium is available
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "playwright",
                    "install",
                    "--dry-run",
                    "chromium",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            # If dry-run succeeds, browsers should be available
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.SubprocessError,
        ):
            return False

    PLAYWRIGHT_AVAILABLE = check_playwright_browsers()

    # If check failed, try to install browsers automatically in CI
    if not PLAYWRIGHT_AVAILABLE:
        ci_indicators = ["CI", "GITHUB_ACTIONS", "GITLAB_CI"]
        if any(os.getenv(indicator) for indicator in ci_indicators):
            try:
                # Note: test_logger not available here, use print for CI debugging
                print("[PLAYWRIGHT] Attempting to install browsers in CI...")
                result = subprocess.run(
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                PLAYWRIGHT_AVAILABLE = result.returncode == 0
                if PLAYWRIGHT_AVAILABLE:
                    print("[PLAYWRIGHT] Browsers installed successfully")
                else:
                    print(f"[PLAYWRIGHT] Browser installation failed: {result.stderr}")
            except Exception as e:
                print(f"[PLAYWRIGHT] Failed to install browsers: {e}")

except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Create test logger instance
test_logger = get_logger("hunyo.test.real_marimo_execution")


class MarimoEditModeWebSocketTester:
    """Test hooks specifically in marimo EDIT MODE with proper WebSocket interception"""

    def __init__(self, notebook_path: str, marimo_port: int | None = None):
        self.notebook_path = notebook_path
        self.marimo_port = marimo_port or self._find_available_port()
        self.websocket_messages = []
        self.hook_events = []
        self.marimo_process = None

    def _find_available_port(self):
        """Find an available port for marimo server"""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]

        test_logger.info(f"[TEST] Found available port: {port}")
        return port

    def _is_ci_environment(self):
        """Detect if running in CI/CD environment"""
        import os

        ci_indicators = [
            "CI",
            "CONTINUOUS_INTEGRATION",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "JENKINS_URL",
            "BUILDKITE",
            "CIRCLECI",
        ]
        return any(os.getenv(indicator) for indicator in ci_indicators)

    def _get_wait_multiplier(self):
        """Get wait time multiplier based on environment"""
        if self._is_ci_environment():
            test_logger.info(
                "[TEST] CI environment detected - using extended wait times"
            )
            return 1.5  # 50% longer waits in CI
        return 1.0

    async def test_edit_mode_hooks(self):
        """Test hooks specifically in marimo EDIT MODE"""

        # Start marimo in edit mode
        await self._start_marimo_edit_mode()

        # Verify marimo is responding via HTTP
        await self._verify_marimo_server()

        try:
            # Use Playwright for proper browser automation
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()

                # Critical: Set up ACTUAL marimo WebSocket interception
                await self._setup_marimo_websocket_interception(page)

                # Navigate to edit mode interface
                test_logger.info(
                    f"[TEST] Navigating to marimo edit interface: http://localhost:{self.marimo_port}"
                )
                await page.goto(f"http://localhost:{self.marimo_port}")

                # Wait for notebook to load in edit mode
                test_logger.info("[TEST] Waiting for notebook to load...")
                try:
                    await page.wait_for_selector(
                        '[data-testid="cell"], .cell, .marimo-cell', timeout=10000
                    )
                    test_logger.info("[TEST] Notebook loaded successfully")
                except Exception as e:
                    test_logger.warning(f"[TEST] Could not find cell selectors: {e}")
                    # Continue anyway - the important part is WebSocket and console monitoring

                # Wait for a bit to capture any automatic execution
                test_logger.info("[TEST] Monitoring for automatic hook activity...")
                monitoring_wait = int(10 * self._get_wait_multiplier())
                test_logger.info(f"[TEST] Monitoring for {monitoring_wait}s...")
                await asyncio.sleep(monitoring_wait)

                # Try simple execution approach without complex button finding
                await self._simple_execution_approach(page)

                await browser.close()

        finally:
            # Clean up marimo process with Windows-compatible handling
            if self.marimo_process:
                self._cleanup_marimo_process()

    def _cleanup_marimo_process(self):
        """Cleanup marimo process with Windows-compatible handling"""
        import platform
        import time

        if not self.marimo_process:
            return

        try:
            # Try graceful termination first
            if self.marimo_process.poll() is None:  # Process is still running
                test_logger.info("[TEST] Terminating marimo process...")
                self.marimo_process.terminate()

                # Wait for graceful shutdown
                try:
                    self.marimo_process.wait(timeout=5)
                    test_logger.info("[TEST] Marimo process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if terminate didn't work
                    test_logger.warning("[TEST] Forcing marimo process shutdown...")
                    self.marimo_process.kill()

                    # Wait for force kill to complete
                    try:
                        self.marimo_process.wait(timeout=3)
                        test_logger.info("[TEST] Marimo process killed")
                    except subprocess.TimeoutExpired:
                        test_logger.error("[TEST] Failed to kill marimo process")

            # Additional Windows-specific cleanup delay
            if platform.system() == "Windows":
                time.sleep(1.0)  # Give Windows time to release file handles

        except Exception as e:
            test_logger.warning(f"[TEST] Error during marimo process cleanup: {e}")
        finally:
            self.marimo_process = None

    async def _start_marimo_edit_mode(self):
        """Start marimo in edit mode"""
        test_logger.info("[TEST] Starting marimo edit mode...")

        # Set environment variables for headless mode
        import os

        env = os.environ.copy()
        env["MARIMO_HEADLESS"] = "1"
        env["DISPLAY"] = ""

        self.marimo_process = subprocess.Popen(
            [
                "marimo",
                "edit",
                str(self.notebook_path),
                "--headless",
                "--no-token",
                "--port",
                str(self.marimo_port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # Wait for marimo to start and capture output
        wait_multiplier = self._get_wait_multiplier()
        startup_wait = int(5 * wait_multiplier)
        test_logger.info(f"[TEST] Waiting {startup_wait}s for marimo to start...")
        await asyncio.sleep(startup_wait)

        # Check if marimo is actually running
        if self.marimo_process.poll() is not None:
            stdout, stderr = self.marimo_process.communicate()
            test_logger.error(
                f"[TEST] Marimo process exited early! Exit code: {self.marimo_process.returncode}"
            )
            test_logger.error(f"[TEST] STDOUT: {stdout}")
            test_logger.error(f"[TEST] STDERR: {stderr}")
            error_msg = "Marimo process failed to start"
            raise RuntimeError(error_msg)

        test_logger.info(f"[TEST] Marimo edit mode started on port {self.marimo_port}")

        # Marimo is running successfully, no need to read stdout which can block
        test_logger.info("[TEST] Marimo process is running")

    async def _verify_marimo_server(self):
        """Verify marimo server is responding via HTTP"""
        import aiohttp

        test_logger.info("[TEST] Verifying marimo server is responding...")

        for attempt in range(5):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{self.marimo_port}", timeout=5
                    ) as response:
                        if response.status == 200:
                            test_logger.info(
                                f"[TEST] Marimo server responding! Status: {response.status}"
                            )
                            return
                        else:
                            test_logger.warning(
                                f"[TEST] Marimo server returned status: {response.status}"
                            )
            except Exception as e:
                test_logger.warning(
                    f"[TEST] Attempt {attempt + 1}: Marimo server not responding: {e}"
                )
                retry_wait = int(2 * self._get_wait_multiplier())
                await asyncio.sleep(retry_wait)

        error_msg = "Marimo server is not responding after multiple attempts"
        raise RuntimeError(error_msg)

    async def _setup_marimo_websocket_interception(self, page):
        """Set up WebSocket interception based on actual marimo protocol"""

        def handle_websocket(websocket):
            test_logger.info(f"[WS] WebSocket connected: {websocket.url}")

            # Monitor frames - this is where cell execution events flow
            def on_frame_sent(payload):
                task = asyncio.create_task(self._process_marimo_frame("SENT", payload))
                # Store task reference to prevent garbage collection
                self._websocket_tasks = getattr(self, "_websocket_tasks", set())
                self._websocket_tasks.add(task)
                task.add_done_callback(self._websocket_tasks.discard)

            def on_frame_received(payload):
                task = asyncio.create_task(
                    self._process_marimo_frame("RECEIVED", payload)
                )
                # Store task reference to prevent garbage collection
                self._websocket_tasks = getattr(self, "_websocket_tasks", set())
                self._websocket_tasks.add(task)
                task.add_done_callback(self._websocket_tasks.discard)

            websocket.on("framesent", on_frame_sent)
            websocket.on("framereceived", on_frame_received)

        page.on("websocket", handle_websocket)

        # Set up console monitoring for hook outputs
        def handle_console(msg):
            self._capture_hook_console_output(msg)

        page.on("console", handle_console)

    async def _process_marimo_frame(self, direction: str, payload):
        """Process WebSocket frames for marimo cell execution detection"""
        try:
            if payload:
                data = json.loads(payload) if isinstance(payload, str) else payload

                if self._is_marimo_cell_execution(data):
                    execution_event = {
                        "direction": direction,
                        "execution_type": self._extract_execution_type(data),
                        "data": data,
                        "timestamp": time.time(),
                    }

                    self.websocket_messages.append(execution_event)
                    test_logger.info(
                        f"[WS] Cell execution detected: {execution_event['execution_type']}"
                    )

        except Exception as e:
            test_logger.warning(f"[WS] Failed to process frame: {e}")

    def _is_marimo_cell_execution(self, data: dict) -> bool:
        """Determine if WebSocket data represents cell execution"""
        if not isinstance(data, dict):
            return False

        # Check for marimo's JSON-RPC patterns
        has_jsonrpc = "jsonrpc" in data
        has_method = "method" in data

        if not (has_jsonrpc or has_method):
            return False

        # Check for execution-related methods
        method = data.get("method", "")
        execution_keywords = [
            "execute",
            "run",
            "cell",
            "kernel",
            "notebook",
            "code",
        ]

        return any(keyword in method.lower() for keyword in execution_keywords)

    def _extract_execution_type(self, data: dict) -> str:
        """Extract the type of execution from WebSocket data"""
        method = data.get("method", "unknown")
        params = data.get("params", {})

        if "execute" in method:
            return "cell_execute"
        elif "run" in method:
            return "cell_run"
        elif params and "cell_id" in params:
            return "cell_operation"
        else:
            return "unknown_execution"

    def _capture_hook_console_output(self, msg):
        """Capture console output that might contain hook information"""
        try:
            text = msg.text

            # Look for hook-related console output
            hook_keywords = ["hook", "tracker", "interceptor", "marimo", "unified"]

            if any(keyword.lower() in text.lower() for keyword in hook_keywords):
                hook_event = {
                    "type": "hook_console",
                    "text": text,
                    "timestamp": time.time(),
                }

                self.hook_events.append(hook_event)
                test_logger.info(f"[HOOK] Console output: {text}")

        except Exception as e:
            test_logger.warning(f"[HOOK] Failed to capture console output: {e}")

    async def _simple_execution_approach(self, page):
        """Try simple execution approach without complex button finding"""
        test_logger.info("[TEST] Trying simple execution approach...")

        # Just try one simple keyboard shortcut and finish
        try:
            await asyncio.wait_for(
                page.keyboard.press("Control+Shift+Enter"), timeout=3.0
            )
            test_logger.info("[TEST] Tried keyboard shortcut")
        except Exception as e:
            test_logger.warning(f"[TEST] Keyboard shortcut failed: {e}")

        # Wait a bit for any outputs
        final_wait = int(3 * self._get_wait_multiplier())
        test_logger.info(f"[TEST] Final wait for {final_wait}s...")
        await asyncio.sleep(final_wait)

        test_logger.info(
            f"[TEST] Captured {len(self.websocket_messages)} WebSocket messages"
        )
        test_logger.info(f"[TEST] Captured {len(self.hook_events)} hook events")

    def correlate_execution_with_hooks(self):
        """Correlate WebSocket execution events with hook events"""
        correlations = []

        for ws_event in self.websocket_messages:
            execution_time = ws_event["timestamp"]

            # Find hook events within 10 seconds
            related_hooks = [
                hook
                for hook in self.hook_events
                if abs(hook["timestamp"] - execution_time) < 10.0
            ]

            correlation = {
                "execution_event": ws_event,
                "hook_events": related_hooks,
                "correlation_strength": len(related_hooks),
                "timing_accuracy": self._calculate_timing_accuracy(
                    ws_event, related_hooks
                ),
            }

            correlations.append(correlation)

            if related_hooks:
                test_logger.info(
                    f"[CORRELATION] Cell execution triggered {len(related_hooks)} hooks"
                )
            else:
                test_logger.warning(
                    "[CORRELATION] Cell execution but no hooks detected"
                )

        return correlations

    def _calculate_timing_accuracy(self, ws_event, hook_events):
        """Calculate timing accuracy of hook firing"""
        if not hook_events:
            return 0.0

        execution_time = ws_event["timestamp"]
        avg_hook_delay = sum(
            hook["timestamp"] - execution_time for hook in hook_events
        ) / len(hook_events)

        return max(0.0, 10.0 - abs(avg_hook_delay)) / 10.0  # Normalize to 0-1


class TestRealMarimoCellExecution:
    """Test real marimo cell execution through marimo's execution pipeline"""

    @pytest.fixture
    def runtime_events_schema(self):
        """Load runtime events schema for validation"""
        schema_path = Path("schemas/json/runtime_events_schema.json")
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def openlineage_events_schema(self):
        """Load OpenLineage events schema for validation"""
        schema_path = Path("schemas/json/openlineage_events_schema.json")
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    def validate_event_against_schema(
        self, event: dict[str, Any], schema: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate a single event against a JSON schema"""
        try:
            jsonschema.validate(event, schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)

    @pytest.mark.integration
    @pytest.mark.playwright
    @pytest.mark.asyncio
    @pytest.mark.timeout(90)  # Extended timeout for CI/CD environments (90 seconds)
    @pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not available")
    async def test_marimo_edit_mode_hooks_with_playwright(
        self, temp_hunyo_dir, runtime_events_schema, openlineage_events_schema
    ):
        """Test hooks firing during real marimo edit mode execution"""

        # Change to temp directory for event files
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(temp_hunyo_dir)

            test_logger.info("[TEST] Starting marimo edit mode hook testing")

            # Create a marimo notebook file with our test cells
            notebook_content = '''#!/usr/bin/env python3
import marimo

app = marimo.App()

@app.cell
def setup_tracking():
    """Set up unified tracking system"""
    import sys
    from pathlib import Path

    # Add src to path for imports
    project_root = Path.cwd().parent.parent  # Go up from temp dir
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Import and enable unified tracking
            from hunyo_capture.unified_marimo_interceptor import enable_unified_tracking

    # Enable tracking with specific files
    tracker = enable_unified_tracking(
        runtime_file="test_runtime_events.jsonl",
        lineage_file="test_lineage_events.jsonl"
    )

    print(f"[SETUP] Tracker installed: {tracker.session_id}")
    print(f"[SETUP] Tracker active: {tracker.interceptor_active}")

    return tracker

@app.cell
def create_dataframes(tracker):
    """Create DataFrames to trigger lineage events"""
    import pandas as pd

    print("[CREATE] Creating DataFrames...")

    # Create DataFrames - should trigger lineage events
    df1 = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    })

    df2 = pd.DataFrame({
        'id': [1, 2, 3],
        'department': ['Engineering', 'Marketing', 'Sales']
    })

    print(f"[CREATE] Created df1 shape: {df1.shape}")
    print(f"[CREATE] Created df2 shape: {df2.shape}")

    return df1, df2

@app.cell
def process_dataframes(df1, df2):
    """Process DataFrames to trigger more events"""
    print("[PROCESS] Processing DataFrames...")

    # Merge DataFrames - should trigger lineage events
    merged_df = df1.merge(df2, on='id', how='left')
    print(f"[PROCESS] Merged shape: {merged_df.shape}")

    return merged_df

if __name__ == "__main__":
    app.run()
'''

            # Write notebook file
            notebook_path = temp_hunyo_dir / "test_notebook.py"
            notebook_path.write_text(notebook_content)

            test_logger.info(f"[TEST] Created notebook: {notebook_path}")

            # Use the corrected WebSocket tester
            tester = MarimoEditModeWebSocketTester(notebook_path)

            # Run the test
            await tester.test_edit_mode_hooks()

            # Correlate results
            correlations = tester.correlate_execution_with_hooks()

            # Analyze results
            test_logger.info(
                f"[ANALYSIS] Found {len(correlations)} execution-hook correlations"
            )

            # Check for event files
            runtime_file = temp_hunyo_dir / "test_runtime_events.jsonl"
            lineage_file = temp_hunyo_dir / "test_lineage_events.jsonl"

            test_logger.info(f"[TEST] Checking for runtime events at: {runtime_file}")
            test_logger.info(f"[TEST] Checking for lineage events at: {lineage_file}")

            # List all files in temp dir for debugging
            test_logger.info("[TEST] Files in temp dir:")
            for f in temp_hunyo_dir.iterdir():
                test_logger.info(f"  {f.name}")

            # Validate events if they exist
            events_found = False

            if runtime_file.exists():
                test_logger.info("[TEST] Runtime events file found! Validating...")
                with open(runtime_file) as f:
                    runtime_events = [
                        json.loads(line.strip()) for line in f if line.strip()
                    ]

                test_logger.info(f"[TEST] Found {len(runtime_events)} runtime events")

                # Validate against schema
                for event in runtime_events:
                    jsonschema.validate(event, runtime_events_schema)

                events_found = True
                test_logger.info("[TEST] Runtime events validation passed!")

            if lineage_file.exists():
                test_logger.info("[TEST] Lineage events file found! Validating...")
                with open(lineage_file) as f:
                    lineage_events = [
                        json.loads(line.strip()) for line in f if line.strip()
                    ]

                test_logger.info(f"[TEST] Found {len(lineage_events)} lineage events")

                # Validate against schema
                for event in lineage_events:
                    jsonschema.validate(event, openlineage_events_schema)

                events_found = True
                test_logger.info("[TEST] Lineage events validation passed!")

            # Assert that we found events
            assert (
                events_found
            ), "No events were generated - hook system may not be working"

            # Validate WebSocket vs Hook activity
            has_websocket_activity = len(tester.websocket_messages) > 0
            has_hook_activity = len(tester.hook_events) > 0
            has_file_events = events_found

            test_logger.info(f"[RESULTS] WebSocket activity: {has_websocket_activity}")
            test_logger.info(f"[RESULTS] Hook activity: {has_hook_activity}")
            test_logger.info(f"[RESULTS] File events: {has_file_events}")

            # The most important validation is that we have file events
            # (WebSocket may not capture everything, but file events prove hooks work)
            assert (
                has_file_events
            ), "File events are required to prove hook system works"

            test_logger.info("[SUCCESS] Real marimo execution with hooks validated!")

        finally:
            # Change back to original directory
            os.chdir(original_cwd)
