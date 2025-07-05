from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockConfig:
    """Mock configuration object for testing"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.events_dir = data_dir / "events"
        self.db_dir = data_dir / "database"
        self.config_dir = data_dir / "config"

        # Ensure directories exist
        for dir_path in [self.events_dir, self.db_dir, self.config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def __truediv__(self, other):
        """Support path operations with / operator"""
        return self.data_dir / other


@pytest.fixture
def temp_hunyo_dir() -> Generator[Path, None, None]:
    """Provides a temporary .hunyo directory for testing"""
    import platform
    import shutil
    import time

    temp_dir = tempfile.mkdtemp()
    hunyo_dir = Path(temp_dir) / ".hunyo"
    hunyo_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (hunyo_dir / "events").mkdir(exist_ok=True)
    (hunyo_dir / "database").mkdir(exist_ok=True)
    (hunyo_dir / "config").mkdir(exist_ok=True)

    try:
        yield hunyo_dir
    finally:
        # Windows-specific cleanup with retry logic
        if platform.system() == "Windows":
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    shutil.rmtree(temp_dir)
                    break
                except PermissionError:
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)  # Wait for file handles to be released
                        continue
                    else:
                        # Last attempt failed, log warning but don't fail test
                        import warnings

                        warnings.warn(
                            f"Could not clean up temp directory {temp_dir} due to file locks. "
                            "This may leave temporary files on disk.",
                            stacklevel=2,
                        )
        else:
            # Unix systems - normal cleanup
            shutil.rmtree(temp_dir)


@pytest.fixture
def config_with_temp_dir(temp_hunyo_dir: Path) -> MockConfig:
    """Provides MockConfig with temporary directory"""
    return MockConfig(temp_hunyo_dir)


@pytest.fixture
def mock_marimo_session():
    """Mock marimo session for testing"""
    session = MagicMock()
    session.session_id = "test-session-123"
    session.app_file_path = "/path/to/test_notebook.py"
    return session


@pytest.fixture
def mock_marimo_hooks():
    """Mock marimo's hook system for testing"""
    hooks = MagicMock()
    hooks.PRE_EXECUTION_HOOKS = []
    hooks.POST_EXECUTION_HOOKS = []
    hooks.ON_FINISH_HOOKS = []
    return hooks


@pytest.fixture
def mock_marimo_cell():
    """Mock marimo cell object for testing"""
    cell = MagicMock()
    cell.cell_id = "test-cell-abc123"
    cell.code = "import pandas as pd\ndf = pd.DataFrame({'test': [1, 2, 3]})"
    cell.name = "test_cell"
    return cell


@pytest.fixture
def mock_marimo_runner():
    """Mock marimo runner object for testing"""
    runner = MagicMock()
    runner.app_file_path = "/path/to/notebook.py"
    runner.session_id = "session-123"
    runner.kernel_id = "kernel-456"
    return runner


@pytest.fixture
def mock_marimo_run_result():
    """Mock marimo run result for testing"""
    result = MagicMock()
    result.success = True
    result.exception = None
    result.output = "Cell executed successfully"
    result.execution_time = 0.123
    return result


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection for testing"""
    ws = AsyncMock()
    ws.remote_address = ("127.0.0.1", 12345)
    ws.send = AsyncMock()
    ws.recv = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def sample_websocket_messages():
    """Sample WebSocket messages that mimic marimo's protocol"""
    return {
        "cell_execution_request": {
            "type": "cell_execution",
            "cell_id": "test-cell-123",
            "code": "df = pd.DataFrame({'a': [1, 2, 3]})",
            "run_id": "run-abc123",
        },
        "cell_execution_result": {
            "type": "cell_execution_result",
            "cell_id": "test-cell-123",
            "success": True,
            "output": "DataFrame created",
            "execution_time": 0.045,
        },
        "cell_execution_error": {
            "type": "cell_execution_error",
            "cell_id": "test-cell-123",
            "error": "NameError: name 'undefined_var' is not defined",
            "traceback": [
                '  File "<cell>", line 1, in <module>',
                "NameError: name 'undefined_var' is not defined",
            ],
        },
    }


@pytest.fixture
def mock_marimo_environment(monkeypatch):
    """Mock a complete marimo execution environment"""
    # Mock environment variables
    monkeypatch.setenv("MARIMO_NOTEBOOK_PATH", "/test/notebook.py")

    # Mock marimo module structure
    mock_marimo = MagicMock()
    mock_runtime = MagicMock()
    mock_hooks = MagicMock()

    mock_hooks.PRE_EXECUTION_HOOKS = []
    mock_hooks.POST_EXECUTION_HOOKS = []
    mock_hooks.ON_FINISH_HOOKS = []

    mock_runtime.get_session.return_value = MagicMock(
        session_id="test-session", app_file_path="/test/notebook.py"
    )

    mock_marimo._runtime = mock_runtime
    mock_marimo._runtime.runner = MagicMock()
    mock_marimo._runtime.runner.hooks = mock_hooks

    # Patch sys.modules to include mocked marimo
    import sys

    original_modules = sys.modules.copy()
    sys.modules.update(
        {
            "marimo": mock_marimo,
            "marimo._runtime": mock_runtime,
            "marimo._runtime.runner": mock_runtime.runner,
            "marimo._runtime.runner.hooks": mock_hooks,
        }
    )

    yield {"marimo": mock_marimo, "runtime": mock_runtime, "hooks": mock_hooks}

    # Restore original modules
    sys.modules.clear()
    sys.modules.update(original_modules)


@pytest.fixture
def sample_dataframe_operations():
    """Sample DataFrame operations for testing"""
    return [
        {
            "operation": "create",
            "df_name": "df1",
            "code": "df1 = pd.DataFrame({'a': [1, 2, 3]})",
            "shape": (3, 1),
            "columns": ["a"],
        },
        {
            "operation": "transform",
            "df_name": "df2",
            "code": "df2 = df1.groupby('a').sum()",
            "shape": (3, 1),
            "columns": ["a"],
        },
    ]


@pytest.fixture
def capture_event_file(temp_hunyo_dir: Path) -> Path:
    """Provides path to capture events file"""
    return temp_hunyo_dir / "events" / "runtime_events.jsonl"


class MockOutputStream:
    """Mock output stream for capturing test output"""

    def __init__(self):
        self.messages = []

    def write(self, message: str):
        self.messages.append(message)

    def clear(self):
        self.messages.clear()


@pytest.fixture
def mock_output_stream():
    """Mock output stream for testing"""
    return MockOutputStream()


@pytest.fixture
def pandas_dataframe_samples():
    """Sample pandas DataFrames for testing lineage tracking"""
    import pandas as pd

    return {
        "simple_df": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        "large_df": pd.DataFrame({"col" + str(i): range(100) for i in range(10)}),
        "string_df": pd.DataFrame(
            {"names": ["Alice", "Bob", "Charlie"], "ages": [25, 30, 35]}
        ),
        "mixed_df": pd.DataFrame(
            {"id": [1, 2, 3], "value": [1.1, 2.2, 3.3], "flag": [True, False, True]}
        ),
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with proper paths and cleanup"""
    import os

    # Set test mode to avoid interfering with actual hunyo installation
    os.environ["HUNYO_DEV_MODE"] = "1"
    os.environ["HUNYO_TEST_MODE"] = "1"

    yield

    # Cleanup
    if "HUNYO_TEST_MODE" in os.environ:
        del os.environ["HUNYO_TEST_MODE"]
