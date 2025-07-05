#!/usr/bin/env python3
"""
Test suite for the unified marimo interceptor
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from capture.unified_marimo_interceptor import (
    UnifiedMarimoInterceptor,
    disable_unified_tracking,
    enable_unified_tracking,
    get_unified_interceptor,
    is_unified_tracking_active,
)


class TestUnifiedMarimoInterceptor:
    """Test cases for the unified marimo interceptor system."""

    def test_interceptor_initialization(self):
        """Test that interceptor initializes correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            interceptor = UnifiedMarimoInterceptor(
                notebook_path=str(Path(tmp_dir) / "test.py"),
                runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                lineage_file=str(Path(tmp_dir) / "lineage.jsonl"),
            )

            assert interceptor.notebook_path == str(Path(tmp_dir) / "test.py")
            assert interceptor.runtime_file == Path(tmp_dir) / "runtime.jsonl"
            assert interceptor.lineage_file == Path(tmp_dir) / "lineage.jsonl"
            assert interceptor.session_id is not None
            assert len(interceptor.session_id) == 8
            assert not interceptor.interceptor_active
            assert interceptor.installed_hooks == []

    def test_interceptor_initialization_with_notebook_path(self):
        """Test that interceptor auto-generates file paths from notebook path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook_path = str(Path(tmp_dir) / "test_notebook.py")

            # Create a mock notebook file
            Path(notebook_path).write_text("print('hello')")

            interceptor = UnifiedMarimoInterceptor(notebook_path=notebook_path)

            assert interceptor.notebook_path == notebook_path
            assert interceptor.runtime_file.exists()
            assert interceptor.lineage_file.exists()
            assert "runtime_events" in interceptor.runtime_file.name
            assert "lineage_events" in interceptor.lineage_file.name

    def test_interceptor_initialization_without_notebook_path(self):
        """Test that interceptor uses defaults when no notebook path provided."""
        interceptor = UnifiedMarimoInterceptor()

        assert interceptor.notebook_path is None
        assert interceptor.runtime_file == Path("marimo_runtime_events.jsonl")
        assert interceptor.lineage_file == Path("marimo_lineage_events.jsonl")

    @patch("marimo._runtime.runner.hooks.PRE_EXECUTION_HOOKS", [])
    @patch("marimo._runtime.runner.hooks.POST_EXECUTION_HOOKS", [])
    @patch("marimo._runtime.runner.hooks.ON_FINISH_HOOKS", [])
    def test_install_marimo_hooks(self):
        """Test that marimo hooks are installed correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            interceptor = UnifiedMarimoInterceptor(
                notebook_path=str(Path(tmp_dir) / "test.py"),
                runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                lineage_file=str(Path(tmp_dir) / "lineage.jsonl"),
            )

            interceptor.install()

            # Verify hooks were installed
            assert interceptor.interceptor_active
            assert len(interceptor.installed_hooks) == 3

    def test_dataframe_monkey_patching(self):
        """Test that DataFrame methods are monkey patched correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            interceptor = UnifiedMarimoInterceptor(
                notebook_path=str(Path(tmp_dir) / "test.py"),
                runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                lineage_file=str(Path(tmp_dir) / "lineage.jsonl"),
            )

            # Store original method
            try:
                import pandas as pd

                original_init = pd.DataFrame.__init__

                interceptor._install_dataframe_patches()

                # Verify that the method was patched
                assert pd.DataFrame.__init__ != original_init
                assert "DataFrame.__init__" in interceptor.original_pandas_methods

                # Restore original method
                pd.DataFrame.__init__ = original_init

            except ImportError:
                pytest.skip("pandas not available")

    def test_runtime_event_emission(self):
        """Test that runtime events are emitted correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_file = Path(tmp_dir) / "runtime.jsonl"
            interceptor = UnifiedMarimoInterceptor(
                notebook_path=str(Path(tmp_dir) / "test.py"),
                runtime_file=str(runtime_file),
                lineage_file=str(Path(tmp_dir) / "lineage.jsonl"),
            )

            # Emit a test runtime event
            test_event = {
                "event_type": "cell_execution_start",
                "execution_id": "test123",
                "cell_id": "cell1",
                "timestamp": "2023-01-01T00:00:00Z",
                "session_id": interceptor.session_id,
            }

            interceptor._emit_runtime_event(test_event)

            # Verify event was written to file
            assert runtime_file.exists()
            with open(runtime_file) as f:
                saved_event = json.loads(f.read().strip())
                assert saved_event["event_type"] == "cell_execution_start"
                assert saved_event["execution_id"] == "test123"
                assert saved_event["session_id"] == interceptor.session_id

    def test_lineage_event_emission(self):
        """Test that lineage events are emitted correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            lineage_file = Path(tmp_dir) / "lineage.jsonl"
            interceptor = UnifiedMarimoInterceptor(
                notebook_path=str(Path(tmp_dir) / "test.py"),
                runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                lineage_file=str(lineage_file),
            )

            # Emit a test lineage event
            test_event = {
                "event_type": "dataframe_creation",
                "execution_id": "test123",
                "dataframe_id": "df1",
                "timestamp": "2023-01-01T00:00:00Z",
                "session_id": interceptor.session_id,
            }

            interceptor._emit_lineage_event(test_event)

            # Verify event was written to file
            assert lineage_file.exists()
            with open(lineage_file) as f:
                saved_event = json.loads(f.read().strip())
                assert saved_event["event_type"] == "dataframe_creation"
                assert saved_event["execution_id"] == "test123"
                assert saved_event["session_id"] == interceptor.session_id

    def test_execution_context_tracking(self):
        """Test that execution context is tracked correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            interceptor = UnifiedMarimoInterceptor(
                notebook_path=str(Path(tmp_dir) / "test.py"),
                runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                lineage_file=str(Path(tmp_dir) / "lineage.jsonl"),
            )

            # Test execution context methods
            assert not interceptor._is_in_marimo_execution()
            assert interceptor._get_current_execution_context() is None

            # Simulate adding execution context
            context_data = {
                "execution_id": "test123",
                "cell_id": "cell1",
                "timestamp": "2023-01-01T00:00:00Z",
            }

            import threading

            thread_id = threading.current_thread().ident
            interceptor._execution_contexts[thread_id] = context_data

            assert interceptor._is_in_marimo_execution()
            assert interceptor._get_current_execution_context() == context_data

    def test_uninstall_functionality(self):
        """Test that interceptor can be uninstalled properly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            interceptor = UnifiedMarimoInterceptor(
                notebook_path=str(Path(tmp_dir) / "test.py"),
                runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                lineage_file=str(Path(tmp_dir) / "lineage.jsonl"),
            )

            # Mock the installed hooks
            interceptor.installed_hooks = [
                ("PRE_EXECUTION_HOOKS", Mock()),
                ("POST_EXECUTION_HOOKS", Mock()),
                ("ON_FINISH_HOOKS", Mock()),
            ]
            interceptor.interceptor_active = True

            # Mock pandas DataFrame method
            try:
                import pandas as pd

                original_init = pd.DataFrame.__init__
                interceptor.original_pandas_methods["DataFrame.__init__"] = (
                    original_init
                )

                interceptor.uninstall()

                assert not interceptor.interceptor_active
                assert len(interceptor.installed_hooks) == 0
                assert pd.DataFrame.__init__ == original_init

            except ImportError:
                pytest.skip("pandas not available")


class TestUnifiedInterceptorGlobalFunctions:
    """Test cases for the global unified interceptor functions."""

    def test_enable_unified_tracking(self):
        """Test enable_unified_tracking function."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook_path = str(Path(tmp_dir) / "test.py")

            # Ensure no interceptor is active
            disable_unified_tracking()
            assert not is_unified_tracking_active()

            # Enable tracking
            interceptor = enable_unified_tracking(notebook_path=notebook_path)

            assert interceptor is not None
            assert is_unified_tracking_active()
            assert get_unified_interceptor() == interceptor

            # Clean up
            disable_unified_tracking()

    def test_disable_unified_tracking(self):
        """Test disable_unified_tracking function."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook_path = str(Path(tmp_dir) / "test.py")

            # Enable tracking first
            _interceptor = enable_unified_tracking(notebook_path=notebook_path)
            assert is_unified_tracking_active()

            # Disable tracking
            disable_unified_tracking()
            assert not is_unified_tracking_active()
            assert get_unified_interceptor() is None

    def test_get_unified_interceptor(self):
        """Test get_unified_interceptor function."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook_path = str(Path(tmp_dir) / "test.py")

            # No interceptor initially
            assert get_unified_interceptor() is None

            # Enable tracking
            interceptor = enable_unified_tracking(notebook_path=notebook_path)
            assert get_unified_interceptor() == interceptor

            # Clean up
            disable_unified_tracking()

    def test_is_unified_tracking_active(self):
        """Test is_unified_tracking_active function."""
        # Start with no tracking
        disable_unified_tracking()
        assert not is_unified_tracking_active()

        # Enable tracking
        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook_path = str(Path(tmp_dir) / "test.py")
            enable_unified_tracking(notebook_path=notebook_path)
            assert is_unified_tracking_active()

            # Clean up
            disable_unified_tracking()
            assert not is_unified_tracking_active()

    def test_enable_unified_tracking_already_active(self):
        """Test that enabling tracking when already active returns existing interceptor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook_path = str(Path(tmp_dir) / "test.py")

            # Enable tracking
            interceptor1 = enable_unified_tracking(notebook_path=notebook_path)

            # Try to enable again
            interceptor2 = enable_unified_tracking(notebook_path=notebook_path)

            # Should return the same interceptor
            assert interceptor1 == interceptor2

            # Clean up
            disable_unified_tracking()

    def test_multiple_disable_calls(self):
        """Test that multiple disable calls don't cause errors."""
        disable_unified_tracking()
        disable_unified_tracking()  # Should not raise error
        assert not is_unified_tracking_active()
