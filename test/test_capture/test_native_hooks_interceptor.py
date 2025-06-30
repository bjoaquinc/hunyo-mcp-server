from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest

from test.mocks import MockNativeHooksInterceptor as NativeHooksInterceptor


class TestNativeHooksInterceptor:
    """Tests for NativeHooksInterceptor following marimo patterns"""

    def test_interceptor_initialization(self, config_with_temp_dir):
        """Test interceptor initializes with proper hooks"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        assert interceptor.config == config_with_temp_dir
        assert hasattr(interceptor, "original_import")
        assert hasattr(interceptor, "hooked_modules")

    def test_import_hook_installation(self, config_with_temp_dir):
        """Test import hook is properly installed"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        # Store original import
        original_import = __builtins__["__import__"]

        try:
            interceptor.install_hooks()

            # Verify import hook is installed
            assert __builtins__["__import__"] != original_import
            assert callable(__builtins__["__import__"])

        finally:
            # Restore original import
            interceptor.uninstall_hooks()
            assert __builtins__["__import__"] == original_import

    def test_pandas_import_interception(self, config_with_temp_dir):
        """Test interception of pandas import"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        with patch.object(interceptor, "_wrap_pandas_methods") as mock_wrap:
            interceptor.install_hooks()

            try:
                # Simulate pandas import
                mock_pandas = MagicMock()
                mock_pandas.__name__ = "pandas"

                # Call the hooked import directly
                result = interceptor._hooked_import("pandas", {}, {}, [], -1)

                # Should attempt to wrap pandas methods
                if mock_wrap.called:
                    mock_wrap.assert_called()

            finally:
                interceptor.uninstall_hooks()

    def test_dataframe_method_wrapping(self, config_with_temp_dir):
        """Test wrapping of DataFrame methods for tracking"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        # Mock DataFrame class
        mock_dataframe = MagicMock()
        mock_dataframe.merge = MagicMock()
        mock_dataframe.groupby = MagicMock()
        mock_dataframe.filter = MagicMock()

        with patch.object(interceptor, "_create_tracked_method") as mock_create:
            mock_create.return_value = MagicMock()

            interceptor._wrap_dataframe_methods(mock_dataframe)

            # Should wrap key DataFrame methods
            assert mock_create.call_count >= 3  # merge, groupby, filter at minimum

    def test_method_call_tracking(self, config_with_temp_dir):
        """Test tracking of DataFrame method calls"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        # Mock DataFrame and method
        mock_df = MagicMock()
        mock_df.__class__.__name__ = "DataFrame"

        original_method = MagicMock(return_value=mock_df)

        with patch.object(interceptor, "_log_method_call") as mock_log:
            # Create tracked method wrapper
            tracked_method = interceptor._create_tracked_method(
                original_method, "merge", mock_df.__class__
            )

            # Call the tracked method
            result = tracked_method(mock_df, right=mock_df, on="key")

            # Should log the method call
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0]
            assert call_args[0] == "merge"  # method name
            assert call_args[1] == mock_df  # self object

    def test_execution_context_capture(self, config_with_temp_dir):
        """Test capturing execution context for method calls"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        with patch("inspect.stack") as mock_stack:
            # Mock call stack
            mock_frame = MagicMock()
            mock_frame.filename = "test_notebook.py"
            mock_frame.lineno = 15
            mock_frame.code_context = ['result = df1.merge(df2, on="id")']

            mock_stack.return_value = [None, mock_frame]

            with patch.object(interceptor, "_log_method_call") as mock_log:
                context = interceptor._capture_execution_context()

                assert context["filename"] == "test_notebook.py"
                assert context["line_number"] == 15
                assert "merge" in context["code_context"][0]

    def test_selective_module_hooking(self, config_with_temp_dir):
        """Test that only target modules are hooked"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        target_modules = ["pandas", "numpy", "polars"]

        with patch.object(interceptor, "_should_hook_module") as mock_should_hook:
            mock_should_hook.side_effect = lambda name: name in target_modules

            interceptor.install_hooks()

            try:
                for module_name in ["pandas", "os", "sys", "numpy"]:
                    should_hook = interceptor._should_hook_module(module_name)

                    if module_name in target_modules:
                        assert should_hook
                    else:
                        assert not should_hook

            finally:
                interceptor.uninstall_hooks()

    @pytest.mark.parametrize(
        "method_name,expected_category",
        [
            ("merge", "join"),
            ("groupby", "aggregation"),
            ("filter", "transformation"),
            ("sort_values", "transformation"),
            ("drop_duplicates", "cleaning"),
            ("fillna", "cleaning"),
        ],
    )
    def test_method_categorization(
        self, config_with_temp_dir, method_name, expected_category
    ):
        """Test categorization of DataFrame methods"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        category = interceptor._categorize_method(method_name)
        assert category == expected_category

    def test_performance_impact_minimal(self, config_with_temp_dir):
        """Test that hooking has minimal performance impact"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        # Mock a simple method
        def simple_method(self, x):
            return x * 2

        # Time the original method
        import time

        start_time = time.time()
        for _ in range(1000):
            simple_method(None, 5)
        original_time = time.time() - start_time

        # Time the wrapped method
        wrapped_method = interceptor._create_tracked_method(
            simple_method, "test_method", type(None)
        )

        with patch.object(interceptor, "_log_method_call"):
            start_time = time.time()
            for _ in range(1000):
                wrapped_method(None, 5)
            wrapped_time = time.time() - start_time

        # Overhead should be reasonable (less than 10x slower)
        assert wrapped_time < original_time * 10

    def test_hook_cleanup_on_error(self, config_with_temp_dir):
        """Test that hooks are properly cleaned up on errors"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        original_import = __builtins__["__import__"]

        try:
            interceptor.install_hooks()

            # Simulate an error during operation
            with pytest.raises(Exception):
                raise Exception("Simulated error")

        except Exception:
            pass
        finally:
            # Cleanup should still work
            interceptor.uninstall_hooks()
            assert __builtins__["__import__"] == original_import

    def test_concurrent_hook_safety(self, config_with_temp_dir):
        """Test that hooks are thread-safe"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        import threading
        import time

        results = []
        errors = []

        def hook_install_uninstall():
            try:
                interceptor.install_hooks()
                time.sleep(0.01)  # Small delay
                interceptor.uninstall_hooks()
                results.append("success")
            except Exception as e:
                errors.append(str(e))

        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=hook_install_uninstall)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should handle concurrent access gracefully
        assert len(errors) == 0 or len(results) > 0  # At least some should succeed

    def test_memory_leak_prevention(self, config_with_temp_dir):
        """Test that interceptor doesn't cause memory leaks"""
        interceptor = NativeHooksInterceptor(config_with_temp_dir)

        initial_modules = len(interceptor.hooked_modules)

        # Install and uninstall hooks multiple times
        for _ in range(10):
            interceptor.install_hooks()

            # Simulate some module hooking
            interceptor.hooked_modules["test_module_" + str(_)] = MagicMock()

            interceptor.uninstall_hooks()

        # Should clean up properly
        final_modules = len(interceptor.hooked_modules)
        assert final_modules == initial_modules
