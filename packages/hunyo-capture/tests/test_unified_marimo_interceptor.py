#!/usr/bin/env python3
"""
Test suite for the unified marimo interceptor
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from hunyo_capture.unified_marimo_interceptor import (
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
        try:
            interceptor = UnifiedMarimoInterceptor()

            assert interceptor.notebook_path is None
            assert interceptor.runtime_file == Path("marimo_runtime_events.jsonl")
            assert interceptor.lineage_file == Path("marimo_lineage_events.jsonl")
        finally:
            # Clean up any files that may have been created during test
            for file_path in [
                "marimo_runtime_events.jsonl",
                "marimo_lineage_events.jsonl",
            ]:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except (OSError, FileNotFoundError):
                    pass  # Ignore cleanup errors for missing or locked files

    def test_install_marimo_hooks(self):
        """Test that marimo hooks are installed correctly."""

        with (
            patch("marimo._runtime.runner.hooks.PRE_EXECUTION_HOOKS", []),
            patch("marimo._runtime.runner.hooks.POST_EXECUTION_HOOKS", []),
            patch("marimo._runtime.runner.hooks.ON_FINISH_HOOKS", []),
        ):

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
            import time

            context_data = {
                "execution_id": "test123",
                "cell_id": "cell1",
                "timestamp": "2023-01-01T00:00:00Z",
                "context_created_at": time.time(),  # Add required timestamp field
            }

            import threading

            thread_id = threading.current_thread().ident
            interceptor._execution_contexts[thread_id] = context_data

            assert interceptor._is_in_marimo_execution()
            assert interceptor._get_current_execution_context() == context_data

    def test_multiple_execution_contexts_timestamp_selection(self):
        """Test that _get_current_execution_context returns the most recent context by timestamp."""
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            interceptor = UnifiedMarimoInterceptor(
                notebook_path=str(Path(tmp_dir) / "test.py"),
                runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                lineage_file=str(Path(tmp_dir) / "lineage.jsonl"),
            )

            # Create multiple execution contexts with different timestamps
            # Simulate the bug scenario: Cell A executes first, then Cell B
            base_time = time.time()

            # First context (Cell A - older)
            context_a = {
                "execution_id": "cell_a_exec",
                "cell_id": "cell_a",
                "cell_code": "df = pd.DataFrame(data)",
                "start_time": base_time,
                "context_created_at": base_time,  # Older timestamp
            }

            # Second context (Cell B - newer)
            context_b = {
                "execution_id": "cell_b_exec",
                "cell_id": "cell_b",
                "cell_code": "df_filtered = df[df['age'] > 25]",
                "start_time": base_time + 1,
                "context_created_at": base_time + 1,  # Newer timestamp
            }

            # Store both contexts (simulating concurrent execution tracking)
            interceptor._execution_contexts["cell_a"] = context_a
            interceptor._execution_contexts["cell_b"] = context_b

            # Test that the most recent context is returned
            current_context = interceptor._get_current_execution_context()

            # Should return context_b (most recent) not context_a (first in dict)
            assert current_context is not None
            assert current_context["execution_id"] == "cell_b_exec"
            assert current_context["cell_id"] == "cell_b"

            # Test edge case: add third context even newer
            context_c = {
                "execution_id": "cell_c_exec",
                "cell_id": "cell_c",
                "cell_code": "result = df_filtered.groupby('col').sum()",
                "start_time": base_time + 2,
                "context_created_at": base_time + 2,  # Newest timestamp
            }

            interceptor._execution_contexts["cell_c"] = context_c

            # Should now return context_c (newest)
            current_context = interceptor._get_current_execution_context()
            assert current_context["execution_id"] == "cell_c_exec"
            assert current_context["cell_id"] == "cell_c"

            # Test that method still works with single context
            interceptor._execution_contexts.clear()
            interceptor._execution_contexts["single"] = context_a

            current_context = interceptor._get_current_execution_context()
            assert current_context["execution_id"] == "cell_a_exec"

            # Test empty contexts
            interceptor._execution_contexts.clear()
            assert interceptor._get_current_execution_context() is None

    def test_uninstall_functionality(self):
        """Test that interceptor can be uninstalled properly."""

        with (
            patch("marimo._runtime.runner.hooks.PRE_EXECUTION_HOOKS", []),
            patch("marimo._runtime.runner.hooks.POST_EXECUTION_HOOKS", []),
            patch("marimo._runtime.runner.hooks.ON_FINISH_HOOKS", []),
        ):

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

    def test_create_openlineage_event(self):
        """Test that OpenLineage events are created with correct structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            interceptor = UnifiedMarimoInterceptor(
                notebook_path=str(Path(tmp_dir) / "test.py"),
                runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                lineage_file=str(Path(tmp_dir) / "lineage.jsonl"),
            )

            # Test basic OpenLineage event creation
            event = interceptor._create_openlineage_event(
                event_type="START",
                job_name="test_job",
                inputs=[],
                outputs=[],
                cell_id="test_cell_123",
            )

            # Verify required OpenLineage fields
            assert event["eventType"] == "START"
            assert event["job"]["name"] == "test_job"
            assert event["job"]["namespace"] == "marimo"
            assert "eventTime" in event
            assert "run" in event
            assert event["run"]["runId"] is not None
            assert event["producer"] == "marimo-lineage-tracker"
            assert (
                event["schemaURL"]
                == "https://openlineage.io/spec/1-0-5/OpenLineage.json"
            )
            assert event["session_id"] == interceptor.session_id
            assert "emitted_at" in event

            # Verify custom fields are included as job facets
            assert event["job"]["facets"]["cell_id"] == "test_cell_123"

            # Verify run facets structure
            assert "facets" in event["run"]
            assert "marimoSession" in event["run"]["facets"]
            assert (
                event["run"]["facets"]["marimoSession"]["sessionId"]
                == interceptor.session_id
            )

    def test_create_dataframe_dataset(self):
        """Test that DataFrame datasets are created with correct schema."""
        try:
            import pandas as pd

            with tempfile.TemporaryDirectory() as tmp_dir:
                interceptor = UnifiedMarimoInterceptor(
                    notebook_path=str(Path(tmp_dir) / "test.py"),
                    runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                    lineage_file=str(Path(tmp_dir) / "lineage.jsonl"),
                )

                # Create test DataFrame
                df = pd.DataFrame(
                    {
                        "col1": [1, 2, 3],
                        "col2": ["a", "b", "c"],
                        "col3": [1.1, 2.2, 3.3],
                    }
                )

                # Test dataset creation
                dataset = interceptor._create_dataframe_dataset(df)

                # Verify dataset structure
                assert dataset["namespace"] == "marimo"
                assert dataset["name"] == f"dataframe_{id(df)}"
                assert "facets" in dataset
                assert "schema" in dataset["facets"]

                # Verify schema fields
                schema_fields = dataset["facets"]["schema"]["fields"]
                assert len(schema_fields) == 3

                # Check specific field types
                field_names = [field["name"] for field in schema_fields]
                assert "col1" in field_names
                assert "col2" in field_names
                assert "col3" in field_names

                # Test custom dataset name
                custom_dataset = interceptor._create_dataframe_dataset(
                    df, "custom_name"
                )
                assert custom_dataset["name"] == "custom_name"

        except ImportError:
            pytest.skip("pandas not available")

    def test_dataframe_modification_capture(self):
        """Test that DataFrame modifications are captured correctly."""
        try:
            import pandas as pd

            with tempfile.TemporaryDirectory() as tmp_dir:
                lineage_file = Path(tmp_dir) / "lineage.jsonl"
                interceptor = UnifiedMarimoInterceptor(
                    notebook_path=str(Path(tmp_dir) / "test.py"),
                    runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                    lineage_file=str(lineage_file),
                )

                # Create test DataFrame
                df = pd.DataFrame({"col1": [1, 2, 3]})

                # Create mock execution context
                execution_context = {
                    "execution_id": "test_exec_123",
                    "cell_id": "test_cell",
                    "start_time": 1234567890,
                    "cell_code": "df['new_col'] = [4, 5, 6]",
                }

                # Test DataFrame modification capture
                interceptor._capture_dataframe_modification(
                    df, execution_context, "column_assignment", "new_col", [4, 5, 6]
                )

                # Verify lineage event was written
                assert lineage_file.exists()
                with open(lineage_file) as f:
                    saved_event = json.loads(f.read().strip())

                    # Verify OpenLineage event structure
                    assert saved_event["eventType"] == "COMPLETE"
                    assert saved_event["job"]["name"] == "pandas_dataframe_modification"
                    assert (
                        saved_event["job"]["facets"]["modification_type"]
                        == "column_assignment"
                    )
                    assert saved_event["job"]["facets"]["key"] == "new_col"
                    assert saved_event["job"]["facets"]["value_type"] == "list"
                    assert saved_event["session_id"] == interceptor.session_id

                    # Verify outputs structure
                    assert len(saved_event["outputs"]) == 1
                    output = saved_event["outputs"][0]
                    assert output["namespace"] == "marimo"
                    assert "schema" in output["facets"]

        except ImportError:
            pytest.skip("pandas not available")

    def test_dataframe_setitem_monkey_patching(self):
        """Test that DataFrame.__setitem__ monkey patching works correctly."""
        try:
            import pandas as pd

            with tempfile.TemporaryDirectory() as tmp_dir:
                lineage_file = Path(tmp_dir) / "lineage.jsonl"
                interceptor = UnifiedMarimoInterceptor(
                    notebook_path=str(Path(tmp_dir) / "test.py"),
                    runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                    lineage_file=str(lineage_file),
                )

                # Store original method
                original_setitem = pd.DataFrame.__setitem__

                # Install monkey patches
                interceptor._install_dataframe_patches()

                # Verify that __setitem__ was patched
                assert pd.DataFrame.__setitem__ != original_setitem
                assert "DataFrame.__setitem__" in interceptor.original_pandas_methods

                # Test that the patched method still works
                df = pd.DataFrame({"a": [1, 2, 3]})

                # This should work without errors (even though no execution context)
                df["b"] = [4, 5, 6]
                assert "b" in df.columns
                assert list(df["b"]) == [4, 5, 6]

                # Restore original method
                pd.DataFrame.__setitem__ = original_setitem

        except ImportError:
            pytest.skip("pandas not available")

    def test_dataframe_failure_capture(self):
        """Test FAIL event capture for DataFrame operations"""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        import json
        import os
        import tempfile

        # Setup interceptor with test files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as lineage_file:
            lineage_file_path = lineage_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as runtime_file:
            runtime_file_path = runtime_file.name

        interceptor = UnifiedMarimoInterceptor(
            notebook_path="test_notebook.py",
            lineage_file=lineage_file_path,
            runtime_file=runtime_file_path,
        )

        # Setup execution context
        execution_context = {
            "execution_id": "test_exec_1",
            "cell_id": "test_cell_1",
            "cell_code": "df = pd.DataFrame({'col': [1, 2, 3]})",
            "start_time": 1234567890.0,
        }

        # Test DataFrame creation failure
        test_df = pd.DataFrame({"col": [1, 2, 3]})
        test_error = ValueError("Test error for DataFrame creation")

        # Call the failure capture method
        interceptor._capture_dataframe_failure(
            df=test_df,
            execution_context=execution_context,
            job_name="pandas_dataframe_creation",
            error=test_error,
            partial_outputs=[test_df],
        )

        # Verify FAIL event was emitted
        with open(lineage_file_path) as f:
            events = [json.loads(line) for line in f.readlines()]

        assert len(events) == 1
        fail_event = events[0]

        # Verify event structure
        assert fail_event["eventType"] == "FAIL"
        assert fail_event["job"]["name"] == "pandas_dataframe_creation"
        assert "errorMessage" in fail_event["run"]["facets"]

        # Verify error facet
        error_facet = fail_event["run"]["facets"]["errorMessage"]
        assert error_facet["message"] == "Test error for DataFrame creation"
        assert error_facet["programmingLanguage"] == "python"
        assert "stackTrace" in error_facet  # Stack trace should be present

        # Verify partial outputs
        assert len(fail_event["outputs"]) == 1
        output_dataset = fail_event["outputs"][0]
        assert output_dataset["namespace"] == "marimo"
        assert "dataframe_" in output_dataset["name"]

        # Cleanup
        os.unlink(lineage_file_path)
        os.unlink(runtime_file_path)

    def test_dataframe_abortion_capture(self):
        """Test ABORT event capture for DataFrame operations"""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        import json
        import os
        import tempfile

        # Setup interceptor with test files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as lineage_file:
            lineage_file_path = lineage_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as runtime_file:
            runtime_file_path = runtime_file.name

        interceptor = UnifiedMarimoInterceptor(
            notebook_path="test_notebook.py",
            lineage_file=lineage_file_path,
            runtime_file=runtime_file_path,
        )

        # Setup execution context
        execution_context = {
            "execution_id": "test_exec_2",
            "cell_id": "test_cell_2",
            "cell_code": "df = pd.DataFrame({'col': [1, 2, 3]})",
            "start_time": 1234567890.0,
        }

        # Test DataFrame modification abortion
        test_df = pd.DataFrame({"col": [1, 2, 3]})
        termination_reason = "User interrupted operation"

        # Call the abortion capture method
        interceptor._capture_dataframe_abortion(
            df=test_df,
            execution_context=execution_context,
            job_name="pandas_dataframe_modification",
            termination_reason=termination_reason,
            partial_outputs=[test_df],
        )

        # Verify ABORT event was emitted
        with open(lineage_file_path) as f:
            events = [json.loads(line) for line in f.readlines()]

        assert len(events) == 1
        abort_event = events[0]

        # Verify event structure
        assert abort_event["eventType"] == "ABORT"
        assert abort_event["job"]["name"] == "pandas_dataframe_modification"
        assert "errorMessage" in abort_event["run"]["facets"]

        # Verify error facet
        error_facet = abort_event["run"]["facets"]["errorMessage"]
        assert (
            error_facet["message"] == "Operation terminated: User interrupted operation"
        )
        assert error_facet["programmingLanguage"] == "python"
        assert "stackTrace" in error_facet  # Stack trace should be present

        # Verify termination context
        assert (
            abort_event["job"]["facets"]["termination_reason"]
            == "User interrupted operation"
        )

        # Verify partial outputs
        assert len(abort_event["outputs"]) == 1
        output_dataset = abort_event["outputs"][0]
        assert output_dataset["namespace"] == "marimo"
        assert "dataframe_" in output_dataset["name"]

        # Cleanup
        os.unlink(lineage_file_path)
        os.unlink(runtime_file_path)

    def test_create_openlineage_event_with_error_info(self):
        """Test OpenLineage event creation with error information"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            interceptor = UnifiedMarimoInterceptor(
                notebook_path=str(Path(tmp_dir) / "test_notebook.py"),
                runtime_file=str(Path(tmp_dir) / "runtime.jsonl"),
                lineage_file=str(Path(tmp_dir) / "lineage.jsonl"),
            )

            # Test FAIL event with error info
            error_info = {
                "error_message": "Test error message",
                "stack_trace": "Test stack trace",
                "error_type": "ValueError",
            }

            fail_event = interceptor._create_openlineage_event(
                event_type="FAIL",
                job_name="test_job",
                error_info=error_info,
                test_field="test_value",
            )

            # Verify base event structure
            assert fail_event["eventType"] == "FAIL"
            assert fail_event["job"]["name"] == "test_job"
            assert fail_event["job"]["facets"]["test_field"] == "test_value"

            # Verify error facet
            assert "errorMessage" in fail_event["run"]["facets"]
            error_facet = fail_event["run"]["facets"]["errorMessage"]
            assert error_facet["message"] == "Test error message"
            assert error_facet["stackTrace"] == "Test stack trace"
            assert error_facet["programmingLanguage"] == "python"

            # Test ABORT event with error info
            abort_event = interceptor._create_openlineage_event(
                event_type="ABORT", job_name="test_job", error_info=error_info
            )

            # Verify ABORT event has error facet
            assert abort_event["eventType"] == "ABORT"
            assert "errorMessage" in abort_event["run"]["facets"]

            # Test event without error info (should not have error facet)
            normal_event = interceptor._create_openlineage_event(
                event_type="START", job_name="test_job"
            )

            assert normal_event["eventType"] == "START"
            assert "errorMessage" not in normal_event["run"]["facets"]

    def test_keyboard_interrupt_handling_dataframe_creation(self):
        """Test KeyboardInterrupt handling in DataFrame creation"""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        import json
        import os
        import tempfile

        # Setup interceptor with test files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as lineage_file:
            lineage_file_path = lineage_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as runtime_file:
            runtime_file_path = runtime_file.name

        interceptor = UnifiedMarimoInterceptor(
            notebook_path="test_notebook.py",
            lineage_file=lineage_file_path,
            runtime_file=runtime_file_path,
        )

        # Setup execution context
        execution_context = {
            "execution_id": "test_exec_3",
            "cell_id": "test_cell_3",
            "cell_code": "df = pd.DataFrame({'col': [1, 2, 3]})",
            "start_time": 1234567890.0,
        }

        # Test that ABORT event is captured when DataFrame operation is interrupted
        test_df = pd.DataFrame({"col": [1, 2, 3]})

        # Directly test the ABORT capture functionality
        interceptor._capture_dataframe_abortion(
            df=test_df,
            execution_context=execution_context,
            job_name="pandas_dataframe_creation",
            termination_reason="User interrupted operation",
        )

        # Verify ABORT event was emitted
        with open(lineage_file_path) as f:
            events = [json.loads(line) for line in f.readlines()]

        assert len(events) == 1
        abort_event = events[0]
        assert abort_event["eventType"] == "ABORT"
        assert abort_event["job"]["name"] == "pandas_dataframe_creation"
        assert (
            abort_event["job"]["facets"]["termination_reason"]
            == "User interrupted operation"
        )

        # Cleanup
        os.unlink(lineage_file_path)
        os.unlink(runtime_file_path)

    def test_cell_execution_error_lineage_events(self):
        """Test FAIL/ABORT lineage events for cell execution errors"""
        import json
        import os
        import tempfile
        from unittest.mock import Mock

        # Setup interceptor with test files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as lineage_file:
            lineage_file_path = lineage_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as runtime_file:
            runtime_file_path = runtime_file.name

        interceptor = UnifiedMarimoInterceptor(
            notebook_path="test_notebook.py",
            lineage_file=lineage_file_path,
            runtime_file=runtime_file_path,
        )

        # Setup execution context
        execution_context = {
            "execution_id": "test_exec_4",
            "cell_id": "test_cell_4",
            "cell_code": "raise ValueError('Test error')",
            "start_time": 1234567890.0,
        }
        interceptor._execution_contexts["test_cell_4"] = execution_context

        # Create post-execution hook
        post_hook = interceptor._create_post_execution_hook()

        # Mock cell and run_result for ValueError
        mock_cell = Mock()
        mock_cell.cell_id = "test_cell_4"
        mock_cell.code = "raise ValueError('Test error')"

        mock_run_result = Mock()
        mock_run_result.exception = ValueError("Test error")

        # Call post-execution hook
        post_hook(mock_cell, None, mock_run_result)

        # Verify FAIL event was emitted
        with open(lineage_file_path) as f:
            events = [json.loads(line) for line in f.readlines()]

        # Should have FAIL event for cell execution
        fail_events = [e for e in events if e.get("eventType") == "FAIL"]
        assert len(fail_events) > 0

        fail_event = fail_events[0]
        assert fail_event["job"]["name"] == "cell_execution"
        assert "errorMessage" in fail_event["run"]["facets"]
        assert fail_event["job"]["facets"]["cell_id"] == "test_cell_4"

        # Test KeyboardInterrupt -> ABORT event by creating a new temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as abort_lineage_file:
            abort_lineage_file_path = abort_lineage_file.name

        # Create new interceptor for ABORT test
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as abort_runtime_file:
            abort_runtime_file_path = abort_runtime_file.name

        abort_interceptor = UnifiedMarimoInterceptor(
            notebook_path="test_notebook.py",
            lineage_file=abort_lineage_file_path,
            runtime_file=abort_runtime_file_path,
        )
        abort_interceptor._execution_contexts["test_cell_4"] = execution_context

        # Create new post-execution hook for ABORT test
        abort_post_hook = abort_interceptor._create_post_execution_hook()

        # Mock KeyboardInterrupt exception
        mock_run_result.exception = KeyboardInterrupt("User interrupted")
        abort_post_hook(mock_cell, None, mock_run_result)

        # Verify ABORT event was emitted
        with open(abort_lineage_file_path) as f:
            abort_events_data = [json.loads(line) for line in f.readlines()]

        abort_events = [e for e in abort_events_data if e.get("eventType") == "ABORT"]
        assert len(abort_events) > 0

        abort_event = abort_events[0]
        assert abort_event["job"]["name"] == "cell_execution"
        assert "errorMessage" in abort_event["run"]["facets"]
        assert abort_event["job"]["facets"]["cell_id"] == "test_cell_4"
        assert (
            abort_event["job"]["facets"]["termination_reason"]
            == "User interrupted cell execution"
        )

        # Cleanup ABORT test file
        os.unlink(abort_lineage_file_path)
        os.unlink(abort_runtime_file_path)

        # Cleanup
        os.unlink(lineage_file_path)
        os.unlink(runtime_file_path)


class TestUnifiedInterceptorGlobalFunctions:
    """Test cases for the global unified interceptor functions."""

    def test_enable_unified_tracking(self):
        """Test enable_unified_tracking function."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook_path = str(Path(tmp_dir) / "test.py")
            runtime_file = str(Path(tmp_dir) / "runtime.jsonl")
            lineage_file = str(Path(tmp_dir) / "lineage.jsonl")

            # Ensure no interceptor is active
            disable_unified_tracking()
            assert not is_unified_tracking_active()

            # Enable tracking with explicit file paths
            interceptor = enable_unified_tracking(
                notebook_path=notebook_path,
                runtime_file=runtime_file,
                lineage_file=lineage_file,
            )

            assert interceptor is not None
            assert is_unified_tracking_active()
            assert get_unified_interceptor() == interceptor

            # Clean up
            disable_unified_tracking()

    def test_disable_unified_tracking(self):
        """Test disable_unified_tracking function."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook_path = str(Path(tmp_dir) / "test.py")
            runtime_file = str(Path(tmp_dir) / "runtime.jsonl")
            lineage_file = str(Path(tmp_dir) / "lineage.jsonl")

            # Enable tracking first
            _interceptor = enable_unified_tracking(
                notebook_path=notebook_path,
                runtime_file=runtime_file,
                lineage_file=lineage_file,
            )
            assert is_unified_tracking_active()

            # Disable tracking
            disable_unified_tracking()
            assert not is_unified_tracking_active()
            assert get_unified_interceptor() is None

    def test_get_unified_interceptor(self):
        """Test get_unified_interceptor function."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook_path = str(Path(tmp_dir) / "test.py")
            runtime_file = str(Path(tmp_dir) / "runtime.jsonl")
            lineage_file = str(Path(tmp_dir) / "lineage.jsonl")

            # No interceptor initially
            assert get_unified_interceptor() is None

            # Enable tracking
            interceptor = enable_unified_tracking(
                notebook_path=notebook_path,
                runtime_file=runtime_file,
                lineage_file=lineage_file,
            )
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
            runtime_file = str(Path(tmp_dir) / "runtime.jsonl")
            lineage_file = str(Path(tmp_dir) / "lineage.jsonl")

            enable_unified_tracking(
                notebook_path=notebook_path,
                runtime_file=runtime_file,
                lineage_file=lineage_file,
            )
            assert is_unified_tracking_active()

            # Clean up
            disable_unified_tracking()
            assert not is_unified_tracking_active()

    def test_enable_unified_tracking_already_active(self):
        """Test that enabling tracking when already active returns existing interceptor."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook_path = str(Path(tmp_dir) / "test.py")
            runtime_file = str(Path(tmp_dir) / "runtime.jsonl")
            lineage_file = str(Path(tmp_dir) / "lineage.jsonl")

            # Enable tracking
            interceptor1 = enable_unified_tracking(
                notebook_path=notebook_path,
                runtime_file=runtime_file,
                lineage_file=lineage_file,
            )

            # Try to enable again
            interceptor2 = enable_unified_tracking(
                notebook_path=notebook_path,
                runtime_file=runtime_file,
                lineage_file=lineage_file,
            )

            # Should return the same interceptor
            assert interceptor1 == interceptor2

            # Clean up
            disable_unified_tracking()

    def test_multiple_disable_calls(self):
        """Test that multiple disable calls don't cause errors."""

        disable_unified_tracking()
        disable_unified_tracking()  # Should not raise error
        assert not is_unified_tracking_active()
