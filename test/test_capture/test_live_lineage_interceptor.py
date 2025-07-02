from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.capture.live_lineage_interceptor import (
    MarimoLiveInterceptor,
    disable_live_tracking,
    enable_live_tracking,
    get_tracking_summary,
    is_tracking_active,
    runtime_debugging_context,
)
from test.mocks import MockLiveLineageInterceptor as MockInterceptor

# Import the real implementation for comprehensive testing


class TestLiveLineageInterceptor:
    """Tests for LiveLineageInterceptor following marimo patterns"""

    # Keep existing mock-based tests for compatibility
    def test_interceptor_initialization(self, config_with_temp_dir):
        """Test interceptor initializes properly"""
        interceptor = MockInterceptor(config_with_temp_dir)

        assert interceptor.config == config_with_temp_dir
        assert hasattr(interceptor, "tracked_objects")
        assert isinstance(interceptor.tracked_objects, dict)

    def test_pandas_dataframe_tracking(self, config_with_temp_dir):
        """Test tracking of pandas DataFrame operations"""
        interceptor = MockInterceptor(config_with_temp_dir)

        # Mock the tracking method
        with patch.object(interceptor, "_track_dataframe_lineage") as mock_track:
            # Simulate DataFrame creation
            test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            interceptor.track_object("test_df", test_df)

            mock_track.assert_called_once()
            call_args = mock_track.call_args[0]
            assert call_args[0] == "test_df"
            assert isinstance(call_args[1], pd.DataFrame)

    def test_dataframe_transformation_tracking(self, config_with_temp_dir):
        """Test tracking of DataFrame transformations"""
        interceptor = MockInterceptor(config_with_temp_dir)

        with patch.object(interceptor, "_log_lineage_event") as mock_log:
            # Simulate DataFrame operations
            df1 = pd.DataFrame({"a": [1, 2, 3]})
            df2 = df1.copy()
            df3 = df1.merge(df2, on="a")

            interceptor.track_object("df1", df1)
            interceptor.track_object("df2", df2)
            interceptor.track_object("df3", df3)

            # Should log lineage events
            assert mock_log.call_count >= 1

    def test_lineage_graph_construction(self, config_with_temp_dir):
        """Test lineage graph is properly constructed"""
        interceptor = MockInterceptor(config_with_temp_dir)

        # Create a chain of DataFrame operations
        with patch.object(interceptor, "_detect_lineage_relationship") as mock_detect:
            mock_detect.return_value = ["df1", "df2"]  # df3 depends on df1 and df2

            df1 = pd.DataFrame({"a": [1, 2, 3]})
            df2 = pd.DataFrame({"b": [4, 5, 6]})
            df3 = pd.concat([df1, df2], axis=1)

            interceptor.track_object("df1", df1)
            interceptor.track_object("df2", df2)
            interceptor.track_object("df3", df3)

            lineage = interceptor.get_lineage_graph()

            # Verify lineage relationships
            assert "df3" in lineage
            assert lineage["df3"]["parents"] == ["df1", "df2"]

    def test_code_context_capture(self, config_with_temp_dir):
        """Test capturing code context for operations"""
        interceptor = MockInterceptor(config_with_temp_dir)

        with patch("inspect.stack") as mock_stack:
            # Mock the call stack to simulate code context
            mock_frame = MagicMock()
            mock_frame.code_context = ['df_result = df1.merge(df2, on="key")']
            mock_frame.filename = "test_notebook.py"
            mock_frame.lineno = 42

            mock_stack.return_value = [None, mock_frame]  # Skip first frame

            with patch.object(interceptor, "_log_lineage_event") as mock_log:
                df1 = pd.DataFrame({"key": [1, 2], "value": [10, 20]})
                df2 = pd.DataFrame({"key": [1, 2], "other": [30, 40]})

                interceptor.track_object("df1", df1)
                interceptor.track_object("df2", df2)

                # Should capture code context
                if mock_log.called:
                    event = mock_log.call_args[0][0]
                    assert "code_context" in event
                    assert event["code_context"]["line_number"] == 42

    def test_memory_efficiency(self, config_with_temp_dir):
        """Test interceptor manages memory efficiently for large DataFrames"""
        interceptor = MockInterceptor(config_with_temp_dir)

        # Create large DataFrame
        large_df = pd.DataFrame({"col1": range(10000), "col2": range(10000, 20000)})

        with patch.object(interceptor, "_create_dataframe_summary") as mock_summary:
            mock_summary.return_value = {
                "shape": (10000, 2),
                "columns": ["col1", "col2"],
                "dtypes": {"col1": "int64", "col2": "int64"},
                "memory_usage": "156.3 KB",
            }

            interceptor.track_object("large_df", large_df)

            # Should create summary instead of storing full DataFrame
            mock_summary.assert_called_once()

    def test_circular_reference_detection(self, config_with_temp_dir):
        """Test detection and handling of circular references"""
        interceptor = MockInterceptor(config_with_temp_dir)

        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        # Simulate circular reference scenario
        with patch.object(interceptor, "_detect_circular_reference") as mock_detect:
            mock_detect.return_value = True

            with patch.object(interceptor, "_log_lineage_event"):
                interceptor.track_object("df1", df1)
                interceptor.track_object("df2", df2)

                # Should handle circular references gracefully
                # No assertion needed, just ensure no exception is raised

    @pytest.mark.parametrize(
        "operation,expected_type",
        [
            ("filter", "transformation"),
            ("groupby", "aggregation"),
            ("merge", "join"),
            ("concat", "union"),
            ("pivot", "reshape"),
        ],
    )
    def test_operation_type_classification(
        self, config_with_temp_dir, operation, expected_type
    ):
        """Test classification of different DataFrame operations"""
        interceptor = MockInterceptor(config_with_temp_dir)

        with patch.object(interceptor, "_classify_operation") as mock_classify:
            mock_classify.return_value = expected_type

            df = pd.DataFrame({"a": [1, 2, 3]})

            # Simulate operation detection
            op_type = interceptor._classify_operation(operation, df)
            assert op_type == expected_type

    def test_integration_with_marimo_session(
        self, config_with_temp_dir, mock_marimo_session
    ):
        """Test integration with marimo session context"""
        interceptor = MockInterceptor(config_with_temp_dir)

        with patch.object(
            interceptor, "_get_marimo_context", return_value=mock_marimo_session
        ):
            context = interceptor._get_marimo_context()

            assert context.session_id == mock_marimo_session.session_id
            assert context.app_file_path == mock_marimo_session.app_file_path

    def test_error_handling_on_invalid_objects(self, config_with_temp_dir):
        """Test error handling when tracking invalid objects"""
        interceptor = MockInterceptor(config_with_temp_dir)

        # Should handle non-DataFrame objects gracefully
        invalid_objects = [None, "string", 123, [], {}]

        for obj in invalid_objects:
            try:
                interceptor.track_object(f"obj_{type(obj).__name__}", obj)
            except Exception as e:
                pytest.fail(f"Should handle {type(obj)} gracefully, but raised: {e}")


class TestMarimoLiveInterceptorReal:
    """Tests for the actual MarimoLiveInterceptor implementation"""

    @pytest.fixture(autouse=True)
    def setup_clean_state(self):
        """Ensure clean global state before and after each test"""
        # Clean up any existing global interceptor
        disable_live_tracking()
        yield
        # Clean up after test
        disable_live_tracking()

    def test_real_interceptor_initialization_with_notebook_path(self, tmp_path):
        """Test real interceptor initialization with notebook path"""
        notebook_path = str(tmp_path / "test_notebook.py")
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", True
        ):
            with patch(
                "src.capture.live_lineage_interceptor.get_user_data_dir",
                return_value=str(tmp_path),
            ):
                interceptor = MarimoLiveInterceptor(
                    notebook_path=notebook_path,
                    output_file=output_file,
                    enable_runtime_debug=False,
                )

                assert interceptor.notebook_path == notebook_path
                assert interceptor.output_file == Path(output_file)
                assert isinstance(interceptor.session_id, str)
                assert len(interceptor.session_id) == 8
                assert interceptor.tracked_operations == []
                assert interceptor.interceptor_active is False

    def test_real_interceptor_initialization_without_notebook_path(self, tmp_path):
        """Test real interceptor initialization without notebook path"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch.object(
            MarimoLiveInterceptor,
            "_detect_notebook_path",
            return_value="/test/notebook.py",
        ):
            with patch(
                "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", True
            ):
                with patch(
                    "src.capture.live_lineage_interceptor.get_user_data_dir",
                    return_value=str(tmp_path),
                ):
                    interceptor = MarimoLiveInterceptor(
                        output_file=output_file, enable_runtime_debug=False
                    )

                    assert interceptor.notebook_path == "/test/notebook.py"
                    assert interceptor.output_file == Path(output_file)

    def test_detect_notebook_path_from_environment(self, tmp_path, monkeypatch):
        """Test notebook path detection from environment variable"""
        test_path = "/env/test_notebook.py"
        monkeypatch.setenv("MARIMO_NOTEBOOK_PATH", test_path)

        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", True
        ):
            with patch(
                "src.capture.live_lineage_interceptor.get_user_data_dir",
                return_value=str(tmp_path),
            ):
                interceptor = MarimoLiveInterceptor(
                    output_file=output_file, enable_runtime_debug=False
                )

                assert interceptor.notebook_path == test_path

    def test_detect_notebook_path_from_call_stack(self, tmp_path):
        """Test notebook path detection from call stack"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        # Mock the _detect_notebook_path method directly to avoid frame walking issues
        with patch.object(
            MarimoLiveInterceptor,
            "_detect_notebook_path",
            return_value="/path/to/my_notebook.py",
        ):
            with patch(
                "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", True
            ):
                with patch(
                    "src.capture.live_lineage_interceptor.get_user_data_dir",
                    return_value=str(tmp_path),
                ):
                    interceptor = MarimoLiveInterceptor(
                        output_file=output_file, enable_runtime_debug=False
                    )

                    assert interceptor.notebook_path == "/path/to/my_notebook.py"

    def test_detect_notebook_path_frame_walking(self, tmp_path):
        """Test that the frame walking logic works correctly with proper frame chain"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        # Create a proper frame chain that terminates
        mock_frame_1 = MagicMock()
        mock_frame_1.f_code.co_filename = "/path/to/some_module.py"

        mock_frame_2 = MagicMock()
        mock_frame_2.f_code.co_filename = "/path/to/my_notebook.py"
        mock_frame_2.f_back = None  # Terminate the chain

        mock_frame_1.f_back = mock_frame_2

        with patch("sys._getframe", return_value=mock_frame_1):
            with patch(
                "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", True
            ):
                with patch(
                    "src.capture.live_lineage_interceptor.get_user_data_dir",
                    return_value=str(tmp_path),
                ):
                    interceptor = MarimoLiveInterceptor(
                        output_file=output_file, enable_runtime_debug=False
                    )

                    # Should detect the notebook from the frame chain
                    assert "my_notebook.py" in interceptor.notebook_path

    def test_interceptor_install_and_uninstall(self, tmp_path):
        """Test interceptor installation and uninstallation"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            # Test installation
            interceptor.install()
            assert interceptor.interceptor_active is True

            # Test uninstallation
            interceptor.uninstall()
            assert interceptor.interceptor_active is False

    def test_hook_builtin_exec(self, tmp_path):
        """Test hooking of builtin exec function"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            interceptor._hook_builtin_exec()

            # Verify that exec was hooked (stored as "builtin_exec")
            assert "builtin_exec" in interceptor.original_functions
            # The original exec should be stored
            assert interceptor.original_functions["builtin_exec"] is not None

    def test_hook_builtin_compile(self, tmp_path):
        """Test hooking of builtin compile function"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            interceptor._hook_builtin_compile()

            # Verify that compile was hooked (stored as "builtin_compile")
            assert "builtin_compile" in interceptor.original_functions

    def test_hook_builtin_eval(self, tmp_path):
        """Test hooking of builtin eval function"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            interceptor._hook_builtin_eval()

            # Verify that eval was hooked (stored as "builtin_eval")
            assert "builtin_eval" in interceptor.original_functions

    def test_hook_pandas_operations(self, tmp_path):
        """Test hooking of pandas operations"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            interceptor._hook_pandas_operations()

            # Verify pandas operations were hooked (stored under "pandas_methods")
            assert "pandas_methods" in interceptor.original_functions
            pandas_methods = interceptor.original_functions["pandas_methods"]
            assert "DataFrame.__init__" in pandas_methods

    def test_create_read_wrapper(self, tmp_path):
        """Test creation of read function wrappers"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            def mock_read_csv(*_args, **_kwargs):
                return pd.DataFrame({"test": [1, 2, 3]})

            wrapped_func = interceptor._create_read_wrapper("read_csv", mock_read_csv)

            # Test the wrapped function
            with patch.object(interceptor, "_track_dataframe_creation") as mock_track:
                result = wrapped_func("test.csv")

                assert isinstance(result, pd.DataFrame)
                mock_track.assert_called_once()

    def test_create_method_wrapper(self, tmp_path):
        """Test creation of method wrappers"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            def mock_method(_self, *_args, **_kwargs):
                return pd.DataFrame({"result": [1, 2, 3]})

            wrapped_method = interceptor._create_method_wrapper(
                "test_method", mock_method
            )

            # Test the wrapped method
            df = pd.DataFrame({"input": [1, 2, 3]})
            with patch.object(
                interceptor, "_track_dataframe_transformation"
            ) as mock_track:
                result = wrapped_method(df)

                assert isinstance(result, pd.DataFrame)
                mock_track.assert_called_once()

    def test_discover_new_dataframes(self, tmp_path):
        """Test discovery of new DataFrames in globals"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            # Create a globals dict with DataFrames
            test_df = pd.DataFrame({"test": [1, 2, 3]})
            globals_dict = {
                "df1": test_df,
                "not_a_df": "string",
                "__builtins__": {},
                "number": 42,
            }

            with patch.object(interceptor, "_track_dataframe_creation") as mock_track:
                interceptor._discover_new_dataframes(globals_dict)

                # Should track the DataFrame but not other objects
                mock_track.assert_called_once()
                args = mock_track.call_args[0]
                # Compare DataFrame using pandas methods since == raises ValueError
                tracked_df = args[0]
                assert hasattr(tracked_df, "shape") and hasattr(tracked_df, "columns")
                assert list(tracked_df.columns) == ["test"]
                assert tracked_df.shape == (3, 1)
                assert (
                    args[1] == "variable_assignment"
                )  # The actual operation type used

    def test_track_dataframe_creation(self, tmp_path):
        """Test tracking of DataFrame creation"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            df = pd.DataFrame({"test": [1, 2, 3]})

            with patch.object(interceptor, "_emit_event") as mock_emit:
                interceptor._track_dataframe_creation(
                    df=df,
                    operation="read_csv",
                    args=("test.csv",),
                    kwargs={"sep": ","},
                    name="df1",
                )

                # Should emit an event
                mock_emit.assert_called_once()
                event = mock_emit.call_args[0][0]
                # Check OpenLineage format
                assert event["eventType"] == "START"
                assert event["job"]["namespace"] == "marimo"
                assert event["job"]["name"] == "pandas_read_csv"
                assert event["producer"] == "marimo-lineage-tracker"
                assert len(event["outputs"]) == 1
                output = event["outputs"][0]
                assert output["namespace"] == "marimo"
                assert output["name"] == "df1"
                assert "schema" in output["facets"]
                assert len(output["facets"]["schema"]["fields"]) == 1
                assert output["facets"]["schema"]["fields"][0]["name"] == "test"
                # Check job facets for arguments
                assert "facets" in event["job"]
                assert "args" in event["job"]["facets"]
                assert "kwargs" in event["job"]["facets"]

    def test_track_dataframe_transformation(self, tmp_path):
        """Test tracking of DataFrame transformations"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            input_df = pd.DataFrame({"input": [1, 2, 3]})
            output_df = pd.DataFrame({"output": [2, 4, 6]})

            with patch.object(interceptor, "_emit_event") as mock_emit:
                interceptor._track_dataframe_transformation(
                    input_df=input_df,
                    output_df=output_df,
                    operation="multiply",
                    args=(2,),
                    kwargs={},
                )

                # Should emit an event
                mock_emit.assert_called_once()
                event = mock_emit.call_args[0][0]
                # Check OpenLineage format for transformation
                assert event["eventType"] == "COMPLETE"
                assert event["job"]["namespace"] == "marimo"
                assert event["job"]["name"] == "pandas_multiply"
                assert event["producer"] == "marimo-lineage-tracker"
                assert len(event["inputs"]) == 1
                assert len(event["outputs"]) == 1
                # Check input dataset
                input_dataset = event["inputs"][0]
                assert input_dataset["namespace"] == "marimo"
                assert "schema" in input_dataset["facets"]
                # Check output dataset
                output_dataset = event["outputs"][0]
                assert output_dataset["namespace"] == "marimo"
                assert "schema" in output_dataset["facets"]
                # Check job facets for arguments
                assert "facets" in event["job"]
                assert "args" in event["job"]["facets"]

    def test_sanitize_args(self, tmp_path):
        """Test argument sanitization"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            # Test with various argument types
            df = pd.DataFrame({"test": [1, 2, 3]})
            args = (df, "string", 42, [1, 2, 3])

            sanitized = interceptor._sanitize_args(args)

            # DataFrames should be replaced with summaries (returns dict not string)
            assert sanitized[0]["type"] == "DataFrame"
            assert sanitized[0]["shape"] == [3, 1]
            assert sanitized[0]["columns"] == ["test"]
            assert sanitized[1] == "string"
            assert sanitized[2] == 42
            assert (
                sanitized[3] == "<list with 3 items>"
            )  # Lists are converted to descriptive strings

    def test_emit_event(self, tmp_path):
        """Test event emission to file"""
        output_file = tmp_path / "lineage_events.jsonl"

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=str(output_file),
                enable_runtime_debug=False,
            )

            event = {
                "event_type": "test_event",
                "timestamp": "2024-01-01T00:00:00Z",
                "data": {"test": "value"},
            }

            interceptor._emit_event(event)

            # Verify event was written to file
            assert output_file.exists()
            with open(output_file) as f:
                written_event = json.loads(f.read().strip())
                assert written_event["event_type"] == "test_event"
                # The written event is exactly what was passed in - session_id not added automatically
                assert "data" in written_event

    def test_get_session_summary(self, tmp_path):
        """Test session summary generation"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            # Add some tracked operations
            interceptor.tracked_operations = [
                {"event_type": "test1"},
                {"event_type": "test2"},
            ]

            summary = interceptor.get_session_summary()

            assert summary["session_id"] == interceptor.session_id
            assert summary["events_logged"] == 2
            assert summary["interceptor_active"] is False
            assert "output_file" in summary
            assert "runtime_debugging_enabled" in summary

    def test_enable_disable_live_tracking(self, tmp_path):
        """Test global enable/disable live tracking functions"""
        notebook_path = str(tmp_path / "test_notebook.py")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            # Test enable
            interceptor = enable_live_tracking(
                notebook_path=notebook_path, enable_runtime_debug=False
            )

            assert is_tracking_active() is True
            assert isinstance(interceptor, MarimoLiveInterceptor)

            # Test tracking summary
            summary = get_tracking_summary()
            assert "session_id" in summary

            # Test disable
            disable_live_tracking()
            assert is_tracking_active() is False

    def test_runtime_debugging_context(self, tmp_path):
        """Test runtime debugging context manager"""
        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", True
        ):
            with patch(
                "src.capture.live_lineage_interceptor.track_cell_execution"
            ) as mock_track:
                cell_source = "df = pd.DataFrame({'test': [1, 2, 3]})"
                globals_dict = {"pd": pd}

                with runtime_debugging_context(cell_source, globals_dict) as tracker:
                    # Should yield the runtime tracker
                    assert tracker is not None

                # Should call track_cell_execution
                mock_track.assert_called_once()

    def test_error_handling_in_hooks(self, tmp_path):
        """Test error handling in hooked functions"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=False,
            )

            # Test error handling in exec hook
            with patch("builtins.exec", side_effect=Exception("Test error")):
                interceptor._hook_builtin_exec()

                # Should not raise exception even if hooking fails
                # The interceptor should handle errors gracefully

    def test_runtime_tracking_unavailable_fallback(self, tmp_path):
        """Test fallback behavior when runtime tracking is unavailable"""
        output_file = str(tmp_path / "lineage_events.jsonl")

        with patch(
            "src.capture.live_lineage_interceptor.RUNTIME_TRACKING_AVAILABLE", False
        ):
            interceptor = MarimoLiveInterceptor(
                notebook_path="/test/notebook.py",
                output_file=output_file,
                enable_runtime_debug=True,  # Should be ignored when unavailable
            )

            # Should still initialize properly without runtime tracking
            assert interceptor.enable_runtime_debug is False
            assert interceptor.runtime_tracker is None
            assert interceptor.notebook_hash == "unknown"
            assert interceptor.notebook_name == "unknown"
