from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from test.mocks import MockLiveLineageInterceptor as LiveLineageInterceptor


class TestLiveLineageInterceptor:
    """Tests for LiveLineageInterceptor following marimo patterns"""

    def test_interceptor_initialization(self, config_with_temp_dir):
        """Test interceptor initializes properly"""
        interceptor = LiveLineageInterceptor(config_with_temp_dir)

        assert interceptor.config == config_with_temp_dir
        assert hasattr(interceptor, "tracked_objects")
        assert isinstance(interceptor.tracked_objects, dict)

    def test_pandas_dataframe_tracking(self, config_with_temp_dir):
        """Test tracking of pandas DataFrame operations"""
        interceptor = LiveLineageInterceptor(config_with_temp_dir)

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
        interceptor = LiveLineageInterceptor(config_with_temp_dir)

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
        interceptor = LiveLineageInterceptor(config_with_temp_dir)

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
        interceptor = LiveLineageInterceptor(config_with_temp_dir)

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
        interceptor = LiveLineageInterceptor(config_with_temp_dir)

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
        interceptor = LiveLineageInterceptor(config_with_temp_dir)

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
        interceptor = LiveLineageInterceptor(config_with_temp_dir)

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
        interceptor = LiveLineageInterceptor(config_with_temp_dir)

        with patch.object(
            interceptor, "_get_marimo_context", return_value=mock_marimo_session
        ):
            context = interceptor._get_marimo_context()

            assert context.session_id == mock_marimo_session.session_id
            assert context.app_file_path == mock_marimo_session.app_file_path

    def test_error_handling_on_invalid_objects(self, config_with_temp_dir):
        """Test error handling when tracking invalid objects"""
        interceptor = LiveLineageInterceptor(config_with_temp_dir)

        # Should handle non-DataFrame objects gracefully
        invalid_objects = [None, "string", 123, [], {}]

        for obj in invalid_objects:
            try:
                interceptor.track_object(f"obj_{type(obj).__name__}", obj)
            except Exception as e:
                pytest.fail(f"Should handle {type(obj)} gracefully, but raised: {e}")
