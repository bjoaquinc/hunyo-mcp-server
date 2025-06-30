from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from test.mocks import MockLightweightRuntimeTracker as LightweightRuntimeTracker


class TestLightweightRuntimeTracker:
    """Tests for LightweightRuntimeTracker following marimo patterns"""

    def test_tracker_initialization(self, config_with_temp_dir):
        """Test tracker initializes with proper configuration"""
        tracker = LightweightRuntimeTracker(config_with_temp_dir)

        assert tracker.config == config_with_temp_dir
        assert tracker.config.events_dir.exists()
        assert tracker.config.db_dir.exists()

    def test_dataframe_operation_tracking(
        self, config_with_temp_dir, sample_dataframe_operations, capture_event_file
    ):
        """Test tracking of DataFrame operations"""
        tracker = LightweightRuntimeTracker(config_with_temp_dir)

        # Mock the event logging
        with patch.object(tracker, "_log_event") as mock_log:
            for operation in sample_dataframe_operations:
                tracker.track_dataframe_operation(
                    operation["df_name"],
                    operation["operation"],
                    operation["code"],
                    operation["shape"],
                    operation["columns"],
                )

        assert mock_log.call_count == len(sample_dataframe_operations)

        # Verify the logged events
        for i, operation in enumerate(sample_dataframe_operations):
            call_args = mock_log.call_args_list[i]
            event = call_args[0][0]

            assert event["df_name"] == operation["df_name"]
            assert event["operation"] == operation["operation"]
            assert event["code"] == operation["code"]

    def test_execution_context_tracking(
        self, config_with_temp_dir, mock_marimo_session
    ):
        """Test execution context is properly tracked"""
        tracker = LightweightRuntimeTracker(config_with_temp_dir)

        with patch.object(
            tracker, "_get_marimo_session", return_value=mock_marimo_session
        ):
            context = tracker._get_execution_context()

            assert context["session_id"] == mock_marimo_session.session_id
            assert context["file_path"] == mock_marimo_session.app_file_path

    def test_event_file_writing(self, config_with_temp_dir, capture_event_file):
        """Test events are properly written to JSONL file"""
        tracker = LightweightRuntimeTracker(config_with_temp_dir)

        test_event = {
            "timestamp": "2024-01-01T00:00:00Z",
            "event_type": "dataframe_operation",
            "df_name": "test_df",
            "operation": "create",
        }

        tracker._log_event(test_event)

        # Verify event was written
        assert capture_event_file.exists()

        with open(capture_event_file) as f:
            written_event = json.loads(f.readline().strip())

        assert written_event["df_name"] == test_event["df_name"]
        assert written_event["operation"] == test_event["operation"]

    def test_error_handling_on_invalid_operation(self, config_with_temp_dir):
        """Test error handling for invalid DataFrame operations"""
        tracker = LightweightRuntimeTracker(config_with_temp_dir)

        # Should not raise exception, should handle gracefully
        try:
            tracker.track_dataframe_operation(
                df_name=None,  # Invalid name
                operation="invalid_op",
                code="",
                shape=None,
                columns=None,
            )
        except Exception as e:
            pytest.fail(
                f"Tracker should handle invalid operations gracefully, but raised: {e}"
            )

    @pytest.mark.parametrize(
        "operation_type", ["create", "transform", "filter", "merge", "group"]
    )
    def test_different_operation_types(self, config_with_temp_dir, operation_type):
        """Test tracking different types of DataFrame operations"""
        tracker = LightweightRuntimeTracker(config_with_temp_dir)

        with patch.object(tracker, "_log_event") as mock_log:
            tracker.track_dataframe_operation(
                df_name=f"df_{operation_type}",
                operation=operation_type,
                code=f"df_{operation_type} = df.{operation_type}()",
                shape=(10, 5),
                columns=["col1", "col2", "col3", "col4", "col5"],
            )

        mock_log.assert_called_once()
        event = mock_log.call_args[0][0]
        assert event["operation"] == operation_type
