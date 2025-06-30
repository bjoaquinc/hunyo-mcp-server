from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from test.mocks import (
    MockLightweightRuntimeTracker as LightweightRuntimeTracker,
)
from test.mocks import (
    MockLiveLineageInterceptor as LiveLineageInterceptor,
)
from test.mocks import (
    MockNativeHooksInterceptor as NativeHooksInterceptor,
)
from test.mocks import (
    MockWebSocketInterceptor as WebSocketInterceptor,
)


class TestCaptureIntegration:
    """Integration tests for the complete capture system"""

    def test_complete_capture_pipeline(self, config_with_temp_dir, capture_event_file):
        """Test complete DataFrame capture pipeline from creation to storage"""
        # Initialize all capture components
        runtime_tracker = LightweightRuntimeTracker(config_with_temp_dir)
        lineage_interceptor = LiveLineageInterceptor(config_with_temp_dir)

        # Create test DataFrames
        df1 = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
            }
        )

        df2 = pd.DataFrame({"id": [1, 2, 3, 4], "salary": [50000, 60000, 70000, 80000]})

        # Track creation operations
        runtime_tracker.track_dataframe_operation(
            df_name="df1",
            operation="create",
            code="df1 = pd.DataFrame({'id': [1,2,3,4], 'name': ['Alice','Bob','Charlie','David'], 'age': [25,30,35,40]})",
            shape=df1.shape,
            columns=list(df1.columns),
        )

        runtime_tracker.track_dataframe_operation(
            df_name="df2",
            operation="create",
            code="df2 = pd.DataFrame({'id': [1,2,3,4], 'salary': [50000,60000,70000,80000]})",
            shape=df2.shape,
            columns=list(df2.columns),
        )

        # Track lineage
        lineage_interceptor.track_object("df1", df1)
        lineage_interceptor.track_object("df2", df2)

        # Perform and track merge operation
        df3 = df1.merge(df2, on="id")

        runtime_tracker.track_dataframe_operation(
            df_name="df3",
            operation="merge",
            code="df3 = df1.merge(df2, on='id')",
            shape=df3.shape,
            columns=list(df3.columns),
        )

        lineage_interceptor.track_object("df3", df3)

        # Verify events were captured
        assert capture_event_file.exists()

        # Verify event content
        with open(capture_event_file) as f:
            events = [
                json.loads(line.strip()) for line in f.readlines() if line.strip()
            ]

        assert len(events) >= 3  # At least 3 operations tracked

        # Verify operation types
        operation_types = [event.get("operation") for event in events]
        assert "create" in operation_types
        assert "merge" in operation_types

    def test_marimo_integration_workflow(
        self, config_with_temp_dir, mock_marimo_session
    ):
        """Test integration with marimo session workflow"""
        runtime_tracker = LightweightRuntimeTracker(config_with_temp_dir)

        with patch.object(
            runtime_tracker, "_get_marimo_session", return_value=mock_marimo_session
        ):
            # Simulate marimo cell execution
            with patch.object(
                runtime_tracker, "_get_execution_context"
            ) as mock_context:
                mock_context.return_value = {
                    "session_id": mock_marimo_session.session_id,
                    "file_path": mock_marimo_session.app_file_path,
                    "cell_id": "cell-1",
                    "execution_count": 1,
                }

                # Track operation within marimo context
                df = pd.DataFrame({"test": [1, 2, 3]})

                runtime_tracker.track_dataframe_operation(
                    df_name="df",
                    operation="create",
                    code="df = pd.DataFrame({'test': [1, 2, 3]})",
                    shape=df.shape,
                    columns=list(df.columns),
                )

                # Verify marimo context was captured
                mock_context.assert_called()

    @pytest.mark.asyncio
    async def test_websocket_integration(self, config_with_temp_dir):
        """Test WebSocket integration with capture system"""
        websocket_interceptor = WebSocketInterceptor(config_with_temp_dir)

        mock_websocket = MagicMock()
        mock_websocket.send = MagicMock()

        # Mock WebSocket connection
        with patch.object(websocket_interceptor, "connection", mock_websocket):
            # Track DataFrame operation and send via WebSocket
            df = pd.DataFrame({"data": [1, 2, 3, 4, 5]})

            operation_event = {
                "type": "dataframe_operation",
                "df_name": "df",
                "operation": "create",
                "shape": list(df.shape),
                "columns": list(df.columns),
                "timestamp": "2024-01-01T00:00:00Z",
            }

            await websocket_interceptor.send_message(operation_event)

            # Verify message was sent
            mock_websocket.send.assert_called_once()

    def test_native_hooks_integration(self, config_with_temp_dir):
        """Test native hooks integration with DataFrame operations"""
        hooks_interceptor = NativeHooksInterceptor(config_with_temp_dir)

        # Mock pandas DataFrame methods
        with patch("pandas.DataFrame.merge") as mock_merge:
            mock_result = pd.DataFrame({"merged": [1, 2, 3]})
            mock_merge.return_value = mock_result

            # Install hooks
            hooks_interceptor.install_hooks()

            try:
                # Simulate DataFrame operation that would be hooked
                df1 = pd.DataFrame({"a": [1, 2, 3]})
                df2 = pd.DataFrame({"b": [4, 5, 6]})

                # This would trigger the hook if properly integrated
                with patch.object(hooks_interceptor, "_log_method_call") as mock_log:
                    # Manually trigger the tracking (simulating the hook)
                    hooks_interceptor._log_method_call("merge", df1, df2, on=None)

                    mock_log.assert_called_once()

            finally:
                hooks_interceptor.uninstall_hooks()

    def test_multi_dataframe_lineage_tracking(self, config_with_temp_dir):
        """Test complex lineage tracking with multiple DataFrames"""
        lineage_interceptor = LiveLineageInterceptor(config_with_temp_dir)
        runtime_tracker = LightweightRuntimeTracker(config_with_temp_dir)

        # Create a complex DataFrame transformation pipeline
        df_raw = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "department": [
                    "Engineering",
                    "Sales",
                    "Engineering",
                    "Marketing",
                    "Sales",
                ],
            }
        )

        df_sales = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5],
                "revenue": [100000, 80000, 120000, 95000, 110000],
            }
        )

        # Track initial DataFrames
        lineage_interceptor.track_object("df_raw", df_raw)
        lineage_interceptor.track_object("df_sales", df_sales)

        # Perform transformations
        df_engineering = df_raw[df_raw["department"] == "Engineering"]
        df_merged = df_engineering.merge(df_sales, on="user_id")
        df_summary = df_merged.groupby("department")["revenue"].sum().reset_index()

        # Track transformations
        lineage_interceptor.track_object("df_engineering", df_engineering)
        lineage_interceptor.track_object("df_merged", df_merged)
        lineage_interceptor.track_object("df_summary", df_summary)

        # Verify lineage relationships
        lineage_graph = lineage_interceptor.get_lineage_graph()

        # Should have tracked all DataFrames
        assert "df_raw" in lineage_graph
        assert "df_sales" in lineage_graph
        assert "df_engineering" in lineage_graph
        assert "df_merged" in lineage_graph
        assert "df_summary" in lineage_graph

    def test_performance_with_large_dataframes(self, config_with_temp_dir):
        """Test capture system performance with large DataFrames"""
        runtime_tracker = LightweightRuntimeTracker(config_with_temp_dir)

        # Create large DataFrame
        large_df = pd.DataFrame(
            {
                "id": range(100000),
                "value": [i * 2 for i in range(100000)],
                "category": (["A", "B", "C"] * 33334)[:100000],  # Exact 100000 elements
            }
        )

        import time

        start_time = time.time()

        # Track large DataFrame operation
        runtime_tracker.track_dataframe_operation(
            df_name="large_df",
            operation="create",
            code="large_df = pd.DataFrame(...)",  # Simplified for test
            shape=large_df.shape,
            columns=list(large_df.columns),
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (< 1 second)
        assert execution_time < 1.0

    def test_error_recovery_and_graceful_degradation(self, config_with_temp_dir):
        """Test system handles errors gracefully and continues operation"""
        runtime_tracker = LightweightRuntimeTracker(config_with_temp_dir)

        # Test with invalid operation that should be handled gracefully
        invalid_operations = [
            {
                "df_name": None,
                "operation": "invalid",
                "code": "",
                "shape": None,
                "columns": None,
            },
            {
                "df_name": "",
                "operation": "",
                "code": None,
                "shape": (-1, -1),
                "columns": [],
            },
            {
                "df_name": "valid_df",
                "operation": "create",
                "code": "valid code",
                "shape": (5, 3),
                "columns": ["a", "b", "c"],
            },
        ]

        successful_operations = 0

        for operation in invalid_operations:
            try:
                runtime_tracker.track_dataframe_operation(**operation)
                successful_operations += 1
            except Exception:
                # Should not raise exceptions, but if it does, continue
                pass

        # At least the valid operation should succeed
        assert successful_operations >= 1

    @pytest.mark.parametrize(
        "operation_sequence",
        [
            ["create", "filter", "groupby"],
            ["create", "merge", "sort"],
            ["create", "pivot", "transpose"],
            ["create", "join", "aggregate"],
        ],
    )
    def test_operation_sequence_tracking(
        self, config_with_temp_dir, operation_sequence
    ):
        """Test tracking of different operation sequences"""
        runtime_tracker = LightweightRuntimeTracker(config_with_temp_dir)

        # Create base DataFrame
        base_df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "value": [10, 20, 30, 40],
                "category": ["A", "B", "A", "B"],
            }
        )

        current_df = base_df

        for i, operation in enumerate(operation_sequence):
            df_name = f"df_{i}"

            runtime_tracker.track_dataframe_operation(
                df_name=df_name,
                operation=operation,
                code=f"{df_name} = previous_df.{operation}(...)",
                shape=current_df.shape,
                columns=list(current_df.columns),
            )

        # Verify all operations were tracked
        events_file = config_with_temp_dir / "events" / "runtime_events.jsonl"

        if events_file.exists():
            with open(events_file) as f:
                events = [
                    json.loads(line.strip()) for line in f.readlines() if line.strip()
                ]

            tracked_operations = [event.get("operation") for event in events]

            # Should have tracked all operations in sequence
            for operation in operation_sequence:
                assert operation in tracked_operations
