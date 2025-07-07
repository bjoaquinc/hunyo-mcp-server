#!/usr/bin/env python3
"""
Test DataFrame lineage event processing integration.

Tests the complete pipeline from schema validation through database insertion
for DataFrame lineage events.
"""

import json
import tempfile
from pathlib import Path

import pytest

from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager
from hunyo_mcp_server.ingestion.event_processor import EventProcessor


class TestDataFrameLineageEventProcessing:
    """Test DataFrame lineage event processing integration."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_hunyo.duckdb"
            yield db_path

    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create a DuckDBManager instance."""
        manager = DuckDBManager(temp_db_path)
        manager.initialize_database()
        yield manager
        manager.close()

    @pytest.fixture
    def event_processor(self, db_manager):
        """Create an EventProcessor instance."""
        return EventProcessor(db_manager)

    @pytest.fixture
    def sample_dataframe_lineage_event(self):
        """Create a sample DataFrame lineage event."""
        return {
            "event_type": "dataframe_lineage",
            "execution_id": "abcd1234",
            "cell_id": "cell_456",
            "session_id": "7890abcd",
            "timestamp": "2023-08-01T12:00:00.000Z",
            "emitted_at": "2023-08-01T12:00:00.001Z",
            "operation_type": "selection",
            "operation_method": "__getitem__",
            "operation_code": "df[df['column'] > 100]",
            "operation_parameters": {
                "columns": ["column"],
                "condition": "column > 100",
                "is_boolean_mask": True,
            },
            "input_dataframes": [
                {
                    "variable_name": "df",
                    "object_id": "df_1",
                    "shape": [1000, 5],
                    "columns": [
                        "column",
                        "other_col",
                        "third_col",
                        "fourth_col",
                        "fifth_col",
                    ],
                    "memory_usage_mb": 0.4,
                }
            ],
            "output_dataframes": [
                {
                    "variable_name": "filtered_df",
                    "object_id": "df_2",
                    "shape": [150, 5],
                    "columns": [
                        "column",
                        "other_col",
                        "third_col",
                        "fourth_col",
                        "fifth_col",
                    ],
                    "memory_usage_mb": 0.06,
                }
            ],
            "column_lineage": {
                "filtered_df.column": ["df.column"],
                "filtered_df.other_col": ["df.other_col"],
                "filtered_df.third_col": ["df.third_col"],
                "filtered_df.fourth_col": ["df.fourth_col"],
                "filtered_df.fifth_col": ["df.fifth_col"],
            },
            "performance": {"overhead_ms": 2.5, "df_size_mb": 0.5, "sampled": False},
        }

    def test_schema_loading(self, event_processor):
        """Test that the DataFrame lineage schema is loaded correctly."""
        assert event_processor.dataframe_lineage_schema is not None
        assert "type" in event_processor.dataframe_lineage_schema
        assert "properties" in event_processor.dataframe_lineage_schema
        assert "event_type" in event_processor.dataframe_lineage_schema["properties"]
        assert (
            "operation_type" in event_processor.dataframe_lineage_schema["properties"]
        )

    def test_event_validation(self, event_processor, sample_dataframe_lineage_event):
        """Test DataFrame lineage event validation."""
        is_valid, error_msg = event_processor._validate_event_against_schema(
            sample_dataframe_lineage_event,
            event_processor.dataframe_lineage_schema,
            "dataframe_lineage",
        )
        assert is_valid, f"Event validation failed: {error_msg}"

    def test_event_transformation(
        self, event_processor, sample_dataframe_lineage_event
    ):
        """Test DataFrame lineage event transformation."""
        transformed = event_processor._transform_dataframe_lineage_event(
            sample_dataframe_lineage_event
        )

        # Check required fields
        assert transformed["event_type"] == "dataframe_lineage"
        assert transformed["execution_id"] == "abcd1234"
        assert transformed["cell_id"] == "cell_456"
        assert transformed["session_id"] == "7890abcd"
        assert transformed["operation_type"] == "selection"
        assert transformed["operation_method"] == "__getitem__"
        assert transformed["operation_code"] == "df[df['column'] > 100]"

        # Check JSON fields
        assert transformed["operation_parameters"] == {
            "columns": ["column"],
            "condition": "column > 100",
            "is_boolean_mask": True,
        }
        assert len(transformed["input_dataframes"]) == 1
        assert len(transformed["output_dataframes"]) == 1
        assert "filtered_df.column" in transformed["column_lineage"]
        assert "filtered_df.other_col" in transformed["column_lineage"]
        assert "filtered_df.third_col" in transformed["column_lineage"]
        assert "filtered_df.fourth_col" in transformed["column_lineage"]
        assert "filtered_df.fifth_col" in transformed["column_lineage"]

        # Check performance fields (now extracted from nested structure)
        assert transformed["overhead_ms"] == 2.5
        assert transformed["df_size_mb"] == 0.5
        assert transformed["sampled"] is False

        # Check timestamp parsing
        assert transformed["timestamp"] is not None
        assert transformed["emitted_at"] is not None

        # Check generated ID
        assert transformed["df_event_id"] is not None
        assert isinstance(transformed["df_event_id"], int)

    def test_single_event_processing(
        self, event_processor, sample_dataframe_lineage_event
    ):
        """Test processing a single DataFrame lineage event."""
        events = [sample_dataframe_lineage_event]
        result = event_processor.process_dataframe_lineage_events(events)

        assert result == 1
        assert event_processor.processed_events == 1
        assert event_processor.failed_events == 0

    def test_database_insertion(self, db_manager, sample_dataframe_lineage_event):
        """Test inserting DataFrame lineage events into the database."""
        # Transform the event first
        processor = EventProcessor(db_manager)
        transformed = processor._transform_dataframe_lineage_event(
            sample_dataframe_lineage_event
        )

        # Insert the event
        db_manager.insert_dataframe_lineage_event(transformed)

        # Verify insertion
        result = db_manager.execute_query(
            "SELECT COUNT(*) as count FROM dataframe_lineage_events"
        )
        assert result[0]["count"] == 1

        # Verify event data
        result = db_manager.execute_query(
            "SELECT * FROM dataframe_lineage_events WHERE df_event_id = ?",
            [transformed["df_event_id"]],
        )
        assert len(result) == 1

        row = result[0]
        assert row["event_type"] == "dataframe_lineage"
        assert row["execution_id"] == "abcd1234"
        assert row["operation_type"] == "selection"
        assert row["operation_method"] == "__getitem__"
        assert row["overhead_ms"] == 2.5
        assert row["df_size_mb"] == 0.5
        assert row["sampled"] is False

    def test_jsonl_file_processing(
        self, event_processor, sample_dataframe_lineage_event
    ):
        """Test processing DataFrame lineage events from a JSONL file."""
        # Create a valid event with proper hex execution_id
        event = sample_dataframe_lineage_event.copy()
        event["execution_id"] = "abc12345"  # Valid 8-char hex

        # Create temporary JSONL file
        jsonl_content = json.dumps(event) + "\n"
        jsonl_content += (
            json.dumps(event) + "\n"
        )  # Add duplicate to test batch processing

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            jsonl_path = f.name

        try:
            # Process the file - convert string path to Path object
            result = event_processor.process_jsonl_file(
                Path(jsonl_path), "dataframe_lineage"
            )
            assert result == 2  # Should process 2 events
        finally:
            # Clean up
            Path(jsonl_path).unlink()

    def test_validation_summary(self, event_processor, sample_dataframe_lineage_event):
        """Test validation summary includes DataFrame lineage schema."""
        # Process some events first
        events = [sample_dataframe_lineage_event]
        event_processor.process_dataframe_lineage_events(events)

        summary = event_processor.get_validation_summary()

        assert summary["processed_events"] == 1
        assert summary["failed_events"] == 0
        assert summary["total_events"] == 1
        assert summary["success_rate"] == 1.0
        assert summary["schemas_loaded"]["dataframe_lineage_schema"] is True

    def test_database_stats_includes_dataframe_lineage(self, temp_db_path):
        """Test that database stats include DataFrame lineage events count."""
        from hunyo_mcp_server.ingestion.duckdb_manager import get_database_stats

        stats = get_database_stats(temp_db_path)

        assert "dataframe_lineage_events_count" in stats
        assert stats["dataframe_lineage_events_count"] == 0  # No events inserted yet

    def test_invalid_event_handling(self, event_processor):
        """Test handling of invalid DataFrame lineage events."""
        # Missing required fields
        invalid_event = {
            "event_type": "dataframe_lineage",
            # Missing session_id, emitted_at, operation_type
        }

        events = [invalid_event]
        result = event_processor.process_dataframe_lineage_events(events)

        assert result == 0
        assert event_processor.processed_events == 0
        assert event_processor.failed_events == 1

    def test_mixed_event_types_processing(
        self, event_processor, sample_dataframe_lineage_event
    ):
        """Test processing different types of DataFrame lineage events."""
        # Selection event
        selection_event = sample_dataframe_lineage_event.copy()
        selection_event["operation_type"] = "selection"
        selection_event["operation_method"] = "__getitem__"
        selection_event["execution_id"] = "aaa12345"  # Valid 8-char hex

        # Aggregation event
        aggregation_event = sample_dataframe_lineage_event.copy()
        aggregation_event["operation_type"] = "aggregation"
        aggregation_event["operation_method"] = "groupby"
        aggregation_event["execution_id"] = "bbb12345"  # Valid 8-char hex

        # Join event
        join_event = sample_dataframe_lineage_event.copy()
        join_event["operation_type"] = "join"
        join_event["operation_method"] = "merge"
        join_event["execution_id"] = "ccc12345"  # Valid 8-char hex

        events = [selection_event, aggregation_event, join_event]
        result = event_processor.process_dataframe_lineage_events(events)

        assert result == 3
        assert event_processor.processed_events == 3
        assert event_processor.failed_events == 0

    def test_view_integration(self, db_manager, sample_dataframe_lineage_event):
        """Test that DataFrame lineage events work with the database view."""
        # Insert a test event
        processor = EventProcessor(db_manager)
        transformed = processor._transform_dataframe_lineage_event(
            sample_dataframe_lineage_event
        )
        db_manager.insert_dataframe_lineage_event(transformed)

        # Query the view
        result = db_manager.execute_query(
            "SELECT * FROM vw_dataframe_lineage WHERE df_event_id = ?",
            [transformed["df_event_id"]],
        )

        assert len(result) == 1
        row = result[0]

        # Check that JSON fields are extracted
        assert row["operation_type"] == "selection"
        assert row["operation_method"] == "__getitem__"

        # Check view counts (simplified for now)
        assert row["df_event_id"] == transformed["df_event_id"]
