#!/usr/bin/env python3
"""
Tests for EventProcessor - Validates and processes JSONL events for database insertion.

Tests cover event validation, transformation, batch processing, and error handling.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager
from hunyo_mcp_server.ingestion.event_processor import EventProcessor


class TestEventProcessor:
    """Tests for EventProcessor following project patterns"""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock DuckDBManager for testing"""
        db_manager = MagicMock(spec=DuckDBManager)
        db_manager.begin_transaction.return_value = None
        db_manager.commit_transaction.return_value = None
        db_manager.rollback_transaction.return_value = None
        db_manager.insert_runtime_event.return_value = None
        db_manager.insert_lineage_event.return_value = None
        return db_manager

    @pytest.fixture
    def event_processor(self, mock_db_manager):
        """Create EventProcessor instance for testing"""
        return EventProcessor(mock_db_manager, strict_validation=False)

    @pytest.fixture
    def sample_runtime_event(self):
        """Sample runtime event for testing with schema-compliant format"""
        return {
            "event_type": "cell_execution_start",  # Must be from enum
            "execution_id": "abcd1234",  # 8-character hex string
            "cell_id": "cell_456",
            "cell_source": "x = 5\nprint(x)",
            "cell_source_lines": 2,
            "start_memory_mb": 100.5,
            "end_memory_mb": 102.3,
            "duration_ms": 250,
            "timestamp": "2024-01-01T00:00:00.000Z",
            "session_id": "5e555678",  # 8-character hex string
            "emitted_at": "2024-01-01T00:00:00.000Z",
            # error_info omitted since there's no error (optional field)
        }

    @pytest.fixture
    def sample_openlineage_event(self):
        """Sample OpenLineage event for testing with all required fields."""
        return {
            "eventType": "COMPLETE",
            "eventTime": "2024-01-01T00:00:00.000Z",
            "run": {
                "runId": str(uuid.uuid4()),
                "facets": {
                    "performance": {
                        "_producer": "marimo-lineage-tracker",
                        "durationMs": 150,
                    },
                    "marimoExecution": {
                        "_producer": "marimo-lineage-tracker",
                        "executionId": "abcd1234",  # 8-character hex string
                        "sessionId": "5e555678",  # 8-character hex string
                    },
                },
            },
            "job": {"namespace": "marimo", "name": "pandas_read_csv"},
            "inputs": [],
            "outputs": [],
            "producer": "marimo-lineage-tracker",
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
            "session_id": "5e555678",  # 8-character hex string
            "emitted_at": "2024-01-01T00:00:00.000Z",
        }

    def test_initialization(self, mock_db_manager):
        """Test EventProcessor initialization"""
        processor = EventProcessor(mock_db_manager)

        assert processor.db_manager == mock_db_manager
        assert processor.processed_events == 0
        assert processor.failed_events == 0

    def test_process_runtime_events_success(
        self, event_processor, sample_runtime_event
    ):
        """Test successful processing of runtime events"""
        events = [sample_runtime_event]

        result = event_processor.process_runtime_events(events)

        assert result == 1
        assert event_processor.processed_events == 1
        assert event_processor.failed_events == 0

        # Verify database calls
        event_processor.db_manager.begin_transaction.assert_called_once()
        event_processor.db_manager.commit_transaction.assert_called_once()
        event_processor.db_manager.insert_runtime_event.assert_called_once()

    def test_process_runtime_events_empty_list(self, event_processor):
        """Test processing empty event list"""
        result = event_processor.process_runtime_events([])

        assert result == 0
        assert event_processor.processed_events == 0
        assert event_processor.failed_events == 0

        # No database calls should be made
        event_processor.db_manager.begin_transaction.assert_not_called()

    def test_process_runtime_events_validation_error(self, event_processor):
        """Test handling of validation errors in runtime events"""
        # Invalid event missing required fields
        invalid_event = {"event_type": "cell_execution_start"}

        result = event_processor.process_runtime_events([invalid_event])

        assert result == 0
        assert event_processor.processed_events == 0
        assert event_processor.failed_events == 1

        # Transaction should still be committed (empty)
        event_processor.db_manager.begin_transaction.assert_called_once()
        event_processor.db_manager.commit_transaction.assert_called_once()

    def test_process_runtime_events_database_error(
        self, event_processor, sample_runtime_event
    ):
        """Test handling of database errors during runtime event processing"""
        # Mock database error
        event_processor.db_manager.insert_runtime_event.side_effect = Exception(
            "DB Error"
        )

        result = event_processor.process_runtime_events([sample_runtime_event])

        assert result == 0
        assert event_processor.failed_events == 1

        # Transaction should be committed even with individual event failures
        event_processor.db_manager.begin_transaction.assert_called_once()
        event_processor.db_manager.commit_transaction.assert_called_once()

    def test_process_runtime_events_transaction_error(
        self, event_processor, sample_runtime_event
    ):
        """Test handling of transaction errors during runtime event processing"""
        # Mock transaction error
        event_processor.db_manager.commit_transaction.side_effect = Exception(
            "Transaction Error"
        )

        with pytest.raises(Exception, match="Transaction Error"):
            event_processor.process_runtime_events([sample_runtime_event])

        event_processor.db_manager.rollback_transaction.assert_called_once()

    def test_process_lineage_events_success(
        self, event_processor, sample_openlineage_event
    ):
        """Test successful processing of lineage events"""
        events = [sample_openlineage_event]

        result = event_processor.process_lineage_events(events)

        assert result == 1
        assert event_processor.processed_events == 1
        assert event_processor.failed_events == 0

        # Verify database calls
        event_processor.db_manager.begin_transaction.assert_called_once()
        event_processor.db_manager.commit_transaction.assert_called_once()
        event_processor.db_manager.insert_lineage_event.assert_called_once()

    def test_process_lineage_events_empty_list(self, event_processor):
        """Test processing empty lineage event list"""
        result = event_processor.process_lineage_events([])

        assert result == 0
        assert event_processor.processed_events == 0
        assert event_processor.failed_events == 0

        # No database calls should be made
        event_processor.db_manager.begin_transaction.assert_not_called()

    def test_process_jsonl_file_runtime(
        self, event_processor, sample_runtime_event, tmp_path
    ):
        """Test processing JSONL file with runtime events"""
        # Create test JSONL file
        jsonl_file = tmp_path / "runtime_events.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps(sample_runtime_event) + "\n")

        result = event_processor.process_jsonl_file(jsonl_file, "runtime")

        assert result == 1
        assert event_processor.processed_events == 1

    def test_process_jsonl_file_lineage(
        self, event_processor, sample_openlineage_event, tmp_path
    ):
        """Test processing JSONL file with lineage events"""
        # Create test JSONL file
        jsonl_file = tmp_path / "lineage_events.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps(sample_openlineage_event) + "\n")

        result = event_processor.process_jsonl_file(jsonl_file, "lineage")

        assert result == 1
        assert event_processor.processed_events == 1

    def test_process_jsonl_file_nonexistent(self, event_processor, tmp_path):
        """Test processing nonexistent JSONL file"""
        nonexistent_file = tmp_path / "does_not_exist.jsonl"

        result = event_processor.process_jsonl_file(nonexistent_file, "runtime")

        assert result == 0

    def test_process_jsonl_file_invalid_json(self, event_processor, tmp_path):
        """Test processing JSONL file with invalid JSON"""
        # Create file with invalid JSON
        jsonl_file = tmp_path / "invalid.jsonl"
        with open(jsonl_file, "w") as f:
            f.write("invalid json line\n")
            f.write('{"valid": "json"}\n')

        result = event_processor.process_jsonl_file(jsonl_file, "runtime")

        # Should process 0 events due to validation failures
        assert result == 0
        assert event_processor.failed_events >= 1

    def test_process_jsonl_file_unknown_type(self, event_processor, tmp_path):
        """Test processing JSONL file with unknown event type"""
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"test": "event"}\n')

        result = event_processor.process_jsonl_file(jsonl_file, "unknown")

        assert result == 0

    def test_transform_runtime_event_success(
        self, event_processor, sample_runtime_event
    ):
        """Test successful runtime event transformation"""
        transformed = event_processor._transform_runtime_event(sample_runtime_event)

        assert transformed["event_type"] == "cell_execution_start"
        assert transformed["execution_id"] == "abcd1234"
        assert transformed["cell_id"] == "cell_456"
        assert transformed["session_id"] == "5e555678"
        assert isinstance(transformed["timestamp"], datetime)

    def test_transform_runtime_event_missing_required_field(self, event_processor):
        """Test runtime event transformation with missing required field"""
        invalid_event = {
            "event_type": "cell_execution_start",
            # Missing session_id and emitted_at
        }

        with pytest.raises(ValueError, match="Missing required field"):
            event_processor._transform_runtime_event(invalid_event)

    def test_transform_lineage_event_openlineage_format(
        self, event_processor, sample_openlineage_event
    ):
        """Test lineage event transformation with OpenLineage format"""
        transformed = event_processor._transform_lineage_event(sample_openlineage_event)

        # Check that run_id is present and is a valid UUID string
        assert "run_id" in transformed
        assert isinstance(transformed["run_id"], str)
        # Validate it's a valid UUID format
        uuid.UUID(transformed["run_id"])  # This will raise ValueError if invalid

        assert transformed["event_type"] == "COMPLETE"
        assert transformed["job_name"] == "pandas_read_csv"
        assert transformed["execution_id"] == "abcd1234"
        assert transformed["session_id"] == "5e555678"
        assert transformed["duration_ms"] == 150

    def test_transform_lineage_event_simple_format(self, event_processor):
        """Test lineage event transformation with simple format"""
        simple_event = {
            "run_id": "simple-run-123",
            "event_type": "data_operation",
            "job_name": "csv_read",
            "session_id": "session-789",
            "emitted_at": "2024-01-01T00:00:00Z",
        }

        transformed = event_processor._transform_lineage_event(simple_event)

        assert transformed["run_id"] == "simple-run-123"
        assert transformed["event_type"] == "data_operation"
        assert transformed["job_name"] == "csv_read"
        assert transformed["session_id"] == "session-789"

    def test_parse_timestamp_iso_format(self, event_processor):
        """Test timestamp parsing with ISO format"""
        timestamp_str = "2024-01-01T12:00:00Z"
        parsed = event_processor._parse_timestamp(timestamp_str)

        assert isinstance(parsed, datetime)
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 1

    def test_parse_timestamp_datetime_object(self, event_processor):
        """Test timestamp parsing with datetime object"""
        dt = datetime.now(timezone.utc)
        parsed = event_processor._parse_timestamp(dt)

        assert parsed == dt

    def test_parse_timestamp_none(self, event_processor):
        """Test timestamp parsing with None"""
        parsed = event_processor._parse_timestamp(None)
        assert parsed is None

    def test_parse_timestamp_invalid(self, event_processor):
        """Test timestamp parsing with invalid format"""
        parsed = event_processor._parse_timestamp("invalid-timestamp")
        assert parsed is None

    def test_extract_column_lineage(self, event_processor):
        """Test column lineage extraction from event"""
        event_with_lineage = {
            "outputs": [
                {
                    "name": "output_df",
                    "facets": {
                        "columnLineage": {
                            "fields": {
                                "col1": {
                                    "inputFields": [
                                        {"namespace": "input", "name": "source_col1"}
                                    ]
                                }
                            }
                        }
                    },
                }
            ]
        }

        lineage = event_processor._extract_column_lineage(event_with_lineage)

        assert lineage is not None
        assert "output_df" in lineage
        assert "fields" in lineage["output_df"]

    def test_extract_column_metrics(self, event_processor):
        """Test column metrics extraction from event"""
        event_with_metrics = {
            "outputs": [
                {
                    "name": "output_df",
                    "facets": {
                        "dataQualityMetrics": {
                            "rowCount": 1000,
                            "nullCount": 5,
                        }
                    },
                }
            ]
        }

        metrics = event_processor._extract_column_metrics(event_with_metrics)

        assert metrics is not None
        assert "output_df" in metrics
        assert metrics["output_df"]["rowCount"] == 1000

    def test_extract_other_facets(self, event_processor):
        """Test extraction of other facets from event"""
        event_with_facets = {
            "run": {
                "facets": {
                    "marimoExecution": {"sessionId": "known"},  # Known facet
                    "customFacet": {"data": "value"},  # Unknown facet
                }
            },
            "job": {"facets": {"jobCustom": {"info": "test"}}},
        }

        other_facets = event_processor._extract_other_facets(event_with_facets)

        assert other_facets is not None
        assert "run.customFacet" in other_facets
        assert "job.jobCustom" in other_facets
        # Known facets should be excluded
        assert "run.marimoExecution" not in other_facets

    def test_get_processing_stats(self, event_processor):
        """Test processing statistics"""
        # Set some stats manually
        event_processor.processed_events = 10
        event_processor.failed_events = 2

        stats = event_processor.get_validation_summary()

        expected_stats = {
            "processed_events": 10,
            "failed_events": 2,
            "total_events": 12,
            "success_rate": 10 / 12,
            "strict_validation": False,
            "schemas_loaded": {
                "runtime_schema": True,
                "openlineage_schema": True,
            },
        }

        assert stats == expected_stats

    def test_multiple_event_processing(self, event_processor, sample_runtime_event):
        """Test processing multiple events in sequence"""
        events = [sample_runtime_event.copy() for _ in range(3)]

        result = event_processor.process_runtime_events(events)

        assert result == 3
        assert event_processor.processed_events == 3
        assert event_processor.failed_events == 0

        # Should have called insert 3 times
        assert event_processor.db_manager.insert_runtime_event.call_count == 3

    def test_mixed_valid_invalid_events(self, event_processor, sample_runtime_event):
        """Test processing mix of valid and invalid events"""
        valid_event = sample_runtime_event
        invalid_event = {"event_type": "invalid"}  # Missing required fields

        events = [valid_event, invalid_event, valid_event]

        result = event_processor.process_runtime_events(events)

        assert result == 2  # Only valid events processed
        assert event_processor.processed_events == 2
        assert event_processor.failed_events == 1

    @pytest.mark.parametrize("event_type", ["runtime", "lineage"])
    def test_process_jsonl_file_empty_lines(
        self, event_processor, tmp_path, event_type
    ):
        """Test processing JSONL file with empty lines"""
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            f.write("\n")  # Empty line
            f.write("  \n")  # Whitespace line
            f.write("\n")  # Another empty line

        result = event_processor.process_jsonl_file(jsonl_file, event_type)

        assert result == 0
        assert (
            event_processor.failed_events == 0
        )  # Empty lines shouldn't count as failures
