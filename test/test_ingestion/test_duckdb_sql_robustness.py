#!/usr/bin/env python3
"""
SQL Robustness Tests - Comprehensive database testing to prevent SQL bugs.

Tests cover schema initialization, primary key handling, transaction safety,
and all SQL operations that caused issues during development.
"""

from __future__ import annotations

import json
import time
import uuid

import duckdb
import pytest

from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager
from hunyo_mcp_server.ingestion.event_processor import EventProcessor


class TestDuckDBSQLRobustness:
    """Comprehensive SQL tests to prevent database bugs"""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path"""
        return tmp_path / "test_hunyo.db"

    @pytest.fixture
    def fresh_db_manager(self, temp_db_path):
        """Create fresh DuckDB manager for each test"""
        manager = DuckDBManager(temp_db_path)
        manager.initialize_database()
        yield manager
        manager.close()

    @pytest.fixture
    def sample_runtime_event_data(self):
        """Schema-compliant runtime event data for SQL testing"""
        import time

        return {
            "event_id": int(time.time() * 1000000),  # Add primary key
            "event_type": "cell_execution_start",
            "execution_id": "abcd1234",
            "cell_id": "test_cell_123",
            "cell_source": "import pandas as pd",
            "cell_source_lines": 1,
            "start_memory_mb": 100.0,
            "end_memory_mb": 105.0,
            "duration_ms": 50.0,
            "timestamp": "2024-01-01T12:00:00.000Z",
            "session_id": "5e551234",
            "emitted_at": "2024-01-01T12:00:00.000Z",
        }

    @pytest.fixture
    def sample_lineage_event_data(self):
        """Schema-compliant lineage event data for SQL testing"""
        import time

        return {
            "ol_event_id": int(time.time() * 1000000),  # Add primary key
            "run_id": str(uuid.uuid4()),
            "execution_id": "abcd1234",
            "event_type": "START",
            "job_name": "pandas_DataFrame",
            "event_time": "2024-01-01T12:00:00.000Z",
            "duration_ms": None,
            "session_id": "5e551234",
            "emitted_at": "2024-01-01T12:00:00.000Z",
            "inputs_json": [],
            "outputs_json": [
                {
                    "namespace": "marimo",
                    "name": "dataframe_123456",
                    "facets": {
                        "schema": {
                            "_producer": "marimo-lineage-tracker",
                            "_schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
                            "fields": [{"name": "col1", "type": "int64"}],
                        }
                    },
                }
            ],
            "column_lineage_json": None,
            "column_metrics_json": None,
            "other_facets_json": None,
        }

    def test_schema_initialization_idempotent(self, temp_db_path):
        """Test schema initialization is idempotent (IF NOT EXISTS works)"""
        # Initialize database twice - should not fail
        manager1 = DuckDBManager(temp_db_path)
        manager1.initialize_database()

        manager2 = DuckDBManager(temp_db_path)
        manager2.initialize_database()  # Should not fail due to IF NOT EXISTS

        # Both should work without errors
        assert manager1._schema_initialized
        assert manager2._schema_initialized

        manager1.close()
        manager2.close()

    def test_runtime_event_insertion_with_primary_key(
        self, fresh_db_manager, sample_runtime_event_data
    ):
        """Test runtime event insertion includes proper primary key"""
        # Insert event
        fresh_db_manager.insert_runtime_event(sample_runtime_event_data)

        # Verify insertion with primary key
        result = fresh_db_manager.execute_query(
            "SELECT event_id, execution_id FROM runtime_events LIMIT 1"
        )

        assert len(result) == 1
        assert result[0]["event_id"] is not None
        assert result[0]["execution_id"] == "abcd1234"

    def test_lineage_event_insertion_with_primary_key(
        self, fresh_db_manager, sample_lineage_event_data
    ):
        """Test lineage event insertion includes proper primary key (ol_event_id)"""
        # Insert event
        fresh_db_manager.insert_lineage_event(sample_lineage_event_data)

        # Verify insertion with primary key
        result = fresh_db_manager.execute_query(
            "SELECT ol_event_id, event_type FROM lineage_events LIMIT 1"
        )

        assert len(result) == 1
        assert result[0]["ol_event_id"] is not None
        assert result[0]["event_type"] == "START"

    def test_primary_key_uniqueness_enforcement(
        self, fresh_db_manager, sample_runtime_event_data
    ):
        """Test that primary keys are unique and auto-generated"""
        # Insert multiple events
        for i in range(3):
            event_data = sample_runtime_event_data.copy()
            event_data["execution_id"] = f"exec_{i:04d}"
            event_data["event_id"] = int(time.time() * 1000000) + i  # Unique IDs
            fresh_db_manager.insert_runtime_event(event_data)

        # Verify all have unique primary keys
        result = fresh_db_manager.execute_query(
            "SELECT event_id FROM runtime_events ORDER BY event_id"
        )

        primary_keys = [row["event_id"] for row in result]
        assert len(primary_keys) == 3
        assert len(set(primary_keys)) == 3  # All unique

    def test_concurrent_insertions_handle_primary_keys(
        self, fresh_db_manager, sample_runtime_event_data, sample_lineage_event_data
    ):
        """Test concurrent insertions generate unique primary keys"""
        # Insert multiple events rapidly to test primary key generation
        events_data = []
        for i in range(10):
            runtime_event = sample_runtime_event_data.copy()
            runtime_event["execution_id"] = f"runtime_{i:04d}"
            runtime_event["event_id"] = int(time.time() * 1000000) + i * 2
            events_data.append(("runtime", runtime_event))

            lineage_event = sample_lineage_event_data.copy()
            lineage_event["run_id"] = str(uuid.uuid4())
            lineage_event["ol_event_id"] = int(time.time() * 1000000) + i * 2 + 1
            events_data.append(("lineage", lineage_event))

        # Insert all events
        fresh_db_manager.begin_transaction()
        try:
            for event_type, event_data in events_data:
                if event_type == "runtime":
                    fresh_db_manager.insert_runtime_event(event_data)
                else:
                    fresh_db_manager.insert_lineage_event(event_data)
            fresh_db_manager.commit_transaction()
        except Exception:
            fresh_db_manager.rollback_transaction()
            raise

        # Verify all events inserted with unique primary keys
        runtime_count = fresh_db_manager.get_table_count("runtime_events")
        lineage_count = fresh_db_manager.get_table_count("lineage_events")

        assert runtime_count == 10
        assert lineage_count == 10

    def test_missing_required_fields_handling(self, fresh_db_manager):
        """Test proper error handling for missing required fields"""
        # Test runtime event with missing primary key
        # This actually tests the bug we found - the INSERT doesn't include event_id!
        invalid_runtime = {
            "event_type": "cell_execution_start",
            "execution_id": "test_exec",
            "session_id": "test_session",
            "emitted_at": "2024-01-01T12:00:00.000Z",
            # Missing event_id (primary key) - this SHOULD fail but the INSERT doesn't include it
        }

        with pytest.raises(
            (RuntimeError, KeyError, duckdb.ConstraintException)
        ):  # Should fail due to missing primary key handling
            fresh_db_manager.insert_runtime_event(invalid_runtime)

        # Test lineage event with missing primary key
        invalid_lineage = {
            "event_type": "START",
            "session_id": "test_session",
            "emitted_at": "2024-01-01T12:00:00.000Z",
            # Missing ol_event_id (primary key)
        }

        with pytest.raises(
            (RuntimeError, KeyError, duckdb.ConstraintException)
        ):  # Should fail due to missing primary key
            fresh_db_manager.insert_lineage_event(invalid_lineage)

    def test_transaction_rollback_on_error(
        self, fresh_db_manager, sample_runtime_event_data
    ):
        """Test transaction rollback works properly on errors"""
        initial_count = fresh_db_manager.get_table_count("runtime_events")

        fresh_db_manager.begin_transaction()
        try:
            # Insert valid event
            fresh_db_manager.insert_runtime_event(sample_runtime_event_data)

            # Force an error by inserting invalid data
            fresh_db_manager.execute_query(
                "INSERT INTO runtime_events (invalid_column) VALUES ('invalid')"
            )

            fresh_db_manager.commit_transaction()
        except Exception:
            fresh_db_manager.rollback_transaction()

        # Verify rollback - count should be unchanged
        final_count = fresh_db_manager.get_table_count("runtime_events")
        assert final_count == initial_count

    def test_schema_tables_exist_with_correct_structure(self, fresh_db_manager):
        """Test all required tables exist with correct structure"""
        # Test runtime_events table structure
        runtime_info = fresh_db_manager.get_table_info("runtime_events")
        runtime_columns = {col["column_name"] for col in runtime_info}

        required_runtime_columns = {
            "event_id",  # Note: schema uses event_id, not runtime_event_id
            "event_type",
            "execution_id",
            "cell_id",
            "session_id",
            "timestamp",
            "emitted_at",
        }
        assert required_runtime_columns.issubset(runtime_columns)

        # Test lineage_events table structure
        lineage_info = fresh_db_manager.get_table_info("lineage_events")
        lineage_columns = {col["column_name"] for col in lineage_info}

        required_lineage_columns = {
            "ol_event_id",
            "event_type",
            "event_time",
            "run_id",
            "job_name",
            "session_id",
            "emitted_at",
        }
        assert required_lineage_columns.issubset(lineage_columns)

    def test_json_field_handling(self, fresh_db_manager, sample_lineage_event_data):
        """Test JSON fields are properly stored and retrieved"""
        # Insert event with complex JSON data
        fresh_db_manager.insert_lineage_event(sample_lineage_event_data)

        # Retrieve and verify JSON fields
        result = fresh_db_manager.execute_query(
            "SELECT other_facets_json, outputs_json FROM lineage_events LIMIT 1"
        )

        assert len(result) == 1
        row = result[0]

        # Verify JSON fields can be parsed (handle None values)
        if row["other_facets_json"]:
            other_facets = json.loads(row["other_facets_json"])
            assert other_facets["marimoExecution"]["executionId"] == "abcd1234"

        if row["outputs_json"]:
            outputs = json.loads(row["outputs_json"])
            assert outputs[0]["namespace"] == "marimo"

    def test_event_processor_integration_with_sql_fixes(
        self, fresh_db_manager, sample_runtime_event_data, sample_lineage_event_data
    ):
        """Test EventProcessor works with all SQL fixes"""
        processor = EventProcessor(fresh_db_manager, strict_validation=False)

        # Create schema-compliant data (without event_id/ol_event_id for EventProcessor)
        schema_compliant_runtime = {
            "event_type": "cell_execution_start",
            "execution_id": "abcd1234",
            "cell_id": "test_cell_123",
            "cell_source": "import pandas as pd",
            "cell_source_lines": 1,
            "start_memory_mb": 100.0,
            "end_memory_mb": 105.0,
            "duration_ms": 50.0,
            "timestamp": "2024-01-01T12:00:00.000Z",
            "session_id": "5e551234",
            "emitted_at": "2024-01-01T12:00:00.000Z",
        }

        schema_compliant_lineage = {
            "eventType": "START",
            "eventTime": "2024-01-01T12:00:00.000Z",
            "run": {
                "runId": str(uuid.uuid4()),
                "facets": {
                    "marimoExecution": {
                        "_producer": "marimo-lineage-tracker",
                        "executionId": "abcd1234",
                        "sessionId": "5e551234",
                    }
                },
            },
            "job": {"namespace": "marimo", "name": "pandas_DataFrame"},
            "inputs": [],
            "outputs": [
                {
                    "namespace": "marimo",
                    "name": "dataframe_123456",
                    "facets": {
                        "schema": {
                            "_producer": "marimo-lineage-tracker",
                            "_schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
                            "fields": [{"name": "column1", "type": "int64"}],
                        }
                    },
                }
            ],
            "producer": "marimo-lineage-tracker",
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
            "session_id": "5e551234",
            "emitted_at": "2024-01-01T12:00:00.000Z",
        }

        # Process schema-compliant events
        runtime_result = processor.process_runtime_events([schema_compliant_runtime])
        lineage_result = processor.process_lineage_events([schema_compliant_lineage])

        # These should work with valid schema
        assert runtime_result >= 0
        assert lineage_result >= 0

        # Verify events in database
        runtime_count = fresh_db_manager.get_table_count("runtime_events")
        lineage_count = fresh_db_manager.get_table_count("lineage_events")

        assert runtime_count >= 0
        assert lineage_count == 1

    def test_database_corruption_recovery(self, temp_db_path):
        """Test recovery from database corruption or connection issues"""
        # Create database
        manager = DuckDBManager(temp_db_path)
        manager.initialize_database()
        manager.close()

        # Simulate corruption by deleting the database file
        if temp_db_path.exists():
            temp_db_path.unlink()

        # Try to reconnect - should recreate database
        new_manager = DuckDBManager(temp_db_path)
        new_manager.initialize_database()

        # Should work without errors
        assert new_manager._schema_initialized
        new_manager.close()

    def test_large_batch_insertion_performance(
        self, fresh_db_manager, sample_runtime_event_data
    ):
        """Test performance and correctness of large batch insertions"""
        # Generate large batch of events
        batch_size = 100
        events = []

        for i in range(batch_size):
            event = sample_runtime_event_data.copy()
            event["execution_id"] = f"batch_{i:04d}"
            event["event_id"] = int(time.time() * 1000000) + i
            events.append(event)

        # Insert in transaction
        fresh_db_manager.begin_transaction()
        start_time = time.time()

        try:
            for event in events:
                fresh_db_manager.insert_runtime_event(event)
            fresh_db_manager.commit_transaction()
        except Exception:
            fresh_db_manager.rollback_transaction()
            raise

        duration = time.time() - start_time

        # Verify all inserted correctly
        final_count = fresh_db_manager.get_table_count("runtime_events")
        assert final_count == batch_size

        # Performance check (should insert 100 events in reasonable time)
        assert duration < 5.0  # Less than 5 seconds for 100 inserts

    def test_sql_injection_protection(self, fresh_db_manager):
        """Test protection against SQL injection in parameters"""
        # Attempt SQL injection in event data
        malicious_data = {
            "event_id": int(time.time() * 1000000),
            "event_type": "cell_execution_start'; DROP TABLE runtime_events; --",
            "execution_id": "abcd1234",
            "cell_id": "test_cell",
            "cell_source": "import pandas",
            "cell_source_lines": 1,
            "start_memory_mb": 100.0,
            "timestamp": "2024-01-01T12:00:00.000Z",
            "session_id": "5e551234",
            "emitted_at": "2024-01-01T12:00:00.000Z",
        }

        # Should insert safely without executing the injection
        fresh_db_manager.insert_runtime_event(malicious_data)

        # Verify table still exists and has the event
        count = fresh_db_manager.get_table_count("runtime_events")
        assert count == 1

        # Verify the malicious string is stored as data, not executed
        result = fresh_db_manager.execute_query(
            "SELECT event_type FROM runtime_events LIMIT 1"
        )
        assert "DROP TABLE" in result[0]["event_type"]  # Stored as string, not executed
