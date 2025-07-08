#!/usr/bin/env python3
"""
Ingestion Pipeline Integration Tests

Tests the complete JSONL → Database pipeline using REAL components:
FileWatcher → EventProcessor → DuckDBManager → Database

This validates component interactions and catches breaking changes in
the ingestion implementation when code is updated.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from hunyo_capture.logger import get_logger

# Import REAL ingestion components (no mocks!)
from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager
from hunyo_mcp_server.ingestion.event_processor import EventProcessor
from hunyo_mcp_server.ingestion.file_watcher import FileWatcher

# Import centralized path utilities
from hunyo_mcp_server.utils.paths import (
    get_safe_temp_database_path,
    get_schema_path,
    setup_cross_platform_directories,
)

# Create test logger instance
test_logger = get_logger("hunyo.test.ingestion_integration")


class TestIngestionPipelineIntegration:
    """Integration tests for JSONL → Database pipeline using real components"""

    @pytest.fixture
    def runtime_events_schema(self):
        """Load runtime events schema for validation using centralized path utilities"""
        schema_path = get_schema_path("runtime_events_schema.json")
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def openlineage_events_schema(self):
        """Load OpenLineage events schema for validation using centralized path utilities"""
        schema_path = get_schema_path("openlineage_events_schema.json")
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def fresh_db_manager(self):
        """Create fresh DuckDBManager with real database using centralized path utilities"""
        # Use centralized safe temp database path instead of ad-hoc path handling
        db_path = get_safe_temp_database_path("integration_test")

        manager = DuckDBManager(db_path)
        manager.initialize_database()
        yield manager
        manager.close()

    @pytest.fixture
    def real_event_processor(self, fresh_db_manager):
        """Create REAL EventProcessor instance for testing"""
        return EventProcessor(fresh_db_manager, strict_validation=False)

    @pytest.fixture
    def events_directory(self, temp_hunyo_dir):
        """Create events directory structure using centralized directory setup"""
        # Use centralized cross-platform directory setup
        directories = setup_cross_platform_directories(str(temp_hunyo_dir))

        # Create specific runtime, lineage, and DataFrame lineage subdirectories within events directory
        events_base = Path(directories["events"])
        runtime_dir = events_base / "runtime"
        lineage_dir = events_base / "lineage"
        dataframe_lineage_dir = events_base / "dataframe_lineage"

        for dir_path in [runtime_dir, lineage_dir, dataframe_lineage_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        return {
            "events": events_base,
            "runtime": runtime_dir,
            "lineage": lineage_dir,
            "dataframe_lineage": dataframe_lineage_dir,
        }

    def validate_event_against_schema(
        self, event: dict[str, Any], schema: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate a single event against a JSON schema"""
        try:
            jsonschema.validate(event, schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)

    def create_sample_runtime_events(self, count: int = 5) -> list[dict[str, Any]]:
        """Create sample runtime events for testing with Windows-compatible unique timestamps"""
        import platform
        import time

        events = []
        base_time = datetime.now(timezone.utc)
        session_id = uuid.uuid4().hex[:8]  # 8 hex chars for session

        for i in range(count):
            execution_id = uuid.uuid4().hex[:8]  # 8 hex chars required by schema

            # Platform-specific timestamp generation to avoid ID collisions
            if platform.system() == "Windows":
                # Windows: Use UUID-based unique microseconds to avoid time resolution issues
                start_microsecond = abs(hash(f"start_{i}_{execution_id}")) % 1000000
                end_microsecond = abs(hash(f"end_{i}_{execution_id}")) % 1000000
                # Add small delay to ensure unique timestamps
                time.sleep(0.001)
            else:
                # Unix: Use incremental microseconds (fine resolution)
                start_microsecond = (i * 2000) % 1000000
                end_microsecond = (i * 2000 + 1000) % 1000000

            # Start event
            start_event = {
                "event_type": "cell_execution_start",
                "execution_id": execution_id,
                "cell_id": f"test_cell_{i:03d}",
                "cell_source": f"df{i} = pd.DataFrame({{'col': [1, 2, 3]}})",
                "cell_source_lines": 1,
                "start_memory_mb": 100.0 + i,
                "timestamp": base_time.replace(
                    microsecond=start_microsecond
                ).isoformat(),
                "session_id": session_id,
                "emitted_at": base_time.replace(
                    microsecond=start_microsecond
                ).isoformat(),
            }
            events.append(start_event)

            # End event
            end_event = {
                "event_type": "cell_execution_end",
                "execution_id": execution_id,
                "cell_id": f"test_cell_{i:03d}",
                "cell_source": f"df{i} = pd.DataFrame({{'col': [1, 2, 3]}})",
                "cell_source_lines": 1,
                "start_memory_mb": 100.0 + i,
                "end_memory_mb": 105.0 + i,
                "duration_ms": 50.0 + i,
                "timestamp": base_time.replace(microsecond=end_microsecond).isoformat(),
                "session_id": session_id,
                "emitted_at": base_time.replace(
                    microsecond=end_microsecond
                ).isoformat(),
            }
            events.append(end_event)

        return events

    def create_sample_openlineage_events(self, count: int = 3) -> list[dict[str, Any]]:
        """Create sample OpenLineage events for testing"""
        events = []
        base_time = datetime.now(timezone.utc)

        for i in range(count):
            event = {
                "eventType": "START",
                "eventTime": (base_time.replace(microsecond=i * 1000)).isoformat(),
                "run": {
                    "runId": str(uuid.uuid4()),
                    "facets": {
                        "marimoExecution": {
                            "_producer": "marimo-lineage-tracker",
                            "executionId": f"exec_{i:03d}",
                            "sessionId": "test_session_123",
                        }
                    },
                },
                "job": {
                    "namespace": "marimo",
                    "name": f"pandas_operation_{i}",
                },
                "inputs": [],
                "outputs": [
                    {
                        "namespace": "marimo",
                        "name": f"dataframe_{i:06d}",
                        "facets": {
                            "schema": {
                                "_producer": "marimo-lineage-tracker",
                                "_schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
                                "fields": [
                                    {"name": f"col_{j}", "type": "int64"}
                                    for j in range(3)
                                ],
                            }
                        },
                    }
                ],
                "producer": "marimo-lineage-tracker",
                "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
                "session_id": uuid.uuid4().hex[:8],
                "emitted_at": (base_time.replace(microsecond=i * 1000)).isoformat(),
            }
            events.append(event)

        return events

    def write_events_to_jsonl(self, events: list[dict], file_path: Path):
        """Write events to JSONL file"""
        with open(file_path, "w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
        test_logger.info(f"[SETUP] Wrote {len(events)} events to {file_path}")

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_runtime_events_file_to_database_pipeline(
        self,
        events_directory,
        real_event_processor,
        fresh_db_manager,
        runtime_events_schema,
    ):
        """Test complete pipeline: JSONL file → EventProcessor → Database"""

        test_logger.info("[TEST] Starting runtime events pipeline integration test")

        # 1. Create test runtime events
        runtime_events = self.create_sample_runtime_events(count=10)
        runtime_file = events_directory["runtime"] / "test_runtime_events.jsonl"
        self.write_events_to_jsonl(runtime_events, runtime_file)

        # 2. Process events through REAL EventProcessor
        test_logger.info("[PROCESS] Processing events with real EventProcessor...")
        processed_count = real_event_processor.process_jsonl_file(
            runtime_file, "runtime"
        )

        # 3. Validate processing results
        assert processed_count > 0, f"Expected processed events, got {processed_count}"
        test_logger.info(f"[OK] Processed {processed_count} runtime events")

        # 4. Verify events in database using REAL DuckDBManager
        test_logger.info("[VALIDATE] Checking database contents...")
        db_count = fresh_db_manager.get_table_count("runtime_events")
        assert db_count > 0, f"Expected events in database, found {db_count}"

        # 5. Retrieve and validate stored events
        stored_events = fresh_db_manager.execute_query(
            "SELECT * FROM runtime_events ORDER BY timestamp LIMIT 20"
        )

        assert (
            len(stored_events) >= 10
        ), f"Expected at least 10 stored events, found {len(stored_events)}"

        # 6. Validate data integrity through pipeline
        test_logger.info("[VALIDATE] Checking data integrity...")

        # Check that execution_ids were preserved
        execution_ids = {event["execution_id"] for event in stored_events}
        assert len(execution_ids) >= 5, "Should have multiple unique execution IDs"

        # Check event types are correct
        event_types = {event["event_type"] for event in stored_events}
        expected_types = {"cell_execution_start", "cell_execution_end"}
        assert expected_types.issubset(
            event_types
        ), f"Missing event types: {expected_types - event_types}"

        test_logger.info(
            f"[OK] Database contains {len(stored_events)} events with {len(execution_ids)} executions"
        )

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_lineage_events_file_to_database_pipeline(
        self,
        events_directory,
        real_event_processor,
        fresh_db_manager,
        openlineage_events_schema,
    ):
        """Test complete pipeline: OpenLineage JSONL → EventProcessor → Database"""

        test_logger.info("[TEST] Starting lineage events pipeline integration test")

        # 1. Create test OpenLineage events
        lineage_events = self.create_sample_openlineage_events(count=8)
        lineage_file = events_directory["lineage"] / "test_lineage_events.jsonl"
        self.write_events_to_jsonl(lineage_events, lineage_file)

        # 2. Process events through REAL EventProcessor
        test_logger.info(
            "[PROCESS] Processing lineage events with real EventProcessor..."
        )
        processed_count = real_event_processor.process_jsonl_file(
            lineage_file, "lineage"
        )

        # 3. Validate processing results
        assert processed_count > 0, f"Expected processed events, got {processed_count}"
        test_logger.info(f"[OK] Processed {processed_count} lineage events")

        # 4. Verify events in database
        test_logger.info("[VALIDATE] Checking lineage events in database...")
        db_count = fresh_db_manager.get_table_count("lineage_events")
        assert db_count > 0, f"Expected lineage events in database, found {db_count}"

        # 5. Retrieve and validate stored lineage events
        stored_events = fresh_db_manager.execute_query(
            "SELECT * FROM lineage_events ORDER BY event_time LIMIT 10"
        )

        assert (
            len(stored_events) >= 8
        ), f"Expected at least 8 stored events, found {len(stored_events)}"

        # 6. Validate OpenLineage data structure preservation
        test_logger.info("[VALIDATE] Checking OpenLineage data integrity...")

        # Check that run_ids were preserved and are valid UUIDs
        run_ids = {event["run_id"] for event in stored_events}
        assert len(run_ids) >= 8, "Should have unique run IDs for each event"

        for run_id in run_ids:
            # Validate UUIDs - run_id should be a valid UUID string
            if isinstance(run_id, str):
                uuid.UUID(run_id)  # This will raise ValueError if invalid UUID
            else:
                # If it's already a UUID object, just check it's valid
                assert isinstance(
                    run_id, uuid.UUID
                ), f"Expected UUID or string, got {type(run_id)}"

        # Check job names were preserved
        job_names = {event["job_name"] for event in stored_events}
        assert len(job_names) > 0, "Should have job names"

        test_logger.info(
            f"[OK] Database contains {len(stored_events)} lineage events with {len(run_ids)} runs"
        )

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_dataframe_lineage_events_file_to_database_pipeline(
        self,
        events_directory,
        real_event_processor,
        fresh_db_manager,
    ):
        """Test complete pipeline: DataFrame lineage JSONL → EventProcessor → Database"""

        test_logger.info(
            "[TEST] Starting DataFrame lineage events pipeline integration test"
        )

        # Load DataFrame lineage schema for validation
        dataframe_lineage_schema_path = Path(
            "schemas/json/dataframe_lineage_schema.json"
        )
        if not dataframe_lineage_schema_path.exists():
            pytest.skip("DataFrame lineage schema not found")

        with open(dataframe_lineage_schema_path, encoding="utf-8") as f:
            json.load(f)  # Validate schema file exists and is readable

        # 1. Create test DataFrame lineage events
        dataframe_lineage_events = [create_dataframe_lineage_event(i) for i in range(6)]
        dataframe_lineage_file = (
            events_directory["dataframe_lineage"]
            / "test_dataframe_lineage_events.jsonl"
        )
        self.write_events_to_jsonl(dataframe_lineage_events, dataframe_lineage_file)

        # 2. Process events through REAL EventProcessor
        test_logger.info(
            "[PROCESS] Processing DataFrame lineage events with real EventProcessor..."
        )
        processed_count = real_event_processor.process_jsonl_file(
            dataframe_lineage_file, "dataframe_lineage"
        )

        # 3. Validate processing results
        assert processed_count > 0, f"Expected processed events, got {processed_count}"
        test_logger.info(f"[OK] Processed {processed_count} DataFrame lineage events")

        # 4. Verify events in database
        test_logger.info("[VALIDATE] Checking DataFrame lineage events in database...")
        db_count = fresh_db_manager.get_table_count("dataframe_lineage_events")
        assert (
            db_count > 0
        ), f"Expected DataFrame lineage events in database, found {db_count}"

        # 5. Retrieve and validate stored DataFrame lineage events
        stored_events = fresh_db_manager.execute_query(
            "SELECT * FROM dataframe_lineage_events ORDER BY timestamp LIMIT 10"
        )

        assert (
            len(stored_events) >= 6
        ), f"Expected at least 6 stored events, found {len(stored_events)}"

        # 6. Validate DataFrame lineage data structure preservation
        test_logger.info("[VALIDATE] Checking DataFrame lineage data integrity...")

        # Check that execution_ids were preserved
        execution_ids = {event["execution_id"] for event in stored_events}
        assert (
            len(execution_ids) >= 6
        ), "Should have unique execution IDs for each event"

        # Check operation types are correct
        operation_types = {event["operation_type"] for event in stored_events}
        assert "selection" in operation_types, "Should have selection operation type"

        # Check operation methods are correct
        operation_methods = {event["operation_method"] for event in stored_events}
        assert (
            "__getitem__" in operation_methods
        ), "Should have __getitem__ operation method"

        # Check that JSON fields are preserved
        for event in stored_events:
            assert (
                event["operation_parameters"] is not None
            ), "Should have operation_parameters"
            assert event["input_dataframes"] is not None, "Should have input_dataframes"
            assert (
                event["output_dataframes"] is not None
            ), "Should have output_dataframes"
            assert event["column_lineage"] is not None, "Should have column_lineage"

        test_logger.info(
            f"[OK] Database contains {len(stored_events)} DataFrame lineage events with {len(execution_ids)} executions"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(45)
    async def test_file_watcher_to_database_integration(
        self,
        events_directory,
        real_event_processor,
        fresh_db_manager,
        runtime_events_schema,
    ):
        """Test FileWatcher → EventProcessor → Database integration"""

        test_logger.info("[TEST] Starting FileWatcher integration test")

        # 1. Set up REAL FileWatcher
        file_watcher = FileWatcher(
            runtime_dir=events_directory["runtime"],
            lineage_dir=events_directory["lineage"],
            dataframe_lineage_dir=events_directory["dataframe_lineage"],
            event_processor=real_event_processor,
            notebook_hash="test_notebook_hash",
            verbose=True,  # Enable verbose logging for testing
        )

        try:
            # 2. Start file watching in background
            test_logger.info("[SETUP] Starting FileWatcher...")
            watch_task = asyncio.create_task(file_watcher.start())

            # Give FileWatcher time to initialize
            await asyncio.sleep(1.0)

            # 3. Create events file while FileWatcher is running
            test_logger.info("[CREATE] Creating runtime events file...")
            runtime_events = self.create_sample_runtime_events(count=5)
            runtime_file = (
                events_directory["runtime"] / "test_notebook_hash_runtime_events.jsonl"
            )
            self.write_events_to_jsonl(runtime_events, runtime_file)

            # 4. Wait for FileWatcher to detect and process
            test_logger.info("[WAIT] Waiting for FileWatcher to process events...")

            # Poll database for results
            max_wait = 15  # seconds
            start_time = time.time()

            while time.time() - start_time < max_wait:
                db_count = fresh_db_manager.get_table_count("runtime_events")
                if db_count > 0:
                    test_logger.info(
                        f"[OK] FileWatcher processed events! Found {db_count} in database"
                    )
                    break
                await asyncio.sleep(0.5)
            else:
                pytest.fail("FileWatcher did not process events within timeout")

            # 5. Validate FileWatcher processing
            stored_events = fresh_db_manager.execute_query(
                "SELECT * FROM runtime_events ORDER BY timestamp"
            )

            assert (
                len(stored_events) >= 10
            ), f"Expected at least 10 events, found {len(stored_events)}"

            # 6. Test real-time file updates
            test_logger.info("[UPDATE] Testing real-time file updates...")

            # Add more events to the same file
            additional_events = self.create_sample_runtime_events(count=3)
            with open(runtime_file, "a", encoding="utf-8") as f:
                for event in additional_events:
                    f.write(json.dumps(event) + "\n")

            # Wait for additional processing
            await asyncio.sleep(3.0)

            final_count = fresh_db_manager.get_table_count("runtime_events")
            assert (
                final_count > db_count
            ), f"Expected more events after update, got {final_count}"

            test_logger.info(
                f"[OK] Real-time updates working! Final count: {final_count}"
            )

        finally:
            # 7. Clean up FileWatcher
            test_logger.info("[CLEANUP] Stopping FileWatcher...")
            await file_watcher.stop()

            # Cancel the watch task
            if not watch_task.done():
                watch_task.cancel()
                try:
                    await watch_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.integration
    @pytest.mark.timeout(40)
    def test_mixed_events_processing_pipeline(
        self,
        events_directory,
        real_event_processor,
        fresh_db_manager,
        runtime_events_schema,
        openlineage_events_schema,
    ):
        """Test processing mixed runtime, lineage, and DataFrame lineage events simultaneously"""

        test_logger.info("[TEST] Starting mixed events pipeline integration test")

        # 1. Create mixed event files for all three event types
        runtime_events = self.create_sample_runtime_events(count=7)
        lineage_events = self.create_sample_openlineage_events(count=5)
        dataframe_lineage_events = [create_dataframe_lineage_event(i) for i in range(4)]

        runtime_file = events_directory["runtime"] / "mixed_runtime_events.jsonl"
        lineage_file = events_directory["lineage"] / "mixed_lineage_events.jsonl"
        dataframe_lineage_file = (
            events_directory["dataframe_lineage"]
            / "mixed_dataframe_lineage_events.jsonl"
        )

        self.write_events_to_jsonl(runtime_events, runtime_file)
        self.write_events_to_jsonl(lineage_events, lineage_file)
        self.write_events_to_jsonl(dataframe_lineage_events, dataframe_lineage_file)

        # 2. Process all three file types
        test_logger.info("[PROCESS] Processing mixed event types...")

        runtime_processed = real_event_processor.process_jsonl_file(
            runtime_file, "runtime"
        )
        lineage_processed = real_event_processor.process_jsonl_file(
            lineage_file, "lineage"
        )
        dataframe_lineage_processed = real_event_processor.process_jsonl_file(
            dataframe_lineage_file, "dataframe_lineage"
        )

        # 3. Validate all were processed
        assert (
            runtime_processed > 0
        ), f"Expected runtime events processed, got {runtime_processed}"
        assert (
            lineage_processed > 0
        ), f"Expected lineage events processed, got {lineage_processed}"
        assert (
            dataframe_lineage_processed > 0
        ), f"Expected DataFrame lineage events processed, got {dataframe_lineage_processed}"

        test_logger.info(
            f"[OK] Processed {runtime_processed} runtime + {lineage_processed} lineage + {dataframe_lineage_processed} DataFrame lineage events"
        )

        # 4. Verify all three tables have data
        runtime_count = fresh_db_manager.get_table_count("runtime_events")
        lineage_count = fresh_db_manager.get_table_count("lineage_events")
        dataframe_lineage_count = fresh_db_manager.get_table_count(
            "dataframe_lineage_events"
        )

        assert (
            runtime_count >= 14
        ), f"Expected at least 14 runtime events, found {runtime_count}"
        assert (
            lineage_count >= 5
        ), f"Expected at least 5 lineage events, found {lineage_count}"
        assert (
            dataframe_lineage_count >= 4
        ), f"Expected at least 4 DataFrame lineage events, found {dataframe_lineage_count}"

        # 5. Test cross-table queries (validate schema consistency)
        test_logger.info("[VALIDATE] Testing cross-table consistency...")

        # Query events to verify data presence (no hardcoded session_id)
        session_query = """
        SELECT
            'runtime' as event_source, COUNT(*) as event_count
        FROM runtime_events
        UNION ALL
        SELECT
            'lineage' as event_source, COUNT(*) as event_count
        FROM lineage_events
        UNION ALL
        SELECT
            'dataframe_lineage' as event_source, COUNT(*) as event_count
        FROM dataframe_lineage_events
        ORDER BY event_source
        """

        session_results = fresh_db_manager.execute_query(session_query)
        assert len(session_results) >= 3, "Should have results from all three tables"

        total_events = sum(result["event_count"] for result in session_results)
        test_logger.info(
            f"[OK] Session consistency validated. Total events: {total_events}"
        )

    @pytest.mark.integration
    @pytest.mark.timeout(25)
    def test_error_handling_pipeline_robustness(
        self,
        events_directory,
        real_event_processor,
        fresh_db_manager,
    ):
        """Test pipeline error handling with malformed events"""

        test_logger.info("[TEST] Starting error handling pipeline test")

        # 1. Create mix of valid and invalid events
        valid_events = self.create_sample_runtime_events(count=3)

        # Add malformed events
        malformed_events = [
            {"invalid": "event", "missing": "required_fields"},
            {"event_type": "cell_execution_start"},  # Missing required fields
            {
                "event_type": "invalid_type",
                "session_id": "test",
                "emitted_at": "2024-01-01T00:00:00Z",
            },
            "not_json_object",  # Invalid JSON structure
        ]

        # 2. Write mixed valid/invalid events
        mixed_file = events_directory["runtime"] / "mixed_validity_events.jsonl"

        with open(mixed_file, "w", encoding="utf-8") as f:
            # Write valid events
            for event in valid_events:
                f.write(json.dumps(event) + "\n")

            # Write malformed events
            for event in malformed_events:
                if isinstance(event, str):
                    f.write(event + "\n")  # Invalid JSON
                else:
                    f.write(json.dumps(event) + "\n")

        test_logger.info(
            f"[SETUP] Created file with {len(valid_events)} valid + {len(malformed_events)} invalid events"
        )

        # 3. Process mixed file - should handle errors gracefully
        processed_count = real_event_processor.process_jsonl_file(mixed_file, "runtime")

        # 4. Validate error handling
        # Should process valid events and skip invalid ones
        assert processed_count <= len(
            valid_events
        ), f"Processed count ({processed_count}) should not exceed valid events ({len(valid_events)})"

        # 5. Check processing statistics
        stats = real_event_processor.get_validation_summary()

        test_logger.info("[STATS] Processing statistics:")
        test_logger.info(f"  [INFO] Processed events: {stats['processed_events']}")
        test_logger.info(f"  [INFO] Failed events: {stats['failed_events']}")
        test_logger.info(f"  [INFO] Success rate: {stats['success_rate']:.2%}")

        # Should have some failures due to malformed events
        assert (
            stats["failed_events"] > 0
        ), "Should have failed events from malformed data"
        assert stats["processed_events"] > 0, "Should have processed some valid events"

        # 6. Verify database integrity despite errors
        db_count = fresh_db_manager.get_table_count("runtime_events")
        assert (
            db_count > 0
        ), "Database should contain valid events despite processing errors"

        test_logger.info(
            f"[OK] Error handling validated. Database contains {db_count} valid events"
        )

    @pytest.mark.integration
    @pytest.mark.timeout(35)
    def test_performance_pipeline_with_large_batch(
        self,
        events_directory,
        real_event_processor,
        fresh_db_manager,
    ):
        """Test pipeline performance with larger event batches"""

        test_logger.info("[TEST] Starting performance pipeline test")

        # 1. Create larger batch of events
        batch_size = 100
        runtime_events = self.create_sample_runtime_events(count=batch_size)

        large_file = events_directory["runtime"] / "large_batch_events.jsonl"
        self.write_events_to_jsonl(runtime_events, large_file)

        test_logger.info(
            f"[SETUP] Created batch of {len(runtime_events)} runtime events"
        )

        # 2. Time the processing
        start_time = time.time()
        processed_count = real_event_processor.process_jsonl_file(large_file, "runtime")
        processing_duration = time.time() - start_time

        # 3. Validate performance
        assert processed_count == len(
            runtime_events
        ), f"Should process all {len(runtime_events)} events"
        assert (
            processing_duration < 10.0
        ), f"Processing took too long: {processing_duration:.2f}s"

        test_logger.info(
            f"[PERFORMANCE] Processed {processed_count} events in {processing_duration:.2f}s"
        )
        test_logger.info(
            f"[PERFORMANCE] Rate: {processed_count/processing_duration:.1f} events/second"
        )

        # 4. Verify database performance
        start_time = time.time()
        db_count = fresh_db_manager.get_table_count("runtime_events")
        query_duration = time.time() - start_time

        assert db_count >= len(
            runtime_events
        ), f"Database should contain at least {len(runtime_events)} events"
        assert (
            query_duration < 2.0
        ), f"Database query took too long: {query_duration:.2f}s"

        test_logger.info(
            f"[PERFORMANCE] Database query completed in {query_duration:.3f}s"
        )

        # 5. Test bulk retrieval performance
        start_time = time.time()
        bulk_events = fresh_db_manager.execute_query(
            "SELECT * FROM runtime_events ORDER BY timestamp LIMIT ?", [batch_size]
        )
        bulk_duration = time.time() - start_time

        assert len(bulk_events) >= batch_size, f"Should retrieve {batch_size} events"
        assert (
            bulk_duration < 3.0
        ), f"Bulk retrieval took too long: {bulk_duration:.2f}s"

        test_logger.info(
            f"[PERFORMANCE] Bulk retrieval of {len(bulk_events)} events in {bulk_duration:.3f}s"
        )
        test_logger.info("[OK] Performance pipeline test completed successfully")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_orchestrator_background_thread_database_operations(
        self,
        events_directory,
        runtime_events_schema,
    ):
        """Test that would have caught the signal.alarm() threading issue.

        This test uses the actual orchestrator threading setup where database operations
        happen in a background thread, which would trigger the signal.alarm() error
        that was missed by other tests.
        """
        from hunyo_mcp_server.orchestrator import HunyoOrchestrator

        test_logger = get_logger("hunyo.test.ingestion_integration")
        test_logger.info("[TEST] Starting orchestrator background thread database test")

        # Use a temporary notebook file for testing
        notebook_file = events_directory["events"].parent / "test_notebook.py"
        notebook_file.write_text("# Test notebook")

        # Create orchestrator (this creates background thread)
        orchestrator = HunyoOrchestrator(notebook_file, verbose=True)

        try:
            test_logger.info(
                "[SETUP] Starting orchestrator (creates background thread)..."
            )
            orchestrator.start()

            # Wait for orchestrator to be ready
            time.sleep(1)

            # Get the notebook hash from the orchestrator
            notebook_hash = orchestrator.notebook_hash

            # Get the file watcher for creating files in the correct directories
            file_watcher = orchestrator.file_watcher

            # Create event files that should trigger filesystem events
            test_logger.info(
                f"[SETUP] Creating event files for notebook hash: {notebook_hash}"
            )

            # Create runtime events file in the actual monitored directory
            runtime_file = (
                file_watcher.runtime_dir
                / f"{notebook_hash}_test_notebook_runtime_events.jsonl"
            )
            runtime_events = [
                create_runtime_event(runtime_events_schema, i) for i in range(5)
            ]
            with runtime_file.open("w") as f:
                for event in runtime_events:
                    f.write(json.dumps(event) + "\n")
            test_logger.info(
                f"[SETUP] Wrote {len(runtime_events)} events to {runtime_file}"
            )

            # Create lineage events file in the actual monitored directory
            lineage_file = (
                file_watcher.lineage_dir
                / f"{notebook_hash}_test_notebook_lineage_events.jsonl"
            )
            lineage_events = [create_lineage_event(notebook_hash, i) for i in range(2)]
            with lineage_file.open("w") as f:
                for event in lineage_events:
                    f.write(json.dumps(event) + "\n")
            test_logger.info(
                f"[SETUP] Wrote {len(lineage_events)} events to {lineage_file}"
            )

            # Create DataFrame lineage events file in the actual monitored directory
            dataframe_lineage_file = (
                file_watcher.dataframe_lineage_dir
                / f"{notebook_hash}_test_notebook_dataframe_lineage_events.jsonl"
            )
            dataframe_lineage_events = [
                create_dataframe_lineage_event(i) for i in range(3)
            ]
            with dataframe_lineage_file.open("w") as f:
                for event in dataframe_lineage_events:
                    f.write(json.dumps(event) + "\n")
            test_logger.info(
                f"[SETUP] Wrote {len(dataframe_lineage_events)} events to {dataframe_lineage_file}"
            )

            # Instead of relying on filesystem events, directly process the files
            # This tests the threading issue more directly
            test_logger.info(
                "[TRIGGER] Directly processing files through background thread..."
            )

            # Process runtime file directly - this will happen in background thread
            test_logger.info(
                "[PROCESS] Processing runtime file in background thread..."
            )
            runtime_count = await file_watcher.process_file_now(runtime_file)
            test_logger.info(
                f"[PROCESS] Runtime file processed: {runtime_count} events"
            )

            # Process lineage file directly - this will happen in background thread
            test_logger.info(
                "[PROCESS] Processing lineage file in background thread..."
            )
            lineage_count = await file_watcher.process_file_now(lineage_file)
            test_logger.info(
                f"[PROCESS] Lineage file processed: {lineage_count} events"
            )

            # Process DataFrame lineage file directly - this will happen in background thread
            test_logger.info(
                "[PROCESS] Processing DataFrame lineage file in background thread..."
            )
            dataframe_lineage_count = await file_watcher.process_file_now(
                dataframe_lineage_file
            )
            test_logger.info(
                f"[PROCESS] DataFrame lineage file processed: {dataframe_lineage_count} events"
            )

            # Verify that events were processed successfully
            if (
                runtime_count == 0
                and lineage_count == 0
                and dataframe_lineage_count == 0
            ):
                test_logger.error(
                    "[FAIL] No events processed by direct file processing"
                )
                pytest.fail(
                    "No events processed by direct file processing. "
                    "This could indicate a threading issue with database operations."
                )

            # Verify events are in the database - this tests database operations from background thread
            try:
                result = orchestrator.get_db_manager().execute_query(
                    "SELECT COUNT(*) as count FROM runtime_events"
                )
                if result and len(result) > 0:
                    # Handle both count formats
                    first_row = result[0]
                    processed_events = first_row.get("count", 0) or first_row.get(
                        "COUNT(*)", 0
                    )
                else:
                    processed_events = 0

                if processed_events == 0:
                    test_logger.error(
                        "[FAIL] No events found in database after processing"
                    )
                    pytest.fail("No events found in database after processing")

            except Exception as e:
                # This is where the original signal.alarm() error would have occurred
                test_logger.error(
                    f"Database operation failed in background thread - threading issue detected: {e}"
                )
                pytest.fail(
                    f"Database operation failed in background thread - threading issue detected: {e}"
                )

            # Verify that the test would have caught the signal.alarm() error
            test_logger.info(
                "[SUCCESS] Background thread database operations completed successfully"
            )
            test_logger.info(
                "[SUCCESS] This test would have caught the signal.alarm() threading issue"
            )

            # Additional verification - test that we can query the database from main thread
            total_events = orchestrator.get_db_manager().execute_query(
                "SELECT COUNT(*) as count FROM runtime_events"
            )[0].get("count", 0) or orchestrator.get_db_manager().execute_query(
                "SELECT COUNT(*) as count FROM runtime_events"
            )[
                0
            ].get(
                "COUNT(*)", 0
            )

            assert total_events >= 5, f"Expected at least 5 events, got {total_events}"
            test_logger.info(f"[SUCCESS] Verified {total_events} events in database")

        finally:
            test_logger.info("[CLEANUP] Stopping orchestrator...")
            orchestrator.stop()


# Helper functions for creating test events
def create_runtime_event(_schema: dict[str, Any], index: int) -> dict[str, Any]:
    """Create a runtime event that validates against the schema"""
    import uuid
    from datetime import datetime, timezone

    base_time = datetime.now(timezone.utc)
    execution_id = uuid.uuid4().hex[:8]
    session_id = uuid.uuid4().hex[:8]

    return {
        "event_type": "cell_execution_start",
        "execution_id": execution_id,
        "cell_id": f"test_cell_{index}",
        "cell_source": f"# Test cell {index}\nimport pandas as pd\ndf_{index} = pd.DataFrame({{'col': [1, 2, 3]}})",
        "cell_source_lines": 3,
        "session_id": session_id,
        "timestamp": base_time.isoformat(),
        "emitted_at": base_time.isoformat(),
        "duration_ms": 100 + index * 10,
        "start_memory_mb": 50.0 + index,
        "end_memory_mb": 55.0 + index,
    }


def create_lineage_event(_notebook_hash: str, index: int) -> dict[str, Any]:
    """Create a lineage event for testing"""
    import uuid
    from datetime import datetime, timezone

    base_time = datetime.now(timezone.utc)

    return {
        "eventType": "START",
        "eventTime": base_time.isoformat(),
        "run": {
            "runId": str(uuid.uuid4()),
            "facets": {
                "parent": {
                    "run": {"runId": str(uuid.uuid4())},
                    "job": {"namespace": "test", "name": f"test_job_{index}"},
                }
            },
        },
        "job": {"namespace": "marimo", "name": f"test_operation_{index}", "facets": {}},
        "inputs": [],
        "outputs": [
            {
                "namespace": "marimo",
                "name": f"test_output_{index}",
                "facets": {
                    "schema": {
                        "_producer": "marimo-lineage-tracker",
                        "_schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
                        "fields": [
                            {"name": "col_1", "type": "int64"},
                            {"name": "col_2", "type": "string"},
                        ],
                    }
                },
            }
        ],
        "producer": "marimo-lineage-tracker",
        "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
        "session_id": uuid.uuid4().hex[:8],
        "emitted_at": base_time.isoformat(),
    }


def create_dataframe_lineage_event(index: int) -> dict[str, Any]:
    """Create a DataFrame lineage event for testing"""
    import uuid
    from datetime import datetime, timezone

    base_time = datetime.now(timezone.utc)
    execution_id = uuid.uuid4().hex[:8]
    session_id = uuid.uuid4().hex[:8]

    return {
        "event_type": "dataframe_lineage",
        "execution_id": execution_id,
        "cell_id": f"test_cell_{index}",
        "session_id": session_id,
        "timestamp": base_time.isoformat(),
        "emitted_at": base_time.isoformat(),
        "operation_type": "selection",
        "operation_method": "__getitem__",
        "operation_code": f"df_{index}[df_{index}['column'] > {index * 10}]",
        "operation_parameters": {
            "columns": ["column"],
            "condition": f"column > {index * 10}",
            "is_boolean_mask": True,
        },
        "input_dataframes": [
            {
                "variable_name": f"df_{index}",
                "object_id": f"df_{index}",
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
                "variable_name": f"filtered_df_{index}",
                "object_id": f"df_{index + 1000}",
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
            "column_mapping": {
                f"filtered_df_{index}.column": [f"df_{index}.column"],
                f"filtered_df_{index}.other_col": [f"df_{index}.other_col"],
                f"filtered_df_{index}.third_col": [f"df_{index}.third_col"],
                f"filtered_df_{index}.fourth_col": [f"df_{index}.fourth_col"],
                f"filtered_df_{index}.fifth_col": [f"df_{index}.fifth_col"],
            },
            "input_columns": [
                f"df_{index}.column",
                f"df_{index}.other_col",
                f"df_{index}.third_col",
                f"df_{index}.fourth_col",
                f"df_{index}.fifth_col",
            ],
            "output_columns": [
                f"filtered_df_{index}.column",
                f"filtered_df_{index}.other_col",
                f"filtered_df_{index}.third_col",
                f"filtered_df_{index}.fourth_col",
                f"filtered_df_{index}.fifth_col",
            ],
            "operation_method": "__getitem__",
            "lineage_type": "selection",
        },
        "performance": {"overhead_ms": 2.5, "df_size_mb": 0.5, "sampled": False},
    }
