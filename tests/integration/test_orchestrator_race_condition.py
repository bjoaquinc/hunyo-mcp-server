#!/usr/bin/env python3
"""
Orchestrator Race Condition Test

This test exactly replicates the failing CI test scenario to reproduce
the race condition locally.
"""

import asyncio
import json

import pytest
from hunyo_capture.logger import get_logger

from hunyo_mcp_server.orchestrator import HunyoOrchestrator

# Test logger
race_logger = get_logger("hunyo.test.race")


class TestOrchestratorRaceCondition:
    """Test the exact orchestrator scenario that fails in CI."""

    @pytest.fixture
    def temp_notebook_file(self, tmp_path):
        """Create a temporary notebook file for testing."""
        notebook_file = tmp_path / "test_notebook.py"
        notebook_file.write_text("# Test notebook for race condition testing")
        return notebook_file

    def create_runtime_event(self, index: int) -> dict:
        """Create a runtime event matching the failing test."""
        import uuid
        from datetime import datetime, timezone

        return {
            "event_type": "cell_execution_start",
            "execution_id": str(uuid.uuid4())[:8],
            "cell_id": f"race_test_cell_{index:03d}",
            "cell_source": f"df{index} = pd.DataFrame({{'col': [1, 2, 3]}})",
            "cell_source_lines": 1,
            "start_memory_mb": 100.0 + index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": str(uuid.uuid4())[:8],  # Fixed: 8 hex characters
            "emitted_at": datetime.now(timezone.utc).isoformat(),
        }

    def create_lineage_event(self, notebook_hash: str, index: int) -> dict:
        """Create a lineage event matching the failing test."""
        import uuid
        from datetime import datetime, timezone

        return {
            "eventType": "START",
            "eventTime": datetime.now(timezone.utc).isoformat(),
            "producer": "marimo-lineage-tracker",  # Fixed: added required producer field
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",  # Fixed: added required schemaURL field
            "session_id": str(uuid.uuid4())[
                :8
            ],  # Fixed: added required session_id field
            "emitted_at": datetime.now(
                timezone.utc
            ).isoformat(),  # Fixed: added required emitted_at field
            "run": {
                "runId": str(uuid.uuid4()),
                "facets": {
                    "marimoExecution": {
                        "_producer": "marimo-lineage-tracker",
                        "executionId": f"race_exec_{index:03d}",
                        "sessionId": str(uuid.uuid4())[:8],  # Fixed: 8 hex characters
                    }
                },
            },
            "job": {
                "namespace": "marimo",
                "name": f"pandas_operation_{index}",
            },
            "inputs": [],
            "outputs": [
                {
                    "namespace": "marimo",
                    "name": f"dataframe_{index:06d}",
                    "facets": {
                        "schema": {
                            "_producer": "marimo-lineage-tracker",
                            "_schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",  # Fixed: added required _schemaURL field
                            "fields": [{"name": "col", "type": "int64"}],
                        }
                    },
                }
            ],
        }

    def create_dataframe_lineage_event(self, index: int) -> dict:
        """Create a DataFrame lineage event matching the failing test."""
        import uuid
        from datetime import datetime, timezone

        return {
            "event_type": "dataframe_lineage",  # Fixed: use required constant value
            "execution_id": str(uuid.uuid4())[:8],
            "cell_id": f"race_df_cell_{index:03d}",  # Fixed: added required cell_id field
            "session_id": str(uuid.uuid4())[:8],  # Fixed: 8 hex characters
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            "operation_type": "selection",  # Fixed: use schema-compliant operation type
            "operation_method": "__getitem__",  # Fixed: use schema-compliant operation method
            "operation_code": f"df{index} = df{index}[['col']]",  # Fixed: match operation method
            "operation_parameters": {
                "columns": ["col"]
            },  # Fixed: match selection operation
            "input_dataframes": [
                {
                    "variable_name": f"df{index}",
                    "shape": [3, 1],
                    "columns": ["col"],
                    "memory_usage_mb": 0.001,  # Fixed: use schema-compliant optional field
                }
            ],
            "output_dataframes": [
                {
                    "variable_name": f"df{index}_selected",
                    "shape": [3, 1],  # Fixed: selection operation keeps same shape
                    "columns": ["col"],  # Fixed: selection operation keeps same columns
                    "memory_usage_mb": 0.001,  # Fixed: use schema-compliant optional field
                }
            ],
            "column_lineage": {
                "column_mapping": {"output_df.col": ["input_df.col"]},
                "input_columns": ["col"],
                "output_columns": ["col"],  # Fixed: match selection operation
                "operation_method": "__getitem__",  # Fixed: use schema-compliant operation method
                "lineage_type": "selection",
            },
            "performance": {
                "overhead_ms": 1.5,
                "df_size_mb": 0.001,
                "sampled": False,
            },
        }

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_orchestrator_race_condition_reproduction(self, temp_notebook_file):
        """
        Exact reproduction of the failing CI test scenario.

        This test replicates the exact sequence from the failing test:
        1. Start orchestrator (creates shared DB connection)
        2. Create multiple event files
        3. Process files concurrently via file_watcher.process_file_now()
        4. This should trigger the race condition
        """
        race_logger.info("[RACE] Starting orchestrator race condition test")

        # Create orchestrator (this creates the shared database connection)
        orchestrator = HunyoOrchestrator(temp_notebook_file, verbose=True)

        try:
            race_logger.info(
                "[RACE] Starting orchestrator (creates background thread)..."
            )
            orchestrator.start()

            # Wait for orchestrator to be ready
            await asyncio.sleep(1)

            # Get the notebook hash and file watcher
            notebook_hash = orchestrator.notebook_hash
            file_watcher = orchestrator.file_watcher

            race_logger.info(f"[RACE] Notebook hash: {notebook_hash}")

            # Create the exact same event files as the failing test
            race_logger.info("[RACE] Creating event files...")

            # Runtime events file
            runtime_file = (
                file_watcher.runtime_dir
                / f"{notebook_hash}_test_notebook_runtime_events.jsonl"
            )
            runtime_events = [self.create_runtime_event(i) for i in range(5)]
            with runtime_file.open("w") as f:
                for event in runtime_events:
                    f.write(json.dumps(event) + "\n")
            race_logger.info(f"[RACE] Created runtime file: {runtime_file}")

            # Lineage events file
            lineage_file = (
                file_watcher.lineage_dir
                / f"{notebook_hash}_test_notebook_lineage_events.jsonl"
            )
            lineage_events = [
                self.create_lineage_event(notebook_hash, i) for i in range(2)
            ]
            with lineage_file.open("w") as f:
                for event in lineage_events:
                    f.write(json.dumps(event) + "\n")
            race_logger.info(f"[RACE] Created lineage file: {lineage_file}")

            # DataFrame lineage events file
            dataframe_lineage_file = (
                file_watcher.dataframe_lineage_dir
                / f"{notebook_hash}_test_notebook_dataframe_lineage_events.jsonl"
            )
            dataframe_lineage_events = [
                self.create_dataframe_lineage_event(i) for i in range(3)
            ]
            with dataframe_lineage_file.open("w") as f:
                for event in dataframe_lineage_events:
                    f.write(json.dumps(event) + "\n")
            race_logger.info(
                f"[RACE] Created dataframe lineage file: {dataframe_lineage_file}"
            )

            # Now trigger the race condition by processing files concurrently
            race_logger.info(
                "[RACE] Processing files concurrently (triggering race condition)..."
            )

            # Create tasks to process all files simultaneously
            # This is the exact pattern from the failing test
            tasks = []

            # Process runtime file
            task1 = asyncio.create_task(file_watcher.process_file_now(runtime_file))
            tasks.append(("runtime", task1))

            # Process lineage file
            task2 = asyncio.create_task(file_watcher.process_file_now(lineage_file))
            tasks.append(("lineage", task2))

            # Process DataFrame lineage file
            task3 = asyncio.create_task(
                file_watcher.process_file_now(dataframe_lineage_file)
            )
            tasks.append(("dataframe_lineage", task3))

            # Wait for all tasks to complete and collect results
            results = {}
            for task_name, task in tasks:
                try:
                    result = await task
                    results[task_name] = result
                    race_logger.info(f"[RACE] {task_name} processed: {result} events")
                except Exception as e:
                    race_logger.error(f"[RACE] {task_name} failed: {e}")
                    results[task_name] = 0

            # Log results
            total_processed = sum(results.values())
            race_logger.info(f"[RACE] Total events processed: {total_processed}")

            # Check for the race condition symptoms
            if total_processed == 0:
                race_logger.error(
                    "[RACE] No events processed - race condition detected!"
                )

            # Verify database state
            try:
                db_manager = orchestrator.get_db_manager()

                # Check runtime events
                runtime_count = db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM runtime_events"
                )
                runtime_actual = runtime_count[0]["count"] if runtime_count else 0

                # Check lineage events
                lineage_count = db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM lineage_events"
                )
                lineage_actual = lineage_count[0]["count"] if lineage_count else 0

                # Check dataframe lineage events
                df_lineage_count = db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM dataframe_lineage_events"
                )
                df_lineage_actual = (
                    df_lineage_count[0]["count"] if df_lineage_count else 0
                )

                race_logger.info("[RACE] Database state:")
                race_logger.info(f"  Runtime events: {runtime_actual}")
                race_logger.info(f"  Lineage events: {lineage_actual}")
                race_logger.info(f"  DataFrame lineage events: {df_lineage_actual}")

                total_in_db = runtime_actual + lineage_actual + df_lineage_actual

                if total_in_db == 0:
                    race_logger.error(
                        "[RACE] RACE CONDITION DETECTED: No events in database!"
                    )
                    race_logger.error("[RACE] This matches the CI failure pattern!")

                    # This is the exact failure from CI - events were "processed" but not in DB
                    pytest.fail(
                        "Race condition reproduced: No events found in database after processing"
                    )

                elif total_in_db < total_processed:
                    race_logger.error(
                        f"[RACE] PARTIAL RACE CONDITION: {total_in_db} in DB, {total_processed} processed"
                    )
                    pytest.fail(
                        f"Partial race condition: {total_in_db} in DB but {total_processed} processed"
                    )

                else:
                    race_logger.info(
                        f"[RACE] SUCCESS: {total_in_db} events successfully stored"
                    )

            except Exception as e:
                race_logger.error(f"[RACE] Database check failed: {e}")
                race_logger.error("[RACE] This indicates a serious race condition!")
                pytest.fail(f"Database check failed (race condition): {e}")

        finally:
            race_logger.info("[RACE] Stopping orchestrator...")
            orchestrator.stop()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_rapid_concurrent_file_processing(self, temp_notebook_file):
        """
        Test rapid concurrent file processing to maximize race condition chances.
        """
        race_logger.info("[RAPID] Starting rapid concurrent file processing test")

        orchestrator = HunyoOrchestrator(temp_notebook_file, verbose=True)

        try:
            orchestrator.start()
            await asyncio.sleep(0.5)

            notebook_hash = orchestrator.notebook_hash
            file_watcher = orchestrator.file_watcher

            # Create multiple small files for rapid processing
            files_to_process = []

            for i in range(10):  # Create 10 files
                # Runtime file
                runtime_file = (
                    file_watcher.runtime_dir
                    / f"{notebook_hash}_rapid_test_{i:02d}_runtime_events.jsonl"
                )
                event = self.create_runtime_event(i)
                with runtime_file.open("w") as f:
                    f.write(json.dumps(event) + "\n")
                files_to_process.append(runtime_file)

            race_logger.info(
                f"[RAPID] Created {len(files_to_process)} files for rapid processing"
            )

            # Process all files simultaneously
            tasks = []
            for i, file_path in enumerate(files_to_process):
                task = asyncio.create_task(file_watcher.process_file_now(file_path))
                tasks.append((f"file_{i}", task))

            # Wait for all tasks with timeout
            results = {}
            for task_name, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=5.0)
                    results[task_name] = result
                except asyncio.TimeoutError:
                    race_logger.error(f"[RAPID] {task_name} timed out")
                    results[task_name] = 0
                except Exception as e:
                    race_logger.error(f"[RAPID] {task_name} failed: {e}")
                    results[task_name] = 0

            # Verify results
            total_processed = sum(results.values())
            race_logger.info(f"[RAPID] Total events processed: {total_processed}")

            # Check database
            db_manager = orchestrator.get_db_manager()
            db_count = db_manager.execute_query(
                "SELECT COUNT(*) as count FROM runtime_events"
            )
            actual_count = db_count[0]["count"] if db_count else 0

            race_logger.info(f"[RAPID] Database contains: {actual_count} events")

            if actual_count == 0:
                race_logger.error("[RAPID] RACE CONDITION: No events in database!")
                pytest.fail("Race condition in rapid processing: No events in database")
            elif actual_count < total_processed:
                race_logger.error(
                    f"[RAPID] PARTIAL RACE CONDITION: {actual_count} in DB, {total_processed} processed"
                )
                pytest.fail(
                    f"Partial race condition: {actual_count} < {total_processed}"
                )
            else:
                race_logger.info(
                    f"[RAPID] SUCCESS: All {actual_count} events processed correctly"
                )

        finally:
            orchestrator.stop()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_repeated_orchestrator_race_attempts(self, temp_notebook_file):
        """
        Run multiple attempts to trigger the race condition.
        """
        race_logger.info("[REPEAT] Starting repeated race condition attempts")

        max_attempts = 5
        failures = []

        for attempt in range(max_attempts):
            race_logger.info(f"[REPEAT] Attempt {attempt + 1}/{max_attempts}")

            orchestrator = HunyoOrchestrator(temp_notebook_file, verbose=False)

            try:
                orchestrator.start()
                await asyncio.sleep(0.2)  # Slightly longer delay for CI stability

                notebook_hash = orchestrator.notebook_hash
                file_watcher = orchestrator.file_watcher

                # Create and process a single file rapidly
                runtime_file = (
                    file_watcher.runtime_dir
                    / f"{notebook_hash}_repeat_test_{attempt}_runtime_events.jsonl"
                )
                event = self.create_runtime_event(attempt)
                with runtime_file.open("w") as f:
                    f.write(json.dumps(event) + "\n")

                # Process the file
                try:
                    # Use await instead of asyncio.run to avoid event loop conflicts
                    result = await file_watcher.process_file_now(runtime_file)

                    # Allow some time for database operations to complete
                    await asyncio.sleep(0.1)

                    # Check database
                    db_manager = orchestrator.get_db_manager()
                    db_count = db_manager.execute_query(
                        "SELECT COUNT(*) as count FROM runtime_events"
                    )
                    actual_count = db_count[0]["count"] if db_count else 0

                    if result > 0 and actual_count == 0:
                        failure_msg = f"Attempt {attempt + 1}: Processed {result} but DB has {actual_count}"
                        failures.append(failure_msg)
                        race_logger.error(f"[REPEAT] {failure_msg}")
                    else:
                        race_logger.info(
                            f"[REPEAT] Attempt {attempt + 1}: OK - {actual_count} events"
                        )

                except Exception as e:
                    failure_msg = f"Attempt {attempt + 1}: Exception - {e}"
                    failures.append(failure_msg)
                    race_logger.error(f"[REPEAT] {failure_msg}")

            finally:
                orchestrator.stop()
                await asyncio.sleep(0.1)  # Brief pause between attempts

        race_logger.info(f"[REPEAT] Completed {max_attempts} attempts")
        race_logger.info(f"[REPEAT] Failures: {len(failures)}")

        if failures:
            race_logger.error("[REPEAT] RACE CONDITIONS DETECTED:")
            for failure in failures:
                race_logger.error(f"  - {failure}")
            pytest.fail(
                f"Race conditions detected in {len(failures)}/{max_attempts} attempts"
            )
        else:
            race_logger.info("[REPEAT] No race conditions detected")
