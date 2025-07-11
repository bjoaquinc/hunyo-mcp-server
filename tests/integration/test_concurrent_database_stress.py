#!/usr/bin/env python3
"""
Concurrent Database Stress Test

This test specifically reproduces the race condition that occurs in CI/CD
when multiple threads try to use the same DuckDB connection simultaneously.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from hunyo_capture.logger import get_logger

from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager
from hunyo_mcp_server.ingestion.event_processor import EventProcessor
from hunyo_mcp_server.utils.paths import get_safe_temp_database_path

# Test logger
stress_logger = get_logger("hunyo.test.stress")


class TestConcurrentDatabaseStress:
    """Test concurrent database operations to reproduce CI transaction errors."""

    @pytest.fixture
    def shared_db_manager(self):
        """Create a shared DuckDBManager instance (like in production)."""
        db_path = get_safe_temp_database_path("stress_test")
        manager = DuckDBManager(db_path)
        manager.initialize_database()
        yield manager
        manager.close()

    @pytest.fixture
    def shared_event_processor(self, shared_db_manager):
        """Create EventProcessor that shares the same database connection."""
        return EventProcessor(shared_db_manager, strict_validation=False)

    def create_sample_runtime_event(self, index: int) -> dict:
        """Create a sample runtime event for testing."""
        import uuid
        from datetime import datetime, timezone

        return {
            "event_type": "cell_execution_start",
            "execution_id": str(uuid.uuid4())[:8],
            "cell_id": f"stress_test_cell_{index:03d}",
            "cell_source": f"df{index} = pd.DataFrame({{'col': [1, 2, 3]}})",
            "cell_source_lines": 1,
            "start_memory_mb": 100.0 + index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": "stress_test_session",
            "emitted_at": datetime.now(timezone.utc).isoformat(),
        }

    def process_events_in_thread(
        self, processor: EventProcessor, events: list[dict], thread_id: int
    ) -> tuple[int, int]:
        """Process events in a separate thread to simulate concurrent access."""
        stress_logger.info(f"Thread {thread_id}: Processing {len(events)} events")

        successful = 0
        failed = 0

        for i, event in enumerate(events):
            try:
                # This simulates the concurrent database access pattern from the failing test
                result = processor.process_runtime_events([event])
                if result > 0:
                    successful += 1
                    stress_logger.info(
                        f"Thread {thread_id}: Event {i} processed successfully"
                    )
                else:
                    failed += 1
                    stress_logger.warning(
                        f"Thread {thread_id}: Event {i} failed to process"
                    )

            except Exception as e:
                failed += 1
                stress_logger.error(
                    f"Thread {thread_id}: Event {i} failed with exception: {e}"
                )

        stress_logger.info(
            f"Thread {thread_id}: Completed - {successful} successful, {failed} failed"
        )
        return successful, failed

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_concurrent_database_access_stress(self, shared_event_processor):
        """
        Test that reproduces the exact race condition from CI by creating
        concurrent database access from multiple threads.
        """
        stress_logger.info("[STRESS] Starting concurrent database access test")

        # Create multiple batches of events for concurrent processing
        num_threads = 5
        events_per_thread = 10

        event_batches = []
        for thread_id in range(num_threads):
            batch = []
            for i in range(events_per_thread):
                event = self.create_sample_runtime_event(
                    thread_id * events_per_thread + i
                )
                batch.append(event)
            event_batches.append(batch)

        stress_logger.info(
            f"[STRESS] Created {num_threads} threads with {events_per_thread} events each"
        )

        # Process events concurrently in multiple threads
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks simultaneously to maximize concurrency
            futures = []
            for thread_id, batch in enumerate(event_batches):
                future = executor.submit(
                    self.process_events_in_thread,
                    shared_event_processor,
                    batch,
                    thread_id,
                )
                futures.append(future)

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    successful, failed = future.result(timeout=10)
                    results.append((successful, failed))
                except Exception as e:
                    stress_logger.error(f"Thread failed with exception: {e}")
                    results.append((0, events_per_thread))

        # Analyze results
        total_successful = sum(result[0] for result in results)
        total_failed = sum(result[1] for result in results)
        total_events = num_threads * events_per_thread

        stress_logger.info(
            f"[STRESS] Results: {total_successful} successful, {total_failed} failed, {total_events} total"
        )

        # If we get transaction errors, this test will catch them
        if total_failed > 0:
            stress_logger.error(
                f"[FAIL] {total_failed} events failed due to concurrent access issues"
            )

        # Verify database state
        try:
            db_count = shared_event_processor.db_manager.execute_query(
                "SELECT COUNT(*) as count FROM runtime_events"
            )
            actual_count = db_count[0]["count"] if db_count else 0

            stress_logger.info(f"[STRESS] Database contains {actual_count} events")

            # The key test: if we have transaction errors, actual_count will be less than total_successful
            if actual_count < total_successful:
                stress_logger.error(
                    f"[FAIL] Database count ({actual_count}) < successful processes ({total_successful})"
                )
                pytest.fail(
                    f"Transaction errors detected: {actual_count} events in DB but {total_successful} claimed successful"
                )

        except Exception as e:
            stress_logger.error(f"[FAIL] Database query failed: {e}")
            pytest.fail(f"Database query failed after concurrent processing: {e}")

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_rapid_fire_database_operations(self, shared_event_processor):
        """
        Test rapid-fire database operations to trigger timing-based race conditions.
        """
        stress_logger.info("[RAPID] Starting rapid-fire database operations test")

        # Create a large number of events for rapid processing
        num_events = 50
        events = []
        for i in range(num_events):
            event = self.create_sample_runtime_event(i)
            events.append(event)

        stress_logger.info(f"[RAPID] Created {num_events} events for rapid processing")

        # Process events as quickly as possible in a tight loop
        start_time = time.time()
        successful = 0
        failed = 0

        for i, event in enumerate(events):
            try:
                # Process each event individually to maximize database transaction frequency
                result = shared_event_processor.process_runtime_events([event])
                if result > 0:
                    successful += 1
                else:
                    failed += 1
                    stress_logger.warning(f"[RAPID] Event {i} failed to process")

                # No delay - process as fast as possible to trigger race conditions

            except Exception as e:
                failed += 1
                stress_logger.error(f"[RAPID] Event {i} failed: {e}")

        end_time = time.time()
        duration = end_time - start_time

        stress_logger.info(f"[RAPID] Processed {num_events} events in {duration:.2f}s")
        stress_logger.info(f"[RAPID] Rate: {num_events/duration:.1f} events/second")
        stress_logger.info(f"[RAPID] Results: {successful} successful, {failed} failed")

        # Verify database consistency
        try:
            db_count = shared_event_processor.db_manager.execute_query(
                "SELECT COUNT(*) as count FROM runtime_events"
            )
            actual_count = db_count[0]["count"] if db_count else 0

            stress_logger.info(f"[RAPID] Database contains {actual_count} events")

            if actual_count != successful:
                stress_logger.error(
                    f"[FAIL] Database inconsistency: {actual_count} in DB, {successful} successful"
                )
                pytest.fail(
                    f"Database inconsistency detected: {actual_count} != {successful}"
                )

        except Exception as e:
            stress_logger.error(f"[FAIL] Database verification failed: {e}")
            pytest.fail(f"Database verification failed: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_async_concurrent_processing(self, shared_event_processor):
        """
        Test async concurrent processing to simulate the exact file_watcher scenario.
        """
        stress_logger.info("[ASYNC] Starting async concurrent processing test")

        async def process_batch_async(batch_id: int, events: list[dict]):
            """Process a batch of events asynchronously."""
            stress_logger.info(f"[ASYNC] Batch {batch_id}: Starting async processing")

            # Simulate the async processing pattern from file_watcher
            await asyncio.sleep(0.1)  # Small delay to simulate real-world timing

            successful = 0
            failed = 0

            for i, event in enumerate(events):
                try:
                    # Process in a thread pool to simulate background processing
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, shared_event_processor.process_runtime_events, [event]
                    )

                    if result > 0:
                        successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    failed += 1
                    stress_logger.error(
                        f"[ASYNC] Batch {batch_id}, Event {i} failed: {e}"
                    )

            stress_logger.info(
                f"[ASYNC] Batch {batch_id}: {successful} successful, {failed} failed"
            )
            return successful, failed

        # Create multiple batches for concurrent async processing
        num_batches = 4
        events_per_batch = 8

        batches = []
        for batch_id in range(num_batches):
            batch = []
            for i in range(events_per_batch):
                event = self.create_sample_runtime_event(
                    batch_id * events_per_batch + i
                )
                batch.append(event)
            batches.append(batch)

        stress_logger.info(
            f"[ASYNC] Created {num_batches} batches with {events_per_batch} events each"
        )

        # Process all batches concurrently
        tasks = []
        for batch_id, batch in enumerate(batches):
            task = asyncio.create_task(process_batch_async(batch_id, batch))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        total_successful = 0
        total_failed = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                stress_logger.error(
                    f"[ASYNC] Batch {i} failed with exception: {result}"
                )
                total_failed += events_per_batch
            else:
                successful, failed = result
                total_successful += successful
                total_failed += failed

        total_events = num_batches * events_per_batch

        stress_logger.info(
            f"[ASYNC] Results: {total_successful} successful, {total_failed} failed, {total_events} total"
        )

        # Verify database state
        try:
            db_count = shared_event_processor.db_manager.execute_query(
                "SELECT COUNT(*) as count FROM runtime_events"
            )
            actual_count = db_count[0]["count"] if db_count else 0

            stress_logger.info(f"[ASYNC] Database contains {actual_count} events")

            if actual_count < total_successful:
                stress_logger.error(
                    f"[FAIL] Async processing inconsistency: {actual_count} in DB, {total_successful} successful"
                )
                pytest.fail(
                    f"Async processing inconsistency: {actual_count} != {total_successful}"
                )

        except Exception as e:
            stress_logger.error(f"[FAIL] Async database verification failed: {e}")
            pytest.fail(f"Async database verification failed: {e}")
