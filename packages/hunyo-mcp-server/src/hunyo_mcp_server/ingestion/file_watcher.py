"""File watcher for monitoring JSONL event files.

This module provides FileWatcher and JSONLFileHandler classes for monitoring
JSONL event files and processing them through the event processor.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from hunyo_mcp_server.logger import get_logger

if TYPE_CHECKING:
    from hunyo_mcp_server.ingestion.event_processor import EventProcessor

# Module logger
watcher_logger = get_logger("hunyo.watcher")


class JSONLFileHandler(FileSystemEventHandler):
    """Handler for JSONL file system events with debouncing."""

    def __init__(self, file_watcher: FileWatcher):
        super().__init__()
        self.file_watcher = file_watcher
        self._pending_files: set[Path] = set()
        self._last_processed: dict[Path, float] = {}
        self._debounce_delay = 1.0  # 1 second debounce
        self._processing_lock = threading.Lock()

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith(".jsonl"):
            file_path = Path(event.src_path)

            # Only process files for the current notebook hash
            if file_path.name.startswith(f"{self.file_watcher.notebook_hash}_"):
                watcher_logger.tracking(f"New JSONL file detected: {file_path.name}")
                self._queue_for_processing(file_path)
            else:
                watcher_logger.debug(
                    f"Ignoring file for different notebook: {file_path.name} (expected hash: {self.file_watcher.notebook_hash})"
                )

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith(".jsonl"):
            file_path = Path(event.src_path)

            # Only process files for the current notebook hash
            if not file_path.name.startswith(f"{self.file_watcher.notebook_hash}_"):
                watcher_logger.debug(
                    f"Ignoring modified file for different notebook: {file_path.name} (expected hash: {self.file_watcher.notebook_hash})"
                )
                return

            # Check if enough time has passed since last processing (debounce)
            current_time = time.time()
            last_processed = self._last_processed.get(file_path, 0)

            if current_time - last_processed >= self._debounce_delay:
                watcher_logger.tracking(f"JSONL file modified: {file_path.name}")
                self._queue_for_processing(file_path)
            else:
                # Add to pending files for later processing
                with self._processing_lock:
                    self._pending_files.add(file_path)

    def _queue_for_processing(self, file_path: Path):
        """Queue a file for processing."""
        try:
            with self._processing_lock:
                self._pending_files.add(file_path)
                self._last_processed[file_path] = time.time()
                watcher_logger.info(f"Queued file for processing: {file_path.name}")
        except Exception as e:
            watcher_logger.error(f"Error queuing file {file_path}: {e}")

    def get_pending_files(self) -> set[Path]:
        """Get all pending files and clear the queue."""
        try:
            with self._processing_lock:
                pending = self._pending_files.copy()
                self._pending_files.clear()
                if pending:
                    watcher_logger.info(f"Retrieved {len(pending)} pending files")
                return pending
        except Exception as e:
            watcher_logger.error(f"Error getting pending files: {e}")
            return set()


class FileWatcher:
    """
    Main file watcher for monitoring JSONL event files.

    Features:
    - Monitors runtime, lineage, and DataFrame lineage event directories
    - Debounced file processing to handle rapid changes
    - Automatic file type detection
    - Error handling and recovery
    - Background processing with asyncio
    - File processing coordination to prevent concurrent access
    """

    def __init__(
        self,
        runtime_dir: Path,
        lineage_dir: Path,
        dataframe_lineage_dir: Path,
        event_processor: EventProcessor,
        notebook_hash: str,
        *,
        verbose: bool = False,
    ):
        # Normalize directory paths for consistent comparison across platforms
        self.runtime_dir = Path(runtime_dir).resolve()
        self.lineage_dir = Path(lineage_dir).resolve()
        self.dataframe_lineage_dir = Path(dataframe_lineage_dir).resolve()
        self.event_processor = event_processor
        self.notebook_hash = notebook_hash
        self.verbose = verbose

        # File watching components
        self.observer = Observer()
        self.handler = JSONLFileHandler(self)
        self.running = False

        # Processing state
        self._processing_queue = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()

        # File processing coordination - prevents concurrent processing of same file
        self._processing_registry: set[Path] = set()
        self._registry_lock = threading.Lock()

        # Statistics
        self.files_processed = 0
        self.events_processed = 0

        watcher_logger.status("[SCAN] Starting file watcher...")
        watcher_logger.info(f"Runtime directory: {self.runtime_dir}")
        watcher_logger.info(f"Lineage directory: {self.lineage_dir}")
        watcher_logger.info(
            f"DataFrame lineage directory: {self.dataframe_lineage_dir}"
        )

    async def start(self) -> None:
        """Start the file watcher and begin monitoring."""
        if self.running:
            watcher_logger.warning("File watcher already running")
            return

        try:
            watcher_logger.info("[SCAN] Starting file watcher...")

            # Create directories if they don't exist
            for directory in [
                self.runtime_dir,
                self.lineage_dir,
                self.dataframe_lineage_dir,
            ]:
                directory.mkdir(parents=True, exist_ok=True)

            # Set up file system monitoring
            self.observer.schedule(self.handler, str(self.runtime_dir), recursive=False)
            self.observer.schedule(self.handler, str(self.lineage_dir), recursive=False)
            self.observer.schedule(
                self.handler, str(self.dataframe_lineage_dir), recursive=False
            )

            # Start the observer
            self.observer.start()
            self.running = True

            watcher_logger.success("[OK] File watcher started")

            # Process any existing files
            await self._process_existing_files()

            # Start background processor as a separate task (don't await it)
            background_task = asyncio.create_task(self._background_processor())
            self._background_tasks.add(background_task)

        except Exception as e:
            watcher_logger.error(f"Error starting file watcher: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the file watcher and clean up resources."""
        if not self.running:
            return

        watcher_logger.info("[SCAN] Stopping file watcher...")

        try:
            self.running = False

            # Stop the observer
            if self.observer.is_alive():
                self.observer.stop()
                self.observer.join(timeout=2.0)

            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete with timeout
            if self._background_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*self._background_tasks, return_exceptions=True),
                    timeout=3.0,
                )

            # Clear processing registry
            with self._registry_lock:
                self._processing_registry.clear()

            watcher_logger.success("[OK] File watcher stopped")

        except asyncio.TimeoutError:
            watcher_logger.warning("[WARN] File watcher stop timeout")
        except Exception as e:
            watcher_logger.error(f"Error stopping file watcher: {e}")

    async def _process_existing_files(self) -> None:
        """Process any existing JSONL files on startup for the current notebook hash."""
        watcher_logger.info(
            f"[SCAN] Checking for existing JSONL files (hash: {self.notebook_hash})..."
        )

        # Process runtime files - only for current notebook hash
        runtime_pattern = f"{self.notebook_hash}_*.jsonl"
        runtime_files = list(self.runtime_dir.glob(runtime_pattern))

        for file_path in runtime_files:
            watcher_logger.info(f"Processing existing runtime file: {file_path.name}")
            try:
                # Register file for processing to prevent conflicts
                if self._register_file_for_processing(file_path):
                    try:
                        await self._process_file(file_path, "runtime")
                    finally:
                        self._unregister_file_from_processing(file_path)
                else:
                    watcher_logger.debug(
                        f"File already being processed, skipping: {file_path.name}"
                    )
            except Exception as e:
                watcher_logger.error(f"Runtime file processing error: {e}")
                self._unregister_file_from_processing(file_path)

        # Process lineage files - only for current notebook hash
        lineage_pattern = f"{self.notebook_hash}_*.jsonl"
        lineage_files = list(self.lineage_dir.glob(lineage_pattern))

        for file_path in lineage_files:
            watcher_logger.info(f"Processing existing lineage file: {file_path.name}")
            try:
                # Register file for processing to prevent conflicts
                if self._register_file_for_processing(file_path):
                    try:
                        await self._process_file(file_path, "lineage")
                    finally:
                        self._unregister_file_from_processing(file_path)
                else:
                    watcher_logger.debug(
                        f"File already being processed, skipping: {file_path.name}"
                    )
            except Exception as e:
                watcher_logger.error(f"Lineage file processing error: {e}")
                self._unregister_file_from_processing(file_path)

        # Process DataFrame lineage files - only for current notebook hash
        dataframe_lineage_pattern = f"{self.notebook_hash}_*.jsonl"
        dataframe_lineage_files = list(
            self.dataframe_lineage_dir.glob(dataframe_lineage_pattern)
        )

        for file_path in dataframe_lineage_files:
            watcher_logger.info(
                f"Processing existing DataFrame lineage file: {file_path.name}"
            )
            try:
                # Register file for processing to prevent conflicts
                if self._register_file_for_processing(file_path):
                    try:
                        await self._process_file(file_path, "dataframe_lineage")
                    finally:
                        self._unregister_file_from_processing(file_path)
                else:
                    watcher_logger.debug(
                        f"File already being processed, skipping: {file_path.name}"
                    )
            except Exception as e:
                watcher_logger.error(f"DataFrame lineage file processing error: {e}")
                self._unregister_file_from_processing(file_path)

        watcher_logger.success(
            f"[OK] Processed {len(runtime_files)} runtime + {len(lineage_files)} lineage + {len(dataframe_lineage_files)} DataFrame lineage files for notebook {self.notebook_hash}"
        )

        if not (runtime_files or lineage_files or dataframe_lineage_files):
            watcher_logger.info(
                f"No existing JSONL files found for notebook hash {self.notebook_hash}"
            )

    async def _background_processor(self) -> None:
        """Background task for processing file changes."""
        watcher_logger.status("[PROC] Background processor started")

        try:
            while self.running:
                # Process pending files every second
                await self._process_pending_files()
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            watcher_logger.info("Background processor cancelled")
        except Exception as e:
            watcher_logger.error(f"Background processor error: {e}")

    async def _process_pending_files(self) -> None:
        """Process all pending files."""
        pending_files = self.handler.get_pending_files()

        if not pending_files:
            return

        watcher_logger.info(f"📋 Processing {len(pending_files)} pending files...")

        for file_path in pending_files:
            try:
                # Check if file is already being processed
                if not self._register_file_for_processing(file_path):
                    watcher_logger.debug(
                        f"File already being processed, skipping: {file_path.name}"
                    )
                    continue

                try:
                    # Determine file type based on directory (resolve paths for proper comparison)
                    file_parent = file_path.parent.resolve()
                    runtime_dir_resolved = self.runtime_dir.resolve()
                    lineage_dir_resolved = self.lineage_dir.resolve()
                    dataframe_lineage_dir_resolved = (
                        self.dataframe_lineage_dir.resolve()
                    )

                    if file_parent == runtime_dir_resolved:
                        event_type = "runtime"
                    elif file_parent == lineage_dir_resolved:
                        event_type = "lineage"
                    elif file_parent == dataframe_lineage_dir_resolved:
                        event_type = "dataframe_lineage"
                    else:
                        watcher_logger.warning(
                            f"[WARN] Unknown file location: {file_path}"
                        )
                        continue

                    await self._process_file(file_path, event_type)

                finally:
                    # Always unregister the file when done
                    self._unregister_file_from_processing(file_path)

            except Exception as e:
                watcher_logger.error(f"Failed to process file {file_path}: {e}")
                # Ensure file is unregistered on error
                self._unregister_file_from_processing(file_path)

    def _register_file_for_processing(self, file_path: Path) -> bool:
        """Register a file for processing. Returns False if already being processed."""
        with self._registry_lock:
            if file_path in self._processing_registry:
                return False
            self._processing_registry.add(file_path)
            return True

    def _unregister_file_from_processing(self, file_path: Path) -> None:
        """Unregister a file from processing."""
        with self._registry_lock:
            self._processing_registry.discard(file_path)

    async def _process_file(self, file_path: Path, event_type: str) -> None:
        """Process a single JSONL file."""
        if not file_path.exists():
            watcher_logger.warning(f"File no longer exists: {file_path}")
            return

        try:
            # Validate the file is actually complete (not currently being written)
            try:
                # Quick check: file size should be stable
                size_before = file_path.stat().st_size
                await asyncio.sleep(0.1)  # Brief wait
                size_after = file_path.stat().st_size

                if size_before != size_after:
                    watcher_logger.debug(
                        f"File {file_path.name} still being written, skipping"
                    )
                    return
            except Exception as e:
                watcher_logger.error(f"Error validating file {file_path}: {e}")
                return

            # Process file with event processor
            events_count = self.event_processor.process_jsonl_file(
                file_path, event_type
            )

            if events_count > 0:
                self.files_processed += 1
                self.events_processed += events_count

                if self.verbose:
                    watcher_logger.success(
                        f"[OK] Processed {events_count} events from {file_path.name}"
                    )
                else:
                    watcher_logger.tracking(
                        f"Processed {events_count} {event_type} events"
                    )
            else:
                watcher_logger.warning(f"No events processed from {file_path.name}")

        except Exception as e:
            watcher_logger.error(f"Error processing file {file_path}: {e}")

    def get_stats(self) -> dict[str, int]:
        """Get file watcher statistics."""
        processor_stats = self.event_processor.get_validation_summary()

        return {
            "files_processed": self.files_processed,
            "events_processed": self.events_processed,
            "total_events": processor_stats["total_events"],
            "failed_events": processor_stats["failed_events"],
        }

    async def process_file_now(self, file_path: Path) -> int:
        """Manually trigger processing of a specific file."""
        watcher_logger.info(f"[PROC] Manual processing: {file_path.name}")

        # Check if file is already being processed
        if not self._register_file_for_processing(file_path):
            watcher_logger.debug(
                f"File already being processed by background watcher, skipping: {file_path.name}"
            )
            return 0

        try:
            # Determine file type (resolve paths for proper comparison)
            file_parent = file_path.parent.resolve()
            runtime_dir_resolved = self.runtime_dir.resolve()
            lineage_dir_resolved = self.lineage_dir.resolve()
            dataframe_lineage_dir_resolved = self.dataframe_lineage_dir.resolve()

            if file_parent == runtime_dir_resolved:
                event_type = "runtime"
            elif file_parent == lineage_dir_resolved:
                event_type = "lineage"
            elif file_parent == dataframe_lineage_dir_resolved:
                event_type = "dataframe_lineage"
            else:
                msg = f"File not in watched directories: {file_path}"
                raise ValueError(msg)

            # Process the file directly with EventProcessor to get accurate count
            # This avoids the race condition with shared global counter
            if not file_path.exists():
                watcher_logger.warning(f"File no longer exists: {file_path}")
                return 0

            try:
                # Validate the file is actually complete (not currently being written)
                try:
                    # Quick check: file size should be stable
                    size_before = file_path.stat().st_size
                    await asyncio.sleep(0.1)  # Brief wait
                    size_after = file_path.stat().st_size

                    if size_before != size_after:
                        watcher_logger.debug(
                            f"File {file_path.name} still being written, skipping"
                        )
                        return 0
                except Exception as e:
                    watcher_logger.error(f"Error validating file {file_path}: {e}")
                    return 0

                # Process file with event processor directly - this returns the actual count
                events_count = self.event_processor.process_jsonl_file(
                    file_path, event_type
                )

                # Update statistics only if events were processed
                if events_count > 0:
                    self.files_processed += 1
                    self.events_processed += events_count

                    if self.verbose:
                        watcher_logger.success(
                            f"[OK] Processed {events_count} events from {file_path.name}"
                        )
                    else:
                        watcher_logger.tracking(
                            f"Processed {events_count} {event_type} events"
                        )
                else:
                    watcher_logger.warning(f"No events processed from {file_path.name}")

                return events_count

            except Exception as e:
                watcher_logger.error(f"Error processing file {file_path}: {e}")
                return 0

        finally:
            # Always unregister the file when done
            self._unregister_file_from_processing(file_path)


# Utility functions for testing and CLI usage
async def watch_directories(
    runtime_dir: Path,
    lineage_dir: Path,
    dataframe_lineage_dir: Path,
    event_processor: EventProcessor,
    notebook_hash: str,
    duration: float | None = None,
) -> FileWatcher:
    """
    Start watching directories for a specified duration (for testing).

    Args:
        runtime_dir: Directory containing runtime JSONL files
        lineage_dir: Directory containing lineage JSONL files
        dataframe_lineage_dir: Directory containing DataFrame lineage JSONL files
        event_processor: Event processor instance
        notebook_hash: Hash of the notebook to filter files
        duration: How long to watch (None = indefinite)

    Returns:
        FileWatcher instance
    """
    watcher = FileWatcher(
        runtime_dir, lineage_dir, dataframe_lineage_dir, event_processor, notebook_hash
    )

    if duration:
        # Watch for specified duration using asyncio.wait_for
        async def timed_watch():
            # Start the watcher in a task so it doesn't block
            start_task = asyncio.create_task(watcher.start())

            # Wait for the specified duration
            await asyncio.sleep(duration)

            # Stop the watcher
            await watcher.stop()

            # Cancel the start task if it's still running
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

        await timed_watch()
    else:
        # Watch indefinitely
        await watcher.start()

    return watcher


if __name__ == "__main__":
    import tempfile
    from unittest.mock import MagicMock

    # Simple test execution
    async def test_file_watcher():
        """Basic test to verify FileWatcher functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            runtime_dir = temp_path / "runtime"
            lineage_dir = temp_path / "lineage"
            dataframe_lineage_dir = temp_path / "dataframe_lineage"

            # Create mock event processor
            mock_processor = MagicMock()
            mock_processor.process_jsonl_file.return_value = 1

            # Test the watcher for a short duration
            watcher = await watch_directories(
                runtime_dir=runtime_dir,
                lineage_dir=lineage_dir,
                dataframe_lineage_dir=dataframe_lineage_dir,
                event_processor=mock_processor,
                notebook_hash="test_hash",
                duration=0.1,
            )

            watcher_logger.info("FileWatcher test completed successfully")
            watcher_logger.info(f"Files processed: {watcher.files_processed}")
            watcher_logger.info(f"Events processed: {watcher.events_processed}")

    # Run the test
    asyncio.run(test_file_watcher())
