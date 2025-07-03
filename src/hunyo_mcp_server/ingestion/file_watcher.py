#!/usr/bin/env python3
"""
File Watcher - Monitors JSONL event files for changes and triggers processing.

Uses watchdog to monitor runtime and lineage event directories for new files
and file modifications, triggering the event processor for real-time ingestion.
"""

import asyncio
import time
from pathlib import Path
from threading import Lock

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Import logging utility
from capture.logger import get_logger
from hunyo_mcp_server.ingestion.event_processor import EventProcessor

watcher_logger = get_logger("hunyo.watcher")


class JSONLFileHandler(FileSystemEventHandler):
    """
    File system event handler for JSONL files.

    Processes file creation and modification events for .jsonl files,
    triggering event processing with debouncing to handle rapid changes.
    """

    def __init__(self, file_watcher: "FileWatcher"):
        super().__init__()
        self.file_watcher = file_watcher
        self._pending_files: set[Path] = set()
        self._processing_lock = Lock()
        self._last_processed: dict[Path, float] = {}
        self._debounce_delay = 1.0  # 1 second debounce

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith(".jsonl"):
            file_path = Path(event.src_path)
            watcher_logger.tracking(f"📄 New JSONL file detected: {file_path.name}")
            self._queue_for_processing(file_path)

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith(".jsonl"):
            file_path = Path(event.src_path)

            # Check if enough time has passed since last processing (debounce)
            current_time = time.time()
            last_processed = self._last_processed.get(file_path, 0)

            if current_time - last_processed >= self._debounce_delay:
                watcher_logger.tracking(f"📝 JSONL file modified: {file_path.name}")
                self._queue_for_processing(file_path)
            else:
                # Add to pending files for later processing
                with self._processing_lock:
                    self._pending_files.add(file_path)

    def _queue_for_processing(self, file_path: Path):
        """Queue a file for processing."""
        with self._processing_lock:
            self._pending_files.add(file_path)
            self._last_processed[file_path] = time.time()

        # Note: Don't create asyncio tasks here - we're in a thread without an event loop.
        # The background processor will pick up pending files automatically.

    def get_pending_files(self) -> set[Path]:
        """Get and clear pending files."""
        with self._processing_lock:
            pending = self._pending_files.copy()
            self._pending_files.clear()
            return pending


class FileWatcher:
    """
    Main file watcher for monitoring JSONL event files.

    Features:
    - Monitors runtime and lineage event directories
    - Debounced file processing to handle rapid changes
    - Automatic file type detection
    - Error handling and recovery
    - Background processing with asyncio
    """

    def __init__(
        self,
        runtime_dir: Path,
        lineage_dir: Path,
        event_processor: EventProcessor,
        *,
        verbose: bool = False,
    ):
        self.runtime_dir = Path(runtime_dir)
        self.lineage_dir = Path(lineage_dir)
        self.event_processor = event_processor
        self.verbose = verbose

        # File watching components
        self.observer = Observer()
        self.handler = JSONLFileHandler(self)
        self.running = False

        # Processing state
        self._processing_queue = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()

        # Statistics
        self.files_processed = 0
        self.events_processed = 0

        watcher_logger.status("File Watcher initialized")
        watcher_logger.info(f"Runtime directory: {self.runtime_dir}")
        watcher_logger.info(f"Lineage directory: {self.lineage_dir}")

    async def start(self) -> None:
        """Start watching for file changes."""
        if self.running:
            watcher_logger.warning("File watcher already running")
            return

        watcher_logger.status("🔍 Starting file watcher...")

        try:
            # Ensure directories exist
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            self.lineage_dir.mkdir(parents=True, exist_ok=True)

            # Set up file system watching
            self.observer.schedule(self.handler, str(self.runtime_dir), recursive=False)
            self.observer.schedule(self.handler, str(self.lineage_dir), recursive=False)

            # Start the observer
            self.observer.start()

            # Start background processing task
            self._processing_task = asyncio.create_task(self._background_processor())

            self.running = True
            watcher_logger.success("✅ File watcher started")

            # Process any existing files
            await self._process_existing_files()

            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            watcher_logger.error(f"Failed to start file watcher: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the file watcher."""
        if not self.running:
            return

        watcher_logger.status("🛑 Stopping file watcher...")

        self.running = False

        # Stop the file system observer
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()

        # Cancel background processing task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        watcher_logger.success("✅ File watcher stopped")

    async def _process_existing_files(self) -> None:
        """Process any existing JSONL files on startup."""
        watcher_logger.info("📂 Checking for existing JSONL files...")

        # Process runtime files
        runtime_files = list(self.runtime_dir.glob("*.jsonl"))
        for file_path in runtime_files:
            watcher_logger.info(f"Processing existing runtime file: {file_path.name}")
            await self._process_file(file_path, "runtime")

        # Process lineage files
        lineage_files = list(self.lineage_dir.glob("*.jsonl"))
        for file_path in lineage_files:
            watcher_logger.info(f"Processing existing lineage file: {file_path.name}")
            await self._process_file(file_path, "lineage")

        if runtime_files or lineage_files:
            watcher_logger.success(
                f"✅ Processed {len(runtime_files)} runtime + {len(lineage_files)} lineage files"
            )
        else:
            watcher_logger.info("No existing JSONL files found")

    async def _background_processor(self) -> None:
        """Background task for processing file changes."""
        watcher_logger.status("🔄 Background processor started")

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
                # Determine file type based on directory
                if file_path.parent == self.runtime_dir:
                    event_type = "runtime"
                elif file_path.parent == self.lineage_dir:
                    event_type = "lineage"
                else:
                    watcher_logger.warning(f"Unknown file location: {file_path}")
                    continue

                await self._process_file(file_path, event_type)

            except Exception as e:
                watcher_logger.error(f"Failed to process file {file_path}: {e}")

    async def _process_file(self, file_path: Path, event_type: str) -> None:
        """Process a single JSONL file."""
        if not file_path.exists():
            watcher_logger.warning(f"File no longer exists: {file_path}")
            return

        try:
            # Process file with event processor
            events_count = self.event_processor.process_jsonl_file(
                file_path, event_type
            )

            if events_count > 0:
                self.files_processed += 1
                self.events_processed += events_count

                if self.verbose:
                    watcher_logger.success(
                        f"✅ Processed {events_count} events from {file_path.name}"
                    )
                else:
                    watcher_logger.tracking(
                        f"Processed {events_count} {event_type} events"
                    )

        except Exception as e:
            watcher_logger.error(f"Error processing file {file_path}: {e}")

    def get_stats(self) -> dict[str, int]:
        """Get file watcher statistics."""
        processor_stats = self.event_processor.get_processing_stats()

        return {
            "files_processed": self.files_processed,
            "events_processed": self.events_processed,
            "total_events": processor_stats["total_events"],
            "failed_events": processor_stats["failed_events"],
        }

    async def process_file_now(self, file_path: Path) -> int:
        """Manually trigger processing of a specific file."""
        watcher_logger.info(f"🔄 Manual processing: {file_path.name}")

        # Determine file type
        if file_path.parent == self.runtime_dir:
            event_type = "runtime"
        elif file_path.parent == self.lineage_dir:
            event_type = "lineage"
        else:
            msg = f"File not in watched directories: {file_path}"
            raise ValueError(msg)

        await self._process_file(file_path, event_type)
        return self.event_processor.process_jsonl_file(file_path, event_type)


# Utility functions for testing and CLI usage
async def watch_directories(
    runtime_dir: Path,
    lineage_dir: Path,
    event_processor: EventProcessor,
    duration: float | None = None,
) -> FileWatcher:
    """
    Start watching directories for a specified duration (for testing).

    Args:
        runtime_dir: Directory containing runtime JSONL files
        lineage_dir: Directory containing lineage JSONL files
        event_processor: Event processor instance
        duration: How long to watch (None = indefinite)

    Returns:
        FileWatcher instance
    """
    watcher = FileWatcher(runtime_dir, lineage_dir, event_processor)

    if duration:
        # Watch for specified duration
        async def timed_watch():
            await watcher.start()
            await asyncio.sleep(duration)
            await watcher.stop()

        await timed_watch()
    else:
        # Watch indefinitely
        await watcher.start()

    return watcher
