#!/usr/bin/env python3
"""
Hunyo MCP Server Orchestrator - Component coordination and lifecycle management.

Manages the startup, coordination, and shutdown of all system components:
- Capture layer integration
- Database initialization and management
- File watching and ingestion pipeline
- MCP tool data access
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

# Import logging utility
from capture.logger import get_logger
from hunyo_mcp_server.config import (
    get_database_path,
    get_event_directories,
    get_hunyo_data_dir,
)
from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager
from hunyo_mcp_server.ingestion.event_processor import EventProcessor
from hunyo_mcp_server.ingestion.file_watcher import FileWatcher

orchestrator_logger = get_logger("hunyo.orchestrator")


class HunyoOrchestrator:
    """
    Main orchestrator for all Hunyo MCP Server components.

    Manages:
    - Database initialization and connections
    - File watching and event ingestion
    - Capture layer integration
    - Component lifecycle and shutdown
    """

    def __init__(self, notebook_path: Path, *, verbose: bool = False):
        self.notebook_path = notebook_path
        self.verbose = verbose
        self.running = False

        # Data paths
        self.data_dir = get_hunyo_data_dir()
        self.database_path = get_database_path()
        self.runtime_dir, self.lineage_dir = get_event_directories()

        # Component instances
        self.db_manager: DuckDBManager | None = None
        self.event_processor: EventProcessor | None = None
        self.file_watcher: FileWatcher | None = None

        # Background tasks
        self._tasks: list[asyncio.Task] = []
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._background_thread: threading.Thread | None = None

        orchestrator_logger.status("Hunyo MCP Orchestrator initialized")
        orchestrator_logger.config(f"Notebook: {self.notebook_path}")
        orchestrator_logger.config(f"Data directory: {self.data_dir}")
        orchestrator_logger.config(f"Database: {self.database_path}")

    def start(self) -> None:
        """Start all components in the correct order."""
        if self.running:
            orchestrator_logger.warning("Orchestrator already running")
            return

        orchestrator_logger.startup("[START] Starting Hunyo MCP Server components...")

        # Create event directories
        runtime_dir = get_hunyo_data_dir() / "events" / "runtime"
        lineage_dir = get_hunyo_data_dir() / "events" / "lineage"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        lineage_dir.mkdir(parents=True, exist_ok=True)

        # Start ingestion components
        try:
            self._start_capture_layer()
            self._start_ingestion_components()

            orchestrator_logger.success("[OK] All components started successfully")

            # Simple marker for test parsing (bypasses rich logger formatting)
            print(  # noqa: T201
                "HUNYO_READY_MARKER: ALL_COMPONENTS_STARTED",
                file=sys.stderr,
                flush=True,
            )

            self.running = True

        except Exception as e:
            orchestrator_logger.error(f"[ERROR] Failed to start components: {e}")
            self.stop()  # Clean up any partial startup
            raise

    def stop(self) -> None:
        """Stop all components and clean up resources."""
        if not self.running:
            return

        orchestrator_logger.startup("[STOP] Stopping Hunyo MCP Server components...")

        try:
            # Stop file watcher first (this sets running=False and stops the main loop)
            if self.file_watcher:
                orchestrator_logger.startup("[STOP] Stopping file watcher...")
                # Schedule the stop method to run in the event loop
                if self._event_loop and not self._event_loop.is_closed():
                    # Create a future to run file_watcher.stop() in the event loop
                    future = asyncio.run_coroutine_threadsafe(
                        self.file_watcher.stop(), self._event_loop
                    )
                    try:
                        future.result(timeout=3.0)  # Wait for file watcher to stop
                    except Exception as e:
                        orchestrator_logger.warning(
                            f"[WARN] File watcher stop error: {e}"
                        )

            # Stop file watcher and background tasks
            if self._event_loop and not self._event_loop.is_closed():
                # Cancel all tasks first
                for task in self._tasks:
                    if not task.done():
                        task.cancel()

                # Stop the event loop gracefully by calling stop
                self._event_loop.call_soon_threadsafe(self._event_loop.stop)

                # Give tasks a moment to complete cancellation
                time.sleep(0.1)

            # Wait for background thread to finish with reasonable timeout
            if self._background_thread and self._background_thread.is_alive():
                orchestrator_logger.startup(
                    "[WAIT] Waiting for background thread to finish..."
                )

                # Use short timeout since it's a daemon thread (will be terminated when main exits)
                self._background_thread.join(timeout=1.0)

                # If thread is still alive after timeout, that's OK since it's daemon
                if self._background_thread.is_alive():
                    orchestrator_logger.info(
                        "[INFO] Background thread still running (daemon - will exit with main process)"
                    )

            # Close database connections
            if self.db_manager:
                orchestrator_logger.startup("[DATA] Closing database connections...")
                self.db_manager.close()

            self.running = False
            orchestrator_logger.success("[OK] All components stopped")

        except Exception as e:
            orchestrator_logger.error(f"Error during shutdown: {e}")
            # Continue with shutdown even if there are errors

    def get_db_manager(self) -> DuckDBManager:
        """Get the database manager instance for MCP tools."""
        if not self.db_manager:
            msg = "Database manager not initialized"
            raise RuntimeError(msg)
        return self.db_manager

    def _start_database(self) -> None:
        """Initialize the DuckDB database and schema."""
        orchestrator_logger.startup("[DATA] Initializing DuckDB database...")

        self.db_manager = DuckDBManager(self.database_path)
        self.db_manager.initialize_database()

        orchestrator_logger.success(f"Database initialized: {self.database_path}")

    def _start_event_processor(self) -> None:
        """Initialize the event processor."""
        orchestrator_logger.startup("[SETUP] Initializing event processor...")

        if not self.db_manager:
            msg = "Database manager must be started before event processor"
            raise RuntimeError(msg)

        self.event_processor = EventProcessor(self.db_manager)
        orchestrator_logger.success("Event processor initialized")

    def _start_file_watcher(self) -> None:
        """Start the file watcher in a background thread."""
        orchestrator_logger.startup("[WATCH] Starting file watcher...")

        if not self.event_processor:
            msg = "Event processor must be started before file watcher"
            raise RuntimeError(msg)

        self.file_watcher = FileWatcher(
            runtime_dir=self.runtime_dir,
            lineage_dir=self.lineage_dir,
            event_processor=self.event_processor,
            verbose=self.verbose,
        )

        # Start file watcher in background thread with asyncio
        def run_file_watcher():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

            try:
                # Start the file watcher
                task = self._event_loop.create_task(self.file_watcher.start())
                self._tasks.append(task)

                # Run the event loop
                self._event_loop.run_forever()
            except Exception as e:
                orchestrator_logger.error(f"File watcher error: {e}")
            finally:
                # Clean up
                for task in self._tasks:
                    if not task.done():
                        task.cancel()
                self._event_loop.close()

        self._background_thread = threading.Thread(target=run_file_watcher, daemon=True)
        self._background_thread.start()

        orchestrator_logger.success("File watcher started in background")

    def _start_capture_layer(self) -> None:
        """Initialize the unified capture layer for both runtime and lineage events."""
        orchestrator_logger.startup("[SETUP] Initializing unified capture layer...")

        try:
            # Import and start unified tracking
            from capture.unified_marimo_interceptor import enable_unified_tracking

            # Enable unified tracking with notebook path for proper convention naming
            interceptor = enable_unified_tracking(notebook_path=str(self.notebook_path))

            if interceptor:
                orchestrator_logger.status(
                    f"Runtime events: {interceptor.runtime_file.name}"
                )
                orchestrator_logger.status(
                    f"Lineage events: {interceptor.lineage_file.name}"
                )
                orchestrator_logger.success(
                    "[OK] Unified capture layer active - marimo hooks enabled for both runtime and lineage events"
                )
            else:
                orchestrator_logger.warning(
                    "[WARN] Failed to create unified interceptor"
                )

        except ImportError as e:
            orchestrator_logger.warning(f"Unified capture layer import failed: {e}")
            orchestrator_logger.warning(
                "[WARN] Unified tracking not available. Please add this import to your notebook manually:"
            )
            orchestrator_logger.status(
                "  from capture.unified_marimo_interceptor import enable_unified_tracking"
            )
            orchestrator_logger.status("  enable_unified_tracking()")

        except Exception as e:
            orchestrator_logger.error(f"Failed to start unified capture layer: {e}")
            orchestrator_logger.warning("Falling back to manual import instructions...")
            orchestrator_logger.status(
                "Please add unified tracking import to your notebook manually (see logs above)"
            )

    def _start_ingestion_components(self) -> None:
        """Start all ingestion components in proper order."""
        orchestrator_logger.startup("[SETUP] Starting ingestion components...")

        # Start components in dependency order
        self._start_database()
        self._start_event_processor()
        self._start_file_watcher()

        orchestrator_logger.success("[OK] All ingestion components started")


# Global orchestrator instance for MCP tools to access
_global_orchestrator: HunyoOrchestrator | None = None


def get_global_orchestrator() -> HunyoOrchestrator:
    """Get the global orchestrator instance for MCP tools."""
    if _global_orchestrator is None:
        msg = "Orchestrator not started. Run hunyo-mcp-server first."
        raise RuntimeError(msg)
    return _global_orchestrator


def set_global_orchestrator(orchestrator: HunyoOrchestrator) -> None:
    """Set the global orchestrator instance."""
    global _global_orchestrator  # noqa: PLW0603
    _global_orchestrator = orchestrator
