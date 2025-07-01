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
import threading
from pathlib import Path
from typing import Optional

from .config import get_database_path, get_event_directories, get_hunyo_data_dir
from .ingestion.duckdb_manager import DuckDBManager
from .ingestion.event_processor import EventProcessor
from .ingestion.file_watcher import FileWatcher

# Import logging utility
try:
    from ..capture.logger import get_logger

    orchestrator_logger = get_logger("hunyo.orchestrator")
except ImportError:
    # Fallback for testing
    class SimpleLogger:
        def status(self, msg):
            print(f"[STATUS] {msg}")

        def success(self, msg):
            print(f"[SUCCESS] {msg}")

        def warning(self, msg):
            print(f"[WARNING] {msg}")

        def error(self, msg):
            print(f"[ERROR] {msg}")

        def config(self, msg):
            print(f"[CONFIG] {msg}")

        def startup(self, msg):
            print(f"[STARTUP] {msg}")

    orchestrator_logger = SimpleLogger()


class HunyoOrchestrator:
    """
    Main orchestrator for all Hunyo MCP Server components.

    Manages:
    - Database initialization and connections
    - File watching and event ingestion
    - Capture layer integration
    - Component lifecycle and shutdown
    """

    def __init__(self, notebook_path: Path, verbose: bool = False):
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

        orchestrator_logger.startup("🚀 Starting Hunyo MCP Server components...")

        try:
            # 1. Initialize database
            self._start_database()

            # 2. Initialize event processor
            self._start_event_processor()

            # 3. Start file watching in background
            self._start_file_watcher()

            # 4. Initialize capture layer (inject into notebook)
            self._start_capture_layer()

            self.running = True
            orchestrator_logger.success("✅ All components started successfully")

        except Exception as e:
            orchestrator_logger.error(f"Failed to start components: {e}")
            self.stop()  # Clean up any partially started components
            raise

    def stop(self) -> None:
        """Stop all components and clean up resources."""
        if not self.running:
            return

        orchestrator_logger.startup("🛑 Stopping Hunyo MCP Server components...")

        try:
            # Stop file watcher and background tasks
            if self._event_loop and not self._event_loop.is_closed():
                for task in self._tasks:
                    if not task.done():
                        task.cancel()

                # Stop the event loop
                self._event_loop.call_soon_threadsafe(self._event_loop.stop)

            # Wait for background thread to finish
            if self._background_thread and self._background_thread.is_alive():
                self._background_thread.join(timeout=5.0)

            # Close database connections
            if self.db_manager:
                self.db_manager.close()

            self.running = False
            orchestrator_logger.success("✅ All components stopped")

        except Exception as e:
            orchestrator_logger.error(f"Error during shutdown: {e}")

    def get_db_manager(self) -> DuckDBManager:
        """Get the database manager instance for MCP tools."""
        if not self.db_manager:
            raise RuntimeError("Database manager not initialized")
        return self.db_manager

    def _start_database(self) -> None:
        """Initialize the DuckDB database and schema."""
        orchestrator_logger.startup("📊 Initializing DuckDB database...")

        self.db_manager = DuckDBManager(self.database_path)
        self.db_manager.initialize_database()

        orchestrator_logger.success(f"Database initialized: {self.database_path}")

    def _start_event_processor(self) -> None:
        """Initialize the event processor."""
        orchestrator_logger.startup("⚙️ Initializing event processor...")

        if not self.db_manager:
            raise RuntimeError(
                "Database manager must be started before event processor"
            )

        self.event_processor = EventProcessor(self.db_manager)
        orchestrator_logger.success("Event processor initialized")

    def _start_file_watcher(self) -> None:
        """Start the file watcher in a background thread."""
        orchestrator_logger.startup("👁️ Starting file watcher...")

        if not self.event_processor:
            raise RuntimeError("Event processor must be started before file watcher")

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
        """Initialize the capture layer for the notebook."""
        orchestrator_logger.startup("🔧 Initializing capture layer...")

        # TODO: Implement notebook injection when AST injector is ready
        # For now, log that manual imports are needed
        orchestrator_logger.warning(
            "⚠️ Notebook auto-injection not yet implemented. "
            "Please add these imports to your notebook manually:"
        )
        orchestrator_logger.config(
            "  from capture.lightweight_runtime_tracker import enable_runtime_tracking"
        )
        orchestrator_logger.config(
            "  from capture.live_lineage_interceptor import enable_live_tracking"
        )
        orchestrator_logger.config("  enable_runtime_tracking()")
        orchestrator_logger.config("  enable_live_tracking()")

        orchestrator_logger.success("Capture layer ready (manual imports required)")


# Global orchestrator instance for MCP tools to access
_global_orchestrator: HunyoOrchestrator | None = None


def get_global_orchestrator() -> HunyoOrchestrator:
    """Get the global orchestrator instance for MCP tools."""
    if _global_orchestrator is None:
        raise RuntimeError("Orchestrator not started. Run hunyo-mcp-server first.")
    return _global_orchestrator


def set_global_orchestrator(orchestrator: HunyoOrchestrator) -> None:
    """Set the global orchestrator instance."""
    global _global_orchestrator
    _global_orchestrator = orchestrator
