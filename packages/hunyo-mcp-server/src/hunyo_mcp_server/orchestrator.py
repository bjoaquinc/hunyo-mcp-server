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
import uuid
from datetime import datetime, timezone
from pathlib import Path

from hunyo_mcp_server.config import (
    ensure_directory_structure,
    get_database_path,
    get_dataframe_lineage_events_dir,
    get_event_directories,
    get_hunyo_data_dir,
    get_notebook_file_hash,
    is_development_mode,
)
from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager
from hunyo_mcp_server.ingestion.event_processor import EventProcessor
from hunyo_mcp_server.ingestion.file_watcher import FileWatcher

# Import logging utility
from hunyo_mcp_server.logger import get_logger

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

        # Calculate notebook hash for notebook-specific database
        self.notebook_hash = get_notebook_file_hash(notebook_path)

        # Data paths
        self.data_dir = get_hunyo_data_dir()
        self.database_path = get_database_path(self.notebook_hash)
        self.runtime_dir, self.lineage_dir = get_event_directories()
        self.dataframe_lineage_dir = get_dataframe_lineage_events_dir()

        # Component instances
        self.db_manager: DuckDBManager | None = None
        self.event_processor: EventProcessor | None = None
        self.file_watcher: FileWatcher | None = None

        # Background tasks
        self._tasks: list[asyncio.Task] = []
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._background_thread: threading.Thread | None = None

        orchestrator_logger.status("Hunyo MCP Orchestrator initialized")
        orchestrator_logger.config(f"Development mode: {is_development_mode()}")
        orchestrator_logger.config(f"Notebook: {self.notebook_path}")
        orchestrator_logger.config(f"Database: {self.database_path}")
        orchestrator_logger.config(f"Runtime events: {self.runtime_dir}")
        orchestrator_logger.config(f"Lineage events: {self.lineage_dir}")
        orchestrator_logger.config(
            f"DataFrame lineage events: {self.dataframe_lineage_dir}"
        )

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
            # Ensure directory structure exists
            ensure_directory_structure()

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

        # Enhanced logging for database path resolution
        orchestrator_logger.config(f"[DATA] Database path: {self.database_path}")
        orchestrator_logger.config(
            f"[DATA] Database path exists: {self.database_path.exists()}"
        )
        orchestrator_logger.config(
            f"[DATA] Database path is absolute: {self.database_path.is_absolute()}"
        )
        orchestrator_logger.config(
            f"[DATA] Database parent directory: {self.database_path.parent}"
        )

        # Ensure database directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        orchestrator_logger.config(
            f"[DATA] Database directory created: {self.database_path.parent}"
        )

        # Initialize database manager
        self.db_manager = DuckDBManager(self.database_path)

        # Initialize schema with validation
        try:
            self.db_manager.initialize_database()
            orchestrator_logger.success(
                f"[DATA] Database schema initialized: {self.database_path}"
            )

            # Verify schema was created successfully
            self._validate_database_schema()

        except Exception as e:
            orchestrator_logger.error(f"[DATA] Database initialization failed: {e}")
            raise

    def _validate_database_schema(self) -> None:
        """Validate that the database schema is properly initialized."""
        orchestrator_logger.startup("[DATA] Validating database schema...")

        if not self.db_manager:
            error_msg = "Database manager not initialized"
            raise RuntimeError(error_msg)

        try:
            # Check if required tables exist
            required_tables = [
                "runtime_events",
                "lineage_events",
                "dataframe_lineage_events",
            ]

            for table_name in required_tables:
                # Use DuckDB system tables to check existence
                result = self.db_manager.execute_query(
                    f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                )

                if not result or result[0]["count_star()"] == 0:
                    error_msg = f"Required table '{table_name}' was not created"
                    raise RuntimeError(error_msg)

                orchestrator_logger.config(f"[DATA] Table '{table_name}' exists")

            orchestrator_logger.success("[DATA] Database schema validation passed")

        except Exception as e:
            orchestrator_logger.error(f"[DATA] Database schema validation failed: {e}")
            raise

    def _test_database_connectivity(self) -> None:
        """Test database connectivity by performing sample operations."""
        orchestrator_logger.startup("[DATA] Testing database connectivity...")

        if not self.db_manager:
            error_msg = "Database manager not initialized"
            raise RuntimeError(error_msg)

        try:
            # Test basic connection
            test_query = "SELECT 1"
            result = self.db_manager.execute_query(test_query)

            if not result:
                error_msg = "Database connection test failed"
                raise RuntimeError(error_msg)

            # Check that all required tables exist by attempting to query them
            required_tables = [
                "runtime_events",
                "lineage_events",
                "dataframe_lineage_events",
            ]
            for table_name in required_tables:
                try:
                    # Just attempt to query the table - if it exists, this will succeed
                    # even if the table is empty (which would return count=0)
                    self.db_manager.execute_query(f"SELECT COUNT(*) FROM {table_name}")
                    orchestrator_logger.config(f"[DATA] Table '{table_name}' exists")
                except Exception as e:
                    error_msg = f"Required table '{table_name}' was not created: {e}"
                    raise RuntimeError(error_msg) from e

            # Test insert and rollback to ensure database is writable
            test_event_id = 999999999  # Use a large integer for testing
            test_execution_id = str(uuid.uuid4())[:8]  # Truncate to 8 chars for CHAR(8)
            test_session_id = str(uuid.uuid4())[:8]  # Truncate to 8 chars for CHAR(8)
            test_timestamp = datetime.now(timezone.utc)

            # Insert test data
            self.db_manager.execute_query(
                """
                INSERT INTO runtime_events
                (event_id, event_type, execution_id, cell_id, session_id,
                 timestamp, emitted_at, duration_ms, start_memory_mb, end_memory_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    test_event_id,
                    "test_event",
                    test_execution_id,
                    "test_cell",
                    test_session_id,
                    test_timestamp,
                    test_timestamp,
                    100,
                    50.0,
                    55.0,
                ),
            )

            # Verify insert
            result = self.db_manager.execute_query(
                "SELECT COUNT(*) FROM runtime_events WHERE event_id = ?",
                (test_event_id,),
            )

            if result and len(result) > 0:
                first_row = result[0]
                try:
                    # Try different possible column names for the count
                    count = first_row.get("count_star()", first_row.get("COUNT(*)", 0))
                except (KeyError, IndexError, TypeError):
                    count = (
                        next(iter(first_row.values()))
                        if isinstance(first_row, dict)
                        else None
                    )

                orchestrator_logger.config("[DATA] Database connectivity test passed")
            else:
                count = 0

            if count != 1:
                error_msg = f"Test insert verification failed - expected count=1, got count={count}"
                raise Exception(error_msg)

            # Clean up test data
            self.db_manager.execute_query(
                "DELETE FROM runtime_events WHERE event_id = ?", (test_event_id,)
            )

        except Exception as e:
            orchestrator_logger.error(f"[DATA] Database connectivity test failed: {e}")
            raise

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
            dataframe_lineage_dir=self.dataframe_lineage_dir,
            event_processor=self.event_processor,
            notebook_hash=self.notebook_hash,
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
        """Provide instructions for enabling the unified capture layer."""
        orchestrator_logger.startup(
            "[SETUP] Unified capture layer setup instructions..."
        )

        orchestrator_logger.info(
            "[INFO] To enable notebook tracking, add this to your marimo notebook:"
        )
        orchestrator_logger.status(
            "  # Install capture layer: pip install hunyo-capture"
        )
        orchestrator_logger.status(
            "  from hunyo_capture.unified_notebook_interceptor import enable_unified_tracking"
        )
        orchestrator_logger.status(
            f"  enable_unified_tracking(notebook_path='{self.notebook_path}')"
        )
        orchestrator_logger.info(
            "[INFO] MCP server will automatically discover and process captured events"
        )
        orchestrator_logger.success(
            "[OK] Capture layer instructions provided - add imports to your notebook"
        )

    def _start_ingestion_components(self) -> None:
        """Start all ingestion components in proper order."""
        orchestrator_logger.startup("[SETUP] Starting ingestion components...")

        # Start components in dependency order
        self._start_database()
        self._test_database_connectivity()  # Test after database initialization
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
