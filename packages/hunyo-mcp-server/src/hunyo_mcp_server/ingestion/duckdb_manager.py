#!/usr/bin/env python3
"""
DuckDB Manager - Database initialization and connection management.

Handles DuckDB database setup, schema creation, and connection pooling
for the Hunyo MCP Server ingestion pipeline.
"""

import concurrent.futures
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import duckdb

# Import project paths
from hunyo_mcp_server.config import get_repository_root

# Import logging utility
from hunyo_mcp_server.logger import get_logger
from hunyo_mcp_server.utils.paths import (
    normalize_database_path,
    validate_path_accessibility,
)

db_logger = get_logger("hunyo.duckdb")

# Constants
WINDOWS_MAX_PATH_LENGTH = 260  # Windows long path limit


class DuckDBManager:
    """
    Manages DuckDB database initialization, connections, and schema setup.

    Features:
    - Automatic schema initialization from SQL files
    - Connection management and reuse with retry logic
    - Platform-specific configuration optimization
    - Query execution with error handling and timeouts
    - Transaction support for batch operations
    """

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.connection: duckdb.DuckDBPyConnection | None = None
        self._schema_initialized = False

        # Platform-specific configuration
        self.platform_config = self._get_platform_config()

        # Normalize database path for cross-platform compatibility
        self.database_path = Path(self._normalize_path(str(self.database_path)))

        # Ensure database directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        db_logger.status("DuckDB Manager initialized")
        db_logger.config(f"Database path: {self.database_path}")
        db_logger.config(f"Platform config: {self.platform_config}")

    def _normalize_path(self, path: str) -> str:
        """Handle cross-platform path normalization with validation."""
        # Use the centralized cross-platform path normalization
        normalized_path = normalize_database_path(path)

        # Validate path accessibility (except for in-memory databases)
        if path != ":memory:" and not validate_path_accessibility(normalized_path):
            db_logger.warning(
                f"[WARN] Database path may not be accessible: {normalized_path}"
            )

        return normalized_path

    def _get_platform_config(self) -> dict[str, Any]:
        """Get platform-optimized DuckDB configuration."""
        system = platform.system().lower()
        cpu_count = os.cpu_count() or 1

        base_config = {
            "threads": cpu_count,
        }

        if system == "windows":
            base_config.update(
                {
                    "memory_limit": "1GB",  # Conservative for Windows - use absolute value
                    "temp_directory": None,  # Let DuckDB auto-detect
                }
            )
            db_logger.config("[WINDOWS] Using conservative memory configuration")
        elif system == "darwin":  # macOS
            base_config.update(
                {
                    "memory_limit": "2GB",  # Respect unified memory - use absolute value
                    "temp_directory": None,  # Let DuckDB auto-detect
                }
            )
            db_logger.config("[MACOS] Using unified memory configuration")
        else:  # Linux and other Unix
            base_config.update(
                {
                    "memory_limit": "4GB",  # Aggressive on Linux - use absolute value
                    "temp_directory": None,  # Let DuckDB auto-detect
                }
            )
            db_logger.config("[LINUX] Using aggressive memory configuration")

        return base_config

    def initialize_database(self) -> None:
        """Initialize database and create schema from SQL files."""
        if self._schema_initialized:
            db_logger.warning("Database already initialized")
            return

        db_logger.startup("[INIT] Initializing DuckDB database and schema...")

        try:
            # Connect to database with retry logic
            self._connect_with_retry()

            # Apply platform-specific configuration
            self._apply_platform_config()

            # Execute schema initialization
            self._create_schema()

            # Verify schema creation
            self._verify_schema()

            # Explicitly commit to ensure database file is created on disk (Windows NTFS compatibility)
            self.connection.commit()

            # Force checkpoint to merge WAL → main file (critical for Windows NTFS directory entry)
            try:
                self.connection.execute("CHECKPOINT")
                db_logger.success(
                    "[OK] Database checkpoint completed - file materialized to disk"
                )
            except Exception as e:
                db_logger.warning(f"Database checkpoint failed: {e}")

            # Enable automatic checkpointing on shutdown for future connections
            try:
                self.connection.execute("PRAGMA enable_checkpoint_on_shutdown")
                db_logger.success("[OK] Auto-checkpoint on shutdown enabled")
            except Exception as e:
                db_logger.warning(f"Failed to enable auto-checkpoint: {e}")

            self._schema_initialized = True
            db_logger.success("[OK] Database schema initialized successfully")

            # Simple marker for test parsing (bypasses rich logger formatting)
            print(  # noqa: T201
                "HUNYO_READY_MARKER: DATABASE_SCHEMA_READY", file=sys.stderr, flush=True
            )

        except Exception as e:
            db_logger.error(f"Failed to initialize database: {e}")
            raise

    def _connect_with_retry(self, max_retries: int = 3) -> None:
        """Connect to database with retry logic for concurrency issues."""
        if self.connection:
            return

        for attempt in range(max_retries):
            try:
                db_logger.config(f"[CONNECT] Attempt {attempt + 1}/{max_retries}")
                self.connection = duckdb.connect(str(self.database_path))
                db_logger.success(f"Connected to database: {self.database_path}")
                return
            except Exception as e:
                error_msg = str(e).lower()

                # Check for concurrency-related errors that warrant retry
                retry_conditions = [
                    "used by another process",
                    "database is locked",
                    "connection failed",
                    "cannot open",
                ]

                should_retry = any(
                    condition in error_msg for condition in retry_conditions
                )

                if should_retry and attempt < max_retries - 1:
                    wait_time = 0.1 * (2**attempt)  # Exponential backoff
                    db_logger.warning(
                        f"[RETRY] Connection failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    db_logger.error(
                        f"Failed to connect to database after {max_retries} attempts: {e}"
                    )
                    raise

    def _apply_platform_config(self) -> None:
        """Apply platform-specific configuration to the database connection."""
        if not self.connection:
            msg = "No database connection available"
            raise RuntimeError(msg)

        db_logger.config("[CONFIG] Applying platform-specific configuration...")

        for setting, value in self.platform_config.items():
            try:
                # Skip settings that are None or not DuckDB configuration options
                if value is None or setting in [
                    "memory_allocator",
                    "enable_external_access",
                    "temp_directory",
                ]:
                    continue

                self.connection.execute(f"SET {setting} = '{value}'")
                db_logger.config(f"[CONFIG] Set {setting} = {value}")
            except Exception as e:
                db_logger.warning(f"[CONFIG] Failed to set {setting}: {e}")

    def _execute_with_timeout(
        self, func, timeout_seconds: float = 30.0, *args, **kwargs
    ):
        """Execute a function with cross-platform timeout handling."""
        if os.name == "nt":  # Windows
            # Use ThreadPoolExecutor for Windows (no signal.alarm support)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError as e:
                    timeout_msg = f"Operation timed out after {timeout_seconds} seconds"
                    db_logger.error(timeout_msg)
                    raise TimeoutError(timeout_msg) from e
        else:  # Unix-like systems
            # Use signal.alarm for Unix systems
            import signal

            def timeout_handler(_signum, _frame):
                timeout_msg = f"Operation timed out after {timeout_seconds} seconds"
                raise TimeoutError(timeout_msg)

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def _connect(self) -> None:
        """Legacy connect method - use _connect_with_retry instead."""
        self._connect_with_retry()

    def _create_schema(self) -> None:
        """Create database schema from SQL files."""
        # Get project root and schema directory
        try:
            project_root = get_repository_root()
            db_logger.config(f"[SCHEMA] Project root: {project_root}")
        except Exception as e:
            db_logger.error(f"[SCHEMA] Failed to get repository root: {e}")
            raise

        schema_dir = project_root / "schemas" / "sql"
        db_logger.config(f"[SCHEMA] Schema directory: {schema_dir}")
        db_logger.config(f"[SCHEMA] Schema directory exists: {schema_dir.exists()}")

        if not schema_dir.exists():
            # Additional debugging for Windows CI
            parent_exists = schema_dir.parent.exists()
            db_logger.error(f"[SCHEMA] Parent directory exists: {parent_exists}")
            if parent_exists:
                try:
                    contents = list(schema_dir.parent.iterdir())
                    db_logger.error(f"[SCHEMA] Parent directory contents: {contents}")
                except Exception as ex:
                    db_logger.error(f"[SCHEMA] Cannot list parent directory: {ex}")

            msg = f"Schema directory not found: {schema_dir}"
            raise FileNotFoundError(msg)

        # Execute main initialization script
        init_script = schema_dir / "init_database.sql"
        db_logger.config(f"[SCHEMA] Init script path: {init_script}")
        db_logger.config(f"[SCHEMA] Init script exists: {init_script.exists()}")

        if init_script.exists():
            db_logger.config(f"Executing schema: {init_script.name}")
            self._execute_sql_file(init_script)
        else:
            # Additional debugging - list schema directory contents
            try:
                schema_files = list(schema_dir.iterdir())
                db_logger.warning(
                    f"[SCHEMA] Available files in {schema_dir}: {schema_files}"
                )
            except Exception as ex:
                db_logger.error(f"[SCHEMA] Cannot list schema directory: {ex}")

            # Fallback: execute individual schema files
            db_logger.warning("Main init script not found, using individual files")

            schema_files = ["runtime_events_table.sql", "openlineage_events_table.sql"]

            for filename in schema_files:
                file_path = schema_dir / filename
                if file_path.exists():
                    db_logger.config(f"Executing schema: {filename}")
                    self._execute_sql_file(file_path)
                else:
                    db_logger.warning(f"Schema file not found: {filename}")

            # Create views
            views_dir = schema_dir / "views"
            if views_dir.exists():
                for view_file in views_dir.glob("*.sql"):
                    db_logger.config(f"Creating view: {view_file.name}")
                    self._execute_sql_file(view_file)

    def _execute_sql_file(self, file_path: Path) -> None:
        """Execute SQL commands from a file."""
        try:
            db_logger.config(f"[SQL] Reading file: {file_path}")
            with open(file_path, encoding="utf-8") as f:
                sql_content = f.read()

            db_logger.config(f"[SQL] File content length: {len(sql_content)} chars")

            # Split on semicolons and execute each statement
            statements = [
                stmt.strip() for stmt in sql_content.split(";") if stmt.strip()
            ]

            db_logger.config(f"[SQL] Found {len(statements)} statements to execute")

            for i, statement in enumerate(statements):
                if statement:
                    # Clean up the statement - remove comments properly
                    lines = statement.split("\n")
                    sql_lines = []

                    for raw_line in lines:
                        clean_line = raw_line.strip()
                        # Skip empty lines and comment-only lines
                        if clean_line and not clean_line.startswith("--"):
                            # Remove inline comments (everything after --)
                            if "--" in clean_line:
                                clean_line = clean_line.split("--")[0].strip()
                            if (
                                clean_line
                            ):  # Only add if there's still content after removing comments
                                sql_lines.append(clean_line)

                    if sql_lines:
                        cleaned_statement = " ".join(sql_lines)
                        db_logger.config(
                            f"[SQL] Executing statement {i + 1}: {cleaned_statement[:50]}..."
                        )
                        try:
                            self.connection.execute(cleaned_statement)
                            db_logger.config(
                                f"[SQL] Statement {i + 1} executed successfully"
                            )
                        except RuntimeError as stmt_error:
                            # Handle interrupted installs gracefully
                            if (
                                "Query interrupted" in str(stmt_error)
                                and "INSTALL" in cleaned_statement.upper()
                            ):
                                db_logger.warning(
                                    f"[SQL] Extension install interrupted: {stmt_error}"
                                )
                                db_logger.warning(
                                    "[SQL] This is likely due to process termination during extension download"
                                )
                                # Re-raise as we can't complete schema initialization
                                raise
                            else:
                                db_logger.error(
                                    f"[SQL] Statement {i + 1} failed: {stmt_error}"
                                )
                                db_logger.error(
                                    f"[SQL] Failing statement: {cleaned_statement}"
                                )
                                raise
                        except Exception as stmt_error:
                            db_logger.error(
                                f"[SQL] Statement {i + 1} failed: {stmt_error}"
                            )
                            db_logger.error(
                                f"[SQL] Failing statement: {cleaned_statement}"
                            )
                            raise

        except Exception as e:
            db_logger.error(f"Failed to execute SQL file {file_path}: {e}")
            raise

    def _verify_schema(self) -> None:
        """Verify that required tables and views exist."""
        required_tables = ["runtime_events", "lineage_events"]
        required_views = ["vw_lineage_io", "vw_performance_metrics"]

        # Check tables
        for table in required_tables:
            try:
                # Check if table exists by running a simple query
                self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                db_logger.success(f"[OK] Table '{table}' exists and accessible")
            except Exception as e:
                db_logger.error(f"[ERROR] Table '{table}' verification failed: {e}")
                raise

        # Check views (optional - they might not all exist yet)
        for view in required_views:
            try:
                # Check if view exists by running a simple query
                self.connection.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
                db_logger.success(f"[OK] View '{view}' exists and accessible")
            except Exception:
                db_logger.warning(
                    f"[WARN] View '{view}' not found (may be created later)"
                )

    def execute_query(
        self, query: str, parameters: list | None = None, timeout: float = 30.0
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dictionaries with timeout support."""
        self._ensure_connected()

        def _execute():
            try:
                if parameters:
                    result = self.connection.execute(query, parameters)
                else:
                    result = self.connection.execute(query)

                # Convert to list of dictionaries
                columns = (
                    [desc[0] for desc in result.description]
                    if result.description
                    else []
                )
                rows = result.fetchall()

                return [dict(zip(columns, row, strict=False)) for row in rows]

            except Exception as e:
                db_logger.error(f"Query execution failed: {e}")
                db_logger.error(f"Query: {query}")
                raise

        # Execute with timeout handling
        try:
            return self._execute_with_timeout(_execute, timeout)
        except TimeoutError:
            db_logger.error(f"Query timed out after {timeout}s: {query[:100]}...")
            # Connection may be unstable after timeout
            self.connection = None
            raise

    def execute_query_with_retry(
        self,
        query: str,
        parameters: list | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> list[dict[str, Any]]:
        """Execute query with both retry logic and timeout handling."""
        last_exception = None

        for attempt in range(max_retries):
            try:
                return self.execute_query(query, parameters, timeout)
            except (TimeoutError, duckdb.TransactionException) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 0.1 * (2**attempt)
                    db_logger.warning(
                        f"[RETRY] Query failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}"
                    )
                    time.sleep(wait_time)
                    # Ensure fresh connection after timeout
                    if isinstance(e, TimeoutError):
                        self._connect_with_retry()
                    continue
                else:
                    db_logger.error(f"Query failed after {max_retries} attempts: {e}")
                    raise

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        else:
            error_msg = f"Query failed after {max_retries} attempts with unknown error"
            raise RuntimeError(error_msg)

    def insert_runtime_event(
        self, event_data: dict[str, Any], timeout: float = 10.0
    ) -> None:
        """Insert a runtime event into the database with timeout."""
        query = """
        INSERT INTO runtime_events (
            event_id, event_type, execution_id, cell_id, cell_source, cell_source_lines,
            start_memory_mb, end_memory_mb, duration_ms, timestamp,
            session_id, emitted_at, error_info
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        parameters = [
            event_data.get("event_id"),
            event_data.get("event_type"),
            event_data.get("execution_id"),
            event_data.get("cell_id"),
            event_data.get("cell_source"),
            event_data.get("cell_source_lines"),
            event_data.get("start_memory_mb"),
            event_data.get("end_memory_mb"),
            event_data.get("duration_ms"),
            event_data.get("timestamp"),
            event_data.get("session_id"),
            event_data.get("emitted_at"),
            (
                json.dumps(event_data.get("error_info"))
                if event_data.get("error_info")
                else None
            ),
        ]

        self._ensure_connected()

        def _insert():
            return self.connection.execute(query, parameters)

        self._execute_with_timeout(_insert, timeout)

    def insert_lineage_event(
        self, event_data: dict[str, Any], timeout: float = 10.0
    ) -> None:
        """Insert a lineage event into the database with timeout."""
        query = """
        INSERT INTO lineage_events (
            ol_event_id, run_id, execution_id, event_type, job_name, event_time,
            duration_ms, session_id, emitted_at, inputs_json, outputs_json,
            column_lineage_json, column_metrics_json, other_facets_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        parameters = [
            event_data.get("ol_event_id"),
            event_data.get("run_id"),
            event_data.get("execution_id"),
            event_data.get("event_type"),
            event_data.get("job_name"),
            event_data.get("event_time"),
            event_data.get("duration_ms"),
            event_data.get("session_id"),
            event_data.get("emitted_at"),
            (
                json.dumps(event_data.get("inputs_json"))
                if event_data.get("inputs_json")
                else None
            ),
            (
                json.dumps(event_data.get("outputs_json"))
                if event_data.get("outputs_json")
                else None
            ),
            (
                json.dumps(event_data.get("column_lineage_json"))
                if event_data.get("column_lineage_json")
                else None
            ),
            (
                json.dumps(event_data.get("column_metrics_json"))
                if event_data.get("column_metrics_json")
                else None
            ),
            (
                json.dumps(event_data.get("other_facets_json"))
                if event_data.get("other_facets_json")
                else None
            ),
        ]

        self._ensure_connected()

        def _insert():
            return self.connection.execute(query, parameters)

        self._execute_with_timeout(_insert, timeout)

    def get_table_info(self, table_name: str) -> list[dict[str, Any]]:
        """Get information about a table's structure."""
        query = f"DESCRIBE {table_name}"
        return self.execute_query(query)

    def get_table_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.execute_query(query)
        return result[0]["count"] if result else 0

    def begin_transaction(self) -> None:
        """Begin a database transaction."""
        self._ensure_connected()
        self.connection.begin()

    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        self._ensure_connected()
        self.connection.commit()

    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        self._ensure_connected()
        self.connection.rollback()

    def _ensure_connected(self) -> None:
        """Ensure database connection is active."""
        if not self.connection:
            self._connect()

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            db_logger.config("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        self._ensure_connected()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions for testing and CLI usage
def create_test_database(database_path: Path) -> DuckDBManager:
    """Create a test database with schema for development/testing."""
    manager = DuckDBManager(database_path)
    manager.initialize_database()
    return manager


def get_database_stats(database_path: Path) -> dict[str, Any]:
    """Get basic statistics about the database."""
    with DuckDBManager(database_path) as db:
        db.initialize_database()

        stats = {
            "runtime_events_count": db.get_table_count("runtime_events"),
            "lineage_events_count": db.get_table_count("lineage_events"),
            "database_size_bytes": (
                database_path.stat().st_size if database_path.exists() else 0
            ),
            "database_path": str(database_path),
        }

        return stats
