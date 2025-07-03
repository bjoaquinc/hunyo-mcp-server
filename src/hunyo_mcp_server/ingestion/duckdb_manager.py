#!/usr/bin/env python3
"""
DuckDB Manager - Database initialization and connection management.

Handles DuckDB database setup, schema creation, and connection pooling
for the Hunyo MCP Server ingestion pipeline.
"""

import json
from pathlib import Path
from typing import Any

import duckdb

# Import logging utility
from capture.logger import get_logger

# Import project paths
from hunyo_mcp_server.config import get_repository_root

db_logger = get_logger("hunyo.duckdb")


class DuckDBManager:
    """
    Manages DuckDB database initialization, connections, and schema setup.

    Features:
    - Automatic schema initialization from SQL files
    - Connection management and reuse
    - Query execution with error handling
    - Transaction support for batch operations
    """

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.connection: duckdb.DuckDBPyConnection | None = None
        self._schema_initialized = False

        # Ensure database directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        db_logger.status("DuckDB Manager initialized")
        db_logger.config(f"Database path: {self.database_path}")

    def initialize_database(self) -> None:
        """Initialize database and create schema from SQL files."""
        if self._schema_initialized:
            db_logger.warning("Database already initialized")
            return

        db_logger.startup("[INIT] Initializing DuckDB database and schema...")

        try:
            # Connect to database
            self._connect()

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

        except Exception as e:
            db_logger.error(f"Failed to initialize database: {e}")
            raise

    def _connect(self) -> None:
        """Establish database connection."""
        if self.connection:
            return

        try:
            self.connection = duckdb.connect(str(self.database_path))
            db_logger.success(f"Connected to database: {self.database_path}")
        except Exception as e:
            db_logger.error(f"Failed to connect to database: {e}")
            raise

    def _create_schema(self) -> None:
        """Create database schema from SQL files."""
        # Get project root and schema directory
        project_root = get_repository_root()
        schema_dir = project_root / "schemas" / "sql"

        if not schema_dir.exists():
            msg = f"Schema directory not found: {schema_dir}"
            raise FileNotFoundError(msg)

        # Execute main initialization script
        init_script = schema_dir / "init_database.sql"
        if init_script.exists():
            db_logger.config(f"Executing schema: {init_script.name}")
            self._execute_sql_file(init_script)
        else:
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
            with open(file_path) as f:
                sql_content = f.read()

            # Split on semicolons and execute each statement
            statements = [
                stmt.strip() for stmt in sql_content.split(";") if stmt.strip()
            ]

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
                            f"Executing statement {i+1}: {cleaned_statement[:50]}..."
                        )
                        self.connection.execute(cleaned_statement)

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
        self, query: str, parameters: list | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dictionaries."""
        self._ensure_connected()

        try:
            if parameters:
                result = self.connection.execute(query, parameters)
            else:
                result = self.connection.execute(query)

            # Convert to list of dictionaries
            columns = (
                [desc[0] for desc in result.description] if result.description else []
            )
            rows = result.fetchall()

            return [dict(zip(columns, row, strict=False)) for row in rows]

        except Exception as e:
            db_logger.error(f"Query execution failed: {e}")
            db_logger.error(f"Query: {query}")
            raise

    def insert_runtime_event(self, event_data: dict[str, Any]) -> None:
        """Insert a runtime event into the database."""
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
        self.connection.execute(query, parameters)

    def insert_lineage_event(self, event_data: dict[str, Any]) -> None:
        """Insert a lineage event into the database."""
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
        self.connection.execute(query, parameters)

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
