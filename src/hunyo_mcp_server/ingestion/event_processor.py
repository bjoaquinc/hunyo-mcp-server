#!/usr/bin/env python3
"""
Event Processor - Validates and processes JSONL events for database insertion.

Handles parsing, validation, transformation, and batch insertion of events
from both runtime and lineage JSONL files into DuckDB.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

# Import logging utility
from capture.logger import get_logger
from hunyo_mcp_server.config import get_repository_root
from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager

processor_logger = get_logger("hunyo.processor")


class EventProcessor:
    """
    Processes and validates events from JSONL files for database insertion.

    Features:
    - Full JSON schema validation using schemas/json/*.json
    - Validates event format and required fields
    - Transforms events to database schema format
    - Batch processing for efficiency
    - Error handling and logging
    """

    def __init__(
        self,
        db_manager: DuckDBManager,
        strict_validation: bool = True,  # noqa: FBT001,FBT002
    ):
        self.db_manager = db_manager
        self.processed_events = 0
        self.failed_events = 0
        self.strict_validation = strict_validation

        # Load JSON schemas at initialization for performance
        self.runtime_schema = self._load_schema("runtime_events_schema.json")
        self.openlineage_schema = self._load_schema("openlineage_events_schema.json")

        processor_logger.status(
            f"Event Processor initialized (strict_validation={strict_validation})"
        )

    def _load_schema(self, schema_filename: str) -> dict[str, Any] | None:
        """Load JSON schema from schemas/json directory."""
        try:
            project_root = get_repository_root()
            schema_path = project_root / "schemas" / "json" / schema_filename

            if not schema_path.exists():
                processor_logger.warning(f"Schema file not found: {schema_path}")
                return None

            with open(schema_path, encoding="utf-8") as f:
                schema = json.load(f)

            processor_logger.success(f"[OK] Loaded schema: {schema_filename}")
            return schema

        except Exception as e:
            processor_logger.error(f"Failed to load schema {schema_filename}: {e}")
            return None

    def _validate_event_against_schema(
        self, event: dict[str, Any], schema: dict[str, Any] | None, event_type: str
    ) -> tuple[bool, str | None]:
        """Validate a single event against its JSON schema."""
        if schema is None:
            if self.strict_validation:
                return False, f"Schema not available for {event_type} events"
            else:
                processor_logger.warning(
                    f"Schema validation skipped for {event_type} (schema not loaded)"
                )
                return True, None

        try:
            jsonschema.validate(event, schema)
            return True, None

        except jsonschema.ValidationError as e:
            error_msg = f"Schema validation failed: {e.message}"
            if hasattr(e, "path") and e.path:
                error_msg += f" (at path: {'.'.join(str(p) for p in e.path)})"
            return False, error_msg

        except Exception as e:
            return False, f"Schema validation error: {e}"

    def process_runtime_events(self, events: list[dict[str, Any]]) -> int:
        """
        Process runtime events with individual transaction boundaries for Windows compatibility.

        Args:
            events: List of runtime event dictionaries

        Returns:
            Number of successfully processed events
        """
        if not events:
            return 0

        successful = 0

        for event in events:
            try:
                # Schema validation first
                is_valid, validation_error = self._validate_event_against_schema(
                    event, self.runtime_schema, "runtime"
                )

                if not is_valid:
                    processor_logger.warning(
                        f"Runtime event schema validation failed: {validation_error}"
                    )
                    self.failed_events += 1
                    continue

                # Transform event
                transformed_event = self._transform_runtime_event(event)

                # Insert with individual transaction and constraint handling
                if self._insert_single_event_with_recovery(
                    transformed_event, "runtime"
                ):
                    successful += 1
                else:
                    self.failed_events += 1

            except Exception as e:
                processor_logger.warning(f"Failed to process runtime event: {e}")
                self.failed_events += 1

        self.processed_events += successful

        if successful > 0:
            processor_logger.success(f"[OK] Processed {successful} runtime events")

        return successful

    def process_lineage_events(self, events: list[dict[str, Any]]) -> int:
        """
        Process lineage events with individual transaction boundaries for Windows compatibility.

        Args:
            events: List of lineage event dictionaries

        Returns:
            Number of successfully processed events
        """
        if not events:
            return 0

        successful = 0

        for event in events:
            try:
                # Schema validation first
                is_valid, validation_error = self._validate_event_against_schema(
                    event, self.openlineage_schema, "openlineage"
                )

                if not is_valid:
                    processor_logger.warning(
                        f"OpenLineage event schema validation failed: {validation_error}"
                    )
                    self.failed_events += 1
                    continue

                # Transform event
                transformed_event = self._transform_lineage_event(event)

                # Insert with individual transaction and constraint handling
                if self._insert_single_event_with_recovery(
                    transformed_event, "lineage"
                ):
                    successful += 1
                else:
                    self.failed_events += 1

            except Exception as e:
                processor_logger.warning(f"Failed to process lineage event: {e}")
                self.failed_events += 1

        self.processed_events += successful

        if successful > 0:
            processor_logger.success(f"[OK] Processed {successful} lineage events")

        return successful

    def process_jsonl_file(self, file_path: Path, event_type: str) -> int:
        """
        Process all events from a JSONL file with schema validation.

        Args:
            file_path: Path to JSONL file
            event_type: 'runtime' or 'lineage'

        Returns:
            Number of successfully processed events
        """
        if not file_path.exists():
            processor_logger.warning(f"JSONL file not found: {file_path}")
            return 0

        processor_logger.info(
            f"📄 Processing {event_type} events from: {file_path.name}"
        )

        events = []
        line_number = 0

        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line_number += 1
                    stripped_line = line.strip()

                    if not stripped_line:
                        continue

                    try:
                        event = json.loads(stripped_line)
                        events.append(event)
                    except json.JSONDecodeError as e:
                        processor_logger.warning(
                            f"Invalid JSON at line {line_number}: {e}"
                        )
                        self.failed_events += 1

        except Exception as e:
            processor_logger.error(f"Failed to read JSONL file {file_path}: {e}")
            return 0

        # Process events based on type with schema validation
        if event_type == "runtime":
            return self.process_runtime_events(events)
        elif event_type == "lineage":
            return self.process_lineage_events(events)
        else:
            processor_logger.error(f"Unknown event type: {event_type}")
            return 0

    def _transform_runtime_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Transform runtime event to database schema format."""
        import platform
        import time
        import uuid

        # With schema validation, we can trust the event structure more
        # But still do basic validation for critical database fields
        required_fields = ["event_type", "session_id", "emitted_at"]
        for field in required_fields:
            if field not in event:
                msg = f"Missing required field: {field}"
                raise ValueError(msg)

        # Generate platform-specific primary key to avoid Windows time resolution issues
        if platform.system() == "Windows":
            # Windows has ~15.6ms system clock ticks - use UUID to avoid duplicates
            event_id = abs(hash(str(uuid.uuid4()))) % (
                10**15
            )  # Ensure positive 15-digit max
        else:
            # Unix systems have microsecond precision - use high-resolution timestamp
            event_id = (
                int(time.perf_counter() * 1000000) + hash(str(uuid.uuid4())) % 1000
            )

        # Transform event to match database schema
        transformed = {
            "event_id": event_id,
            "event_type": event.get("event_type"),
            "execution_id": event.get("execution_id"),
            "cell_id": event.get("cell_id"),
            "cell_source": event.get("cell_source"),
            "cell_source_lines": event.get("cell_source_lines"),
            "start_memory_mb": event.get("start_memory_mb"),
            "end_memory_mb": event.get("end_memory_mb"),
            "duration_ms": event.get("duration_ms"),
            "timestamp": self._parse_timestamp(event.get("timestamp")),
            "session_id": event.get("session_id"),
            "emitted_at": self._parse_timestamp(event.get("emitted_at")),
            "error_info": event.get("error_info"),
        }

        return transformed

    def _transform_lineage_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Transform lineage event to database schema format."""
        import platform
        import time
        import uuid

        # Generate platform-specific primary key to avoid Windows time resolution issues
        if platform.system() == "Windows":
            # Windows has ~15.6ms system clock ticks - use UUID to avoid duplicates
            ol_event_id = abs(hash(str(uuid.uuid4()))) % (
                10**15
            )  # Ensure positive 15-digit max
        else:
            # Unix systems have microsecond precision - use high-resolution timestamp
            ol_event_id = (
                int(time.perf_counter() * 1000000) + hash(str(uuid.uuid4())) % 1000
            )

        # Handle OpenLineage event structure
        if "run" in event and "job" in event:
            # OpenLineage format
            run_info = event.get("run", {})
            job_info = event.get("job", {})

            # Extract execution context from facets
            facets = run_info.get("facets", {})
            execution_info = facets.get("marimoExecution", {})
            performance_info = facets.get("performance", {})

            transformed = {
                "ol_event_id": ol_event_id,
                "run_id": run_info.get("runId"),
                "execution_id": execution_info.get("executionId"),
                "event_type": event.get("eventType"),
                "job_name": job_info.get("name"),
                "event_time": self._parse_timestamp(event.get("eventTime")),
                "duration_ms": performance_info.get("durationMs"),
                "session_id": execution_info.get("sessionId"),
                "emitted_at": self._parse_timestamp(event.get("emitted_at")),
                "inputs_json": event.get("inputs", []),
                "outputs_json": event.get("outputs", []),
                "column_lineage_json": self._extract_column_lineage(event),
                "column_metrics_json": self._extract_column_metrics(event),
                "other_facets_json": self._extract_other_facets(event),
            }
        else:
            # Simple lineage format (fallback)
            transformed = {
                "ol_event_id": ol_event_id,
                "run_id": event.get("run_id"),
                "execution_id": event.get("execution_id"),
                "event_type": event.get("event_type"),
                "job_name": event.get("job_name"),
                "event_time": self._parse_timestamp(event.get("event_time")),
                "duration_ms": event.get("duration_ms"),
                "session_id": event.get("session_id"),
                "emitted_at": self._parse_timestamp(event.get("emitted_at")),
                "inputs_json": event.get("inputs", []),
                "outputs_json": event.get("outputs", []),
                "column_lineage_json": event.get("column_lineage"),
                "column_metrics_json": event.get("column_metrics"),
                "other_facets_json": event.get("other_facets"),
            }

        return transformed

    def _parse_timestamp(self, timestamp: str | datetime | None) -> datetime | None:
        """Parse timestamp from various formats."""
        if timestamp is None:
            return None

        if isinstance(timestamp, datetime):
            return timestamp

        if isinstance(timestamp, str):
            try:
                # Try ISO format first
                if "T" in timestamp:
                    # Handle timezone info
                    if timestamp.endswith("Z"):
                        timestamp = timestamp[:-1] + "+00:00"
                    return datetime.fromisoformat(timestamp)
                else:
                    # Try parsing as float (Unix timestamp)
                    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
            except (ValueError, TypeError) as e:
                processor_logger.warning(
                    f"Failed to parse timestamp '{timestamp}': {e}"
                )
                return None

        return None

    def _extract_column_lineage(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Extract column lineage information from event."""
        # Look for column lineage in outputs
        outputs = event.get("outputs", [])
        column_lineage = {}

        for output in outputs:
            facets = output.get("facets", {})
            if "columnLineage" in facets:
                column_lineage[output.get("name", "unknown")] = facets["columnLineage"]

        return column_lineage if column_lineage else None

    def _extract_column_metrics(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Extract column metrics from event."""
        # Look for data quality metrics in outputs
        outputs = event.get("outputs", [])
        column_metrics = {}

        for output in outputs:
            facets = output.get("facets", {})
            if "dataQualityMetrics" in facets:
                column_metrics[output.get("name", "unknown")] = facets[
                    "dataQualityMetrics"
                ]

        return column_metrics if column_metrics else None

    def _extract_other_facets(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Extract other facets from the event."""
        other_facets = {}

        # Extract run facets (excluding known ones)
        run_facets = event.get("run", {}).get("facets", {})
        known_run_facets = {"marimoExecution", "performance"}

        for key, value in run_facets.items():
            if key not in known_run_facets:
                other_facets[f"run.{key}"] = value

        # Extract job facets
        job_facets = event.get("job", {}).get("facets", {})
        for key, value in job_facets.items():
            other_facets[f"job.{key}"] = value

        return other_facets if other_facets else None

    def _execute_with_retry(self, operation_func, max_retries: int = 3) -> Any:
        """Execute database operation with retry logic for transaction conflicts."""
        for attempt in range(max_retries):
            try:
                return operation_func()
            except Exception as e:
                # Check for specific DuckDB exceptions following best practices guide
                if "used by another process" in str(e) and attempt < max_retries - 1:
                    # File locking issue - retry with exponential backoff
                    wait_time = 0.1 * (2**attempt)
                    processor_logger.warning(
                        f"Database locked, retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                elif "Current transaction is aborted" in str(e):
                    # Transaction already aborted - cannot continue
                    processor_logger.error(
                        "Transaction aborted - operation cannot continue"
                    )
                    raise
                else:
                    # Other errors or final attempt - re-raise
                    raise

    def _insert_single_event_with_recovery(
        self, event_data: dict[str, Any], event_type: str
    ) -> bool:
        """Insert single event with individual transaction and constraint handling."""
        import duckdb

        def insert_operation():
            """Nested function for retry mechanism."""
            # Each event gets its own transaction to prevent cascade failures
            self.db_manager.begin_transaction()

            try:
                if event_type == "runtime":
                    self.db_manager.insert_runtime_event(event_data)
                elif event_type == "lineage":
                    self.db_manager.insert_lineage_event(event_data)
                else:
                    error_msg = f"Unknown event type: {event_type}"
                    raise ValueError(error_msg)

                self.db_manager.commit_transaction()
                return True

            except Exception as e:
                self.db_manager.rollback_transaction()

                # Handle specific DuckDB exceptions per best practices guide
                if isinstance(e, duckdb.ConstraintException):
                    # Primary key violation - skip this event but continue processing
                    processor_logger.warning(
                        f"Skipping duplicate event (constraint violation): {e}"
                    )
                    return False
                elif isinstance(e, duckdb.BinderException):
                    # Column or function not found
                    processor_logger.error(f"Database binding error: {e}")
                    return False
                elif isinstance(e, duckdb.CatalogException):
                    # Table/schema not found
                    processor_logger.error(f"Database catalog error: {e}")
                    return False
                elif isinstance(e, duckdb.InternalException):
                    # Internal errors trigger restricted mode - connection must be restarted
                    processor_logger.error(
                        f"DuckDB internal error (restricted mode): {e}"
                    )
                    self.db_manager.close()
                    raise
                else:
                    # Unknown error - return False (already rolled back)
                    return False

        try:
            return self._execute_with_retry(insert_operation)
        except Exception as e:
            processor_logger.error(
                f"Failed to insert {event_type} event after retries: {e}"
            )
            return False

    def get_validation_summary(self) -> dict[str, Any]:
        """Get validation and processing summary."""
        return {
            "processed_events": self.processed_events,
            "failed_events": self.failed_events,
            "total_events": self.processed_events + self.failed_events,
            "success_rate": (
                self.processed_events / (self.processed_events + self.failed_events)
                if (self.processed_events + self.failed_events) > 0
                else 0.0
            ),
            "strict_validation": self.strict_validation,
            "schemas_loaded": {
                "runtime_schema": self.runtime_schema is not None,
                "openlineage_schema": self.openlineage_schema is not None,
            },
        }
