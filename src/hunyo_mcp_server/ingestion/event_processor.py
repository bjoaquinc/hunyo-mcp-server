#!/usr/bin/env python3
"""
Event Processor - Validates and processes JSONL events for database insertion.

Handles parsing, validation, transformation, and batch insertion of events
from both runtime and lineage JSONL files into DuckDB.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hunyo_mcp_server.ingestion.duckdb_manager import DuckDBManager

# Import logging utility
from ...capture.logger import get_logger

processor_logger = get_logger("hunyo.processor")


class EventProcessor:
    """
    Processes and validates events from JSONL files for database insertion.

    Features:
    - Validates event format and required fields
    - Transforms events to database schema format
    - Batch processing for efficiency
    - Error handling and logging
    """

    def __init__(self, db_manager: DuckDBManager):
        self.db_manager = db_manager
        self.processed_events = 0
        self.failed_events = 0

        processor_logger.status("Event Processor initialized")

    def process_runtime_events(self, events: list[dict[str, Any]]) -> int:
        """
        Process a batch of runtime events.

        Args:
            events: List of runtime event dictionaries

        Returns:
            Number of successfully processed events
        """
        if not events:
            return 0

        successful = 0

        try:
            self.db_manager.begin_transaction()

            for event in events:
                try:
                    # Validate and transform event
                    transformed_event = self._transform_runtime_event(event)

                    # Insert into database
                    self.db_manager.insert_runtime_event(transformed_event)
                    successful += 1

                except Exception as e:
                    processor_logger.warning(f"Failed to process runtime event: {e}")
                    self.failed_events += 1

            self.db_manager.commit_transaction()
            self.processed_events += successful

            if successful > 0:
                processor_logger.success(f"✅ Processed {successful} runtime events")

        except Exception as e:
            self.db_manager.rollback_transaction()
            processor_logger.error(f"Failed to process runtime events batch: {e}")
            raise

        return successful

    def process_lineage_events(self, events: list[dict[str, Any]]) -> int:
        """
        Process a batch of lineage events.

        Args:
            events: List of lineage event dictionaries

        Returns:
            Number of successfully processed events
        """
        if not events:
            return 0

        successful = 0

        try:
            self.db_manager.begin_transaction()

            for event in events:
                try:
                    # Validate and transform event
                    transformed_event = self._transform_lineage_event(event)

                    # Insert into database
                    self.db_manager.insert_lineage_event(transformed_event)
                    successful += 1

                except Exception as e:
                    processor_logger.warning(f"Failed to process lineage event: {e}")
                    self.failed_events += 1

            self.db_manager.commit_transaction()
            self.processed_events += successful

            if successful > 0:
                processor_logger.success(f"✅ Processed {successful} lineage events")

        except Exception as e:
            self.db_manager.rollback_transaction()
            processor_logger.error(f"Failed to process lineage events batch: {e}")
            raise

        return successful

    def process_jsonl_file(self, file_path: Path, event_type: str) -> int:
        """
        Process all events from a JSONL file.

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
            with open(file_path) as f:
                for line in f:
                    line_number += 1
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError as e:
                        processor_logger.warning(
                            f"Invalid JSON at line {line_number}: {e}"
                        )
                        self.failed_events += 1

        except Exception as e:
            processor_logger.error(f"Failed to read JSONL file {file_path}: {e}")
            return 0

        # Process events based on type
        if event_type == "runtime":
            return self.process_runtime_events(events)
        elif event_type == "lineage":
            return self.process_lineage_events(events)
        else:
            processor_logger.error(f"Unknown event type: {event_type}")
            return 0

    def _transform_runtime_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Transform runtime event to database schema format."""
        # Validate required fields
        required_fields = ["event_type", "session_id", "emitted_at"]
        for field in required_fields:
            if field not in event:
                msg = f"Missing required field: {field}"
                raise ValueError(msg)

        # Transform event to match database schema
        transformed = {
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

    def _parse_timestamp(
        self, timestamp: str | datetime | None
    ) -> datetime | None:
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

    def _extract_column_lineage(
        self, event: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract column lineage information from event."""
        # Look for column lineage in outputs
        outputs = event.get("outputs", [])
        column_lineage = {}

        for output in outputs:
            facets = output.get("facets", {})
            if "columnLineage" in facets:
                column_lineage[output.get("name", "unknown")] = facets["columnLineage"]

        return column_lineage if column_lineage else None

    def _extract_column_metrics(
        self, event: dict[str, Any]
    ) -> dict[str, Any] | None:
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

    def get_processing_stats(self) -> dict[str, int]:
        """Get processing statistics."""
        return {
            "processed_events": self.processed_events,
            "failed_events": self.failed_events,
            "total_events": self.processed_events + self.failed_events,
        }
