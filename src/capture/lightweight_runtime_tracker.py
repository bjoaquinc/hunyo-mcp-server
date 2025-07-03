#!/usr/bin/env python3
"""Lightweight Runtime Tracker for OpenLineage Events"""

import inspect
import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from capture.logger import get_logger

# Initialize logger with dedicated name
tracker_logger = get_logger("hunyo.tracker.runtime")

# Type definitions
EventProcessor = Callable[[dict[str, Any]], None]

# Constants for configuration
DEFAULT_BUFFER_SIZE = 100
MAX_STACK_DEPTH = 10
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_ARG_SIZE = 500
MAX_VAR_SIZE = 1000
ARG_PREVIEW_SIZE = 200
VAR_PREVIEW_SIZE = 100
MAX_PREVIEW_VARS = 5


class LightweightRuntimeTracker:
    """
    Lightweight runtime tracker for monitoring code execution.

    Generates OpenLineage-compatible events and runtime metadata with minimal
    performance overhead. Tracks variables, function calls, and execution context.
    """

    def __init__(
        self,
        output_file: str | Path | None = None,
        *,
        enable_tracking: bool = True,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        """
        Initialize the lightweight runtime tracker.

        Args:
            output_file: File path for event output (None for in-memory only)
            enable_tracking: Whether to enable event tracking
            buffer_size: Size of the event buffer before flushing
        """
        self.output_file = Path(output_file) if output_file else None
        self.enable_tracking = enable_tracking
        self.buffer_size = buffer_size

        # Event tracking state
        self.events: list[dict[str, Any]] = []
        self.is_active = False
        self.session_id = f"runtime_session_{int(time.time())}"
        self.total_events_processed = 0  # Track total events across all flushes

        # Execution context
        self.execution_context: dict[str, Any] = {}
        self.variable_tracker: dict[str, Any] = {}

        # Performance tracking
        self.start_time: float | None = None
        self.last_flush_time: float = time.time()

        tracker_logger.info(f"Initialized LightweightRuntimeTracker: {self.session_id}")

    def start_tracking(self, *, force_restart: bool = False) -> None:
        """
        Start runtime tracking.

        Args:
            force_restart: Whether to force restart if already active
        """
        if self.is_active and not force_restart:
            tracker_logger.warning("Tracking is already active")
            return

        self.is_active = True
        self.start_time = time.time()

        # NOTE: Removed session event creation to comply with runtime_events_schema.json
        # The schema only allows: cell_execution_start, cell_execution_end, cell_execution_error

        tracker_logger.info(f"Started runtime tracking: {self.session_id}")

    def stop_tracking(self, *, flush_events: bool = True) -> dict[str, Any]:
        """
        Stop runtime tracking and optionally flush events.

        Args:
            flush_events: Whether to flush remaining events to output

        Returns:
            Summary of tracking session
        """
        if not self.is_active:
            tracker_logger.warning("Tracking is not active")
            return {"status": "not_active", "events_captured": 0}

        self.is_active = False
        end_time = time.time()

        # NOTE: Removed session end event creation to comply with runtime_events_schema.json
        # The schema only allows: cell_execution_start, cell_execution_end, cell_execution_error

        if flush_events:
            self.flush_events()

        # Calculate session summary
        duration = end_time - (self.start_time or end_time)
        summary = {
            "status": "completed",
            "session_id": self.session_id,
            "duration_seconds": round(duration, 3),
            "events_captured": self.total_events_processed,
            "output_file": str(self.output_file) if self.output_file else None,
        }

        tracker_logger.info(f"Stopped runtime tracking: {summary}")
        return summary

    def _create_session_event(self) -> dict[str, Any]:
        """
        DEPRECATED: Session events don't comply with runtime_events_schema.json

        The runtime schema only allows: cell_execution_start, cell_execution_end, cell_execution_error
        Session tracking should be handled at a higher level or in a separate event stream.
        """
        msg = (
            "Session events are not supported by runtime_events_schema.json. "
            "Use only cell_execution_start, cell_execution_end, cell_execution_error events."
        )
        raise NotImplementedError(msg)

    def _create_session_end_event(self, end_time: float) -> dict[str, Any]:
        """
        DEPRECATED: Session events don't comply with runtime_events_schema.json

        The runtime schema only allows: cell_execution_start, cell_execution_end, cell_execution_error
        Session tracking should be handled at a higher level or in a separate event stream.
        """
        msg = (
            "Session events are not supported by runtime_events_schema.json. "
            "Use only cell_execution_start, cell_execution_end, cell_execution_error events."
        )
        raise NotImplementedError(msg)

    def _add_event(self, event: dict[str, Any]) -> None:
        """Add an event to the buffer."""
        if not self.enable_tracking:
            return

        self.events.append(event)

        # Auto-flush if buffer is full
        if len(self.events) >= self.buffer_size:
            self.flush_events()

    def flush_events(self) -> None:
        """Flush events to output file."""
        if not self.output_file or not self.events:
            return

        try:
            # Ensure output directory exists
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            # Write events to file
            with self.output_file.open("a", encoding="utf-8") as f:
                for event in self.events:
                    f.write(json.dumps(event, default=str) + "\n")

            # Update counters
            events_flushed = len(self.events)
            self.total_events_processed += events_flushed
            tracker_logger.info(
                f"Flushed {events_flushed} events to {self.output_file}"
            )
            self.events.clear()
            self.last_flush_time = time.time()

        except Exception as e:
            tracker_logger.error(f"Failed to flush events: {e}")

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of current tracking session."""
        duration = 0
        if self.start_time:
            end_time = time.time() if self.is_active else (self.start_time + duration)
            duration = end_time - self.start_time

        return {
            "session_id": self.session_id,
            "is_active": self.is_active,
            "duration_seconds": round(duration, 3),
            "events_captured": self.total_events_processed
            + len(self.events),  # Include both flushed and pending
            "output_file": str(self.output_file) if self.output_file else None,
            "last_flush_time": self.last_flush_time,
        }

    # NOTE: DataFrame operations should be tracked by live_lineage_interceptor.py, not here!
    # This runtime tracker ONLY handles cell execution events per the runtime_events_schema.json

    # NOTE: Execution context methods removed - they were mixing DataFrame concerns.
    # Cell execution context is now properly handled by the schema-compliant methods.

    def generate_schema_compliant_event(
        self,
        event_type: str,
        cell_id: str,
        cell_source: str,
        execution_id: str | None = None,
        start_memory_mb: float | None = None,
        end_memory_mb: float | None = None,
        duration_ms: float | None = None,
        error_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a schema-compliant runtime event.

        Args:
            event_type: One of cell_execution_start, cell_execution_end, cell_execution_error
            cell_id: Unique identifier for the cell
            cell_source: Source code of the cell
            execution_id: 8-character hex execution ID
            start_memory_mb: Memory usage at start (MB)
            end_memory_mb: Memory usage at end (MB)
            duration_ms: Execution duration (milliseconds)
            error_info: Error information if applicable

        Returns:
            Schema-compliant event dictionary
        """
        if not execution_id:
            execution_id = f"{int(time.time() * 1000) % 0xFFFFFFFF:08x}"

        if start_memory_mb is None:
            try:
                import psutil

                process = psutil.Process()
                start_memory_mb = process.memory_info().rss / 1024 / 1024
            except (ImportError, Exception):
                start_memory_mb = 0.0

        now = datetime.now(timezone.utc).isoformat()

        # Generate schema-compliant session ID (8-char hex)
        schema_session_id = f"{abs(hash(self.session_id)) % 0xFFFFFFFF:08x}"

        event = {
            "event_type": event_type,
            "execution_id": execution_id,
            "cell_id": cell_id,
            "cell_source": cell_source,
            "cell_source_lines": len(cell_source.splitlines()),
            "start_memory_mb": start_memory_mb,
            "timestamp": now,
            "session_id": schema_session_id,
            "emitted_at": now,
        }

        # Add optional fields based on event type
        if event_type in {"cell_execution_end", "cell_execution_error"}:
            if end_memory_mb is not None:
                event["end_memory_mb"] = end_memory_mb
            if duration_ms is not None:
                event["duration_ms"] = duration_ms

        if event_type == "cell_execution_error" and error_info:
            event["error_info"] = error_info

        return event

    def track_cell_execution_start(
        self,
        cell_id: str,
        cell_source: str,
        execution_id: str | None = None,
    ) -> str:
        """
        Track the start of cell execution (schema-compliant).

        Args:
            cell_id: Unique identifier for the cell
            cell_source: Source code of the cell
            execution_id: Optional execution ID (generated if not provided)

        Returns:
            The execution ID for this cell execution
        """
        if not execution_id:
            execution_id = f"{int(time.time() * 1000) % 0xFFFFFFFF:08x}"

        event = self.generate_schema_compliant_event(
            event_type="cell_execution_start",
            cell_id=cell_id,
            cell_source=cell_source,
            execution_id=execution_id,
        )

        self._add_event(event)
        return execution_id

    def track_cell_execution_end(
        self,
        execution_id: str,
        cell_id: str,
        cell_source: str,
        start_time: float,
        end_memory_mb: float | None = None,
    ) -> None:
        """
        Track the end of cell execution (schema-compliant).

        Args:
            execution_id: The execution ID from track_cell_execution_start
            cell_id: Unique identifier for the cell
            cell_source: Source code of the cell
            start_time: Start time from time.time()
            end_memory_mb: Memory usage at end (MB)
        """
        duration_ms = (time.time() - start_time) * 1000

        event = self.generate_schema_compliant_event(
            event_type="cell_execution_end",
            cell_id=cell_id,
            cell_source=cell_source,
            execution_id=execution_id,
            duration_ms=duration_ms,
            end_memory_mb=end_memory_mb,
        )

        self._add_event(event)

    def track_cell_execution_error(
        self,
        execution_id: str,
        cell_id: str,
        cell_source: str,
        start_time: float,
        error: Exception,
        traceback_str: str | None = None,
    ) -> None:
        """
        Track a cell execution error (schema-compliant).

        Args:
            execution_id: The execution ID from track_cell_execution_start
            cell_id: Unique identifier for the cell
            cell_source: Source code of the cell
            start_time: Start time from time.time()
            error: The exception that occurred
            traceback_str: Full traceback string
        """
        duration_ms = (time.time() - start_time) * 1000

        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if traceback_str:
            error_info["traceback"] = traceback_str

        event = self.generate_schema_compliant_event(
            event_type="cell_execution_error",
            cell_id=cell_id,
            cell_source=cell_source,
            execution_id=execution_id,
            duration_ms=duration_ms,
            error_info=error_info,
        )

        self._add_event(event)

    def get_current_execution_id(self) -> str | None:
        """
        Get the current execution ID if cell execution is in progress.

        Returns:
            Current execution ID or None if no execution is active
        """
        # For now, return None since we don't track active executions
        # This could be enhanced to track active execution state
        return None


# Global runtime tracker instance
_runtime_tracker: LightweightRuntimeTracker | None = None


def enable_runtime_tracking(
    output_file: str | Path | None = None,
    *,
    notebook_path: str | None = None,
    enable_tracking: bool = True,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> LightweightRuntimeTracker:
    """
    Enable lightweight runtime tracking with naming convention matching lineage files.

    Args:
        output_file: Path to output file (None for auto-generated convention-based path)
        notebook_path: Path to notebook file (for convention-based naming)
        enable_tracking: Whether to enable event tracking
        buffer_size: Size of event buffer before flushing
    """
    global _runtime_tracker  # noqa: PLW0603

    # If no output_file specified, use same convention as lineage files
    if output_file is None:
        try:
            # Try to import the naming convention functions
            from capture import get_event_filenames, get_user_data_dir

            if notebook_path:
                # Use exact same convention as lineage files
                data_dir = get_user_data_dir()
                runtime_events_file, _ = get_event_filenames(notebook_path, data_dir)
                output_file = runtime_events_file
            else:
                # Fallback to generic naming in same directory structure
                data_dir = get_user_data_dir()
                output_file = str(
                    Path(data_dir) / "events" / "runtime" / "runtime_events.jsonl"
                )
        except ImportError:
            # Fallback: use generic file in current directory
            output_file = "runtime_events.jsonl"

    if _runtime_tracker is None:
        _runtime_tracker = LightweightRuntimeTracker(
            output_file=output_file,
            enable_tracking=enable_tracking,
            buffer_size=buffer_size,
        )
        _runtime_tracker.start_tracking()

    return _runtime_tracker


def get_runtime_tracker() -> LightweightRuntimeTracker | None:
    """Get the current runtime tracker instance."""
    return _runtime_tracker


def disable_runtime_tracking() -> dict[str, Any]:
    """
    Disable runtime tracking.

    Returns:
        Summary of the disabled session
    """
    global _runtime_tracker  # noqa: PLW0603

    if _runtime_tracker:
        summary = _runtime_tracker.stop_tracking()
        tracker_logger.info("Runtime tracking disabled")
        _runtime_tracker = None
        return summary

    return {"status": "not_active"}


@contextmanager
def track_cell_execution(cell_source: str, cell_id: str | None = None):
    """
    Context manager for tracking cell execution (for live_lineage_interceptor compatibility).

    Args:
        cell_source: Source code of the cell
        cell_id: Optional cell ID (generated if not provided)

    Yields:
        Execution context object with execution_id and methods
    """
    tracker = get_runtime_tracker()
    if not tracker or not tracker.is_active:
        # Return a dummy context when tracking is disabled
        class DummyContext:
            def __init__(self):
                self.execution_id = f"{int(time.time() * 1000) % 0xFFFFFFFF:08x}"

            def set_result(self, result):
                pass

            def snapshot(self, name):
                pass

        yield DummyContext()
        return

    if not cell_id:
        cell_id = f"cell_{int(time.time() * 1000000) % 0xFFFFFFFF:08x}"

    # Start cell execution tracking
    execution_id = tracker.track_cell_execution_start(cell_id, cell_source)
    start_time = time.time()

    class ExecutionContext:
        def __init__(self):
            self.execution_id = execution_id
            self.result = None

        def set_result(self, result):
            """Set the result of the cell execution"""
            self.result = result

        def snapshot(self, name: str):
            """Take a snapshot during execution (placeholder)"""
            pass

    ctx = ExecutionContext()

    try:
        yield ctx
        # Cell execution completed successfully
        tracker.track_cell_execution_end(execution_id, cell_id, cell_source, start_time)
    except Exception as e:
        # Cell execution failed
        import traceback

        tracker.track_cell_execution_error(
            execution_id, cell_id, cell_source, start_time, e, traceback.format_exc()
        )
        raise


@contextmanager
def track_execution_context(context_name: str, *, capture_vars: bool = False):
    """
    Context manager for tracking execution context.

    Args:
        context_name: Name/description of the execution context
        capture_vars: Whether to capture local variables
    """
    tracker = get_runtime_tracker()
    if not tracker or not tracker.is_active:
        yield
        return

    start_time = time.time()
    context_id = f"ctx_{int(start_time * 1000000)}"

    # Create start event
    start_event = {
        "event_type": "context_start",
        "context_id": context_id,
        "context_name": context_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": tracker.session_id,
    }

    if capture_vars:
        try:
            # Capture caller's local variables safely
            frame = inspect.currentframe().f_back
            if frame:
                local_vars = {
                    k: str(v)[:VAR_PREVIEW_SIZE]
                    for k, v in frame.f_locals.items()
                    if not k.startswith("_") and len(str(v)) < MAX_VAR_SIZE
                }
                start_event["local_vars_count"] = len(local_vars)
                start_event["local_vars_preview"] = dict(
                    list(local_vars.items())[:MAX_PREVIEW_VARS]
                )
        except Exception as e:
            tracker_logger.warning(f"Failed to capture context variables: {e}")

    tracker._add_event(start_event)

    try:
        yield
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
        raise
    finally:
        end_time = time.time()
        duration = end_time - start_time

        # Create end event
        end_event = {
            "event_type": "context_end",
            "context_id": context_id,
            "context_name": context_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": tracker.session_id,
            "duration_seconds": round(duration, 3),
            "success": success,
        }

        if error:
            end_event["error"] = error

        tracker._add_event(end_event)
