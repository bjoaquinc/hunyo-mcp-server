from __future__ import annotations

import json
import time

import pytest

from capture.lightweight_runtime_tracker import LightweightRuntimeTracker


class TestLightweightRuntimeTracker:
    """Tests for LightweightRuntimeTracker - focused on cell execution events only"""

    @pytest.fixture(autouse=True)
    def setup_clean_tracker_state(self):
        """Ensure clean global tracker state before and after each test"""
        from capture.lightweight_runtime_tracker import disable_runtime_tracking

        # Clean up before test
        disable_runtime_tracking()

        yield

        # Clean up after test
        disable_runtime_tracking()

    def test_tracker_initialization(self, tmp_path):
        """Test tracker initializes with proper configuration"""
        output_file = tmp_path / "runtime_events.jsonl"
        tracker = LightweightRuntimeTracker(output_file=output_file)

        assert not tracker.is_active
        assert tracker.output_file == output_file
        assert tracker.session_id is not None

    def test_cell_execution_start_tracking(self, tmp_path):
        """Test tracking of cell execution start events"""
        output_file = tmp_path / "runtime_events.jsonl"
        tracker = LightweightRuntimeTracker(output_file=output_file)
        tracker.start_tracking()

        cell_id = "test_cell_1"
        cell_source = "x = 1 + 2"

        execution_id = tracker.track_cell_execution_start(cell_id, cell_source)

        assert execution_id is not None
        assert len(execution_id) > 0

        # Check event was added
        assert len(tracker.events) > 0

        # Find the cell execution start event
        start_event = None
        for event in tracker.events:
            if event.get("event_type") == "cell_execution_start":
                start_event = event
                break

        assert start_event is not None
        assert start_event["cell_id"] == cell_id
        assert start_event["cell_source"] == cell_source
        assert start_event["execution_id"] == execution_id

    def test_cell_execution_end_tracking(self, tmp_path):
        """Test tracking of cell execution end events"""
        output_file = tmp_path / "runtime_events.jsonl"
        tracker = LightweightRuntimeTracker(output_file=output_file)
        tracker.start_tracking()

        cell_id = "test_cell_1"
        cell_source = "x = 1 + 2"
        execution_id = tracker.track_cell_execution_start(cell_id, cell_source)

        start_time = time.time()
        tracker.track_cell_execution_end(execution_id, cell_id, cell_source, start_time)

        # Find the cell execution end event
        end_event = None
        for event in tracker.events:
            if event.get("event_type") == "cell_execution_end":
                end_event = event
                break

        assert end_event is not None
        assert end_event["cell_id"] == cell_id
        assert end_event["execution_id"] == execution_id
        assert "duration_ms" in end_event
        assert end_event["duration_ms"] >= 0

    def test_cell_execution_error_tracking(self, tmp_path):
        """Test tracking of cell execution error events"""
        output_file = tmp_path / "runtime_events.jsonl"
        tracker = LightweightRuntimeTracker(output_file=output_file)
        tracker.start_tracking()

        cell_id = "test_cell_1"
        cell_source = "x = 1 / 0"  # Division by zero
        execution_id = tracker.track_cell_execution_start(cell_id, cell_source)

        start_time = time.time()
        error = ZeroDivisionError("division by zero")
        traceback_str = 'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\nZeroDivisionError: division by zero'

        tracker.track_cell_execution_error(
            execution_id, cell_id, cell_source, start_time, error, traceback_str
        )

        # Find the cell execution error event
        error_event = None
        for event in tracker.events:
            if event.get("event_type") == "cell_execution_error":
                error_event = event
                break

        assert error_event is not None
        assert error_event["cell_id"] == cell_id
        assert error_event["execution_id"] == execution_id
        assert "error_info" in error_event
        assert error_event["error_info"]["error_type"] == "ZeroDivisionError"
        assert error_event["error_info"]["error_message"] == "division by zero"

    def test_event_file_writing(self, tmp_path):
        """Test events are properly written to JSONL file"""
        output_file = tmp_path / "runtime_events.jsonl"
        tracker = LightweightRuntimeTracker(output_file=output_file)
        tracker.start_tracking()

        cell_id = "test_cell_1"
        cell_source = "x = 1 + 2"
        tracker.track_cell_execution_start(cell_id, cell_source)

        # Flush events to file
        tracker.flush_events()

        # Verify events were written
        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        assert (
            len(lines) >= 1
        )  # cell_execution_start (session events removed for schema compliance)

        # Parse and verify at least one event
        event = json.loads(lines[-1].strip())
        assert "event_type" in event
        assert "timestamp" in event
        assert "session_id" in event

    def test_schema_compliance(self, tmp_path):
        """Test that generated events comply with the runtime events schema"""
        output_file = tmp_path / "runtime_events.jsonl"
        tracker = LightweightRuntimeTracker(output_file=output_file)
        tracker.start_tracking()

        cell_id = "test_cell_1"
        cell_source = "x = 1 + 2"
        execution_id = tracker.track_cell_execution_start(cell_id, cell_source)

        start_time = time.time()
        tracker.track_cell_execution_end(execution_id, cell_id, cell_source, start_time)

        # Check that all events have required schema fields
        for event in tracker.events:
            event_type = event.get("event_type")

            # All events should have these basic fields
            assert "timestamp" in event
            assert "session_id" in event
            assert "event_type" in event

            # Cell execution events should have specific fields
            if event_type in [
                "cell_execution_start",
                "cell_execution_end",
                "cell_execution_error",
            ]:
                assert "cell_id" in event
                assert "cell_source" in event
                if "execution_id" in event:  # Not all events might have this
                    assert isinstance(event["execution_id"], str)

    def test_session_summary(self, tmp_path):
        """Test session summary functionality"""
        output_file = tmp_path / "runtime_events.jsonl"
        tracker = LightweightRuntimeTracker(output_file=output_file)
        tracker.start_tracking()

        # Add some events
        tracker.track_cell_execution_start("cell_1", "x = 1")
        tracker.track_cell_execution_start("cell_2", "y = 2")

        summary = tracker.get_session_summary()

        assert "session_id" in summary
        assert "events_captured" in summary
        assert "is_active" in summary
        assert summary["events_captured"] >= 2  # At least the events we added
        assert summary["is_active"] is True

    def test_error_handling_on_tracking_disabled(self, tmp_path):
        """Test graceful handling when tracking is not enabled"""
        output_file = tmp_path / "runtime_events.jsonl"
        tracker = LightweightRuntimeTracker(
            output_file=output_file, enable_tracking=False
        )

        # Should not crash when tracking is disabled
        execution_id = tracker.track_cell_execution_start("test_cell", "x = 1")
        assert execution_id is not None  # Should still return an ID for compatibility

    def test_context_manager_functionality(self, tmp_path):
        """Test the track_cell_execution context manager"""
        from capture.lightweight_runtime_tracker import (
            disable_runtime_tracking,
            enable_runtime_tracking,
            track_cell_execution,
        )

        # Clean up any existing global tracker
        disable_runtime_tracking()

        output_file = tmp_path / "runtime_events.jsonl"
        tracker = enable_runtime_tracking(output_file=output_file)

        # Verify tracker is active
        assert (
            tracker.is_active
        ), "Tracker should be active after enable_runtime_tracking"

        cell_source = "result = 1 + 1"

        with track_cell_execution(cell_source) as ctx:
            assert ctx.execution_id is not None
            ctx.set_result(2)

        # Should have generated start and end events
        start_events = [
            e for e in tracker.events if e.get("event_type") == "cell_execution_start"
        ]
        end_events = [
            e for e in tracker.events if e.get("event_type") == "cell_execution_end"
        ]

        assert (
            len(start_events) >= 1
        ), f"Expected at least 1 start event, got {len(start_events)}"
        assert (
            len(end_events) >= 1
        ), f"Expected at least 1 end event, got {len(end_events)}"

        # Clean up
        disable_runtime_tracking()
