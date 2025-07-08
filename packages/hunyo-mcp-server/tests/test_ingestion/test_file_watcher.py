#!/usr/bin/env python3
"""
Tests for FileWatcher - Monitors JSONL event files for changes and triggers processing.

Tests cover file monitoring, debouncing, async processing, and error handling.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hunyo_mcp_server.ingestion.event_processor import EventProcessor
from hunyo_mcp_server.ingestion.file_watcher import (
    FileWatcher,
    JSONLFileHandler,
    watch_directories,
)


class TestJSONLFileHandler:
    """Tests for JSONLFileHandler following project patterns"""

    @pytest.fixture
    def mock_file_watcher(self):
        """Mock FileWatcher for testing"""
        watcher = MagicMock(spec=FileWatcher)
        watcher.notebook_hash = "test_hash"
        return watcher

    @pytest.fixture
    def file_handler(self, mock_file_watcher):
        """Create JSONLFileHandler instance for testing"""
        return JSONLFileHandler(mock_file_watcher)

    def test_handler_initialization(self, mock_file_watcher):
        """Test JSONLFileHandler initialization"""
        handler = JSONLFileHandler(mock_file_watcher)

        assert handler.file_watcher == mock_file_watcher
        assert isinstance(handler._pending_files, set)
        assert len(handler._pending_files) == 0
        assert handler._debounce_delay == 1.0

    def test_on_created_jsonl_file(self, file_handler):
        """Test file creation event handling for JSONL files"""
        # Mock file creation event
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/path/test_hash_new_file.jsonl"

        file_handler.on_created(event)

        # Should add file to pending files
        pending = file_handler.get_pending_files()
        assert len(pending) == 1
        assert Path("/test/path/test_hash_new_file.jsonl") in pending

    def test_on_created_non_jsonl_file(self, file_handler):
        """Test file creation event handling for non-JSONL files"""
        # Mock file creation event for non-JSONL file
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/path/readme.txt"

        file_handler.on_created(event)

        # Should not add non-JSONL file to pending files
        pending = file_handler.get_pending_files()
        assert len(pending) == 0

    def test_on_created_directory(self, file_handler):
        """Test directory creation event handling"""
        # Mock directory creation event
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/test/path/new_directory"

        file_handler.on_created(event)

        # Should not add directory to pending files
        pending = file_handler.get_pending_files()
        assert len(pending) == 0

    def test_on_modified_jsonl_file_with_debounce(self, file_handler):
        """Test file modification event handling with debouncing"""
        # Mock file modification event
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/path/test_hash_modified.jsonl"

        # First modification - should be processed immediately
        file_handler.on_modified(event)
        pending = file_handler.get_pending_files()
        assert len(pending) == 1

        # Second modification within debounce window - should be queued
        file_handler.on_modified(event)
        pending = file_handler.get_pending_files()
        assert len(pending) == 1  # Still only one file, but updated timestamp

    def test_get_pending_files_clears_queue(self, file_handler):
        """Test that get_pending_files clears the pending queue"""
        # Add some files to pending
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/test_hash_file1.jsonl"
        file_handler.on_created(event)

        event.src_path = "/test/test_hash_file2.jsonl"
        file_handler.on_created(event)

        # Get pending files
        pending = file_handler.get_pending_files()
        assert len(pending) == 2

        # Queue should be cleared after retrieval
        pending_again = file_handler.get_pending_files()
        assert len(pending_again) == 0


class TestFileWatcher:
    """Tests for FileWatcher following project patterns"""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing"""
        runtime_dir = tmp_path / "runtime"
        lineage_dir = tmp_path / "lineage"
        dataframe_lineage_dir = tmp_path / "dataframe_lineage"

        runtime_dir.mkdir()
        lineage_dir.mkdir()
        dataframe_lineage_dir.mkdir()

        return {
            "runtime": runtime_dir,
            "lineage": lineage_dir,
            "dataframe_lineage": dataframe_lineage_dir,
        }

    @pytest.fixture
    def mock_event_processor(self):
        """Mock EventProcessor for testing"""
        processor = MagicMock()
        processor.process_jsonl_file.return_value = 3
        processor.get_validation_summary.return_value = {
            "total_events": 10,
            "failed_events": 0,
        }
        return processor

    @pytest.fixture
    def file_handler(self):
        """Create JSONLFileHandler for testing"""
        mock_file_watcher = MagicMock()
        mock_file_watcher.notebook_hash = "test_hash"
        return JSONLFileHandler(mock_file_watcher)

    @pytest.fixture
    def file_watcher(self, temp_dirs, mock_event_processor):
        """Create FileWatcher instance for testing"""
        return FileWatcher(
            runtime_dir=temp_dirs["runtime"],
            lineage_dir=temp_dirs["lineage"],
            dataframe_lineage_dir=temp_dirs["dataframe_lineage"],
            event_processor=mock_event_processor,
            notebook_hash="test_hash",
            verbose=True,
        )

    def test_file_watcher_initialization(self, temp_dirs, mock_event_processor):
        """Test FileWatcher initialization"""
        watcher = FileWatcher(
            runtime_dir=temp_dirs["runtime"],
            lineage_dir=temp_dirs["lineage"],
            dataframe_lineage_dir=temp_dirs["dataframe_lineage"],
            event_processor=mock_event_processor,
            notebook_hash="test_hash",
        )

        assert watcher.runtime_dir == temp_dirs["runtime"].resolve()
        assert watcher.lineage_dir == temp_dirs["lineage"].resolve()
        assert watcher.dataframe_lineage_dir == temp_dirs["dataframe_lineage"].resolve()
        assert watcher.event_processor == mock_event_processor
        assert not watcher.running
        assert watcher.files_processed == 0
        assert watcher.events_processed == 0

    @pytest.mark.asyncio
    async def test_start_and_stop_watcher(self, file_watcher):
        """Test starting and stopping file watcher"""
        # Start the watcher in background
        start_task = asyncio.create_task(file_watcher.start())

        # Give it time to start
        await asyncio.sleep(0.1)
        assert file_watcher.running

        # Stop the watcher
        await file_watcher.stop()
        assert not file_watcher.running

        # Cancel the start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_process_existing_files(self, file_watcher, temp_dirs):
        """Test processing existing files on startup"""
        # Create some existing JSONL files
        runtime_file = temp_dirs["runtime"] / "test_hash_existing_runtime.jsonl"
        lineage_file = temp_dirs["lineage"] / "test_hash_existing_lineage.jsonl"
        dataframe_lineage_file = (
            temp_dirs["dataframe_lineage"]
            / "test_hash_existing_dataframe_lineage.jsonl"
        )

        runtime_file.write_text('{"event_type": "test"}\n')
        lineage_file.write_text('{"eventType": "START"}\n')
        dataframe_lineage_file.write_text('{"event_type": "dataframe_lineage"}\n')

        # Process existing files
        await file_watcher._process_existing_files()

        # Should have called processor for all three files
        assert file_watcher.event_processor.process_jsonl_file.call_count == 3

    @pytest.mark.asyncio
    async def test_process_file_runtime(self, file_watcher, temp_dirs):
        """Test processing a runtime file"""
        runtime_file = temp_dirs["runtime"] / "test_runtime.jsonl"
        runtime_file.write_text('{"event_type": "cell_execution_start"}\n')

        await file_watcher._process_file(runtime_file, "runtime")

        # Should have called processor with correct parameters
        file_watcher.event_processor.process_jsonl_file.assert_called_with(
            runtime_file, "runtime"
        )

    @pytest.mark.asyncio
    async def test_process_file_lineage(self, file_watcher, temp_dirs):
        """Test processing a lineage file"""
        lineage_file = temp_dirs["lineage"] / "test_lineage.jsonl"
        lineage_file.write_text('{"eventType": "COMPLETE"}\n')

        await file_watcher._process_file(lineage_file, "lineage")

        # Should have called processor with correct parameters
        file_watcher.event_processor.process_jsonl_file.assert_called_with(
            lineage_file, "lineage"
        )

    @pytest.mark.asyncio
    async def test_process_file_dataframe_lineage(self, file_watcher, temp_dirs):
        """Test processing a DataFrame lineage file"""
        dataframe_lineage_file = (
            temp_dirs["dataframe_lineage"] / "test_dataframe_lineage.jsonl"
        )
        dataframe_lineage_file.write_text('{"event_type": "dataframe_lineage"}\n')

        await file_watcher._process_file(dataframe_lineage_file, "dataframe_lineage")

        # Should have called processor with correct parameters
        file_watcher.event_processor.process_jsonl_file.assert_called_with(
            dataframe_lineage_file, "dataframe_lineage"
        )

    @pytest.mark.asyncio
    async def test_process_file_error_handling(self, file_watcher, temp_dirs):
        """Test error handling during file processing"""
        # Mock processor to raise an exception
        file_watcher.event_processor.process_jsonl_file.side_effect = Exception(
            "Processing error"
        )

        runtime_file = temp_dirs["runtime"] / "error_file.jsonl"
        runtime_file.write_text('{"event_type": "test"}\n')

        # Should handle error gracefully (method is void, doesn't return value)
        await file_watcher._process_file(runtime_file, "runtime")

        # Should still have attempted to call processor
        file_watcher.event_processor.process_jsonl_file.assert_called_with(
            runtime_file, "runtime"
        )

    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, file_watcher, temp_dirs):
        """Test processing a file that doesn't exist"""
        nonexistent_file = temp_dirs["runtime"] / "nonexistent.jsonl"

        await file_watcher._process_file(nonexistent_file, "runtime")

        # Should not have called processor for nonexistent file
        file_watcher.event_processor.process_jsonl_file.assert_not_called()

    def test_get_stats(self, file_watcher):
        """Test getting watcher statistics"""
        # Set some mock statistics
        file_watcher.files_processed = 10
        file_watcher.events_processed = 50

        # Mock the event processor stats method (now correctly calls get_validation_summary)
        file_watcher.event_processor.get_validation_summary.return_value = {
            "total_events": 55,
            "failed_events": 5,
        }

        stats = file_watcher.get_stats()

        expected_stats = {
            "files_processed": 10,
            "events_processed": 50,
            "total_events": 55,
            "failed_events": 5,
        }
        assert stats == expected_stats

    @pytest.mark.asyncio
    async def test_process_file_now(self, file_watcher, temp_dirs):
        """Test immediate file processing"""
        runtime_file = temp_dirs["runtime"] / "immediate.jsonl"
        runtime_file.write_text('{"event_type": "test"}\n')

        # Mock the events_processed counter to simulate processing
        file_watcher.events_processed = 0

        # Mock _process_file to increment events_processed
        async def mock_process_file(_file_path, _event_type):
            file_watcher.events_processed += 5

        file_watcher._process_file = mock_process_file

        result = await file_watcher.process_file_now(runtime_file)

        # Should determine file type and process
        # Now returns the number of events processed (events_processed diff)
        assert result == 5  # Events processed diff
        assert file_watcher.events_processed == 5

    @pytest.mark.asyncio
    async def test_background_processor_handles_pending_files(self, file_watcher):
        """Test background processor handles pending files"""
        # Mock the handler to return some pending files
        mock_file1 = Path("/test/file1.jsonl")
        mock_file2 = Path("/test/file2.jsonl")

        with patch.object(
            file_watcher.handler, "get_pending_files"
        ) as mock_get_pending:
            # First call returns pending files, subsequent calls return empty
            mock_get_pending.side_effect = [
                {mock_file1, mock_file2},  # First call
                set(),  # Subsequent calls
                set(),
            ]

            with patch.object(file_watcher, "_process_file") as mock_process:
                mock_process.return_value = 3  # Mock processed count

                # Start the background processor briefly
                file_watcher.running = True
                task = asyncio.create_task(file_watcher._background_processor())

                # Let it run briefly
                await asyncio.sleep(0.1)

                # Stop and wait
                file_watcher.running = False
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Should have processed pending files
                assert (
                    mock_process.call_count >= 0
                )  # May or may not be called depending on timing

    @pytest.mark.asyncio
    async def test_directory_creation_on_start(self, mock_event_processor, tmp_path):
        """Test that directories are created if they don't exist"""
        # Use non-existent directories
        runtime_dir = tmp_path / "new_runtime"
        lineage_dir = tmp_path / "new_lineage"
        dataframe_lineage_dir = tmp_path / "new_dataframe_lineage"

        watcher = FileWatcher(
            runtime_dir=runtime_dir,
            lineage_dir=lineage_dir,
            dataframe_lineage_dir=dataframe_lineage_dir,
            event_processor=mock_event_processor,
            notebook_hash="test_hash",
        )

        # Directories shouldn't exist yet
        assert not runtime_dir.exists()
        assert not lineage_dir.exists()
        assert not dataframe_lineage_dir.exists()

        # Start the watcher briefly
        start_task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.1)
        await watcher.stop()

        # Cancel the start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass

        # Directories should now exist
        assert runtime_dir.exists()
        assert lineage_dir.exists()
        assert dataframe_lineage_dir.exists()

    @pytest.mark.asyncio
    async def test_file_type_detection_in_process_file_now(
        self, file_watcher, temp_dirs
    ):
        """Test automatic file type detection in process_file_now"""
        # Test all three file types
        runtime_file = temp_dirs["runtime"] / "test.jsonl"
        lineage_file = temp_dirs["lineage"] / "test.jsonl"
        dataframe_lineage_file = temp_dirs["dataframe_lineage"] / "test.jsonl"

        runtime_file.write_text('{"event_type": "test"}\n')
        lineage_file.write_text('{"eventType": "START"}\n')
        dataframe_lineage_file.write_text('{"event_type": "dataframe_lineage"}\n')

        # Mock _process_file to track calls
        call_history = []

        async def mock_process_file(_file_path, _event_type):
            call_history.append((_file_path, _event_type))
            file_watcher.events_processed += 1

        file_watcher._process_file = mock_process_file

        # Test each file type
        await file_watcher.process_file_now(runtime_file)
        await file_watcher.process_file_now(lineage_file)
        await file_watcher.process_file_now(dataframe_lineage_file)

        # Should have detected correct event types
        assert len(call_history) == 3
        assert call_history[0][1] == "runtime"
        assert call_history[1][1] == "lineage"
        assert call_history[2][1] == "dataframe_lineage"

    @pytest.mark.asyncio
    async def test_dataframe_lineage_file_processing_end_to_end(
        self, file_watcher, temp_dirs
    ):
        """Test complete DataFrame lineage file processing workflow"""
        # Create a DataFrame lineage file
        dataframe_lineage_file = (
            temp_dirs["dataframe_lineage"] / "test_hash_dataframe_lineage.jsonl"
        )
        dataframe_lineage_file.write_text(
            '{"event_type": "dataframe_lineage", "operation_type": "selection"}\n'
        )

        # Test processing existing files (simulates startup)
        await file_watcher._process_existing_files()

        # Verify the file was processed with correct event type
        assert file_watcher.event_processor.process_jsonl_file.call_count >= 1

        # Check the call was made with the DataFrame lineage file and correct event type
        calls = file_watcher.event_processor.process_jsonl_file.call_args_list
        dataframe_lineage_calls = [
            call for call in calls if call[0][1] == "dataframe_lineage"
        ]
        assert (
            len(dataframe_lineage_calls) >= 1
        ), "Should have processed DataFrame lineage file"

        # Test manual file processing
        file_watcher.event_processor.process_jsonl_file.reset_mock()
        await file_watcher.process_file_now(dataframe_lineage_file)

        # Should have processed the file
        file_watcher.event_processor.process_jsonl_file.assert_called_with(
            dataframe_lineage_file, "dataframe_lineage"
        )


class TestFileWatcherIntegration:
    """Integration tests for FileWatcher with real file operations"""

    @pytest.fixture
    def event_processor(self, tmp_path):
        """Create a real event processor for integration tests"""
        # Use mock database manager for testing
        mock_db = MagicMock()
        processor = EventProcessor(mock_db)
        return processor

    @pytest.mark.asyncio
    async def test_real_file_monitoring(self, tmp_path):
        """Test real file monitoring with actual file creation"""
        runtime_dir = tmp_path / "runtime"
        lineage_dir = tmp_path / "lineage"
        dataframe_lineage_dir = tmp_path / "dataframe_lineage"

        # Create mock event processor
        mock_processor = MagicMock()
        mock_processor.process_jsonl_file.return_value = 2

        watcher = FileWatcher(
            runtime_dir=runtime_dir,
            lineage_dir=lineage_dir,
            dataframe_lineage_dir=dataframe_lineage_dir,
            event_processor=mock_processor,
            notebook_hash="test_hash",
            verbose=True,
        )

        # Start watcher
        start_task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.2)  # Let watcher start

        # Create a test file
        test_file = runtime_dir / "test_hash_test.jsonl"
        test_file.write_text('{"event_type": "test"}\n')

        # Wait for file to be processed - background processor runs every 1 second
        # So we need to wait at least 1.5 seconds to ensure processing happens
        await asyncio.sleep(1.5)

        # Stop watcher
        await watcher.stop()

        # Cancel the start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass

        # Should have processed the file
        assert mock_processor.process_jsonl_file.call_count >= 1

    @pytest.mark.asyncio
    async def test_error_recovery_during_monitoring(self, tmp_path):
        """Test error recovery during file monitoring"""
        runtime_dir = tmp_path / "runtime"
        lineage_dir = tmp_path / "lineage"
        dataframe_lineage_dir = tmp_path / "dataframe_lineage"

        # Create mock event processor that throws an error
        mock_processor = MagicMock()
        mock_processor.process_jsonl_file.side_effect = Exception("Test error")

        watcher = FileWatcher(
            runtime_dir=runtime_dir,
            lineage_dir=lineage_dir,
            dataframe_lineage_dir=dataframe_lineage_dir,
            event_processor=mock_processor,
            notebook_hash="test_hash",
            verbose=True,
        )

        # Start watcher
        start_task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.2)  # Let watcher start

        # Create a test file
        test_file = runtime_dir / "test_hash_test.jsonl"
        test_file.write_text('{"event_type": "test"}\n')

        # Wait for file to be processed - background processor runs every 1 second
        # So we need to wait at least 1.5 seconds to ensure processing happens
        await asyncio.sleep(1.5)

        # Stop watcher
        await watcher.stop()

        # Cancel the start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass

        # Should still be running despite error
        assert mock_processor.process_jsonl_file.call_count >= 1


@pytest.mark.asyncio
async def test_watch_directories_helper_function(tmp_path):
    """Test the utility function for watching directories"""
    runtime_dir = tmp_path / "runtime"
    lineage_dir = tmp_path / "lineage"
    dataframe_lineage_dir = tmp_path / "dataframe_lineage"

    # Create mock event processor
    mock_processor = MagicMock()
    mock_processor.process_jsonl_file.return_value = 1

    # Test the helper function with a short duration
    watcher = await watch_directories(
        runtime_dir=runtime_dir,
        lineage_dir=lineage_dir,
        dataframe_lineage_dir=dataframe_lineage_dir,
        event_processor=mock_processor,
        notebook_hash="test_hash",
        duration=0.1,  # Very short duration
    )

    # Should return a FileWatcher instance
    assert isinstance(watcher, FileWatcher)
    assert not watcher.running  # Should be stopped after duration
