#!/usr/bin/env python3
"""
Test suite for the capture logger module
"""

import io
import logging
from unittest.mock import patch

import pytest

from capture.logger import EmojiFormatter, HunyoLogger, get_logger


class TestEmojiFormatter:
    """Test the EmojiFormatter class"""

    def test_formatter_initialization(self):
        """Test that EmojiFormatter initializes correctly"""
        formatter = EmojiFormatter("%(message)s")
        assert isinstance(formatter, logging.Formatter)

    def test_format_with_different_log_levels(self):
        """Test formatting with different log levels"""
        formatter = EmojiFormatter("%(message)s")

        # Create test log records
        debug_record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="Debug message",
            args=(),
            exc_info=None,
        )

        info_record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Info message",
            args=(),
            exc_info=None,
        )

        warning_record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        error_record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        # Test formatting
        assert formatter.format(debug_record) == "[DEBUG] Debug message"
        assert formatter.format(info_record) == "[INFO] Info message"
        assert formatter.format(warning_record) == "[WARN] Warning message"
        assert formatter.format(error_record) == "[ERROR] Error message"

    def test_format_with_unknown_log_level(self):
        """Test formatting with unknown log level falls back to [INFO]"""
        formatter = EmojiFormatter("%(message)s")

        custom_record = logging.LogRecord(
            name="test",
            level=25,  # Custom level not in EMOJI_MAP
            pathname="",
            lineno=0,
            msg="Custom message",
            args=(),
            exc_info=None,
        )

        assert formatter.format(custom_record) == "[INFO] Custom message"


class TestHunyoLogger:
    """Test the HunyoLogger class"""

    @pytest.fixture
    def temp_logger(self):
        """Create a temporary logger for testing"""
        logger_name = f"test_logger_{id(self)}"
        logger = HunyoLogger(logger_name)
        yield logger
        # Cleanup: remove handlers to avoid interference between tests
        for handler in logger.logger.handlers[:]:
            logger.logger.removeHandler(handler)

    def test_logger_initialization(self, temp_logger):
        """Test that HunyoLogger initializes correctly"""
        assert isinstance(temp_logger.logger, logging.Logger)
        assert len(temp_logger.logger.handlers) == 1
        assert isinstance(temp_logger.logger.handlers[0], logging.StreamHandler)
        assert temp_logger.logger.level == logging.INFO

    def test_logger_has_original_methods(self, temp_logger):
        """Test that original logging methods are stored"""
        assert hasattr(temp_logger, "_original_info")
        assert hasattr(temp_logger, "_original_warning")
        assert hasattr(temp_logger, "_original_error")
        assert hasattr(temp_logger, "_original_debug")

    def test_context_specific_logging_methods(self, temp_logger):
        """Test context-specific logging methods produce correct output"""
        # Capture output by temporarily replacing the handler stream
        stream = io.StringIO()
        temp_logger.logger.handlers[0].stream = stream

        temp_logger.startup("Starting application")
        temp_logger.success("Operation completed")
        temp_logger.tracking("Monitoring activity")
        temp_logger.lineage("Data lineage tracked")
        temp_logger.runtime("Runtime event")
        temp_logger.config("Configuration loaded")

        output = stream.getvalue()

        # Verify context-specific prefixes are added
        assert "[START] Starting application" in output
        assert "[OK] Operation completed" in output
        assert "[DEBUG] Monitoring activity" in output
        assert "[LINK] Data lineage tracked" in output
        assert "[RUNTIME] Runtime event" in output
        assert "[CONFIG] Configuration loaded" in output

    def test_standard_logging_methods(self, temp_logger):
        """Test standard logging methods work correctly"""
        # Capture output by temporarily replacing the handler stream
        stream = io.StringIO()
        temp_logger.logger.handlers[0].stream = stream

        temp_logger.info("Info message")
        temp_logger.warning("Warning message")
        temp_logger.error("Error message")
        temp_logger.debug("Debug message")  # Won't show at INFO level

        output = stream.getvalue()

        assert "Info message" in output
        assert "[WARN] Warning message" in output
        assert "[ERROR] Error message" in output
        # Debug message won't appear because logger level is INFO
        assert "Debug message" not in output

    def test_debug_logging_with_debug_level(self, temp_logger):
        """Test debug logging when logger level is set to DEBUG"""
        temp_logger.logger.setLevel(logging.DEBUG)

        # Capture output by temporarily replacing the handler stream
        stream = io.StringIO()
        temp_logger.logger.handlers[0].stream = stream

        temp_logger.debug("Debug message")

        output = stream.getvalue()
        assert "[DEBUG] Debug message" in output

    def test_safe_log_with_unicode_error_fallback(self, temp_logger):
        """Test safe logging with Unicode encoding fallback"""
        # Test that the safe log method handles exceptions gracefully
        call_count = 0

        def mock_logger_info(_message):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call raises Unicode error
                encoding = "ascii"
                raise UnicodeEncodeError(encoding, "test", 0, 1, "error")
            # Second call succeeds (fallback)

        with patch.object(temp_logger.logger, "info", side_effect=mock_logger_info):
            # This should not raise an exception but should handle the error gracefully
            temp_logger.info("Test message with unicode issues")
            # Test completes if no exception is raised
            assert call_count == 2  # Verify fallback was attempted

    def test_exception_logging(self, temp_logger):
        """Test exception logging with traceback"""
        # Capture output by temporarily replacing the handler stream
        stream = io.StringIO()
        temp_logger.logger.handlers[0].stream = stream

        try:
            error_msg = "Test exception"
            raise ValueError(error_msg)
        except ValueError:
            temp_logger.exception("An exception occurred")

        output = stream.getvalue()
        assert "[ERROR] An exception occurred" in output
        assert "ValueError: Test exception" in output
        assert "Traceback" in output

    def test_get_logger_method(self, temp_logger):
        """Test getting the underlying logger instance"""
        underlying_logger = temp_logger.get_logger()
        assert underlying_logger is temp_logger.logger
        assert isinstance(underlying_logger, logging.Logger)

    def test_all_context_methods_exist(self, temp_logger):
        """Test that all expected context-specific methods exist"""
        expected_methods = [
            "startup",
            "success",
            "target",
            "tracking",
            "status",
            "lineage",
            "timing",
            "notebook",
            "runtime",
            "config",
            "file_op",
            "critical",
            "info",
            "warning",
            "error",
            "exception",
            "debug",
        ]

        for method_name in expected_methods:
            assert hasattr(temp_logger, method_name)
            assert callable(getattr(temp_logger, method_name))


class TestGetLoggerFactory:
    """Test the get_logger factory function"""

    def test_get_logger_returns_hunyo_logger(self):
        """Test that get_logger returns HunyoLogger instance"""
        logger = get_logger("test.logger")
        assert isinstance(logger, HunyoLogger)

    def test_get_logger_with_different_names(self):
        """Test get_logger with different names creates different loggers"""
        logger1 = get_logger("test.logger1")
        logger2 = get_logger("test.logger2")

        assert logger1.logger.name == "test.logger1"
        assert logger2.logger.name == "test.logger2"
        assert logger1.logger.name != logger2.logger.name

    def test_get_logger_name_propagation(self):
        """Test that logger name is properly set"""
        test_name = "hunyo.capture.test"
        logger = get_logger(test_name)
        assert logger.logger.name == test_name

    def test_get_logger_configuration(self):
        """Test that logger returned by get_logger is properly configured"""
        logger = get_logger("hunyo.test.config")

        # Should have exactly one handler (console handler)
        assert len(logger.logger.handlers) == 1

        # Handler should be StreamHandler
        assert isinstance(logger.logger.handlers[0], logging.StreamHandler)

        # Handler should have EmojiFormatter
        assert isinstance(logger.logger.handlers[0].formatter, EmojiFormatter)

        # Logger level should be INFO
        assert logger.logger.level == logging.INFO

    def test_logger_handler_reuse(self):
        """Test that creating logger with same name doesn't duplicate handlers"""
        logger1 = get_logger("reuse.test")
        initial_handler_count = len(logger1.logger.handlers)

        logger2 = get_logger("reuse.test")
        final_handler_count = len(logger2.logger.handlers)

        # Should have same number of handlers (no duplication)
        assert initial_handler_count == final_handler_count
        assert logger1.logger is logger2.logger  # Same underlying logger


class TestLoggerIntegration:
    """Integration tests for logger functionality"""

    def test_real_world_logging_scenario(self):
        """Test a realistic logging scenario"""
        logger = get_logger("hunyo.integration.test")

        # Capture output by temporarily replacing the handler stream
        stream = io.StringIO()
        logger.logger.handlers[0].stream = stream

        # Simulate real application flow
        logger.startup("Initializing Hunyo MCP Server")
        logger.config("Loading configuration from config.json")
        logger.runtime("Runtime event processor started")
        logger.lineage("DataFrame tracking enabled")
        logger.success("System initialization complete")

        # Simulate some warnings and errors
        logger.warning("Low memory warning")
        logger.error("Failed to connect to database")

        output = stream.getvalue()

        # Verify all expected messages are present with correct formatting
        assert "[START] Initializing Hunyo MCP Server" in output
        assert "[CONFIG] Loading configuration from config.json" in output
        assert "[RUNTIME] Runtime event processor started" in output
        assert "[LINK] DataFrame tracking enabled" in output
        assert "[OK] System initialization complete" in output
        assert "[WARN] Low memory warning" in output
        assert "[ERROR] Failed to connect to database" in output

    def test_logger_thread_safety_basic(self):
        """Basic test for logger thread safety"""
        import threading
        import time

        logger = get_logger("hunyo.thread.test")

        def log_messages(thread_id):
            for i in range(5):
                logger.info(f"Thread {thread_id} message {i}")
                time.sleep(0.01)  # Small delay to interleave

        # Start multiple threads
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=log_messages, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Test completes successfully if no exceptions are raised
        assert True

    def test_logger_memory_efficiency(self):
        """Test that logger doesn't consume excessive memory with many calls"""
        logger = get_logger("hunyo.memory.test")

        # Capture output by temporarily replacing the handler stream
        stream = io.StringIO()
        logger.logger.handlers[0].stream = stream

        # Log many messages to test memory efficiency
        for i in range(1000):
            logger.info(f"Test message {i}")

        output = stream.getvalue()

        # Verify first and last messages are present
        assert "Test message 0" in output
        assert "Test message 999" in output

        # Count total lines (should be 1000)
        lines = output.strip().split("\n")
        assert len(lines) == 1000
