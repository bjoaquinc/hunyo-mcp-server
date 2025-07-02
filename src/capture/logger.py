"""
Logging utilities for hunyo-notebook-memories-mcp
Provides structured logging with emoji-rich user feedback
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar


class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emojis to log levels"""

    EMOJI_MAP: ClassVar[dict[int, str]] = {
        logging.DEBUG: "🔍",
        logging.INFO: "💡",  # Changed from ambiguous info symbol to light bulb
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Add emoji prefix
        emoji = self.EMOJI_MAP.get(record.levelno, "📝")

        # Format the base message
        formatted = super().format(record)

        # Add emoji prefix to the message
        return f"{emoji} {formatted}"


class HunyoLogger:
    """Custom logger for hunyo capture modules with emoji support"""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup console handler with emoji formatting"""
        # Detect marimo context and avoid logging conflicts
        if self._is_marimo_context():
            # Don't add handlers in marimo - use direct print instead
            return

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Use custom emoji formatter
        formatter = EmojiFormatter(
            fmt="%(message)s", datefmt="%H:%M:%S"  # Just the message for clean output
        )
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def _is_marimo_context(self) -> bool:
        """Check if we're running inside a marimo notebook"""
        try:
            import sys

            # Check if marimo is in the call stack
            frame = sys._getframe()
            while frame:
                if "marimo" in str(frame.f_code.co_filename):
                    return True
                frame = frame.f_back
            return False
        except (AttributeError, ValueError):  # Specific exceptions for frame inspection
            return False

    def setup_file_logging(self, log_file: Path | None = None) -> Path:
        """Add file logging for debugging"""
        if log_file is None:
            from capture import get_user_data_dir

            log_dir = Path(get_user_data_dir()) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = (
                log_dir / f"hunyo_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log"
            )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Detailed format for file logs
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(file_handler)
        return log_file

    def _safe_log(self, level: str, message: str) -> None:
        """Safely log messages, using print in marimo context"""
        try:
            if self._is_marimo_context():
                # Use print for marimo to avoid stream conflicts
                print(message)  # noqa: T201
            else:
                # Use regular logging outside marimo
                getattr(self.logger, level)(message)
        except (OSError, ValueError, AttributeError) as e:
            # Handle I/O errors (closed stdout/stderr during shutdown)
            # Silently ignore these errors to prevent crash during cleanup
            if (
                "I/O operation on closed file" in str(e)
                or "closed file" in str(e)
                or "stream" in str(e).lower()
            ):
                pass  # Expected during shutdown
            else:
                # Re-raise unexpected errors
                raise
        except Exception:  # noqa: S110
            # Catch any other logging-related errors during shutdown
            # This is a last resort to prevent crashes during cleanup
            # We intentionally don't log here to avoid recursive logging errors
            pass

    # Convenience methods with emoji themes
    def startup(self, message: str) -> None:
        """Log startup messages with rocket emoji"""
        self._safe_log("info", f"🚀 {message}")

    def success(self, message: str) -> None:
        """Log success messages with checkmark emoji"""
        self._safe_log("info", f"✅ {message}")

    def status(self, message: str) -> None:
        """Log status messages with target emoji"""
        self._safe_log("info", f"🎯 {message}")

    def tracking(self, message: str) -> None:
        """Log tracking messages with magnifying glass emoji"""
        self._safe_log("info", f"🔍 {message}")

    def notebook(self, message: str) -> None:
        """Log notebook-related messages with notebook emoji"""
        self._safe_log("info", f"📝 {message}")

    def lineage(self, message: str) -> None:
        """Log lineage messages with link emoji"""
        self._safe_log("info", f"🔗 {message}")

    def runtime(self, message: str) -> None:
        """Log runtime messages with stopwatch emoji"""
        self._safe_log("info", f"⏱️ {message}")

    def config(self, message: str) -> None:
        """Log configuration messages with gear emoji"""
        self._safe_log("info", f"🔧 {message}")

    def file_op(self, message: str) -> None:
        """Log file operations with folder emoji"""
        self._safe_log("info", f"📁 {message}")

    def warning(self, message: str) -> None:
        """Log warnings"""
        self._safe_log("warning", f"⚠️ {message}")

    def error(self, message: str) -> None:
        """Log errors"""
        self._safe_log("error", f"❌ {message}")

    def info(self, message: str) -> None:
        """Log info messages"""
        self._safe_log("info", message)

    def debug(self, message: str) -> None:
        """Log debug messages"""
        self._safe_log("debug", f"🔍 {message}")

    def critical(self, message: str) -> None:
        """Log critical messages"""
        self._safe_log("critical", f"🚨 {message}")


# Global logger instances for different modules
_loggers = {}


def get_logger(name: str) -> HunyoLogger:
    """Get or create a logger for a module"""
    if name not in _loggers:
        _loggers[name] = HunyoLogger(name)
    return _loggers[name]


# Common loggers
capture_logger = get_logger("hunyo.capture")
runtime_logger = get_logger("hunyo.runtime")
lineage_logger = get_logger("hunyo.lineage")
hooks_logger = get_logger("hunyo.hooks")
