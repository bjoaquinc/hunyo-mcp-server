"""
Logging utilities for hunyo-notebook-memories-mcp
Provides structured logging with emoji-rich user feedback
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emojis to log levels"""

    EMOJI_MAP = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨",
    }

    def format(self, record):
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

    def _setup_handlers(self):
        """Setup console handler with emoji formatting"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Use custom emoji formatter
        formatter = EmojiFormatter(
            fmt="%(message)s", datefmt="%H:%M:%S"  # Just the message for clean output
        )
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def setup_file_logging(self, log_file: Path | None = None):
        """Add file logging for debugging"""
        if log_file is None:
            from . import get_user_data_dir

            log_dir = Path(get_user_data_dir()) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"hunyo_{datetime.now().strftime('%Y%m%d')}.log"

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

    # Convenience methods with emoji themes
    def startup(self, message: str):
        """Log startup messages with rocket emoji"""
        self.logger.info(f"🚀 {message}")

    def success(self, message: str):
        """Log success messages with checkmark emoji"""
        self.logger.info(f"✅ {message}")

    def status(self, message: str):
        """Log status messages with target emoji"""
        self.logger.info(f"🎯 {message}")

    def tracking(self, message: str):
        """Log tracking messages with magnifying glass emoji"""
        self.logger.info(f"🔍 {message}")

    def notebook(self, message: str):
        """Log notebook-related messages with notebook emoji"""
        self.logger.info(f"📝 {message}")

    def lineage(self, message: str):
        """Log lineage messages with link emoji"""
        self.logger.info(f"🔗 {message}")

    def runtime(self, message: str):
        """Log runtime messages with stopwatch emoji"""
        self.logger.info(f"⏱️ {message}")

    def config(self, message: str):
        """Log configuration messages with gear emoji"""
        self.logger.info(f"🔧 {message}")

    def file_op(self, message: str):
        """Log file operations with folder emoji"""
        self.logger.info(f"📁 {message}")

    def warning(self, message: str):
        """Log warnings"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log errors"""
        self.logger.error(message)

    def info(self, message: str):
        """Log info messages"""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug messages"""
        self.logger.debug(f"🔍 {message}")

    def critical(self, message: str):
        """Log critical messages"""
        self.logger.critical(message)


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
