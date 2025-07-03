"""
Hunyo MCP Server - Enhanced logging with context-aware emoji formatting.

Provides structured logging with:
- Context-specific emoji formatting (marimo-aware fallbacks)
- Proper file handling and rotation
- Production-ready log levels and formatting
- Windows-compatible ASCII-safe output
"""

from __future__ import annotations

import logging
import sys

# Windows-safe emoji mapping (ASCII alternatives)
EMOJI_MAP = {
    logging.DEBUG: "[DEBUG]",
    logging.INFO: "[INFO]",
    logging.WARNING: "[WARN]",
    logging.ERROR: "[ERROR]",
    logging.CRITICAL: "[CRITICAL]",
}


class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds context-appropriate emojis to log messages"""

    def format(self, record):
        # Add emoji prefix based on log level (Windows-safe)
        emoji = EMOJI_MAP.get(record.levelno, "[INFO]")
        original_msg = super().format(record)
        return f"{emoji} {original_msg}"


def get_logger(name: str) -> HunyoLogger:
    """
    Get a context-aware logger instance.

    Args:
        name: Logger name (e.g., 'hunyo.capture.lineage')

    Returns:
        HunyoLogger instance with emoji formatting
    """
    return HunyoLogger(name)


class HunyoLogger:
    """
    Hunyo-specific logger with context-aware emoji formatting.

    Provides enhanced logging methods with emoji prefixes that degrade
    gracefully in different environments (marimo notebooks, CLI, etc.).
    """

    def __init__(self, name: str):
        """Initialize logger with emoji-enhanced formatting"""
        self.logger = logging.getLogger(name)

        # Only configure if not already configured
        if not self.logger.handlers:
            # Console handler with emoji formatting
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                EmojiFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

        # Store original logging methods for fallback
        self._original_info = self.logger.info
        self._original_warning = self.logger.warning
        self._original_error = self.logger.error
        self._original_debug = self.logger.debug

    def _safe_log(self, level: str, message: str):
        """Safely log messages with fallback handling"""
        try:
            getattr(self.logger, level)(message)
        except (UnicodeEncodeError, OSError):
            # Fallback for environments that can't handle emojis
            fallback_message = message.encode("ascii", errors="replace").decode("ascii")
            getattr(self.logger, level)(fallback_message)

    # Context-specific logging methods (Windows-safe)
    def startup(self, message: str):
        """Log startup/initialization messages"""
        self._safe_log("info", f"[START] {message}")

    def success(self, message: str):
        """Log successful operations"""
        self._safe_log("info", f"[OK] {message}")

    def target(self, message: str):
        """Log target/goal achievement"""
        self._safe_log("info", f"[TARGET] {message}")

    def tracking(self, message: str):
        """Log tracking/monitoring activities"""
        self._safe_log("info", f"[DEBUG] {message}")

    def status(self, message: str):
        """Log status updates"""
        self._safe_log("info", f"[INFO] {message}")

    def lineage(self, message: str):
        """Log lineage-related operations"""
        self._safe_log("info", f"[LINK] {message}")

    def timing(self, message: str):
        """Log timing/performance information"""
        self._safe_log("info", f"[TIME] {message}")

    def notebook(self, message: str):
        """Log notebook-related messages"""
        self._safe_log("info", f"[NOTEBOOK] {message}")

    def runtime(self, message: str):
        """Log runtime messages"""
        self._safe_log("info", f"[RUNTIME] {message}")

    def config(self, message: str):
        """Log configuration messages"""
        self._safe_log("info", f"[CONFIG] {message}")

    def file_op(self, message: str):
        """Log file operations"""
        self._safe_log("info", f"[FILE] {message}")

    def critical(self, message: str):
        """Log critical messages"""
        self._safe_log("critical", f"[CRITICAL] {message}")

    # Standard logging methods with emoji enhancement
    def info(self, message: str):
        """Log info messages"""
        self._safe_log("info", message)

    def warning(self, message: str):
        """Log warning messages"""
        self._safe_log("warning", f"[WARN] {message}")

    def error(self, message: str):
        """Log error messages"""
        self._safe_log("error", f"[ERROR] {message}")

    def exception(self, message: str):
        """Log exception with traceback"""
        self.logger.exception(f"[ERROR] {message}")

    def debug(self, message: str):
        """Log debug messages"""
        self._safe_log("debug", f"[DEBUG] {message}")

    # Direct access to underlying logger
    def get_logger(self):
        """Get the underlying logger instance"""
        return self.logger


# Logger registry to avoid duplicate loggers
_loggers: dict[str, HunyoLogger] = {}
