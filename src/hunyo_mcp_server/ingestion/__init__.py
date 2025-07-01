"""
Ingestion Module for Hunyo MCP Server.

Handles the data pipeline from JSONL event files to DuckDB storage:
- DuckDBManager: Database initialization and management
- EventProcessor: Event validation and transformation
- FileWatcher: Real-time monitoring of event files
"""

from .duckdb_manager import DuckDBManager
from .event_processor import EventProcessor
from .file_watcher import FileWatcher

__all__ = ["DuckDBManager", "EventProcessor", "FileWatcher"]
