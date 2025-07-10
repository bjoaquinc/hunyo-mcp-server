#!/usr/bin/env python3
"""
Constants and configuration values for hunyo-capture.

This module consolidates all constants, magic numbers, and configuration values
that were previously scattered across multiple files, following DRY principles.
"""

import os
from typing import Final

# ========================================
# Version and Package Information
# ========================================

VERSION: Final[str] = "0.1.0"

# ========================================
# Data Capture Limits and Constraints
# ========================================

# DataFrame and object size limits for performance
MAX_OBJECT_SIZE: Final[int] = 1000
MAX_COLLECTION_SIZE: Final[int] = 10
MAX_DICT_SIZE: Final[int] = 5

# Hash and ID generation
MAX_HASH_LENGTH: Final[int] = 8
UUID_TRUNCATE_LENGTH: Final[int] = 8

# DataFrame operation limits
DATAFRAME_DIMENSIONS: Final[int] = (
    2  # DataFrame should have 2 dimensions (rows, columns)
)
MAX_OPERATION_HISTORY: Final[int] = 50  # Maximum operations to keep in lineage chain
OPERATION_HISTORY_TRIM: Final[int] = 25  # Keep last 25 operations when trimming

# ========================================
# File and Directory Constants
# ========================================

# Data directory structure
DATA_DIR_NAME: Final[str] = ".hunyo"
EVENTS_DIR_NAME: Final[str] = "events"
RUNTIME_DIR_NAME: Final[str] = "runtime"
LINEAGE_DIR_NAME: Final[str] = "lineage"
DATAFRAME_LINEAGE_DIR_NAME: Final[str] = "dataframe_lineage"
DATABASE_DIR_NAME: Final[str] = "database"
CONFIG_DIR_NAME: Final[str] = "config"

# File extensions and suffixes
JSONL_EXTENSION: Final[str] = ".jsonl"
RUNTIME_SUFFIX: Final[str] = "_runtime_events"
LINEAGE_SUFFIX: Final[str] = "_lineage_events"
DATAFRAME_LINEAGE_SUFFIX: Final[str] = "_dataframe_lineage_events"

# ========================================
# Environment and Configuration
# ========================================

# Environment variable names
ENV_HUNYO_DEV_MODE: Final[str] = "HUNYO_DEV_MODE"
ENV_HUNYO_DATAFRAME_LINEAGE: Final[str] = "HUNYO_DATAFRAME_LINEAGE"

# Environment detection markers
DEV_MARKERS: Final[list[str]] = [
    ".git",  # Git repository
    "src",  # Source code directory
    "requirements.txt",  # Python project
    "pyproject.toml",  # Modern Python project
    "setup.py",  # Python package
    ".env",  # Environment config
    "venv",  # Virtual environment
    "test",  # Test directory
    "tests",  # Test directory
]

# Project detection markers
PROJECT_MARKERS: Final[list[str]] = [
    ".git",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "src",
]

# ========================================
# OpenLineage and Schema Constants
# ========================================

# OpenLineage producer and schema URLs
OPENLINEAGE_PRODUCER: Final[str] = "marimo-lineage-tracker"
OPENLINEAGE_SCHEMA_URL: Final[str] = (
    "https://openlineage.io/spec/1-0-5/OpenLineage.json"
)
ERROR_MESSAGE_SCHEMA_URL: Final[str] = (
    "https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json"
)

# Programming language identifier
PROGRAMMING_LANGUAGE: Final[str] = "python"

# Schema-compliant event types
RUNTIME_EVENT_TYPES: Final[dict[str, str]] = {
    "START": "cell_execution_start",
    "END": "cell_execution_end",
    "ERROR": "cell_execution_error",
}

# OpenLineage event types
OPENLINEAGE_EVENT_TYPES: Final[dict[str, str]] = {
    "START": "START",
    "COMPLETE": "COMPLETE",
    "FAIL": "FAIL",
    "ABORT": "ABORT",
}

# DataFrame operation types
DATAFRAME_OPERATION_TYPES: Final[dict[str, str]] = {
    "selection": "selection",
    "aggregation": "aggregation",
    "join": "join",
    "transformation": "transformation",
}

# ========================================
# Performance and Monitoring
# ========================================

# Performance tracking defaults
DEFAULT_PERFORMANCE_OVERHEAD_MS: Final[float] = 0.5
MEMORY_CONVERSION_FACTOR: Final[float] = 1024 * 1024  # Bytes to MB
MILLISECONDS_CONVERSION_FACTOR: Final[float] = 1000  # Seconds to milliseconds

# ========================================
# Configuration Defaults
# ========================================

# Default configuration values
DEFAULT_DATAFRAME_LINEAGE_ENABLED: Final[bool] = True
DEFAULT_ENVIRONMENT_PREFIX: Final[str] = "marimo"

# Truth value mappings for environment variables
TRUTH_VALUES: Final[set[str]] = {"1", "true", "yes", "on"}
FALSE_VALUES: Final[set[str]] = {"0", "false", "no", "off"}

# ========================================
# Helper Functions for Configuration
# ========================================


def get_dataframe_lineage_config() -> bool:
    """Get DataFrame lineage configuration from environment."""
    return os.getenv(ENV_HUNYO_DATAFRAME_LINEAGE, "true").lower() == "true"


def get_dev_mode_config() -> bool | None:
    """Get development mode configuration from environment."""
    env_mode = os.environ.get(ENV_HUNYO_DEV_MODE, "").lower()
    if env_mode in TRUTH_VALUES:
        return True
    elif env_mode in FALSE_VALUES:
        return False
    return None  # Use auto-detection


def generate_truncated_uuid() -> str:
    """Generate a truncated UUID for consistent ID generation."""
    import uuid

    return str(uuid.uuid4())[:UUID_TRUNCATE_LENGTH]
