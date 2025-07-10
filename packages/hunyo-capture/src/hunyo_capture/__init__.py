"""
Marimo Lineage MCP - Data Capture Layer

This module handles capturing runtime and lineage events from Marimo notebooks
with unique file naming based on notebook file paths.
"""

import hashlib
from pathlib import Path

# Phase 1 abstractions - available for internal use but not exposed to users
from .adapters.base import ExecutionContextAdapter  # noqa: F401
from .adapters.marimo_adapter import MarimoContextAdapter  # noqa: F401
from .constants import *  # noqa: F403
from .constants import (
    DATAFRAME_LINEAGE_SUFFIX,
    EVENTS_DIR_NAME,
    LINEAGE_SUFFIX,
    MAX_HASH_LENGTH,
    RUNTIME_SUFFIX,
    VERSION,
)
from .environments.base import EnvironmentDetector, NotebookEnvironment  # noqa: F401
from .factory import (  # noqa: F401
    ComponentFactory,
    EnvironmentNotAvailableError,
    EnvironmentNotSupportedError,
    create_components,
    create_context_adapter,
    create_hook_manager,
)
from .hooks.base import NotebookHooks  # noqa: F401
from .hooks.marimo_hooks import MarimoHooks  # noqa: F401
from .logger import get_logger
from .unified_notebook_interceptor import (
    UnifiedNotebookInterceptor,  # noqa: F401
    disable_unified_tracking,
    enable_unified_tracking,
    get_unified_interceptor,  # noqa: F401
    is_unified_tracking_active,
)

# Set package version
__version__ = VERSION

# Backward compatibility: Create common logger instances that were previously exported
# These are maintained for existing code that depends on them
capture_logger = get_logger("hunyo.capture")
runtime_logger = get_logger("hunyo.runtime")
lineage_logger = get_logger("hunyo.lineage")
hooks_logger = get_logger("hunyo.hooks")


def get_notebook_file_hash(notebook_path: str) -> str:
    """Generate a short hash from the notebook file path for unique identification.

    Args:
        notebook_path: Full path to the notebook file

    Returns:
        8-character hex hash of the file path
    """
    path_bytes = str(Path(notebook_path).resolve()).encode("utf-8")
    return hashlib.sha256(path_bytes).hexdigest()[:MAX_HASH_LENGTH]


def get_notebook_name(notebook_path: str) -> str:
    """Extract clean notebook name from file path.

    Args:
        notebook_path: Full path to the notebook file

    Returns:
        Notebook filename without extension, filesystem-safe
    """
    path = Path(notebook_path)
    name = path.stem  # filename without extension
    # Make filesystem-safe by replacing problematic characters
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return safe_name


def get_event_filenames(
    notebook_path: str, data_dir: str, environment: str = "marimo"  # noqa: ARG001
) -> tuple[str, str, str]:
    """Generate unique event filenames for a notebook.

    Args:
        notebook_path: Full path to the notebook file
        data_dir: Base directory for storing event files
        environment: Environment prefix for file naming (default: "marimo" for backward compatibility)

    Returns:
        Tuple of (runtime_events_file, lineage_events_file, dataframe_lineage_events_file) paths
    """
    file_hash = get_notebook_file_hash(notebook_path)
    notebook_name = get_notebook_name(notebook_path)

    # Generate filenames without environment prefix to maintain compatibility with MCP server
    runtime_file = f"{file_hash}_{notebook_name}{RUNTIME_SUFFIX}.jsonl"
    lineage_file = f"{file_hash}_{notebook_name}{LINEAGE_SUFFIX}.jsonl"
    dataframe_lineage_file = (
        f"{file_hash}_{notebook_name}{DATAFRAME_LINEAGE_SUFFIX}.jsonl"
    )

    data_path = Path(data_dir)
    return (
        str(data_path / EVENTS_DIR_NAME / "runtime" / runtime_file),
        str(data_path / EVENTS_DIR_NAME / "lineage" / lineage_file),
        str(data_path / EVENTS_DIR_NAME / "dataframe_lineage" / dataframe_lineage_file),
    )


def get_user_data_dir() -> str:
    """Get data directory for marimo-lineage-mcp.

    In development mode (when in a git repo or with DEV environment variable):
        Uses .hunyo in project root for easy validation
    In production mode:
        Uses .hunyo in user's home directory

    Returns:
        Path to .hunyo directory, creating it if necessary
    """
    # Use new environment detection system while maintaining backward compatibility

    data_dir = EnvironmentDetector.get_data_directory()
    EnvironmentDetector.ensure_data_directory_structure(data_dir)

    return str(data_dir)


__all__ = [
    # Essential tracking functions and utilities (alphabetically sorted)
    "disable_unified_tracking",
    "enable_unified_tracking",
    "get_event_filenames",
    "get_notebook_file_hash",
    "get_notebook_name",
    "get_user_data_dir",
    "is_unified_tracking_active",
]
