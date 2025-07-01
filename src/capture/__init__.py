"""
Marimo Lineage MCP - Data Capture Layer

This module handles capturing runtime and lineage events from Marimo notebooks
with unique file naming based on notebook file paths.
"""

import hashlib
import os
from pathlib import Path

from .logger import (
    capture_logger,
    get_logger,
    hooks_logger,
    lineage_logger,
    runtime_logger,
)


def get_notebook_file_hash(notebook_path: str) -> str:
    """Generate a short hash from the notebook file path for unique identification.

    Args:
        notebook_path: Full path to the notebook file

    Returns:
        8-character hex hash of the file path
    """
    path_bytes = str(Path(notebook_path).resolve()).encode("utf-8")
    return hashlib.sha256(path_bytes).hexdigest()[:8]


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


def get_event_filenames(notebook_path: str, data_dir: str) -> tuple[str, str]:
    """Generate unique event filenames for a notebook.

    Args:
        notebook_path: Full path to the notebook file
        data_dir: Base directory for storing event files

    Returns:
        Tuple of (runtime_events_file, lineage_events_file) paths
    """
    file_hash = get_notebook_file_hash(notebook_path)
    notebook_name = get_notebook_name(notebook_path)

    runtime_file = f"{file_hash}_{notebook_name}_runtime_events.jsonl"
    lineage_file = f"{file_hash}_{notebook_name}_lineage_events.jsonl"

    data_path = Path(data_dir)
    return (
        str(data_path / "events" / "runtime" / runtime_file),
        str(data_path / "events" / "lineage" / lineage_file),
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

    # Check if we're in development mode
    is_dev_mode = False

    # Method 1: Check for environment variable override
    env_mode = os.environ.get("HUNYO_DEV_MODE", "").lower()
    if env_mode in ("1", "true", "yes", "on"):
        is_dev_mode = True
    elif env_mode in ("0", "false", "no", "off"):
        is_dev_mode = False  # Explicitly force production mode
    else:
        # Method 2: Auto-detect based on development indicators
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            # Check for common development indicators
            dev_markers = [
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
            if any((parent / marker).exists() for marker in dev_markers):
                is_dev_mode = True
                break

    if is_dev_mode:
        # Development mode: use project root .hunyo
        current_dir = Path.cwd()
        project_root = current_dir

        # Look for project markers going up the directory tree
        markers = [".git", "requirements.txt", "pyproject.toml", "setup.py", "src"]
        for parent in [current_dir] + list(current_dir.parents):
            if any((parent / marker).exists() for marker in markers):
                project_root = parent
                break

        data_dir = project_root / ".hunyo"
    else:
        # Production mode: use home directory .hunyo
        data_dir = Path.home() / ".hunyo"

    # Create directory structure
    (data_dir / "events" / "runtime").mkdir(parents=True, exist_ok=True)
    (data_dir / "events" / "lineage").mkdir(parents=True, exist_ok=True)
    (data_dir / "database").mkdir(parents=True, exist_ok=True)
    (data_dir / "config").mkdir(parents=True, exist_ok=True)

    return str(data_dir)


__all__ = [
    "capture_logger",
    "get_event_filenames",
    "get_logger",
    "get_notebook_file_hash",
    "get_notebook_name",
    "get_user_data_dir",
    "hooks_logger",
    "lineage_logger",
    "runtime_logger",
]
