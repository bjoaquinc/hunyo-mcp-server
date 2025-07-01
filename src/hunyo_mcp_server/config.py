"""
Configuration management for Hunyo MCP Server.

Handles data directory resolution, environment detection, and path management
for both development and production environments.
"""

import os
from pathlib import Path

# Import logging utility
try:
    from ..capture.logger import get_logger

    mcp_logger = get_logger("hunyo.mcp")
except ImportError:
    # Fallback for testing (silent for style compliance)
    class SimpleLogger:
        def status(self, msg):
            pass

        def success(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            pass

        def config(self, msg):
            pass

        def file_op(self, msg):
            pass

        def debug(self, msg):
            pass

    mcp_logger = SimpleLogger()


def get_hunyo_data_dir() -> Path:
    """
    Returns the appropriate data directory for Hunyo events and database.

    Logic:
    - Production mode: ~/.hunyo
    - Development mode: .hunyo in repository root

    Returns:
        Path: The data directory path
    """
    if is_development_mode():
        # Development: use .hunyo in repository root
        repo_root = get_repository_root()
        return repo_root / ".hunyo"
    else:
        # Production: use ~/.hunyo in user home directory
        return Path.home() / ".hunyo"


def is_development_mode() -> bool:
    """
    Check if running from repository (development) vs installed package (production).

    Detection logic:
    1. Check if we're in a git repository (has .git directory)
    2. Check if pyproject.toml exists in current directory tree
    3. Check HUNYO_DEV_MODE environment variable override

    Returns:
        bool: True if in development mode, False if in production
    """
    # Environment variable override
    env_dev_mode = os.environ.get("HUNYO_DEV_MODE")
    if env_dev_mode is not None:
        return env_dev_mode.lower() in ("1", "true", "yes", "on")

    # Check if we're in a repository
    try:
        repo_root = get_repository_root()

        # Look for development indicators
        has_git = (repo_root / ".git").exists()
        has_pyproject = (repo_root / "pyproject.toml").exists()
        has_src_dir = (repo_root / "src" / "hunyo_mcp_server").exists()

        return has_git and has_pyproject and has_src_dir

    except Exception:
        # If we can't determine, assume production
        return False


def get_repository_root() -> Path:
    """
    Find the repository root by looking for .git directory or pyproject.toml.

    Returns:
        Path: Repository root directory

    Raises:
        RuntimeError: If repository root cannot be found
    """
    current = Path(__file__).resolve()

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume current working directory
    cwd = Path.cwd()
    if (cwd / ".git").exists() or (cwd / "pyproject.toml").exists():
        return cwd

    raise RuntimeError(
        "Could not find repository root (no .git or pyproject.toml found)"
    )


def get_event_directories() -> tuple[Path, Path]:
    """
    Get the runtime and lineage event directories.

    Returns:
        Tuple[Path, Path]: (runtime_events_dir, lineage_events_dir)
    """
    data_dir = get_hunyo_data_dir()
    runtime_dir = data_dir / "events" / "runtime"
    lineage_dir = data_dir / "events" / "lineage"
    return runtime_dir, lineage_dir


def get_database_path() -> Path:
    """
    Get the DuckDB database file path.

    Returns:
        Path: Database file path
    """
    data_dir = get_hunyo_data_dir()
    return data_dir / "database" / "lineage.duckdb"


def get_config_path() -> Path:
    """
    Get the configuration file path.

    Returns:
        Path: Configuration file path
    """
    data_dir = get_hunyo_data_dir()
    return data_dir / "config" / "settings.yaml"


def ensure_directory_structure() -> None:
    """
    Ensure all necessary directories exist with proper permissions.

    Creates:
    - Main data directory (.hunyo)
    - Events subdirectories (runtime, lineage)
    - Database directory
    - Config directory
    """
    data_dir = get_hunyo_data_dir()

    # Create main directory structure
    directories_to_create = [
        data_dir,
        data_dir / "events",
        data_dir / "events" / "runtime",
        data_dir / "events" / "lineage",
        data_dir / "database",
        data_dir / "config",
    ]

    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)

        # Ensure reasonable permissions (owner read/write/execute)
        try:
            directory.chmod(0o755)
        except (OSError, NotImplementedError):
            # Some filesystems don't support chmod, skip silently
            pass


def get_event_file_path(event_type: str, notebook_path: str | None = None) -> Path:
    """
    Get the full path for an event file.

    Args:
        event_type: 'runtime' or 'lineage'
        notebook_path: Optional notebook path for unique naming

    Returns:
        Path: Full path to the event file

    Raises:
        ValueError: If event_type is not 'runtime' or 'lineage'
    """
    if event_type not in ("runtime", "lineage"):
        raise ValueError(
            f"event_type must be 'runtime' or 'lineage', got: {event_type}"
        )

    data_dir = get_hunyo_data_dir()
    events_dir = data_dir / "events" / event_type

    if notebook_path:
        # Create unique filename based on notebook
        notebook_name = Path(notebook_path).stem
        filename = f"{notebook_name}_{event_type}_events.jsonl"
    else:
        # Default filename
        filename = f"{event_type}_events.jsonl"

    return events_dir / filename


def get_environment_info() -> dict:
    """
    Get comprehensive environment information for debugging.

    Returns:
        dict: Environment information including paths and mode
    """
    try:
        data_dir = get_hunyo_data_dir()
        db_path = get_database_path()
        config_path = get_config_path()
        runtime_dir, lineage_dir = get_event_directories()

        return {
            "development_mode": is_development_mode(),
            "data_directory": str(data_dir),
            "database_path": str(db_path),
            "config_path": str(config_path),
            "runtime_events_dir": str(runtime_dir),
            "lineage_events_dir": str(lineage_dir),
            "data_dir_exists": data_dir.exists(),
            "database_exists": db_path.exists(),
            "config_exists": config_path.exists(),
        }
    except Exception as e:
        return {
            "error": str(e),
            "development_mode": None,
        }


class HunyoConfig:
    """
    Centralized configuration management for Hunyo MCP Server.
    
    Provides easy access to all configuration settings and paths.
    """

    def __init__(self):
        """Initialize configuration with current environment detection."""
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all directories exist during initialization."""
        try:
            ensure_directory_structure()
        except Exception as e:
            mcp_logger.warning(f"Failed to create directory structure: {e}")

    @property
    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return is_development_mode()

    @property
    def data_directory(self) -> Path:
        """Get the main data directory."""
        return get_hunyo_data_dir()

    @property
    def database_path(self) -> Path:
        """Get the DuckDB database file path."""
        return get_database_path()

    @property
    def config_path(self) -> Path:
        """Get the configuration file path."""
        return get_config_path()

    @property
    def runtime_events_dir(self) -> Path:
        """Get the runtime events directory."""
        runtime_dir, _ = get_event_directories()
        return runtime_dir

    @property
    def lineage_events_dir(self) -> Path:
        """Get the lineage events directory."""
        _, lineage_dir = get_event_directories()
        return lineage_dir

    def get_event_file_path(self, event_type: str, notebook_path: str | None = None) -> Path:
        """Get the full path for an event file."""
        return get_event_file_path(event_type, notebook_path)

    def get_environment_info(self) -> dict:
        """Get comprehensive environment information."""
        return get_environment_info()

    def __repr__(self) -> str:
        """String representation of config."""
        return f"HunyoConfig(dev_mode={self.is_development_mode}, data_dir={self.data_directory})"


# Environment detection for debugging
if __name__ == "__main__":
    """CLI for testing configuration detection."""
    import json

    mcp_logger.config("Hunyo MCP Server Configuration")
    mcp_logger.config("=" * 50)

    env_info = get_environment_info()
    mcp_logger.config(json.dumps(env_info, indent=2))

    mcp_logger.file_op("Directory Structure")
    mcp_logger.config("=" * 50)

    try:
        ensure_directory_structure()
        mcp_logger.success("Directory structure created successfully")

        # Show the structure
        data_dir = get_hunyo_data_dir()
        mcp_logger.file_op(f"Data directory: {data_dir}")
        for item in data_dir.rglob("*"):
            if item.is_dir():
                mcp_logger.file_op(f"  {item.relative_to(data_dir)}/")

    except Exception as e:
        mcp_logger.error(f"Error creating directory structure: {e}")
