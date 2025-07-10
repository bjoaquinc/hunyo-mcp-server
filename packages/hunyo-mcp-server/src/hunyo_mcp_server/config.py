"""
Configuration management for Hunyo MCP Server.

Handles data directory resolution, environment detection, and path management
for both development and production environments.
"""

import hashlib
import os
from pathlib import Path
from typing import Any

# Import cross-platform path utilities
from hunyo_mcp_server.utils.paths import (
    normalize_database_path,
    setup_cross_platform_directories,
)

# Import logging utility
try:
    from hunyo_mcp_server.logger import get_logger

    mcp_logger = get_logger("hunyo.mcp")
except ImportError:
    # Fallback for testing (silent for style compliance)
    class SimpleLogger:
        def status(self, msg: str) -> None:
            pass

        def success(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            pass

        def config(self, msg: str) -> None:
            pass

        def file_op(self, msg: str) -> None:
            pass

        def debug(self, msg: str) -> None:
            pass

    mcp_logger = SimpleLogger()


def get_hunyo_data_dir() -> Path:
    """
    Returns the appropriate data directory for Hunyo events and database.

    Logic:
    1. Check HUNYO_DATA_DIR environment variable (for testing/custom paths)
    2. Development mode: .hunyo in repository root
    3. Production mode: ~/.hunyo in user home directory

    Returns:
        Path: The data directory path
    """
    # Environment variable override (for testing or custom installations)
    env_data_dir = os.environ.get("HUNYO_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir)

    if is_development_mode():
        # Development: use .hunyo in repository root
        # Check if HUNYO_DEV_MODE is explicitly set (for testing)
        env_dev_mode = os.environ.get("HUNYO_DEV_MODE")
        if env_dev_mode is not None:
            # Use current working directory when explicitly set for testing
            return Path.cwd() / ".hunyo"
        else:
            # Use repository root for normal development
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
        return env_dev_mode.lower() in {"1", "true", "yes", "on"}

    # Check if we're in a repository
    try:
        repo_root = get_repository_root()

        # Look for development indicators
        has_git = (repo_root / ".git").exists()
        has_pyproject = (repo_root / "pyproject.toml").exists()

        # Check for monorepo structure (packages/hunyo-mcp-server/src/hunyo_mcp_server)
        # or standard structure (src/hunyo_mcp_server)
        has_src_dir = (
            repo_root / "packages" / "hunyo-mcp-server" / "src" / "hunyo_mcp_server"
        ).exists() or (repo_root / "src" / "hunyo_mcp_server").exists()

        return has_git and has_pyproject and has_src_dir

    except Exception:
        # If we can't determine, assume production
        return False


def get_repository_root() -> Path:
    """
    Find the repository root by looking for .git directory or pyproject.toml.
    For monorepo structure, prefers the root containing schemas/ directory.

    Returns:
        Path: Repository root directory

    Raises:
        RuntimeError: If repository root cannot be found
    """
    current = Path(__file__).resolve()
    mcp_logger.config(f"[ROOT] Starting from: {current}")

    # Walk up the directory tree - collect all potential roots
    candidate_roots = []
    for i, parent in enumerate([current, *list(current.parents)]):
        mcp_logger.config(f"[ROOT] Checking {i}: {parent}")
        git_exists = (parent / ".git").exists()
        pyproject_exists = (parent / "pyproject.toml").exists()
        mcp_logger.config(
            f"[ROOT] .git exists: {git_exists}, pyproject.toml exists: {pyproject_exists}"
        )

        if git_exists or pyproject_exists:
            candidate_roots.append(parent)

    # Check cwd as fallback
    cwd = Path.cwd()
    mcp_logger.config(f"[ROOT] Fallback to cwd: {cwd}")
    git_exists = (cwd / ".git").exists()
    pyproject_exists = (cwd / "pyproject.toml").exists()
    mcp_logger.config(
        f"[ROOT] CWD .git exists: {git_exists}, pyproject.toml exists: {pyproject_exists}"
    )

    if git_exists or pyproject_exists:
        candidate_roots.append(cwd)

    if not candidate_roots:
        mcp_logger.error(
            f"[ROOT] Repository root not found. Current: {current}, CWD: {cwd}"
        )
        msg = "Could not find repository root (no .git or pyproject.toml found)"
        raise RuntimeError(msg)

    # Prefer the root that contains schemas/ directory (monorepo root)
    for root in candidate_roots:
        if (root / "schemas").exists():
            mcp_logger.config(f"[ROOT] Found monorepo root with schemas: {root}")
            return root

    # Fallback: return the highest level root found
    root = candidate_roots[0]
    mcp_logger.config(f"[ROOT] Using first found root: {root}")
    return root


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


def get_notebook_file_hash(notebook_path: str | Path) -> str:
    """
    Generate 8-character SHA256 hash from notebook file path.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        str: 8-character hash for notebook identification
    """
    # Enhanced logging for hash calculation debugging
    mcp_logger.config("[HASH] Hash calculation started")
    mcp_logger.config(f"[HASH] Input path: {notebook_path}")
    mcp_logger.config(f"[HASH] Input path type: {type(notebook_path)}")

    # Convert to Path object first
    path_obj = Path(notebook_path)
    mcp_logger.config(f"[HASH] Path object: {path_obj}")
    mcp_logger.config(f"[HASH] Path exists: {path_obj.exists()}")
    mcp_logger.config(f"[HASH] Path is absolute: {path_obj.is_absolute()}")

    # Resolve the path
    resolved_path = path_obj.resolve()
    mcp_logger.config(f"[HASH] Resolved path: {resolved_path}")

    # Convert to string and encode
    path_str = str(resolved_path)
    mcp_logger.config(f"[HASH] Path string: {path_str}")
    mcp_logger.config(f"[HASH] Path string length: {len(path_str)}")

    path_bytes = path_str.encode("utf-8")
    mcp_logger.config(f"[HASH] Path bytes length: {len(path_bytes)}")

    # Generate hash
    full_hash = hashlib.sha256(path_bytes).hexdigest()
    truncated_hash = full_hash[:8]

    mcp_logger.config(f"[HASH] Full hash: {full_hash}")
    mcp_logger.config(f"[HASH] Truncated hash: {truncated_hash}")
    mcp_logger.config("[HASH] Hash calculation complete")

    return truncated_hash


def get_database_path(notebook_hash: str | None = None) -> Path:
    """
    Get the DuckDB database file path with cross-platform normalization.

    Args:
        notebook_hash: Optional notebook hash for notebook-specific database

    Returns:
        Path: Normalized database file path
    """
    data_dir = get_hunyo_data_dir()

    if notebook_hash:
        db_name = f"{notebook_hash}.duckdb"
    else:
        db_name = "lineage.duckdb"  # Keep existing default for backward compatibility

    db_path = data_dir / "database" / db_name
    normalized_path = normalize_database_path(str(db_path))
    return Path(normalized_path)


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
    Ensure all necessary directories exist with cross-platform permissions.

    Creates:
    - Main data directory (.hunyo)
    - Events subdirectories (runtime, lineage, dataframe_lineage)
    - Database directory
    - Config directory

    Uses cross-platform directory setup for proper permissions.
    """
    data_dir = get_hunyo_data_dir()

    # Use cross-platform directory setup utility
    setup_cross_platform_directories(str(data_dir))  # Create base directories

    # Create additional event subdirectories not covered by base setup
    event_subdirs = [
        data_dir / "events" / "runtime",
        data_dir / "events" / "lineage",
        data_dir / "events" / "dataframe_lineage",
    ]

    for subdir in event_subdirs:
        subdir.mkdir(parents=True, exist_ok=True)

        # Apply cross-platform permissions
        try:
            if os.name != "nt":  # Not Windows
                subdir.chmod(0o755)
        except (OSError, NotImplementedError):
            # Some filesystems don't support chmod, skip silently
            pass

    mcp_logger.config(f"[OK] Directory structure created at: {data_dir}")

    # Additional specific setup for DataFrame lineage
    if is_dataframe_lineage_enabled():
        ensure_dataframe_lineage_directory()


def get_event_file_path(event_type: str, notebook_path: str | None = None) -> Path:
    """
    Get the full path for an event file.

    Args:
        event_type: 'runtime', 'lineage', or 'dataframe_lineage'
        notebook_path: Optional notebook path for unique naming

    Returns:
        Path: Full path to the event file

    Raises:
        ValueError: If event_type is not supported
    """
    if event_type not in {"runtime", "lineage", "dataframe_lineage"}:
        msg = f"event_type must be 'runtime', 'lineage', or 'dataframe_lineage', got: {event_type}"
        raise ValueError(msg)

    data_dir = get_hunyo_data_dir()
    events_dir = data_dir / "events" / event_type

    if notebook_path:
        # Create unique filename based on notebook with hash prefix for compatibility
        notebook_hash = get_notebook_file_hash(notebook_path)
        notebook_name = Path(notebook_path).stem
        filename = f"{notebook_hash}_{notebook_name}_{event_type}_events.jsonl"
    else:
        # Default filename
        filename = f"{event_type}_events.jsonl"

    return events_dir / filename


def get_dataframe_lineage_config() -> dict[str, Any]:
    """
    Get DataFrame lineage configuration settings from environment variables.

    Returns:
        dict: DataFrame lineage configuration with the following keys:
            - enabled: Whether DataFrame lineage tracking is enabled
            - sample_large_dataframes: Whether to sample large DataFrames
            - size_threshold_mb: Size threshold for sampling (MB)
            - sample_rate: Sampling rate for large DataFrames (0.0-1.0)
            - max_overhead_ms: Maximum acceptable overhead per operation (ms)
    """
    try:
        return {
            "enabled": os.environ.get("HUNYO_TRACK_DATAFRAME_LINEAGE", "true").lower()
            == "true",
            "sample_large_dataframes": os.environ.get(
                "HUNYO_SAMPLE_LARGE_DF", "true"
            ).lower()
            == "true",
            "size_threshold_mb": float(
                os.environ.get("HUNYO_DF_SIZE_THRESHOLD", "10.0")
            ),
            "sample_rate": float(os.environ.get("HUNYO_DF_SAMPLE_RATE", "0.1")),
            "max_overhead_ms": float(os.environ.get("HUNYO_DF_MAX_OVERHEAD", "5.0")),
        }
    except (ValueError, TypeError) as e:
        mcp_logger.warning(f"[CONFIG] Invalid DataFrame lineage config value: {e}")
        # Return safe defaults
        return {
            "enabled": True,
            "sample_large_dataframes": True,
            "size_threshold_mb": 10.0,
            "sample_rate": 0.1,
            "max_overhead_ms": 5.0,
        }


def is_dataframe_lineage_enabled() -> bool:
    """
    Check if DataFrame lineage tracking is enabled.

    Returns:
        bool: True if DataFrame lineage tracking is enabled, False otherwise
    """
    return get_dataframe_lineage_config()["enabled"]


def get_dataframe_lineage_events_dir() -> Path:
    """
    Get the DataFrame lineage events directory.

    Returns:
        Path: Directory path for DataFrame lineage events
    """
    data_dir = get_hunyo_data_dir()
    return data_dir / "events" / "dataframe_lineage"


def ensure_dataframe_lineage_directory() -> None:
    """
    Ensure the DataFrame lineage events directory exists.

    Creates the directory with proper cross-platform permissions.
    """
    df_lineage_dir = get_dataframe_lineage_events_dir()
    df_lineage_dir.mkdir(parents=True, exist_ok=True)

    # Apply cross-platform permissions
    try:
        if os.name != "nt":  # Not Windows
            df_lineage_dir.chmod(0o755)
    except (OSError, NotImplementedError):
        # Some filesystems don't support chmod, skip silently
        pass

    mcp_logger.config(f"[OK] DataFrame lineage directory created: {df_lineage_dir}")


def get_environment_info() -> dict[str, Any]:
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
        df_lineage_dir = get_dataframe_lineage_events_dir()
        df_lineage_config = get_dataframe_lineage_config()

        return {
            "development_mode": is_development_mode(),
            "data_directory": str(data_dir),
            "database_path": str(db_path),
            "config_path": str(config_path),
            "runtime_events_dir": str(runtime_dir),
            "lineage_events_dir": str(lineage_dir),
            "dataframe_lineage_dir": str(df_lineage_dir),
            "data_dir_exists": data_dir.exists(),
            "database_exists": db_path.exists(),
            "config_exists": config_path.exists(),
            "dataframe_lineage_config": df_lineage_config,
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

    def __init__(self) -> None:
        """Initialize configuration with current environment detection."""
        self._ensure_directories()

    def _ensure_directories(self) -> None:
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

    @property
    def dataframe_lineage_dir(self) -> Path:
        """Get the DataFrame lineage events directory."""
        return get_dataframe_lineage_events_dir()

    @property
    def dataframe_lineage_config(self) -> dict[str, Any]:
        """Get DataFrame lineage configuration settings."""
        return get_dataframe_lineage_config()

    @property
    def is_dataframe_lineage_enabled(self) -> bool:
        """Check if DataFrame lineage tracking is enabled."""
        return is_dataframe_lineage_enabled()

    def get_event_file_path(
        self, event_type: str, notebook_path: str | None = None
    ) -> Path:
        """Get the full path for an event file."""
        return get_event_file_path(event_type, notebook_path)

    def get_environment_info(self) -> dict[str, Any]:
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
