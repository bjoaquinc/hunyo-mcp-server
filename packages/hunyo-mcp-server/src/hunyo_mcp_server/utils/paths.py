"""
Cross-platform path utilities following DuckDB best practices.

Provides robust path handling for Windows, Linux, and macOS environments
with proper Unicode support, long path handling, and permission management.
"""

import os
import platform
from pathlib import Path

try:
    from hunyo_mcp_server.logger import get_logger

    path_logger = get_logger("hunyo.utils.paths")
except ImportError:
    # Fallback for testing (silent for style compliance)
    class SimpleLogger:
        def config(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            pass

    path_logger = SimpleLogger()


# Constants for cross-platform path handling
WINDOWS_MAX_PATH_LENGTH = 260  # Windows long path limit


def normalize_database_path(path: str) -> str:
    """
    Normalize database path across platforms following DuckDB best practices.

    Handles Windows long path support and proper path resolution for all platforms.

    Args:
        path: Database path (can be ':memory:' or file path)

    Returns:
        str: Normalized absolute path with platform-specific optimizations
    """
    if path in [":memory:", None]:
        return ":memory:"

    # Convert to absolute path
    db_path = Path(path).resolve()

    # Windows long path handling (DuckDB requirement for paths > WINDOWS_MAX_PATH_LENGTH chars)
    if platform.system() == "Windows" and len(str(db_path)) > WINDOWS_MAX_PATH_LENGTH:
        normalized_path = f"\\\\?\\{db_path}"
        path_logger.config(f"[WINDOWS] Applied long path prefix: {normalized_path}")
        return normalized_path

    return str(db_path)


def get_cross_platform_temp_dir() -> str:
    """
    Get platform-appropriate temporary directory following DuckDB guidelines.

    Returns:
        str: Platform-optimized temporary directory path
    """
    system = platform.system().lower()

    if system == "windows":
        # Windows: Use system temp or fallback to C:\\temp
        temp_dir = os.environ.get("TEMP", "C:\\temp\\hunyo")
        path_logger.config(f"[WINDOWS] Using temp directory: {temp_dir}")
        return temp_dir
    elif system == "darwin":  # macOS
        # macOS: Respect system temp with fallback
        temp_dir = os.environ.get("TMPDIR", "/tmp/hunyo")  # noqa: S108
        path_logger.config(f"[MACOS] Using temp directory: {temp_dir}")
        return temp_dir
    else:  # Linux and other Unix
        # Linux: Use /tmp with proper permissions
        temp_dir = "/tmp/hunyo"  # noqa: S108
        path_logger.config(f"[LINUX] Using temp directory: {temp_dir}")
        return temp_dir


def get_project_root() -> Path:
    """
    Find the project root by looking for .git directory or pyproject.toml.
    For monorepo structure, prefers the root containing schemas/ directory.

    Returns:
        Path: Project root directory

    Raises:
        RuntimeError: If project root cannot be found
    """
    # Use module's __file__ attribute to allow mocking in tests
    current_file = Path(globals()["__file__"]).resolve()

    # Walk up the directory tree - collect all potential roots
    candidate_roots = []
    for parent in [current_file, *list(current_file.parents)]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            candidate_roots.append(parent)

    # Check cwd as fallback
    cwd = Path.cwd()
    if (cwd / ".git").exists() or (cwd / "pyproject.toml").exists():
        candidate_roots.append(cwd)

    if not candidate_roots:
        msg = "Could not find project root (no .git or pyproject.toml found)"
        raise RuntimeError(msg)

    # Prefer the root that contains schemas/ directory (monorepo root)
    for root in candidate_roots:
        if (root / "schemas").exists():
            return root

    # Fallback: return the highest level root found
    return candidate_roots[0]


def get_schema_path(schema_name: str) -> Path:
    """
    Get absolute path to JSON schema file with cross-platform resolution.

    Args:
        schema_name: Name of schema file (e.g., 'runtime_events_schema.json')

    Returns:
        Path: Absolute path to schema file

    Raises:
        FileNotFoundError: If schema file doesn't exist
    """
    project_root = get_project_root()
    schema_path = project_root / "schemas" / "json" / schema_name

    if not schema_path.exists():
        msg = f"Schema file not found: {schema_path}"
        path_logger.error(f"[ERROR] {msg}")
        raise FileNotFoundError(msg)

    path_logger.config(f"[SCHEMA] Resolved schema path: {schema_path}")
    return schema_path


def setup_cross_platform_directories(base_path: str) -> dict[str, str]:
    """
    Setup required directories with cross-platform permissions following DuckDB practices.

    Args:
        base_path: Base directory path

    Returns:
        Dict[str, str]: Mapping of directory names to their absolute paths
    """
    base = Path(base_path).resolve()

    directories = {
        "data": base / "data",
        "temp": base / "temp",
        "logs": base / "logs",
        "backup": base / "backup",
        "events": base / "events",
        "database": base / "database",
        "config": base / "config",
    }

    system = platform.system()
    created_dirs = {}

    for name, dir_path in directories.items():
        try:
            dir_path.mkdir(parents=True, exist_ok=True)

            # Set appropriate permissions (Unix only, as per DuckDB best practices)
            if system != "Windows":
                try:
                    os.chmod(dir_path, 0o755)  # noqa: S103
                    path_logger.config(f"[{system.upper()}] Created {name}: {dir_path}")
                except (OSError, NotImplementedError):
                    # Some filesystems don't support chmod, skip silently
                    path_logger.warning(
                        f"[{system.upper()}] Could not set permissions for {dir_path}"
                    )
            else:
                path_logger.config(f"[WINDOWS] Created {name}: {dir_path}")

            created_dirs[name] = str(dir_path)

        except (OSError, PermissionError) as e:
            path_logger.error(f"[ERROR] Failed to create {name} directory: {e}")
            # Continue with other directories even if one fails
            continue

    return created_dirs


def get_safe_temp_database_path(suffix: str = "") -> str:
    """
    Generate a safe temporary database path for testing across platforms.

    Args:
        suffix: Optional suffix for the database name

    Returns:
        str: Cross-platform safe temporary database path
    """
    temp_dir = get_cross_platform_temp_dir()

    # Ensure temp directory exists
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Generate safe filename
    if suffix:
        db_name = f"test_hunyo_{suffix}.duckdb"
    else:
        db_name = "test_hunyo.duckdb"

    temp_db_path = Path(temp_dir) / db_name
    return normalize_database_path(str(temp_db_path))


def validate_path_accessibility(path: str) -> bool:
    """
    Validate that a path is accessible for DuckDB operations.

    Args:
        path: Path to validate

    Returns:
        bool: True if path is accessible, False otherwise
    """
    if path == ":memory:":
        return True

    try:
        path_obj = Path(path)

        # Check if parent directory exists and is writable
        parent = path_obj.parent
        if not parent.exists():
            return False

        # Check write permissions on parent directory
        return os.access(parent, os.R_OK | os.W_OK)

    except (OSError, ValueError):
        return False
