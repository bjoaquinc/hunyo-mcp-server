"""
Base environment detection system for hunyo-capture.

Extracted from existing get_user_data_dir() logic and enhanced with
notebook environment detection capabilities.
"""

from enum import Enum
from pathlib import Path

from ..constants import (
    CONFIG_DIR_NAME,
    DATABASE_DIR_NAME,
    DATAFRAME_LINEAGE_DIR_NAME,
    DEV_MARKERS,
    EVENTS_DIR_NAME,
    LINEAGE_DIR_NAME,
    PROJECT_MARKERS,
    RUNTIME_DIR_NAME,
    get_dev_mode_config,
)


class NotebookEnvironment(Enum):
    """Supported notebook environments."""

    MARIMO = "marimo"
    JUPYTER = "jupyter"
    UNKNOWN = "unknown"


class DeploymentMode(Enum):
    """Deployment modes for data directory selection."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class EnvironmentDetector:
    """
    Environment detection system extracted from existing get_user_data_dir() logic.

    Provides both deployment mode detection (development/production) and
    notebook environment detection (Marimo/Jupyter/Unknown).
    """

    @staticmethod
    def detect_environment() -> NotebookEnvironment:
        """
        Detect current notebook environment.

        Returns:
            NotebookEnvironment: Detected environment (MARIMO, JUPYTER, or UNKNOWN)
        """
        try:
            # Check for Marimo (existing detection logic)
            import marimo  # noqa: F401

            # Validate marimo hooks are available
            from marimo._runtime.runner.hooks import PRE_EXECUTION_HOOKS  # noqa: F401

            return NotebookEnvironment.MARIMO
        except ImportError:
            pass

        try:
            # Check for Jupyter/IPython environment
            from IPython import get_ipython

            if get_ipython() is not None:
                return NotebookEnvironment.JUPYTER
        except ImportError:
            pass

        return NotebookEnvironment.UNKNOWN

    @staticmethod
    def validate_environment(environment: NotebookEnvironment) -> bool:
        """
        Validate that environment is available and functional.

        Args:
            environment: Environment to validate

        Returns:
            bool: True if environment is available and functional
        """
        if environment == NotebookEnvironment.MARIMO:
            try:
                from marimo._runtime.runner.hooks import (
                    PRE_EXECUTION_HOOKS,  # noqa: F401
                )

                return True
            except ImportError:
                return False
        elif environment == NotebookEnvironment.JUPYTER:
            try:
                from IPython import get_ipython

                return get_ipython() is not None
            except ImportError:
                return False
        return False

    @staticmethod
    def detect_deployment_mode() -> DeploymentMode:
        """
        Detect deployment mode (development/production).

        Extracted from existing get_user_data_dir() logic.

        Returns:
            DeploymentMode: DEVELOPMENT or PRODUCTION
        """
        # Method 1: Check for environment variable override
        dev_mode_config = get_dev_mode_config()
        if dev_mode_config is not None:
            return (
                DeploymentMode.DEVELOPMENT
                if dev_mode_config
                else DeploymentMode.PRODUCTION
            )

        # Method 2: Auto-detect based on development indicators
        current_dir = Path.cwd()
        for parent in [current_dir, *list(current_dir.parents)]:
            # Check for common development indicators
            if any((parent / marker).exists() for marker in DEV_MARKERS):
                return DeploymentMode.DEVELOPMENT

        return DeploymentMode.PRODUCTION

    @staticmethod
    def get_project_root() -> Path:
        """
        Get project root directory for development mode.

        Extracted from existing get_user_data_dir() logic.

        Returns:
            Path: Project root directory
        """
        current_dir = Path.cwd()
        project_root = current_dir

        # Look for project markers going up the directory tree
        for parent in [current_dir, *list(current_dir.parents)]:
            if any((parent / marker).exists() for marker in PROJECT_MARKERS):
                project_root = parent
                break

        return project_root

    @staticmethod
    def get_data_directory() -> Path:
        """
        Get data directory based on deployment mode.

        Maintains exact same behavior as existing get_user_data_dir().

        Returns:
            Path: Data directory (.hunyo)
        """
        deployment_mode = EnvironmentDetector.detect_deployment_mode()

        if deployment_mode == DeploymentMode.DEVELOPMENT:
            # Development mode: use project root .hunyo
            project_root = EnvironmentDetector.get_project_root()
            from ..constants import DATA_DIR_NAME

            data_dir = project_root / DATA_DIR_NAME
        else:
            # Production mode: use home directory .hunyo
            from ..constants import DATA_DIR_NAME

            data_dir = Path.home() / DATA_DIR_NAME

        return data_dir

    @staticmethod
    def ensure_data_directory_structure(data_dir: Path) -> None:
        """
        Ensure data directory structure exists.

        Extracted from existing get_user_data_dir() logic.

        Args:
            data_dir: Base data directory
        """
        # Create directory structure (same as existing get_user_data_dir)
        (data_dir / EVENTS_DIR_NAME / RUNTIME_DIR_NAME).mkdir(
            parents=True, exist_ok=True
        )
        (data_dir / EVENTS_DIR_NAME / LINEAGE_DIR_NAME).mkdir(
            parents=True, exist_ok=True
        )
        (data_dir / EVENTS_DIR_NAME / DATAFRAME_LINEAGE_DIR_NAME).mkdir(
            parents=True, exist_ok=True
        )
        (data_dir / DATABASE_DIR_NAME).mkdir(parents=True, exist_ok=True)
        (data_dir / CONFIG_DIR_NAME).mkdir(parents=True, exist_ok=True)
