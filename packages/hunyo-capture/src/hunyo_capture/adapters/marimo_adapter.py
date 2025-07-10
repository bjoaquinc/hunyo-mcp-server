"""
Marimo-specific execution context adapter.

Extracted from existing UnifiedMarimoInterceptor execution context logic
in _create_pre_execution_hook() and _create_post_execution_hook() methods.
"""

from typing import Any

from hunyo_capture.logger import get_logger

from .base import ExecutionContextAdapter

unified_logger = get_logger("hunyo_capture.adapters.marimo")


class MarimoContextAdapter(ExecutionContextAdapter):
    """
    Marimo-specific execution context adapter.

    Extracted from existing UnifiedMarimoInterceptor execution context logic
    to provide Marimo-specific cell handling while preserving exact behavior.
    """

    def extract_cell_info(
        self, cell: Any, runner: Any = None  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Extract cell information from Marimo cell object.

        Extracted from existing logic in pre_execution_hook and post_execution_hook:
        cell_id = cell.cell_id
        cell_code = cell.code

        Args:
            cell: Marimo cell object
            runner: Marimo runner object (unused but kept for interface compatibility)

        Returns:
            dict[str, Any]: Dictionary with cell_id and cell_code keys

        Raises:
            AttributeError: If cell object doesn't have expected Marimo attributes
        """
        try:
            # Extract cell information from marimo - exact logic from existing code
            cell_id = cell.cell_id
            cell_code = cell.code

            return {
                "cell_id": cell_id,
                "cell_code": cell_code,
            }

        except AttributeError as e:
            unified_logger.error(
                f"[ERROR] Failed to extract cell info from Marimo cell: {e}"
            )
            raise
        except Exception as e:
            unified_logger.error(f"[ERROR] Unexpected error extracting cell info: {e}")
            raise

    def detect_error(self, run_result: Any = None) -> Exception | None:
        """
        Detect execution errors from Marimo run result.

        Extracted from existing logic in post_execution_hook:
        error = getattr(run_result, "exception", None) if run_result else None

        Args:
            run_result: Marimo execution result object

        Returns:
            Exception | None: Exception if error occurred, None otherwise
        """
        try:
            # Check for errors - exact logic from existing post_execution_hook
            error = getattr(run_result, "exception", None) if run_result else None

            return error

        except Exception as e:
            unified_logger.error(f"[ERROR] Failed to detect error from run result: {e}")
            # Return None to avoid cascading errors in error detection
            return None
