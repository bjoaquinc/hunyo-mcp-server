"""
Abstract execution context adapter for notebook environments.

Extracted from existing UnifiedMarimoInterceptor execution context logic
to provide a generic interface for different notebook environments.
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any


class ExecutionContextAdapter(ABC):
    """
    Abstract base class for execution context adapters.

    Extracted from the existing Marimo execution context logic in
    _create_pre_execution_hook() and _create_post_execution_hook() methods.

    Provides standard interface for:
    - Cell information extraction (cell_id, cell_code)
    - Error detection from execution results
    - Execution ID generation
    - Context creation and retrieval
    """

    @abstractmethod
    def extract_cell_info(self, cell: Any, runner: Any = None) -> dict[str, Any]:
        """
        Extract cell information from environment-specific cell object.

        Extracted from existing logic:
        cell_id = cell.cell_id
        cell_code = cell.code

        Args:
            cell: Environment-specific cell object
            runner: Environment-specific runner object (optional)

        Returns:
            dict[str, Any]: Dictionary with cell_id and cell_code keys
        """
        pass

    @abstractmethod
    def detect_error(self, run_result: Any = None) -> Exception | None:
        """
        Detect execution errors from environment-specific run result.

        Extracted from existing logic:
        error = getattr(run_result, "exception", None) if run_result else None

        Args:
            run_result: Environment-specific execution result object

        Returns:
            Optional[Exception]: Exception if error occurred, None otherwise
        """
        pass

    def generate_execution_id(self) -> str:
        """
        Generate unique execution ID.

        Extracted from existing logic:
        execution_id = str(uuid.uuid4())[:8]

        Returns:
            str: Unique execution ID (8 characters)
        """
        return str(uuid.uuid4())[:8]

    def create_execution_context(
        self, cell_id: str, cell_code: str, execution_id: str
    ) -> dict[str, Any]:
        """
        Create execution context dictionary.

        Extracted from existing logic in pre_execution_hook:
        context_data = {
            "execution_id": execution_id,
            "start_time": time.time(),
            "cell_code": cell_code,
            "cell_id": cell_id,
            "context_created_at": time.time(),
        }

        Args:
            cell_id: Unique cell identifier
            cell_code: Cell source code
            execution_id: Unique execution identifier

        Returns:
            Dict[str, Any]: Execution context dictionary
        """
        current_time = time.time()
        return {
            "execution_id": execution_id,
            "start_time": current_time,
            "cell_code": cell_code,
            "cell_id": cell_id,
            "context_created_at": current_time,
        }

    def calculate_duration_ms(self, context: dict[str, Any]) -> float:
        """
        Calculate execution duration in milliseconds.

        Extracted from existing logic in post_execution_hook:
        from ..constants import MILLISECONDS_CONVERSION_FACTOR
        duration_ms = (time.time() - start_time) * MILLISECONDS_CONVERSION_FACTOR

        Args:
            context: Execution context dictionary with start_time

        Returns:
            float: Duration in milliseconds
        """
        start_time = context.get("start_time", time.time())
        return (time.time() - start_time) * 1000

    def extract_error_info(self, error: Exception) -> dict[str, Any]:
        """
        Extract error information for event generation.

        Extracted from existing logic in post_execution_hook:
        event_data["error_info"] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        Args:
            error: Exception that occurred during execution

        Returns:
            Dict[str, Any]: Error information dictionary
        """
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    def get_cell_source_lines(self, cell_code: str) -> int:
        """
        Get number of lines in cell source code.

        Extracted from existing logic:
        "cell_source_lines": len(cell_code.split("\n")) if cell_code else 0

        Args:
            cell_code: Cell source code

        Returns:
            int: Number of lines in source code
        """
        return len(cell_code.split("\n")) if cell_code else 0
