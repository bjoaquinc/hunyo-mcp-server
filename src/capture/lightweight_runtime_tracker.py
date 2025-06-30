#!/usr/bin/env python3
"""
Lightweight Marimo Runtime Tracker - Links execution context to DataFrame lineage events
Focuses on timing, errors, and execution flow - DataFrame analysis handled by lineage system
"""

import json
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import psutil

# Import utilities and logger
try:
    from . import (
        get_event_filenames,
        get_notebook_file_hash,
        get_notebook_name,
        get_user_data_dir,
    )
    from .logger import runtime_logger
except ImportError:
    # Fallback for standalone testing
    def get_event_filenames(notebook_path, data_dir):
        name = Path(notebook_path).stem
        return f"{name}_runtime.jsonl", f"{name}_lineage.jsonl"
    
    def get_notebook_file_hash(notebook_path):
        return "test_hash"
    
    def get_notebook_name(notebook_path):
        return Path(notebook_path).stem
    
    def get_user_data_dir():
        return str(Path.home() / ".hunyo")

    # Create a simple fallback logger (silent for style compliance)
    class SimpleLogger:
        def status(self, msg): pass
        def success(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass
        def config(self, msg): pass
        def runtime(self, msg): pass
        def tracking(self, msg): pass
        def debug(self, msg): pass
        def startup(self, msg): pass
        def notebook(self, msg): pass
        def file_op(self, msg): pass
    
    runtime_logger = SimpleLogger()


class LightweightRuntimeTracker:
    """Lightweight execution tracking that links to DataFrame lineage events"""

    def __init__(self, notebook_path: str = None, output_file: str = None):
        # Determine notebook path and create unique file names
        if notebook_path is None:
            # Try to detect from call stack
            notebook_path = self._detect_notebook_path()

        self.notebook_path = notebook_path
        self.notebook_hash = (
            get_notebook_file_hash(notebook_path) if notebook_path else "unknown"
        )
        self.notebook_name = (
            get_notebook_name(notebook_path) if notebook_path else "unknown"
        )

        # Set up data directory and file paths
        self.data_dir = get_user_data_dir()

        if output_file is None:
            if notebook_path:
                self.output_file, _ = get_event_filenames(notebook_path, self.data_dir)
            else:
                # Fallback to old behavior
                self.output_file = os.path.join(
                    self.data_dir, "events", "runtime", "marimo_runtime.jsonl"
                )
        else:
            self.output_file = output_file

        self.session_id = str(uuid.uuid4())[:8]
        self._lock = threading.Lock()

        # Execution tracking
        self.cell_executions = {}
        self.execution_stack = []

        # System monitoring (lightweight)
        try:
            self.process = psutil.Process()
        except:
            self.process = None

        runtime_logger.status("Lightweight Runtime Tracker v2.0")
        runtime_logger.notebook(f"Notebook: {self.notebook_name} ({self.notebook_hash})")
        runtime_logger.file_op(f"Runtime logs: {self.output_file}")
        runtime_logger.config(f"Session: {self.session_id}")

    def _detect_notebook_path(self) -> str:
        """Attempt to detect the notebook path from the call stack or environment"""
        # Try to get from environment variables first
        if "MARIMO_NOTEBOOK_PATH" in os.environ:
            return os.environ["MARIMO_NOTEBOOK_PATH"]

        # Try to detect from call stack
        frame = sys._getframe()
        while frame:
            filename = frame.f_code.co_filename
            if filename.endswith(".py") and not filename.endswith("__init__.py"):
                # Check if this looks like a notebook file
                if any(
                    keyword in filename.lower()
                    for keyword in ["notebook", "marimo", "analysis"]
                ):
                    return filename
            frame = frame.f_back

        # Fallback to current working directory + generic name
        return os.path.join(os.getcwd(), "marimo_notebook.py")

    def start_cell_execution(self, cell_id: str, cell_source: str) -> str:
        """Start tracking a cell execution - lightweight version"""
        execution_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Lightweight system state
        memory_mb = 0
        if self.process:
            try:
                memory_mb = round(self.process.memory_info().rss / 1024 / 1024, 2)
            except:
                pass

        # Store execution context
        with self._lock:
            self.cell_executions[execution_id] = {
                "execution_id": execution_id,
                "cell_id": cell_id,
                "cell_source": cell_source,
                "start_time": start_time,
                "start_memory_mb": memory_mb,
            }
            self.execution_stack.append(execution_id)

        # Emit start event
        self._emit_runtime_event(
            {
                "event_type": "cell_execution_start",
                "execution_id": execution_id,
                "cell_id": cell_id,
                "cell_source": cell_source,
                "cell_source_lines": cell_source.count("\n") + 1,
                "start_memory_mb": memory_mb,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return execution_id

    def end_cell_execution(
        self,
        execution_id: str,
        success: bool = True,
        error: Exception = None,
        result: Any = None,
    ) -> dict:
        """End tracking a cell execution"""
        end_time = time.time()

        with self._lock:
            if execution_id not in self.cell_executions:
                return {}

            exec_ctx = self.cell_executions[execution_id]

            # Remove from execution stack
            if execution_id in self.execution_stack:
                self.execution_stack.remove(execution_id)

        # Calculate duration and memory change
        duration = end_time - exec_ctx["start_time"]

        end_memory_mb = 0
        memory_delta_mb = 0
        if self.process:
            try:
                end_memory_mb = round(self.process.memory_info().rss / 1024 / 1024, 2)
                memory_delta_mb = round(
                    end_memory_mb - exec_ctx.get("start_memory_mb", 0), 2
                )
            except:
                pass

        # Capture error context if there was an error
        error_context = None
        if error:
            error_context = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exception(
                    type(error), error, error.__traceback__
                ),
            }

        # Capture result/output (safely)
        result_info = None
        if result is not None:
            try:
                result_info = self._capture_result_safely(result)
            except Exception as e:
                result_info = {
                    "capture_error": str(e),
                    "result_type": str(type(result).__name__),
                }

        # Update execution context
        with self._lock:
            exec_ctx.update(
                {
                    "end_time": end_time,
                    "duration_seconds": duration,
                    "success": success,
                    "error": str(error) if error else None,
                    "end_memory_mb": end_memory_mb,
                    "memory_delta_mb": memory_delta_mb,
                    "error_context": error_context,
                    "result": result_info,
                }
            )

        # Emit completion event
        completion_event = {
            "event_type": "cell_execution_complete",
            "execution_id": execution_id,
            "cell_id": exec_ctx["cell_id"],
            "duration_seconds": round(duration, 3),
            "success": success,
            "end_memory_mb": end_memory_mb,
            "memory_delta_mb": memory_delta_mb,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if error_context:
            completion_event["error_context"] = error_context

        if result_info:
            completion_event["result"] = result_info

        self._emit_runtime_event(completion_event)

        return exec_ctx

    def get_current_execution_id(self) -> str | None:
        """Get the current execution ID for linking with lineage events"""
        with self._lock:
            return self.execution_stack[-1] if self.execution_stack else None

    def _capture_result_safely(self, result: Any) -> dict:
        """Safely capture execution result/output for debugging"""
        try:
            result_info = {
                "type": type(result).__name__,
                "module": getattr(type(result), "__module__", None),
            }

            # Handle different result types
            if result is None:
                result_info["value"] = None
            elif isinstance(result, (str, int, float, bool)):
                # Simple types - capture directly (with length limit for strings)
                if isinstance(result, str) and len(result) > 200:
                    result_info["value"] = result[:200] + "... (truncated)"
                    result_info["full_length"] = len(result)
                else:
                    result_info["value"] = result
            elif isinstance(result, (list, tuple)):
                # Collections - smart handling based on size
                length = len(result)
                if length > 100:  # Large collection - describe only
                    element_types = (
                        list(set(type(x).__name__ for x in result[:10]))
                        if result
                        else []
                    )
                    result_info.update(
                        {
                            "length": length,
                            "element_types": element_types,
                            "description": f"{type(result).__name__} with {length} items (types: {', '.join(element_types)})",
                        }
                    )
                else:  # Small collection - show preview
                    str_repr = str(result)
                    result_info.update(
                        {
                            "length": length,
                            "preview": (
                                str_repr[:200] + "..."
                                if len(str_repr) > 200
                                else str_repr
                            ),
                            "element_types": (
                                list(set(type(x).__name__ for x in result))
                                if result
                                else []
                            ),
                        }
                    )
            elif isinstance(result, dict):
                # Dictionaries - smart handling based on size
                key_count = len(result)
                if key_count > 50:  # Large dictionary - describe only
                    result_info.update(
                        {
                            "key_count": key_count,
                            "keys_preview": list(result.keys())[:5],  # First 5 keys
                            "description": f"Dictionary with {key_count} keys (not stored due to size)",
                        }
                    )
                else:  # Small dictionary - show preview
                    str_repr = str(result)
                    result_info.update(
                        {
                            "key_count": key_count,
                            "keys_preview": list(result.keys()),
                            "preview": (
                                str_repr[:200] + "..."
                                if len(str_repr) > 200
                                else str_repr
                            ),
                        }
                    )
            elif hasattr(result, "shape") and hasattr(result, "columns"):
                # DataFrames - basic description only (detailed analysis is in lineage events)
                df_id = id(result)  # Same ID used by lineage system for linking
                result_info.update(
                    {
                        "is_dataframe": True,
                        "dataframe_id": df_id,  # Key for linking to OpenLineage events
                        "shape": list(result.shape),
                        "columns_count": len(result.columns),
                        "description": f"DataFrame with {result.shape[0]} rows and {result.shape[1]} columns",
                    }
                )
            elif hasattr(result, "shape"):
                # NumPy arrays or similar - describe, don't store data
                shape = list(result.shape)
                dtype = str(getattr(result, "dtype", "unknown"))
                size = result.size if hasattr(result, "size") else "unknown"
                result_info.update(
                    {
                        "is_array": True,
                        "shape": shape,
                        "dtype": dtype,
                        "description": f"Array of shape {shape} with {size} elements ({dtype})",
                    }
                )
            elif callable(result):
                # Functions
                result_info.update(
                    {
                        "is_callable": True,
                        "name": getattr(result, "__name__", "<unknown>"),
                        "file": getattr(result, "__file__", None),
                        "description": f"Function: {getattr(result, '__name__', '<unknown>')}",
                    }
                )
            else:
                # Check for matplotlib/plotly figures and other large objects
                type_name = type(result).__name__
                module_name = getattr(type(result), "__module__", "")

                if "matplotlib" in module_name and "Figure" in type_name:
                    # Matplotlib figure
                    axes_count = len(getattr(result, "axes", []))
                    result_info.update(
                        {
                            "is_plot": True,
                            "plot_type": "matplotlib",
                            "axes_count": axes_count,
                            "description": f"Matplotlib figure with {axes_count} axes",
                        }
                    )
                elif "plotly" in module_name:
                    # Plotly figure
                    result_info.update(
                        {
                            "is_plot": True,
                            "plot_type": "plotly",
                            "description": f"Plotly {type_name} chart",
                        }
                    )
                elif "seaborn" in module_name or "sns" in module_name:
                    # Seaborn plot
                    result_info.update(
                        {
                            "is_plot": True,
                            "plot_type": "seaborn",
                            "description": f"Seaborn {type_name} plot",
                        }
                    )
                elif hasattr(result, "__len__") and len(result) > 1000:
                    # Large collections - describe, don't store
                    length = len(result)
                    result_info.update(
                        {
                            "is_large_collection": True,
                            "length": length,
                            "description": f"Large {type_name} with {length} items (not stored due to size)",
                        }
                    )
                else:
                    # Generic objects - check size before storing string representation
                    try:
                        size_bytes = (
                            sys.getsizeof(result) if hasattr(sys, "getsizeof") else 0
                        )
                        str_repr = str(result)

                        if size_bytes > 10000 or len(str_repr) > 500:  # Large object
                            result_info.update(
                                {
                                    "is_large_object": True,
                                    "size_bytes": size_bytes,
                                    "description": f"{type_name} object ({size_bytes} bytes, not stored due to size)",
                                }
                            )
                        else:
                            # Small object - safe to store representation
                            result_info.update(
                                {
                                    "string_repr": (
                                        str_repr[:200] + "..."
                                        if len(str_repr) > 200
                                        else str_repr
                                    ),
                                    "size_bytes": size_bytes,
                                }
                            )
                    except:
                        result_info.update(
                            {
                                "description": f"{type_name} object (representation unavailable)"
                            }
                        )

            return result_info

        except Exception as e:
            return {
                "capture_failed": True,
                "error": str(e),
                "type": str(type(result).__name__) if result is not None else "None",
            }

    def _emit_runtime_event(self, event: dict):
        """Emit a runtime event to the log file"""
        try:
            event["session_id"] = self.session_id
            event["emitted_at"] = datetime.now(UTC).isoformat()

            with open(self.output_file, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")

        except Exception as e:
            runtime_logger.warning(f"Failed to emit runtime event: {e}")

    def get_session_summary(self) -> dict:
        """Get summary of current session runtime data"""
        with self._lock:
            completed_executions = [
                exec_ctx
                for exec_ctx in self.cell_executions.values()
                if "end_time" in exec_ctx
            ]

            total_execution_time = sum(
                exec_ctx.get("duration_seconds", 0) for exec_ctx in completed_executions
            )

            error_count = sum(
                1
                for exec_ctx in completed_executions
                if not exec_ctx.get("success", True)
            )

            return {
                "session_id": self.session_id,
                "total_cell_executions": len(self.cell_executions),
                "completed_executions": len(completed_executions),
                "active_executions": len(self.execution_stack),
                "total_execution_time_seconds": round(total_execution_time, 3),
                "error_count": error_count,
                "success_rate": round(
                    (len(completed_executions) - error_count)
                    / max(len(completed_executions), 1),
                    2,
                ),
            }


# Global runtime tracker instance
_runtime_tracker: LightweightRuntimeTracker | None = None
_tracker_lock = threading.Lock()


def enable_runtime_tracking(
    notebook_path: str = None, output_file: str = None
) -> LightweightRuntimeTracker:
    """Enable lightweight cell runtime tracking

    Args:
        notebook_path: Path to the notebook file (auto-detected if None)
        output_file: Custom output file path (auto-generated if None)
    """
    global _runtime_tracker

    with _tracker_lock:
        if _runtime_tracker is None:
            _runtime_tracker = LightweightRuntimeTracker(notebook_path, output_file)
        return _runtime_tracker


def get_runtime_tracker() -> LightweightRuntimeTracker | None:
    """Get the current runtime tracker instance"""
    return _runtime_tracker


def get_current_execution_id() -> str | None:
    """Get the current execution ID for linking with lineage events"""
    tracker = get_runtime_tracker()
    return tracker.get_current_execution_id() if tracker else None


def disable_runtime_tracking():
    """Disable runtime tracking"""
    global _runtime_tracker

    with _tracker_lock:
        if _runtime_tracker:
            runtime_logger.config("Runtime tracking disabled")
            _runtime_tracker = None


@contextmanager
def track_cell_execution(cell_source: str, cell_id: str = None):
    """Context manager for tracking a cell execution"""
    tracker = get_runtime_tracker()
    if not tracker:
        yield None
        return

    cell_id = cell_id or str(uuid.uuid4())[:8]
    execution_id = tracker.start_cell_execution(cell_id, cell_source)

    class ExecutionResult:
        def __init__(self, execution_id):
            self.execution_id = execution_id
            self.result = None

        def set_result(self, result):
            """Set the result of the cell execution"""
            self.result = result

    exec_result = ExecutionResult(execution_id)

    try:
        yield exec_result
    except Exception as e:
        tracker.end_cell_execution(
            execution_id, success=False, error=e, result=exec_result.result
        )
        raise
    else:
        tracker.end_cell_execution(
            execution_id, success=True, result=exec_result.result
        )


if __name__ == "__main__":
    # Test the lightweight tracker
    runtime_logger.startup("Testing Lightweight Runtime Tracker")
    runtime_logger.status("=" * 50)

    # Test with explicit notebook path
    test_notebook_path = os.path.join(os.getcwd(), "test_notebook.py")
    tracker = enable_runtime_tracking(notebook_path=test_notebook_path)

    # Test 1: Simple execution with result capture
    runtime_logger.tracking("1. Testing simple execution with result capture...")
    with track_cell_execution("x = 1 + 1") as ctx:
        x = 1 + 1
        ctx.set_result(x)  # Capture the result
        runtime_logger.tracking(f"   Execution ID: {ctx.execution_id}")
    runtime_logger.success(f"   x = {x}")

    # Test 2: DataFrame operations with result capture
    runtime_logger.tracking("2. Testing DataFrame operations with result capture...")
    with track_cell_execution(
        "df = pd.DataFrame({'a': range(100), 'b': range(100)})"
    ) as ctx:
        import pandas as pd

        df = pd.DataFrame({"a": range(100), "b": range(100)})
        ctx.set_result(df)  # Capture the DataFrame result
        runtime_logger.tracking(f"   Execution ID: {ctx.execution_id}")
    runtime_logger.success(f"   DataFrame created: {df.shape}")

    # Test 3: Complex result capture
    runtime_logger.tracking("3. Testing complex result capture...")
    with track_cell_execution("result = {'data': [1,2,3], 'summary': 'test'}") as ctx:
        result = {"data": [1, 2, 3], "summary": "test"}
        ctx.set_result(result)  # Capture complex result
        runtime_logger.tracking(f"   Execution ID: {ctx.execution_id}")
    runtime_logger.success(f"   Complex result: {result}")

    # Test 4: Error handling
    runtime_logger.tracking("4. Testing error handling...")
    try:
        with track_cell_execution("result = undefined_variable + 42") as ctx:
            runtime_logger.tracking(f"   Execution ID: {ctx.execution_id}")
            result = undefined_variable + 42
            ctx.set_result(result)
    except NameError as e:
        runtime_logger.success(f"   Expected error captured: {e}")

    # Summary
    summary = tracker.get_session_summary()
    runtime_logger.runtime("Summary:")
    runtime_logger.runtime(f"   Executions: {summary['total_cell_executions']}")
    runtime_logger.runtime(f"   Success rate: {summary['success_rate']:.1%}")
    runtime_logger.runtime(f"   Total time: {summary['total_execution_time_seconds']:.3f}s")

    runtime_logger.success("Lightweight tracker test completed!")
