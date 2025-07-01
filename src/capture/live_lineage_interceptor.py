#!/usr/bin/env python3
"""
Marimo Live Lineage Interceptor
Zero-configuration live DataFrame tracking for Marimo notebooks

This module provides automatic tracking of DataFrame operations in Marimo notebooks
by monkey-patching pandas operations and intercepting cell execution.
"""

import builtins
import json
import sys
import threading
import uuid
import weakref
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Import runtime tracker, utilities, and logger
try:
    from . import (
        get_event_filenames,
        get_notebook_file_hash,
        get_notebook_name,
        get_user_data_dir,
    )
    from .lightweight_runtime_tracker import (
        enable_runtime_tracking,
        track_cell_execution,
    )
    from .logger import lineage_logger

    RUNTIME_TRACKING_AVAILABLE = True
    lineage_logger.lineage("Runtime tracking integration enabled")
except ImportError:
    RUNTIME_TRACKING_AVAILABLE = False

    # Fallback logger (silent for style compliance)
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

        def lineage(self, msg):
            pass

        def tracking(self, msg):
            pass

        def debug(self, msg):
            pass

        def startup(self, msg):
            pass

        def notebook(self, msg):
            pass

        def file_op(self, msg):
            pass

        def runtime(self, msg):
            pass

    lineage_logger = SimpleLogger()
    lineage_logger.warning(
        "Runtime tracking not available - running with basic lineage only"
    )

# Global tracking state
_tracked_dataframes = weakref.WeakKeyDictionary()
_cell_execution_context = {}
_global_interceptor = None


class MarimoLiveInterceptor:
    """
    Live interceptor for Marimo notebook execution
    Tracks DataFrame operations and provides real-time lineage data
    """

    def __init__(
        self,
        notebook_path: str = None,
        output_file: str = None,
        enable_runtime_debug: bool = True,
    ):
        # Determine notebook path and create unique file names
        if notebook_path is None:
            notebook_path = self._detect_notebook_path()

        self.notebook_path = notebook_path
        self.notebook_hash = (
            get_notebook_file_hash(notebook_path)
            if RUNTIME_TRACKING_AVAILABLE and notebook_path
            else "unknown"
        )
        self.notebook_name = (
            get_notebook_name(notebook_path)
            if RUNTIME_TRACKING_AVAILABLE and notebook_path
            else "unknown"
        )

        # Set up data directory and file paths
        if RUNTIME_TRACKING_AVAILABLE:
            self.data_dir = get_user_data_dir()
            if output_file is None:
                if notebook_path:
                    _, self.output_file = get_event_filenames(
                        notebook_path, self.data_dir
                    )
                else:
                    self.output_file = str(
                        Path(self.data_dir)
                        / "events"
                        / "lineage"
                        / "marimo_live_lineage.jsonl"
                    )
            else:
                self.output_file = output_file
        else:
            self.output_file = output_file or "marimo_live_lineage.jsonl"

        self.output_file = Path(self.output_file)
        self.session_id = str(uuid.uuid4())[:8]
        self.tracked_operations = []
        self.original_functions = {}
        self.interceptor_active = False
        self._lock = threading.Lock()

        # Runtime debugging
        self.enable_runtime_debug = enable_runtime_debug and RUNTIME_TRACKING_AVAILABLE
        self.runtime_tracker = None

        # Ensure output file exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.touch()

        lineage_logger.status("Marimo Live Interceptor v3.0")
        lineage_logger.notebook(
            f"Notebook: {self.notebook_name} ({self.notebook_hash})"
        )
        lineage_logger.file_op(f"Lineage logs: {self.output_file.name}")
        lineage_logger.config(f"Session: {self.session_id}")

        if self.enable_runtime_debug:
            self.runtime_tracker = enable_runtime_tracking(notebook_path=notebook_path)
            lineage_logger.tracking("Runtime debugging enabled")

    def _detect_notebook_path(self) -> str:
        """Attempt to detect the notebook path from the call stack or environment"""
        import os

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

    def install(self):
        """Install all hooks for live tracking"""
        if self.interceptor_active:
            lineage_logger.warning("Interceptor already active")
            return

        lineage_logger.config("Installing live tracking hooks...")

        try:
            self._hook_builtin_exec()
            self._hook_builtin_compile()
            self._hook_builtin_eval()
            self._hook_pandas_operations()
            self.interceptor_active = True
            lineage_logger.success("Live tracking active!")

        except Exception as e:
            lineage_logger.error(f"Failed to install hooks: {e}")
            self.uninstall()

    def _hook_builtin_exec(self):
        """Hook into Python's builtin exec function to detect cell execution"""
        original_exec = builtins.exec

        def monitored_exec(source, globals_dict=None, locals_dict=None):
            # More permissive detection of actual user cell execution
            is_user_code = False

            if (
                isinstance(source, str)
                and globals_dict is not None
                and len(source.strip()) > 0
            ):
                # Exclude obvious internal code patterns
                exclude_patterns = [
                    "def __create_fn__",
                    "__dataclass_",
                    "FrozenInstanceError",
                    "__generated_with",
                    "__qualname__",
                    "recursive_repr",
                    "_feature_version",
                    "from __future__ import",
                    "__annotations__",
                    "typing.get_type_hints",
                    "import typing",
                    "__pydantic",
                    "field_validator",
                ]

                is_internal = any(pattern in source for pattern in exclude_patterns)

                # Much more permissive user code detection
                has_reasonable_size = 5 <= len(source) <= 50000  # Very wide range
                has_reasonable_lines = len(source.split("\n")) <= 500
                not_just_whitespace = source.strip() != ""

                # Check if this looks like a cell execution context
                has_marimo_context = (
                    globals_dict
                    and any(key.startswith("__") for key in globals_dict.keys())
                    and len(globals_dict) > 10  # Typical marimo cell has many builtins
                )

                # Determine if this is user code - much more permissive
                is_user_code = (
                    not is_internal
                    and has_reasonable_size
                    and has_reasonable_lines
                    and not_just_whitespace
                    and has_marimo_context
                )

            # Debug: Log ALL potential user code for analysis
            if (
                isinstance(source, str)
                and globals_dict is not None
                and len(source.strip()) > 0
            ):
                source_preview = source[:80].replace("\n", "\\n")
                globals_count = len(globals_dict) if globals_dict else 0

                if is_user_code:
                    lineage_logger.tracking(
                        f"CAPTURING USER CODE: {source_preview}... (globals: {globals_count})"
                    )
                else:
                    # Show why we rejected it
                    rejection_reasons = []
                    if any(
                        pattern in source
                        for pattern in ["__dataclass_", "def __create_fn__"]
                    ):
                        rejection_reasons.append("internal_pattern")
                    if not (5 <= len(source) <= 50000):
                        rejection_reasons.append(f"size_{len(source)}")
                    if not globals_dict or len(globals_dict) <= 10:
                        rejection_reasons.append(f"context_{globals_count}")

                    lineage_logger.debug(
                        f"REJECTED: {source_preview}... ({', '.join(rejection_reasons)})"
                    )

            # Track what looks like actual user code with much broader criteria
            if is_user_code:
                # Generate cell ID and set execution context
                cell_id = str(uuid.uuid4())[:8]
                _cell_execution_context.update(
                    {
                        "cell_id": cell_id,
                        "source": source,
                        "timestamp": datetime.now().isoformat(),
                        "globals_keys": (
                            list(globals_dict.keys())
                            if len(globals_dict) < 50
                            else f"<{len(globals_dict)} variables>"
                        ),
                    }
                )

                try:
                    # Execute with runtime tracking if enabled
                    if self.enable_runtime_debug and self.runtime_tracker:
                        with track_cell_execution(source, cell_id) as ctx:
                            result = original_exec(source, globals_dict, locals_dict)

                            # Scan for new DataFrames after execution
                            if globals_dict:
                                self._discover_new_dataframes(globals_dict)

                            # Capture result if possible (for DataFrame linking)
                            if (
                                globals_dict
                                and "_" in globals_dict
                                and globals_dict["_"] is not None
                            ):
                                ctx.set_result(globals_dict["_"])

                            return result
                    else:
                        # No runtime tracking - just execute normally
                        result = original_exec(source, globals_dict, locals_dict)

                        # Scan for new DataFrames after execution
                        if globals_dict:
                            self._discover_new_dataframes(globals_dict)

                        return result

                finally:
                    # Clear context after execution
                    _cell_execution_context.clear()
            else:
                return original_exec(source, globals_dict, locals_dict)

        # Install the hook
        builtins.exec = monitored_exec
        self.original_functions["builtin_exec"] = original_exec

    def _hook_builtin_compile(self):
        """Hook into Python's builtin compile function"""
        original_compile = builtins.compile

        def monitored_compile(
            source, filename, mode, flags=0, dont_inherit=False, optimize=-1, **kwargs
        ):
            # Debug compile calls
            if isinstance(source, str) and len(source.strip()) > 0:
                source_preview = source[:50].replace("\n", "\\n")
                lineage_logger.debug(
                    f"COMPILE: {source_preview}... (filename: {filename}, mode: {mode})"
                )

            return original_compile(
                source, filename, mode, flags, dont_inherit, optimize, **kwargs
            )

        builtins.compile = monitored_compile
        self.original_functions["builtin_compile"] = original_compile

    def _hook_builtin_eval(self):
        """Hook into Python's builtin eval function"""
        original_eval = builtins.eval

        def monitored_eval(expression, globals_dict=None, locals_dict=None):
            # Debug eval calls
            if isinstance(expression, str):
                expr_preview = expression[:50].replace("\n", "\\n")
                globals_count = len(globals_dict) if globals_dict else 0
                lineage_logger.debug(
                    f"EVAL: {expr_preview}... (globals: {globals_count})"
                )

            return original_eval(expression, globals_dict, locals_dict)

        builtins.eval = monitored_eval
        self.original_functions["builtin_eval"] = original_eval

    def _hook_pandas_operations(self):
        """Hook into pandas DataFrame operations"""
        try:
            import pandas as pd

            # Store original methods before hooking
            original_methods = {}

            # Hook DataFrame creation functions
            for func_name in ["read_csv", "read_json", "read_excel", "read_parquet"]:
                if hasattr(pd, func_name):
                    original_func = getattr(pd, func_name)
                    original_methods[func_name] = original_func
                    setattr(
                        pd,
                        func_name,
                        self._create_read_wrapper(func_name, original_func),
                    )

            # Hook DataFrame constructor
            original_init = pd.DataFrame.__init__
            original_methods["DataFrame.__init__"] = original_init

            def tracked_init(df_self, *args, **kwargs):
                result = original_init(df_self, *args, **kwargs)
                self._track_dataframe_creation(df_self, "DataFrame", args, kwargs)
                return result

            pd.DataFrame.__init__ = tracked_init

            # Hook DataFrame methods that create new DataFrames
            transformation_methods = [
                "merge",
                "join",
                "concat",
                "groupby",
                "drop",
                "dropna",
                "fillna",
                "query",
                "assign",
                "pipe",
                "transform",
                "apply",
            ]

            for method_name in transformation_methods:
                if hasattr(pd.DataFrame, method_name):
                    original_method = getattr(pd.DataFrame, method_name)
                    original_methods[f"DataFrame.{method_name}"] = original_method
                    setattr(
                        pd.DataFrame,
                        method_name,
                        self._create_method_wrapper(method_name, original_method),
                    )

            # Store all original methods for restoration
            self.original_functions["pandas_methods"] = original_methods

        except ImportError:
            lineage_logger.warning("pandas not available - DataFrame tracking disabled")
        except Exception as e:
            lineage_logger.warning(f"pandas hooking failed: {e}")

    def _create_read_wrapper(self, func_name: str, original_func):
        """Create a wrapper for pandas read functions"""

        def wrapper(*args, **kwargs):
            try:
                result = original_func(*args, **kwargs)

                # Track if result is a DataFrame
                if hasattr(result, "shape") and hasattr(result, "columns"):
                    self._track_dataframe_creation(result, func_name, args, kwargs)

                    # Capture runtime snapshot if enabled (simplified for basic tracker)
                    if self.enable_runtime_debug and self.runtime_tracker:
                        # Note: Simple tracker doesn't maintain execution stack
                        # Snapshots will be captured at cell level instead
                        pass

                return result

            except Exception as e:
                lineage_logger.warning(f"Error in {func_name} wrapper: {e}")
                return original_func(*args, **kwargs)

        return wrapper

    def _create_method_wrapper(self, method_name: str, original_method):
        """Create a wrapper for DataFrame methods"""

        def wrapper(df_self, *args, **kwargs):
            try:
                result = original_method(df_self, *args, **kwargs)

                # Track if result is a DataFrame
                if hasattr(result, "shape") and hasattr(result, "columns"):
                    self._track_dataframe_transformation(
                        input_df=df_self,
                        output_df=result,
                        operation=method_name,
                        args=args,
                        kwargs=kwargs,
                    )

                    # Capture runtime snapshot if enabled (simplified for basic tracker)
                    if self.enable_runtime_debug and self.runtime_tracker:
                        # Note: Simple tracker doesn't maintain execution stack
                        # Snapshots will be captured at cell level instead
                        pass

                return result

            except Exception as e:
                lineage_logger.warning(f"Error in {method_name} wrapper: {e}")
                return original_method(df_self, *args, **kwargs)

        return wrapper

    def _discover_new_dataframes(self, globals_dict: dict[str, Any]):
        """Scan global namespace for new DataFrames after cell execution"""
        try:
            for var_name, obj in globals_dict.items():
                if (
                    hasattr(obj, "shape")
                    and hasattr(obj, "columns")
                    and not var_name.startswith("_")
                ):

                    # Check if already tracked (safely)
                    already_tracked = False
                    try:
                        already_tracked = obj in _tracked_dataframes
                    except (TypeError, ValueError):
                        pass

                    if not already_tracked:
                        # Track newly discovered DataFrame
                        self._track_dataframe_creation(
                            obj, "variable_assignment", name=var_name
                        )

        except Exception as e:
            lineage_logger.warning(f"Error discovering DataFrames: {e}")

    def _track_dataframe_creation(
        self, df, operation: str, args=None, kwargs=None, name=None
    ):
        """Track creation of a new DataFrame"""
        try:
            df_id = id(df)

            # Store in weak reference dictionary
            try:
                _tracked_dataframes[df] = {
                    "id": df_id,
                    "operation": operation,
                    "created_at": datetime.now(),
                    "name": name,
                }
            except (TypeError, ValueError):
                # DataFrame might not be suitable for weak reference, continue anyway
                pass

            # TRIGGER RUNTIME TRACKING - Since DataFrame operations happen during user interaction
            # Create a pseudo cell execution for runtime tracking if no current execution
            runtime_execution_id = None
            if self.enable_runtime_debug and self.runtime_tracker:
                current_execution_id = self.runtime_tracker.get_current_execution_id()
                if not current_execution_id:
                    # No current execution - create one for this DataFrame operation
                    cell_source = f"# DataFrame operation: {operation}"
                    with track_cell_execution(cell_source) as ctx:
                        ctx.set_result(df)
                        runtime_execution_id = ctx.execution_id
                else:
                    runtime_execution_id = current_execution_id

            # Generate lineage event
            event = {
                "event_id": str(uuid.uuid4()),
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "event_type": "dataframe_created",
                "operation": operation,
                "dataframe_id": df_id,
                "dataframe_name": name,
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage_mb": round(
                    df.memory_usage(deep=True).sum() / 1024 / 1024, 2
                ),
                "cell_context": dict(_cell_execution_context),
            }

            # Add runtime debugging context
            if runtime_execution_id:
                event["runtime_execution_id"] = runtime_execution_id
                event["runtime_debugging_enabled"] = True

            # Add sanitized arguments
            if args:
                event["creation_args"] = self._sanitize_args(args)
            if kwargs:
                event["creation_kwargs"] = self._sanitize_args(kwargs)

            self._emit_event(event)

        except Exception as e:
            lineage_logger.warning(f"Error tracking DataFrame creation: {e}")

    def _track_dataframe_transformation(
        self, input_df, output_df, operation: str, args=None, kwargs=None
    ):
        """Track transformation from one DataFrame to another"""
        try:
            input_id = id(input_df)
            output_id = id(output_df)

            # Track the output DataFrame
            try:
                _tracked_dataframes[output_df] = {
                    "id": output_id,
                    "operation": operation,
                    "created_at": datetime.now(),
                    "parent_id": input_id,
                }
            except (TypeError, ValueError):
                # DataFrame might not be suitable for weak reference
                pass

            # Generate transformation event
            event = {
                "event_id": str(uuid.uuid4()),
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "event_type": "dataframe_transformed",
                "operation": operation,
                "input_dataframe_id": input_id,
                "output_dataframe_id": output_id,
                "input_shape": list(input_df.shape),
                "output_shape": list(output_df.shape),
                "input_columns": list(input_df.columns),
                "output_columns": list(output_df.columns),
                "shape_change": {
                    "rows_delta": output_df.shape[0] - input_df.shape[0],
                    "cols_delta": output_df.shape[1] - input_df.shape[1],
                },
                "memory_delta_mb": round(
                    (
                        output_df.memory_usage(deep=True).sum()
                        - input_df.memory_usage(deep=True).sum()
                    )
                    / 1024
                    / 1024,
                    2,
                ),
                "cell_context": dict(_cell_execution_context),
            }

            # Add runtime debugging context if available
            if self.enable_runtime_debug and self.runtime_tracker:
                execution_id = self.runtime_tracker.get_current_execution_id()
                if execution_id:
                    event["runtime_execution_id"] = execution_id
                event["runtime_debugging_enabled"] = True

            # Add sanitized arguments
            if args:
                event["operation_args"] = self._sanitize_args(args)
            if kwargs:
                event["operation_kwargs"] = self._sanitize_args(kwargs)

            self._emit_event(event)

        except Exception as e:
            lineage_logger.warning(f"Error tracking DataFrame transformation: {e}")

    def _sanitize_args(self, args):
        """Sanitize arguments for JSON serialization"""
        sanitized = []
        for arg in args:
            try:
                if hasattr(arg, "shape") and hasattr(arg, "columns"):
                    # DataFrame argument
                    sanitized.append(
                        {
                            "type": "DataFrame",
                            "shape": list(arg.shape),
                            "columns": list(arg.columns),
                        }
                    )
                elif isinstance(arg, (str, int, float, bool, type(None))):
                    sanitized.append(arg)
                elif isinstance(arg, (list, tuple)):
                    sanitized.append(f"<{type(arg).__name__} with {len(arg)} items>")
                else:
                    sanitized.append(f"<{type(arg).__name__}>")
            except:
                sanitized.append("<unparseable>")
        return sanitized

    def _emit_event(self, event):
        """Emit a lineage event to the output file"""
        try:
            with self._lock:
                with open(self.output_file, "a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            lineage_logger.warning(f"Failed to emit event: {e}")

    def uninstall(self):
        """Remove all hooks and restore original functions"""
        if not self.interceptor_active:
            return

        lineage_logger.config("Uninstalling live tracking hooks...")

        try:
            # Restore builtin exec
            if "builtin_exec" in self.original_functions:
                builtins.exec = self.original_functions["builtin_exec"]

            # Restore builtin compile
            if "builtin_compile" in self.original_functions:
                builtins.compile = self.original_functions["builtin_compile"]

            # Restore builtin eval
            if "builtin_eval" in self.original_functions:
                builtins.eval = self.original_functions["builtin_eval"]

            # Restore pandas methods
            if "pandas_methods" in self.original_functions:
                import pandas as pd

                original_methods = self.original_functions["pandas_methods"]

                # Restore pandas functions
                for func_name in [
                    "read_csv",
                    "read_json",
                    "read_excel",
                    "read_parquet",
                ]:
                    if func_name in original_methods:
                        setattr(pd, func_name, original_methods[func_name])

                # Restore DataFrame methods
                if "DataFrame.__init__" in original_methods:
                    pd.DataFrame.__init__ = original_methods["DataFrame.__init__"]

                for method_name in [
                    "merge",
                    "join",
                    "concat",
                    "groupby",
                    "drop",
                    "dropna",
                    "fillna",
                    "query",
                    "assign",
                    "pipe",
                    "transform",
                    "apply",
                ]:
                    key = f"DataFrame.{method_name}"
                    if key in original_methods:
                        setattr(pd.DataFrame, method_name, original_methods[key])

            self.interceptor_active = False
            lineage_logger.success("Live tracking uninstalled")

        except Exception as e:
            lineage_logger.warning(f"Error during uninstall: {e}")

    def get_session_summary(self):
        """Get summary of current tracking session"""
        summary = {
            "session_id": self.session_id,
            "events_logged": len(self.tracked_operations),
            "dataframes_tracked": len(_tracked_dataframes),
            "interceptor_active": self.interceptor_active,
            "output_file": str(self.output_file),
            "runtime_debugging_enabled": self.enable_runtime_debug,
        }

        # Add runtime tracker summary if available
        if self.enable_runtime_debug and self.runtime_tracker:
            runtime_summary = self.runtime_tracker.get_session_summary()
            summary["runtime_summary"] = runtime_summary

        return summary


def enable_live_tracking(
    notebook_path: str = None,
    output_file: str = None,
    enable_runtime_debug: bool = True,
):
    """Enable live DataFrame tracking for Marimo notebooks

    Args:
        notebook_path: Path to the notebook file (auto-detected if None)
        output_file: Custom output file path (auto-generated if None)
        enable_runtime_debug: Whether to enable runtime performance tracking
    """
    global _global_interceptor

    if _global_interceptor is not None:
        lineage_logger.warning("Live tracking already enabled")
        return _global_interceptor

    _global_interceptor = MarimoLiveInterceptor(
        notebook_path, output_file, enable_runtime_debug
    )
    _global_interceptor.install()
    return _global_interceptor


def disable_live_tracking():
    """Disable live DataFrame tracking"""
    global _global_interceptor

    if _global_interceptor is not None:
        _global_interceptor.uninstall()

        # Disable runtime tracking if enabled
        if _global_interceptor.enable_runtime_debug:
            try:
                from .lightweight_runtime_tracker import disable_runtime_tracking

                disable_runtime_tracking()
            except ImportError:
                pass

        _global_interceptor = None
        lineage_logger.config("Live tracking disabled")
    else:
        lineage_logger.warning("Live tracking was not active")


def is_tracking_active():
    """Check if live tracking is currently active"""
    return _global_interceptor is not None and _global_interceptor.interceptor_active


def get_tracking_summary():
    """Get summary of current tracking session"""
    if _global_interceptor is not None:
        return _global_interceptor.get_session_summary()
    else:
        return {"status": "inactive", "message": "Live tracking is not enabled"}


def get_current_runtime_tracker():
    """Get the current runtime tracker instance if available"""
    if _global_interceptor and _global_interceptor.enable_runtime_debug:
        return _global_interceptor.runtime_tracker
    return None


@contextmanager
def runtime_debugging_context(cell_source: str, globals_dict: dict = None):
    """
    Context manager for manual runtime debugging of specific code blocks

    Usage:
        with runtime_debugging_context("df = pd.read_csv('data.csv')", globals()) as ctx:
            df = pd.read_csv('data.csv')
            ctx.snapshot("checkpoint")
    """
    if RUNTIME_TRACKING_AVAILABLE:
        with track_cell_execution(cell_source, globals_dict) as ctx:
            yield ctx
    else:
        yield None


# Auto-enable if imported directly
if __name__ == "__main__":
    # Example usage
    interceptor = enable_live_tracking(enable_runtime_debug=True)

    # Simulate some DataFrame operations
    import pandas as pd

    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = df1.merge(pd.DataFrame({"A": [1, 2], "C": [7, 8]}), on="A")

    lineage_logger.runtime("\n" + "=" * 50)
    lineage_logger.runtime("SESSION SUMMARY")
    lineage_logger.runtime("=" * 50)
    lineage_logger.runtime(json.dumps(get_tracking_summary(), indent=2))
else:
    # Auto-enable when imported
    enable_live_tracking()
