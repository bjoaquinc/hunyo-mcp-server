#!/usr/bin/env python3
"""
Unified Marimo Interceptor
Production-ready interceptor using marimo's native hooks with execution context scoping

This module provides a single system that handles:
1. Runtime events (cell execution performance, timing, memory)
2. Lineage events (DataFrame operations, OpenLineage compliance)

Both event types are captured using marimo's native hooks but written to separate files
with their respective schemas.
"""

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import the logger
from hunyo_capture.logger import get_logger

# Initialize logger
unified_logger = get_logger("hunyo.unified.marimo")

# Global tracking state
_tracked_dataframes = {}  # Use df_id as key instead of DataFrame object
_global_unified_interceptor = None

# Constants for data capture limits
MAX_OBJECT_SIZE = 1000
MAX_COLLECTION_SIZE = 10
MAX_DICT_SIZE = 5
MAX_HASH_LENGTH = 8


class UnifiedMarimoInterceptor:
    """
    Unified marimo interceptor for both runtime and lineage event tracking.

    Uses marimo's native hook system to capture cell execution events and
    DataFrame operations within the proper execution context.
    """

    def __init__(
        self,
        notebook_path: str | None = None,
        runtime_file: str | None = None,
        lineage_file: str | None = None,
        dataframe_lineage_file: str | None = None,
    ):
        # If specific file paths are provided, use them directly
        if runtime_file or lineage_file or dataframe_lineage_file:
            self.runtime_file = Path(runtime_file or "marimo_runtime_events.jsonl")
            self.lineage_file = Path(lineage_file or "marimo_lineage_events.jsonl")
            self.dataframe_lineage_file = Path(
                dataframe_lineage_file or "marimo_dataframe_lineage_events.jsonl"
            )
        elif notebook_path:
            # Generate proper file paths using naming convention
            from hunyo_capture import get_event_filenames, get_user_data_dir

            data_dir = get_user_data_dir()
            runtime_file_path, lineage_file_path, dataframe_lineage_file_path = (
                get_event_filenames(notebook_path, data_dir)
            )
            self.runtime_file = Path(runtime_file_path)
            self.lineage_file = Path(lineage_file_path)
            self.dataframe_lineage_file = Path(dataframe_lineage_file_path)
        else:
            # Fallback to defaults
            self.runtime_file = Path("marimo_runtime_events.jsonl")
            self.lineage_file = Path("marimo_lineage_events.jsonl")
            self.dataframe_lineage_file = Path("marimo_dataframe_lineage_events.jsonl")

        self.notebook_path = notebook_path
        self.session_id = str(uuid.uuid4())[:8]
        self.interceptor_active = False
        self._lock = threading.Lock()

        # Hook references
        self.installed_hooks = []

        # Execution context tracking
        self._execution_contexts = {}

        # DataFrame monkey patching state
        self.original_pandas_methods = {}

        # NEW: DataFrame lineage infrastructure (Phase 2.1)
        self.dataframe_lineage_enabled = self._get_dataframe_lineage_config()
        self.object_to_variable = {}  # Simple object ID to variable name mapping
        self.dataframe_lineage_config = {}  # Full configuration dict
        self.performance_tracking = {}  # Performance monitoring state

        # Constants for style compliance
        self.DATAFRAME_DIMENSIONS = (
            2  # DataFrame should have 2 dimensions (rows, columns)
        )
        self.MAX_OPERATION_HISTORY = 50  # Maximum operations to keep in lineage chain
        self.OPERATION_HISTORY_TRIM = 25  # Keep last 25 operations when trimming

        # Ensure all three directories exist
        self.runtime_file.parent.mkdir(parents=True, exist_ok=True)
        self.lineage_file.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe_lineage_file.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_file.touch()
        self.lineage_file.touch()
        self.dataframe_lineage_file.touch()

        unified_logger.status("Unified Marimo Interceptor v1.0")
        unified_logger.runtime(f"Runtime events: {self.runtime_file.name}")
        unified_logger.lineage(f"Lineage events: {self.lineage_file.name}")
        unified_logger.config(
            f"DataFrame lineage events: {self.dataframe_lineage_file.name}"
        )
        unified_logger.config(f"Session: {self.session_id}")

        # Log DataFrame lineage configuration
        if self.dataframe_lineage_enabled:
            unified_logger.config("[CONFIG] DataFrame lineage tracking enabled")
        else:
            unified_logger.config("[CONFIG] DataFrame lineage tracking disabled")

    def install(self):
        """Install marimo's native execution hooks and DataFrame monkey patches"""
        if self.interceptor_active:
            unified_logger.warning("[WARN] Unified interceptor already installed")
            return

        try:
            # Install marimo hooks
            self._install_marimo_hooks()

            # Install DataFrame monkey patches with execution context
            self._install_dataframe_patches()

            self.interceptor_active = True
            unified_logger.success("[OK] Unified marimo interceptor installed!")

        except ImportError as e:
            unified_logger.error(f"[ERROR] Failed to import marimo hooks: {e}")
        except Exception as e:
            unified_logger.error(f"[ERROR] Failed to install unified interceptor: {e}")

    def _install_marimo_hooks(self):
        """Install marimo's native execution hooks"""
        try:
            from marimo._runtime.runner.hooks import (
                ON_FINISH_HOOKS,
                POST_EXECUTION_HOOKS,
                PRE_EXECUTION_HOOKS,
            )
        except ImportError as e:
            unified_logger.error(f"[ERROR] Failed to import marimo hooks: {e}")
            raise

        unified_logger.info("[INSTALL] Installing marimo native hooks...")

        # Install pre-execution hook
        pre_hook = self._create_pre_execution_hook()
        PRE_EXECUTION_HOOKS.append(pre_hook)
        self.installed_hooks.append(("PRE_EXECUTION_HOOKS", pre_hook))

        # Install post-execution hook
        post_hook = self._create_post_execution_hook()
        POST_EXECUTION_HOOKS.append(post_hook)
        self.installed_hooks.append(("POST_EXECUTION_HOOKS", post_hook))

        # Install finish hook (for cleanup)
        finish_hook = self._create_finish_hook()
        ON_FINISH_HOOKS.append(finish_hook)
        self.installed_hooks.append(("ON_FINISH_HOOKS", finish_hook))

    def _install_dataframe_patches(self):
        """Install DataFrame monkey patches that only activate during marimo execution"""
        try:
            import pandas as pd

            # Store original methods
            self.original_pandas_methods["DataFrame.__init__"] = pd.DataFrame.__init__
            self.original_pandas_methods["DataFrame.__setitem__"] = (
                pd.DataFrame.__setitem__
            )
            # NEW: Store __getitem__ for DataFrame lineage tracking (Step 2.2)
            self.original_pandas_methods["DataFrame.__getitem__"] = (
                pd.DataFrame.__getitem__
            )
            # NEW: Store groupby for DataFrame lineage tracking (Step 2.4)
            self.original_pandas_methods["DataFrame.groupby"] = pd.DataFrame.groupby
            # NEW: Store merge for DataFrame lineage tracking (Step 2.5)
            self.original_pandas_methods["DataFrame.merge"] = pd.DataFrame.merge

            # Create execution-context-aware DataFrame.__init__
            def tracked_dataframe_init(df_self, *args, **kwargs):
                # Call original method first
                result = self.original_pandas_methods["DataFrame.__init__"](
                    df_self, *args, **kwargs
                )

                # Only capture if we're in active marimo execution
                if self._is_in_marimo_execution():
                    execution_context = self._get_current_execution_context()
                    if execution_context:
                        unified_logger.debug(
                            "[HOOK] In execution context, capturing DataFrame creation..."
                        )
                        try:
                            self._capture_dataframe_creation(
                                df_self, execution_context, *args, **kwargs
                            )
                        except KeyboardInterrupt:
                            # Emit ABORT event for interrupted DataFrame creation
                            self._capture_dataframe_abortion(
                                df=df_self,
                                execution_context=execution_context,
                                job_name="pandas_dataframe_creation",
                                termination_reason="User interrupted operation",
                                partial_outputs=(
                                    [df_self] if df_self is not None else None
                                ),
                            )
                            raise
                    else:
                        unified_logger.debug("[HOOK] No execution context found")
                else:
                    unified_logger.debug(
                        "[HOOK] Not in marimo execution, skipping DataFrame capture"
                    )

                return result

            # Create execution-context-aware DataFrame.__setitem__
            def tracked_dataframe_setitem(df_self, key, value):
                # Call original method first
                result = self.original_pandas_methods["DataFrame.__setitem__"](
                    df_self, key, value
                )

                # Only capture if we're in active marimo execution
                if self._is_in_marimo_execution():
                    execution_context = self._get_current_execution_context()
                    if execution_context:
                        unified_logger.debug(
                            f"[HOOK] Capturing DataFrame column assignment: {key}"
                        )
                        try:
                            self._capture_dataframe_modification(
                                df_self,
                                execution_context,
                                "column_assignment",
                                key,
                                value,
                            )
                        except KeyboardInterrupt:
                            # Emit ABORT event for interrupted DataFrame modification
                            self._capture_dataframe_abortion(
                                df=df_self,
                                execution_context=execution_context,
                                job_name="pandas_dataframe_modification",
                                termination_reason="User interrupted operation",
                                partial_outputs=(
                                    [df_self] if df_self is not None else None
                                ),
                            )
                            raise
                    else:
                        unified_logger.debug("[HOOK] No execution context found")
                else:
                    unified_logger.debug(
                        "[HOOK] Not in marimo execution, skipping DataFrame modification capture"
                    )

                return result

            # Apply the monkey patches
            pd.DataFrame.__init__ = tracked_dataframe_init
            pd.DataFrame.__setitem__ = tracked_dataframe_setitem

            unified_logger.info("[INSTALL] DataFrame monkey patches installed")

            # NEW: Setup DataFrame lineage tracking if enabled (Phase 2.1)
            if self.dataframe_lineage_enabled:
                self._setup_dataframe_lineage_tracking()
                unified_logger.info(
                    "[INSTALL] DataFrame lineage tracking setup initiated"
                )

        except ImportError:
            unified_logger.warning("[WARN] pandas not available for DataFrame tracking")
        except Exception as e:
            unified_logger.error(f"[ERROR] Failed to install DataFrame patches: {e}")

    def _setup_dataframe_lineage_tracking(self):
        """Setup DataFrame lineage tracking for computational operations.

        Phase 2.2: Implement DataFrame.__getitem__ monkey patch for selection operations.
        This is the PRIMARY GAP FIX - captures filtering like df[df['age'] > 25].
        """
        try:
            import pandas as pd

            unified_logger.info("[SETUP] Installing DataFrame lineage tracking...")

            # Phase 2.2: PRIMARY GAP FIX - DataFrame.__getitem__ for selection operations
            def tracked_dataframe_getitem(df_self, key):
                """Wrapper for DataFrame.__getitem__ to capture selection operations"""
                # Call original method first
                result = self.original_pandas_methods["DataFrame.__getitem__"](
                    df_self, key
                )

                # Only capture if we're in active marimo execution and lineage is enabled
                # AND the result is actually a DataFrame (not a Series)
                if (
                    self._is_in_marimo_execution()
                    and self.dataframe_lineage_enabled
                    and hasattr(result, "shape")
                    and len(result.shape) == self.DATAFRAME_DIMENSIONS
                    and hasattr(result, "columns")
                ):

                    execution_context = self._get_current_execution_context()
                    if execution_context:
                        unified_logger.debug(
                            "[LINEAGE] Capturing DataFrame.__getitem__ operation (DataFrame result)"
                        )
                        try:
                            # Capture DataFrame operation using Step 2.1 infrastructure
                            self._capture_dataframe_operation(
                                input_df=df_self,
                                output_df=result,
                                operation_method="__getitem__",
                                execution_context=execution_context,
                                key=key,
                            )
                        except Exception as e:
                            unified_logger.error(
                                f"[ERROR] Failed to capture DataFrame.__getitem__ operation: {e}"
                            )
                elif (
                    self._is_in_marimo_execution()
                    and self.dataframe_lineage_enabled
                    and hasattr(result, "shape")
                    and len(result.shape) == 1
                ):
                    unified_logger.debug(
                        "[LINEAGE] Skipping Series result from DataFrame.__getitem__"
                    )

                return result

            # Apply the monkey patch
            pd.DataFrame.__getitem__ = tracked_dataframe_getitem

            unified_logger.success(
                "[SETUP] DataFrame.__getitem__ monkey patch installed"
            )

            # Phase 2.4: DataFrame.groupby for aggregation operations
            def tracked_dataframe_groupby(df_self, *args, **kwargs):
                """Wrapper for DataFrame.groupby to capture GroupBy setup operations"""
                # Call original method first
                result = self.original_pandas_methods["DataFrame.groupby"](
                    df_self, *args, **kwargs
                )

                # Only capture if we're in active marimo execution and lineage is enabled
                if self._is_in_marimo_execution() and self.dataframe_lineage_enabled:
                    execution_context = self._get_current_execution_context()
                    if execution_context:
                        unified_logger.debug(
                            "[LINEAGE] Capturing DataFrame.groupby operation"
                        )
                        try:
                            # Store GroupBy object for later aggregation tracking
                            groupby_obj_id = id(result)
                            self.object_to_variable[groupby_obj_id] = (
                                self._get_variable_name_from_object_id(
                                    id(df_self), "groupby_setup"
                                )
                            )

                            # Track GroupBy setup (without output DataFrame yet)
                            self._track_groupby_setup(
                                df_self,
                                result,
                                execution_context,
                                *args,
                                **kwargs,
                            )
                        except Exception as e:
                            unified_logger.error(
                                f"[ERROR] Failed to capture DataFrame.groupby operation: {e}"
                            )

                return result

            # Apply the GroupBy monkey patch
            pd.DataFrame.groupby = tracked_dataframe_groupby

            # Setup GroupBy aggregation method tracking
            self._setup_groupby_aggregation_tracking()

            unified_logger.success("[SETUP] DataFrame.groupby monkey patch installed")

            # Phase 2.5: DataFrame.merge for join operations
            def tracked_dataframe_merge(df_self, *args, **kwargs):
                """Wrapper for DataFrame.merge to capture join operations"""
                # Call original method first
                result = self.original_pandas_methods["DataFrame.merge"](
                    df_self, *args, **kwargs
                )

                # Only capture if we're in active marimo execution and lineage is enabled
                if self._is_in_marimo_execution() and self.dataframe_lineage_enabled:
                    execution_context = self._get_current_execution_context()
                    if execution_context:
                        unified_logger.debug(
                            "[LINEAGE] Capturing DataFrame.merge operation"
                        )
                        try:
                            # Capture the merge operation
                            self._capture_dataframe_operation(
                                input_df=df_self,
                                output_df=result,
                                operation_method="merge",
                                execution_context=execution_context,
                                merge_args=args,
                                merge_kwargs=kwargs,
                            )
                        except Exception as e:
                            unified_logger.error(
                                f"[ERROR] Failed to capture DataFrame.merge operation: {e}"
                            )

                return result

            # Apply the merge monkey patch
            pd.DataFrame.merge = tracked_dataframe_merge

            unified_logger.success("[SETUP] DataFrame.merge monkey patch installed")
            unified_logger.debug(
                f"[CONFIG] DataFrame lineage config: {self.dataframe_lineage_config}"
            )

        except ImportError:
            unified_logger.warning(
                "[WARN] pandas not available for DataFrame lineage tracking"
            )
        except Exception as e:
            unified_logger.error(
                f"[ERROR] Failed to setup DataFrame lineage tracking: {e}"
            )

    def _track_groupby_setup(
        self, input_df, groupby_obj, execution_context, *args, **kwargs
    ):
        """Track GroupBy setup operation for aggregation lineage.

        Phase 2.4: Track the GroupBy setup phase before aggregation methods are called.
        This provides context for subsequent aggregation operations.
        """
        try:
            # Generate GroupBy setup event (informational, not a full DataFrame operation)
            unified_logger.debug(
                "[LINEAGE] Tracking GroupBy setup for aggregation operations"
            )

            # Store GroupBy context for later aggregation tracking
            groupby_context = {
                "input_df_id": id(input_df),
                "groupby_obj_id": id(groupby_obj),
                "execution_context": execution_context,
                "groupby_args": args,
                "groupby_kwargs": kwargs,
            }

            # Store context in object mapping for aggregation methods to use
            self.object_to_variable[f"groupby_context_{id(groupby_obj)}"] = (
                groupby_context
            )

            unified_logger.debug(
                f"[LINEAGE] GroupBy setup context stored for object {id(groupby_obj)}"
            )

        except Exception as e:
            unified_logger.error(f"[ERROR] Failed to track GroupBy setup: {e}")

    def _setup_groupby_aggregation_tracking(self):
        """Setup GroupBy aggregation method tracking.

        Phase 2.4: Track both GroupBy creation AND aggregation operations.
        Patches GroupBy.sum(), GroupBy.mean(), GroupBy.count() methods.
        """
        try:
            import pandas as pd

            unified_logger.info("[SETUP] Setting up GroupBy aggregation tracking...")

            # Store original GroupBy aggregation methods
            if "GroupBy.sum" not in self.original_pandas_methods:
                self.original_pandas_methods["GroupBy.sum"] = (
                    pd.core.groupby.DataFrameGroupBy.sum
                )
                self.original_pandas_methods["GroupBy.mean"] = (
                    pd.core.groupby.DataFrameGroupBy.mean
                )
                self.original_pandas_methods["GroupBy.count"] = (
                    pd.core.groupby.DataFrameGroupBy.count
                )

            def create_tracked_aggregation_method(method_name, original_method):
                """Create a tracked version of a GroupBy aggregation method"""

                def tracked_aggregation(groupby_self, *args, **kwargs):
                    """Wrapper for GroupBy aggregation methods"""
                    # Call original method first
                    result = original_method(groupby_self, *args, **kwargs)

                    # Only capture if we're in active marimo execution and lineage is enabled
                    if (
                        self._is_in_marimo_execution()
                        and self.dataframe_lineage_enabled
                    ):
                        execution_context = self._get_current_execution_context()
                        if execution_context:
                            unified_logger.debug(
                                f"[LINEAGE] Capturing GroupBy.{method_name} operation"
                            )
                            try:
                                # Get the original DataFrame from the GroupBy object
                                input_df = groupby_self.obj

                                # Capture the aggregation operation
                                self._capture_dataframe_operation(
                                    input_df,
                                    result,
                                    method_name,
                                    execution_context,
                                    *args,
                                    groupby_obj=groupby_self,
                                    **kwargs,
                                )
                            except Exception as e:
                                unified_logger.error(
                                    f"[ERROR] Failed to capture GroupBy.{method_name} operation: {e}"
                                )

                    return result

                return tracked_aggregation

            # Apply patches to aggregation methods
            pd.core.groupby.DataFrameGroupBy.sum = create_tracked_aggregation_method(
                "sum", self.original_pandas_methods["GroupBy.sum"]
            )
            pd.core.groupby.DataFrameGroupBy.mean = create_tracked_aggregation_method(
                "mean", self.original_pandas_methods["GroupBy.mean"]
            )
            pd.core.groupby.DataFrameGroupBy.count = create_tracked_aggregation_method(
                "count", self.original_pandas_methods["GroupBy.count"]
            )

            unified_logger.success(
                "[SETUP] GroupBy aggregation method tracking installed"
            )

        except ImportError:
            unified_logger.warning(
                "[WARN] pandas not available for GroupBy aggregation tracking"
            )
        except Exception as e:
            unified_logger.error(
                f"[ERROR] Failed to setup GroupBy aggregation tracking: {e}"
            )

    def _create_pre_execution_hook(self):
        """Create pre-execution hook function"""

        def pre_execution_hook(cell, _runner):
            """Called before each cell execution"""
            try:
                # Extract cell information from marimo
                cell_id = cell.cell_id
                cell_code = cell.code

                unified_logger.debug(f"[HOOK] PRE-EXECUTION HOOK FIRED: {cell_id[:8]}")

                # Generate execution ID for this cell run
                execution_id = str(uuid.uuid4())[:8]

                # Emit runtime event
                self._emit_runtime_event(
                    {
                        "event_type": "cell_execution_start",
                        "execution_id": execution_id,
                        "cell_id": cell_id,
                        "cell_source": cell_code,
                        "cell_source_lines": (
                            len(cell_code.split("\n")) if cell_code else 0
                        ),
                        "start_memory_mb": self._get_memory_usage(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "session_id": self.session_id,
                        "emitted_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

                # Store execution context for DataFrame scoping
                context_data = {
                    "execution_id": execution_id,
                    "start_time": time.time(),
                    "cell_code": cell_code,
                    "cell_id": cell_id,
                    "context_created_at": time.time(),  # NEW: Track when context was created
                }

                # Store by cell_id and code hash for safety
                if cell_id:
                    self._execution_contexts[cell_id] = context_data

                if cell_code:
                    try:
                        code_hash = str(hash(cell_code))[:MAX_HASH_LENGTH]
                        self._execution_contexts[f"code_{code_hash}"] = context_data
                    except Exception as e:
                        unified_logger.error(f"Failed to store execution context: {e}")

            except Exception as e:
                unified_logger.error(f"Error in pre-execution hook: {e}")

        return pre_execution_hook

    def _create_post_execution_hook(self):
        """Create post-execution hook function"""

        def post_execution_hook(cell, _runner, run_result):
            """Called after each cell execution"""
            try:
                cell_id = cell.cell_id
                cell_code = cell.code

                # Retrieve execution context
                execution_context = self._execution_contexts.get(cell_id)
                if not execution_context:
                    # Try by code hash
                    if cell_code:
                        code_hash = str(hash(cell_code))[:MAX_HASH_LENGTH]
                        execution_context = self._execution_contexts.get(
                            f"code_{code_hash}"
                        )

                if execution_context:
                    execution_id = execution_context["execution_id"]
                    start_time = execution_context["start_time"]
                    duration_ms = (time.time() - start_time) * 1000

                    # Check for errors
                    error = (
                        getattr(run_result, "exception", None) if run_result else None
                    )

                    # Emit runtime event
                    event_type = (
                        "cell_execution_error" if error else "cell_execution_end"
                    )
                    event_data = {
                        "event_type": event_type,
                        "execution_id": execution_id,
                        "cell_id": cell_id,
                        "cell_source": cell_code,
                        "cell_source_lines": (
                            len(cell_code.split("\n")) if cell_code else 0
                        ),
                        "start_memory_mb": self._get_memory_usage(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "session_id": self.session_id,
                        "emitted_at": datetime.now(timezone.utc).isoformat(),
                        "duration_ms": duration_ms,
                    }

                    if error:
                        event_data["error_info"] = {
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }

                        # Emit FAIL lineage event for cell execution error
                        try:
                            import traceback

                            error_info = {
                                "error_message": str(error),
                                "stack_trace": traceback.format_exc(),
                                "error_type": type(error).__name__,
                            }

                            # Check if it's a KeyboardInterrupt for ABORT event
                            if isinstance(error, KeyboardInterrupt):
                                abort_event = self._create_openlineage_event(
                                    event_type="ABORT",
                                    job_name="cell_execution",
                                    error_info=error_info,
                                    execution_context=execution_context,
                                    termination_reason="User interrupted cell execution",
                                    cell_id=cell_id,
                                    cell_source=cell_code,
                                )
                                self._emit_lineage_event(abort_event)
                            else:
                                fail_event = self._create_openlineage_event(
                                    event_type="FAIL",
                                    job_name="cell_execution",
                                    error_info=error_info,
                                    execution_context=execution_context,
                                    failure_context=f"Cell execution failed: {type(error).__name__}",
                                    cell_id=cell_id,
                                    cell_source=cell_code,
                                )
                                self._emit_lineage_event(fail_event)
                        except Exception as lineage_error:
                            unified_logger.error(
                                f"Failed to emit lineage error event: {lineage_error}"
                            )

                    self._emit_runtime_event(event_data)

                    # Clean up execution context
                    if cell_id in self._execution_contexts:
                        del self._execution_contexts[cell_id]
                    if cell_code:
                        code_hash = str(hash(cell_code))[:MAX_HASH_LENGTH]
                        context_key = f"code_{code_hash}"
                        if context_key in self._execution_contexts:
                            del self._execution_contexts[context_key]

            except Exception as e:
                unified_logger.error(f"Error in post-execution hook: {e}")

        return post_execution_hook

    def _create_finish_hook(self):
        """Create finish hook for cleanup"""

        def finish_hook(_runner):
            """Called when marimo session finishes"""
            try:
                unified_logger.info("[CLEANUP] Marimo session finished")
                # Clean up any remaining execution contexts
                self._execution_contexts.clear()
            except Exception as e:
                unified_logger.error(f"Error in finish hook: {e}")

        return finish_hook

    def _is_in_marimo_execution(self) -> bool:
        """Check if we're currently in a marimo cell execution"""
        return len(self._execution_contexts) > 0

    def _get_current_execution_context(self) -> dict[str, Any] | None:
        """Get the current execution context"""
        if not self._execution_contexts:
            return None

        # Return the most recent execution context based on timestamp
        # FIX: Use timestamp-based selection instead of dictionary iteration order
        most_recent_context = None
        most_recent_timestamp = 0

        for context in self._execution_contexts.values():
            context_timestamp = context.get("context_created_at", 0)
            if context_timestamp > most_recent_timestamp:
                most_recent_timestamp = context_timestamp
                most_recent_context = context

        return most_recent_context

    def _create_openlineage_event(
        self,
        event_type,
        job_name,
        inputs=None,
        outputs=None,
        error_info=None,
        execution_context=None,
        **extra_fields,
    ):
        """Create a standardized OpenLineage event"""
        event_time = datetime.now(timezone.utc).isoformat()
        run_id = str(uuid.uuid4())

        # Build run facets
        run_facets = {
            "marimoSession": {
                "_producer": "marimo-lineage-tracker",
                "sessionId": self.session_id,
            }
        }

        # Add execution context as a run facet if provided
        if execution_context:
            run_facets["marimoExecution"] = {
                "_producer": "marimo-lineage-tracker",
                "executionId": execution_context.get("execution_id", ""),
                "sessionId": self.session_id,
            }

        # Add error facet for FAIL/ABORT events
        if error_info and event_type in ("FAIL", "ABORT"):
            run_facets["errorMessage"] = {
                "_producer": "marimo-lineage-tracker",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json",
                "message": error_info.get("error_message", ""),
                "programmingLanguage": "python",
                "stackTrace": error_info.get("stack_trace", ""),
            }

        # Create base OpenLineage event
        lineage_event = {
            "eventType": event_type,
            "eventTime": event_time,
            "run": {"runId": run_id, "facets": run_facets},
            "job": {"namespace": "marimo", "name": job_name},
            "inputs": inputs or [],
            "outputs": outputs or [],
            "producer": "marimo-lineage-tracker",
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
            "session_id": self.session_id,
            "emitted_at": event_time,
        }

        # Only add valid schema fields from extra_fields
        # Don't add execution_context or other non-schema fields
        schema_allowed_fields = {
            "cell_id",
            "cell_source",
            "termination_reason",
            "failure_context",
            "modification_type",
            "key",
            "value_type",
            "test_field",
            "custom_field",
            "abortion_context",
        }

        for field_name, field_value in extra_fields.items():
            if field_name in schema_allowed_fields:
                # Add these as job facets instead of top-level properties
                if "facets" not in lineage_event["job"]:
                    lineage_event["job"]["facets"] = {}
                lineage_event["job"]["facets"][field_name] = field_value

        return lineage_event

    def _create_dataframe_dataset(self, df, dataset_name=None):
        """Create a dataset representation for a DataFrame"""
        if dataset_name is None:
            dataset_name = f"dataframe_{id(df)}"

        # Build schema fields from DataFrame
        schema_fields = []
        try:
            for col, dtype in df.dtypes.items():
                schema_fields.append({"name": str(col), "type": str(dtype)})
        except Exception as e:
            unified_logger.debug(f"Failed to build schema fields: {e}")

        return {
            "namespace": "marimo",
            "name": dataset_name,
            "facets": {
                "schema": {
                    "_producer": "marimo-lineage-tracker",
                    "_schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
                    "fields": schema_fields,
                }
            },
        }

    def _capture_dataframe_creation(self, df, execution_context, *_args, **_kwargs):
        """Capture DataFrame creation as OpenLineage event"""
        try:
            unified_logger.debug(
                f"[CAPTURE] Starting to capture DataFrame creation for df_{id(df)}"
            )

            # Create output dataset
            output_dataset = self._create_dataframe_dataset(df)

            # Create OpenLineage event
            lineage_event = self._create_openlineage_event(
                event_type="START",
                job_name="pandas_dataframe_creation",
                outputs=[output_dataset],
                execution_context=execution_context,
            )

            # Add to tracked DataFrames
            df_id = f"df_{id(df)}"
            _tracked_dataframes[df_id] = {
                "df_id": df_id,
                "created_at": lineage_event["eventTime"],
                "execution_id": execution_context["execution_id"],
            }

            self._emit_lineage_event(lineage_event)

        except Exception as e:
            unified_logger.error(f"Failed to capture DataFrame creation: {e}")
            # Emit FAIL event for DataFrame creation failure
            try:
                self._capture_dataframe_failure(
                    df=df,
                    execution_context=execution_context,
                    job_name="pandas_dataframe_creation",
                    error=e,
                    partial_outputs=[df] if df is not None else None,
                )
            except Exception as fail_error:
                unified_logger.error(
                    f"Failed to capture DataFrame creation failure: {fail_error}"
                )

    def _capture_dataframe_modification(
        self, df, execution_context, modification_type, key, value
    ):
        """Capture DataFrame modification as OpenLineage event"""
        try:
            unified_logger.debug(
                f"[CAPTURE] Starting to capture DataFrame modification for df_{id(df)}"
            )

            # Create output dataset
            output_dataset = self._create_dataframe_dataset(df)

            # Create OpenLineage event
            lineage_event = self._create_openlineage_event(
                event_type="COMPLETE",
                job_name="pandas_dataframe_modification",
                outputs=[output_dataset],
                execution_context=execution_context,
                modification_type=modification_type,
                key=str(key),
                value_type=str(type(value).__name__),
            )

            # Add to tracked DataFrames
            df_id = f"df_{id(df)}"
            _tracked_dataframes[df_id] = {
                "df_id": df_id,
                "created_at": lineage_event["eventTime"],
                "execution_id": execution_context["execution_id"],
            }

            self._emit_lineage_event(lineage_event)

        except Exception as e:
            unified_logger.error(f"Failed to capture DataFrame modification: {e}")
            # Emit FAIL event for DataFrame modification failure
            try:
                self._capture_dataframe_failure(
                    df=df,
                    execution_context=execution_context,
                    job_name="pandas_dataframe_modification",
                    error=e,
                    partial_outputs=[df] if df is not None else None,
                )
            except Exception as fail_error:
                unified_logger.error(
                    f"Failed to capture DataFrame modification failure: {fail_error}"
                )

    def _capture_dataframe_failure(
        self, df, execution_context, job_name, error, partial_outputs=None
    ):
        """Capture DataFrame operation failure as OpenLineage FAIL event"""
        try:
            unified_logger.debug(
                f"[CAPTURE] Starting to capture DataFrame failure for df_{id(df)}"
            )

            # Create partial output datasets if any data was produced
            outputs = []
            if partial_outputs:
                for partial_df in partial_outputs:
                    outputs.append(self._create_dataframe_dataset(partial_df))

            # Build error information
            import traceback

            error_info = {
                "error_message": str(error),
                "stack_trace": traceback.format_exc(),
                "error_type": type(error).__name__,
            }

            # Create OpenLineage FAIL event
            lineage_event = self._create_openlineage_event(
                event_type="FAIL",
                job_name=job_name,
                outputs=outputs,
                error_info=error_info,
                execution_context=execution_context,
                failure_context=f"DataFrame operation failed: {type(error).__name__}",
            )

            self._emit_lineage_event(lineage_event)

        except Exception as e:
            unified_logger.error(f"Failed to capture DataFrame failure: {e}")

    def _capture_dataframe_abortion(
        self, df, execution_context, job_name, termination_reason, partial_outputs=None
    ):
        """Capture DataFrame operation abortion as OpenLineage ABORT event"""
        try:
            unified_logger.debug(
                f"[CAPTURE] Starting to capture DataFrame abortion for df_{id(df)}"
            )

            # Create partial output datasets if any data was produced
            outputs = []
            if partial_outputs:
                for partial_df in partial_outputs:
                    outputs.append(self._create_dataframe_dataset(partial_df))

            # Build termination information
            error_info = {
                "error_message": f"Operation terminated: {termination_reason}",
                "stack_trace": f"Termination reason: {termination_reason}",
                "error_type": "OperationAborted",
            }

            # Create OpenLineage ABORT event
            lineage_event = self._create_openlineage_event(
                event_type="ABORT",
                job_name=job_name,
                outputs=outputs,
                error_info=error_info,
                execution_context=execution_context,
                termination_reason=termination_reason,
                abortion_context=f"DataFrame operation aborted: {termination_reason}",
            )

            self._emit_lineage_event(lineage_event)

        except Exception as e:
            unified_logger.error(f"Failed to capture DataFrame abortion: {e}")

    def _emit_runtime_event(self, event_data):
        """Emit runtime event to runtime file"""
        try:
            with open(self.runtime_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data, default=str) + "\n")
        except Exception as e:
            unified_logger.error(f"Failed to emit runtime event: {e}")

    def _emit_lineage_event(self, event_data):
        """Emit lineage event to lineage file"""
        try:
            with open(self.lineage_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data, default=str) + "\n")
        except Exception as e:
            unified_logger.error(f"Failed to emit lineage event: {e}")

    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _get_dataframe_lineage_config(self) -> bool:
        """Check if DataFrame lineage tracking is enabled via environment variables.

        Following existing patterns for configuration management.

        Returns:
            bool: True if DataFrame lineage tracking is enabled, False otherwise
        """
        try:
            import os

            # Get configuration from environment (default: enabled)
            enabled = (
                os.environ.get("HUNYO_TRACK_DATAFRAME_LINEAGE", "true").lower()
                == "true"
            )

            # Store full configuration for future use
            self.dataframe_lineage_config = {
                "enabled": enabled,
                "sample_large_dataframes": os.environ.get(
                    "HUNYO_SAMPLE_LARGE_DF", "true"
                ).lower()
                == "true",
                "size_threshold_mb": float(
                    os.environ.get("HUNYO_DF_SIZE_THRESHOLD", "10.0")
                ),
                "sample_rate": float(os.environ.get("HUNYO_DF_SAMPLE_RATE", "0.1")),
                "max_overhead_ms": float(
                    os.environ.get("HUNYO_DF_MAX_OVERHEAD", "5.0")
                ),
            }

            return enabled

        except Exception as e:
            unified_logger.warning(
                f"[CONFIG] Error reading DataFrame lineage config: {e}"
            )
            # Safe fallback
            self.dataframe_lineage_config = {
                "enabled": True,
                "sample_large_dataframes": True,
                "size_threshold_mb": 10.0,
                "sample_rate": 0.1,
                "max_overhead_ms": 5.0,
            }
            return True

    def _emit_dataframe_lineage_event(self, event_data):
        """Emit DataFrame lineage event to DataFrame lineage file"""
        try:
            with open(self.dataframe_lineage_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data, default=str) + "\n")
        except Exception as e:
            unified_logger.error(f"Failed to emit DataFrame lineage event: {e}")

    def _get_dataframe_memory_usage(self, df) -> float:
        """Get DataFrame memory usage in MB.

        Args:
            df: pandas DataFrame object

        Returns:
            float: Memory usage in MB, or 0.0 if unable to calculate
        """
        try:
            return df.memory_usage(deep=True).sum() / 1024 / 1024
        except Exception:
            return 0.0

    def _get_variable_name_from_object_id(
        self, obj_id: int, operation_hint: str = ""
    ) -> str:
        """Get variable name from object ID with semantic naming.

        Phase 2.3 implementation with enhanced semantic naming logic.

        Args:
            obj_id: Object ID from id(dataframe)
            operation_hint: Hint about the operation for semantic naming

        Returns:
            str: Variable name (existing or generated)
        """
        # Check if we already have a name for this object
        if obj_id in self.object_to_variable:
            return self.object_to_variable[obj_id]

        # Try to get meaningful variable name from current execution context
        meaningful_name = self._extract_meaningful_variable_name(obj_id, operation_hint)
        if meaningful_name:
            # Store mapping with meaningful name
            self.object_to_variable[obj_id] = meaningful_name
            return meaningful_name

        # Generate semantic name based on operation hint
        if operation_hint:
            # Map operation hints to meaningful names
            operation_map = {
                "selection": "filtered",
                "aggregation": "grouped",
                "join": "merged",
                "input": "source",
                "output": "result",
            }
            semantic_hint = operation_map.get(operation_hint, operation_hint)
            base_name = f"df_{semantic_hint}_{str(obj_id)[-6:]}"  # Use last 6 digits for readability
        else:
            base_name = f"df_{str(obj_id)[-6:]}"

        # Handle collisions by appending counter
        counter = 1
        var_name = base_name
        while var_name in self.object_to_variable.values():
            var_name = f"{base_name}_{counter}"
            counter += 1

        # Store mapping
        self.object_to_variable[obj_id] = var_name
        return var_name

    def _extract_meaningful_variable_name(
        self, obj_id: int, operation_hint: str = ""  # noqa: ARG002
    ) -> str | None:
        """Extract meaningful variable name from execution context.

        Phase 2.3 implementation to find actual variable names in the execution context.

        Args:
            obj_id: Object ID from id(dataframe)
            operation_hint: Hint about the operation for context

        Returns:
            str | None: Meaningful variable name if found, None otherwise
        """
        try:
            # Get current execution context
            execution_context = self._get_current_execution_context()
            if not execution_context:
                return None

            # Try to find variable name in the current cell execution
            # This is a best-effort approach using inspect module
            import inspect

            # Get the current frame (this will be the marimo cell frame)
            frame = inspect.currentframe()
            if frame:
                # Go up the stack to find the marimo cell execution frame
                for _ in range(10):  # Limit stack traversal
                    frame = frame.f_back
                    if frame and frame.f_code.co_name == "<module>":
                        # Check local and global variables in the frame
                        for var_name, var_value in frame.f_locals.items():
                            if (
                                hasattr(var_value, "__class__")
                                and "DataFrame" in str(var_value.__class__)
                                and id(var_value) == obj_id
                            ):
                                return var_name

                        for var_name, var_value in frame.f_globals.items():
                            if (
                                hasattr(var_value, "__class__")
                                and "DataFrame" in str(var_value.__class__)
                                and id(var_value) == obj_id
                            ):
                                return var_name
                        break

            return None
        except Exception:
            # If anything goes wrong, return None to fall back to generated names
            return None

    def _track_variable_relationship(
        self,
        input_var: str,
        output_var: str,
        operation_type: str,
        operation_method: str,
    ) -> None:
        """Track variable relationship for lineage.

        Phase 2.3 implementation to maintain variable relationships and lineage chains.

        Args:
            input_var: Input variable name
            output_var: Output variable name
            operation_type: Type of operation (selection, aggregation, join)
            operation_method: Method name (__getitem__, groupby, merge)
        """
        try:
            # Initialize variable lineage tracking if not exists
            if not hasattr(self, "variable_lineage"):
                self.variable_lineage = {}

            # Track the relationship
            if output_var not in self.variable_lineage:
                self.variable_lineage[output_var] = {
                    "parents": [],
                    "operation_history": [],
                }

            # Add parent relationship
            if input_var not in self.variable_lineage[output_var]["parents"]:
                self.variable_lineage[output_var]["parents"].append(input_var)

            # Add operation to history
            operation_record = {
                "operation_type": operation_type,
                "operation_method": operation_method,
                "input_variable": input_var,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.variable_lineage[output_var]["operation_history"].append(
                operation_record
            )

            # Maintain lineage chain (limit to prevent memory issues)
            if (
                len(self.variable_lineage[output_var]["operation_history"])
                > self.MAX_OPERATION_HISTORY
            ):
                self.variable_lineage[output_var]["operation_history"] = (
                    self.variable_lineage[output_var]["operation_history"][
                        -self.OPERATION_HISTORY_TRIM :
                    ]
                )

            unified_logger.debug(
                f"[LINEAGE] Tracked relationship: {input_var} -> {output_var} ({operation_type})"
            )

        except Exception as e:
            unified_logger.error(f"[ERROR] Failed to track variable relationship: {e}")

    def _detect_operation_type(self, operation_method: str) -> str:
        """Detect operation type from method name.

        Phase 2.4 implementation with support for GroupBy aggregation operations.

        Args:
            operation_method: Method name (e.g., '__getitem__', 'groupby', 'sum', 'mean', 'count', 'merge')

        Returns:
            str: Operation type ('selection', 'aggregation', 'join', 'unknown')
        """
        # Phase 2.4: Enhanced operation detection for GroupBy aggregation
        if operation_method in ["__getitem__"]:
            return "selection"
        elif operation_method in ["groupby", "sum", "mean", "count"]:
            return "aggregation"
        elif operation_method in ["merge"]:
            return "join"
        else:
            return "unknown"

    def _generate_basic_column_lineage(
        self, input_df, output_df, operation_method: str
    ) -> dict:
        """Generate basic column lineage for MVP operations.

        Phase 2.4 implementation with enhanced support for GroupBy aggregation operations.

        Args:
            input_df: Input DataFrame
            output_df: Output DataFrame
            operation_method: Method name

        Returns:
            dict: Column lineage information
        """
        try:
            input_cols = list(input_df.columns) if hasattr(input_df, "columns") else []
            output_cols = (
                list(output_df.columns) if hasattr(output_df, "columns") else []
            )

            # Generate column-level lineage for __getitem__ operations
            if operation_method == "__getitem__":
                # For selection operations, output columns are a subset of input columns
                lineage_map = {}
                for col in output_cols:
                    if col in input_cols:
                        lineage_map[f"output.{col}"] = [f"input.{col}"]

                return {
                    "column_mapping": lineage_map,
                    "input_columns": input_cols,
                    "output_columns": output_cols,
                    "operation_method": operation_method,
                    "lineage_type": "selection",
                }
            elif operation_method in ["sum", "mean", "count"]:
                # For aggregation operations, output columns are computed from input columns
                lineage_map = {}
                for col in output_cols:
                    if col in input_cols:
                        # Direct aggregation of same column
                        lineage_map[f"output.{col}"] = [f"input.{col}"]
                    else:
                        # Computed column from aggregation
                        lineage_map[f"output.{col}"] = input_cols

                return {
                    "column_mapping": lineage_map,
                    "input_columns": input_cols,
                    "output_columns": output_cols,
                    "operation_method": operation_method,
                    "lineage_type": "aggregation",
                }
            elif operation_method == "merge":
                # For merge operations, output columns come from both DataFrames
                lineage_map = {}

                # Basic merge lineage - columns from left DataFrame
                for col in output_cols:
                    if col in input_cols:
                        # Column from left DataFrame (input_df)
                        lineage_map[f"output.{col}"] = [f"left.{col}"]
                    else:
                        # Column likely from right DataFrame or computed
                        # Note: We don't have access to right_df here, so this is simplified
                        lineage_map[f"output.{col}"] = [f"right.{col}"]

                return {
                    "column_mapping": lineage_map,
                    "input_columns": input_cols,  # Left DataFrame columns
                    "output_columns": output_cols,
                    "operation_method": operation_method,
                    "lineage_type": "join",
                    "join_note": "Simplified lineage - right DataFrame columns not tracked in MVP",
                }
            else:
                # Basic lineage for other operations
                return {
                    "input_columns": input_cols,
                    "output_columns": output_cols,
                    "operation_method": operation_method,
                    "lineage_type": "basic",
                }
        except Exception:
            return {"lineage_type": "error", "operation_method": operation_method}

    def _extract_operation_code(self, operation_method: str, *args, **kwargs) -> str:
        """Extract operation code representation for DataFrame lineage events.

        Args:
            operation_method: Method name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            str: Operation code representation
        """
        try:
            if operation_method == "__getitem__":
                if args:
                    key = args[0]
                    if hasattr(key, "dtype") and key.dtype == bool:
                        return "df[boolean_mask]"
                    elif isinstance(key, str):
                        return f"df['{key}']"
                    elif isinstance(key, list):
                        return f"df[{key}]"
                    else:
                        return f"df[{key!r}]"
                elif "key" in kwargs:
                    key = kwargs["key"]
                    if hasattr(key, "dtype") and key.dtype == bool:
                        return "df[boolean_mask]"
                    elif isinstance(key, str):
                        return f"df['{key}']"
                    else:
                        return f"df[{key!r}]"
                else:
                    return "df[unknown_key]"
            elif operation_method == "groupby":
                # Extract groupby columns from arguments
                if args:
                    groupby_cols = args[0]
                    if isinstance(groupby_cols, str):
                        return f"df.groupby('{groupby_cols}')"
                    elif isinstance(groupby_cols, list):
                        return f"df.groupby({groupby_cols})"
                    else:
                        return f"df.groupby({groupby_cols!r})"
                else:
                    return "df.groupby(...)"
            elif operation_method in ["sum", "mean", "count"]:
                # Extract GroupBy aggregation with context
                if "groupby_obj" in kwargs:
                    return f"df.groupby(...).{operation_method}()"
                else:
                    return f"df.{operation_method}()"
            elif operation_method == "merge":
                # Extract merge operation code
                if "merge_kwargs" in kwargs:
                    merge_kwargs = kwargs["merge_kwargs"]
                    how = merge_kwargs.get("how", "inner")

                    if "on" in merge_kwargs:
                        on_col = merge_kwargs["on"]
                        if isinstance(on_col, str):
                            return f"df.merge(other_df, on='{on_col}', how='{how}')"
                        else:
                            return f"df.merge(other_df, on={on_col}, how='{how}')"
                    elif "left_on" in merge_kwargs or "right_on" in merge_kwargs:
                        left_on = merge_kwargs.get("left_on", "...")
                        right_on = merge_kwargs.get("right_on", "...")
                        return f"df.merge(other_df, left_on='{left_on}', right_on='{right_on}', how='{how}')"
                    else:
                        return f"df.merge(other_df, how='{how}')"
                else:
                    return "df.merge(other_df)"
            else:
                return f"df.{operation_method}(...)"
        except Exception:
            return f"df.{operation_method}(...)"

    def _analyze_dataframe(self, df, variable_name: str) -> dict:
        """Analyze DataFrame for lineage event.

        Args:
            df: DataFrame to analyze
            variable_name: Variable name for the DataFrame

        Returns:
            dict: DataFrame analysis
        """
        try:
            return {
                "variable_name": variable_name,
                "object_id": str(id(df)),
                "shape": list(df.shape),
                "columns": list(df.columns),
                "memory_usage_mb": self._get_dataframe_memory_usage(df),
            }
        except Exception:
            return {
                "variable_name": variable_name,
                "object_id": str(id(df)),
                "shape": [0, 0],
                "columns": [],
                "memory_usage_mb": 0.0,
            }

    def _extract_operation_parameters(
        self, operation_method: str, *args, **kwargs
    ) -> dict:
        """Extract operation parameters for DataFrame lineage events.

        Phase 2.4 implementation with enhanced support for GroupBy operations.

        Args:
            operation_method: Method name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            dict: Operation parameters
        """
        try:
            if operation_method == "__getitem__":
                # Enhanced parameter extraction for __getitem__ operations
                params = {"method": operation_method}

                if args:
                    key = args[0]
                    if hasattr(key, "dtype") and key.dtype == bool:
                        params["selection_type"] = "boolean_mask"
                        params["mask_true_count"] = (
                            int(key.sum()) if hasattr(key, "sum") else 0
                        )
                    elif isinstance(key, str):
                        params["selection_type"] = "single_column"
                        params["column_name"] = key
                    elif isinstance(key, list):
                        params["selection_type"] = "multi_column"
                        params["column_names"] = key
                        params["column_count"] = len(key)
                    else:
                        params["selection_type"] = "other"
                        params["key_type"] = type(key).__name__
                elif "key" in kwargs:
                    key = kwargs["key"]
                    if hasattr(key, "dtype") and key.dtype == bool:
                        params["selection_type"] = "boolean_mask"
                        params["mask_true_count"] = (
                            int(key.sum()) if hasattr(key, "sum") else 0
                        )
                    elif isinstance(key, str):
                        params["selection_type"] = "single_column"
                        params["column_name"] = key
                    else:
                        params["selection_type"] = "other"
                        params["key_type"] = type(key).__name__
                else:
                    params["selection_type"] = "unknown"

                params["extraction_type"] = "enhanced"
                return params
            elif operation_method == "groupby":
                # Enhanced parameter extraction for GroupBy operations
                params = {"method": operation_method}

                if args:
                    groupby_cols = args[0]
                    if isinstance(groupby_cols, str):
                        params["groupby_type"] = "single_column"
                        params["groupby_column"] = groupby_cols
                    elif isinstance(groupby_cols, list):
                        params["groupby_type"] = "multi_column"
                        params["groupby_columns"] = groupby_cols
                        params["groupby_count"] = len(groupby_cols)
                    else:
                        params["groupby_type"] = "other"
                        params["groupby_key_type"] = type(groupby_cols).__name__
                else:
                    params["groupby_type"] = "unknown"

                # Add common GroupBy parameters
                params.update(
                    {
                        "as_index": kwargs.get("as_index", True),
                        "sort": kwargs.get("sort", True),
                        "group_keys": kwargs.get("group_keys", True),
                        "extraction_type": "enhanced",
                    }
                )
                return params
            elif operation_method in ["sum", "mean", "count"]:
                # Enhanced parameter extraction for GroupBy aggregation operations
                params = {"method": operation_method}

                # Add aggregation-specific parameters
                params.update(
                    {
                        "aggregation_type": operation_method,
                        "numeric_only": kwargs.get("numeric_only", None),
                        "min_count": (
                            kwargs.get("min_count", 0)
                            if operation_method == "sum"
                            else None
                        ),
                        "extraction_type": "enhanced",
                    }
                )

                # Add GroupBy context if available
                if "groupby_obj" in kwargs:
                    params["has_groupby_context"] = True
                else:
                    params["has_groupby_context"] = False

                return params
            elif operation_method == "merge":
                # Enhanced parameter extraction for DataFrame.merge operations
                params = {"method": operation_method}

                # Extract merge parameters from kwargs (passed as merge_kwargs)
                if "merge_kwargs" in kwargs:
                    merge_kwargs = kwargs["merge_kwargs"]

                    # Join type (how parameter)
                    params["join_type"] = merge_kwargs.get("how", "inner")

                    # Join keys
                    if "on" in merge_kwargs:
                        params["on"] = merge_kwargs["on"]
                        params["key_type"] = "common_column"
                    elif "left_on" in merge_kwargs or "right_on" in merge_kwargs:
                        params["left_on"] = merge_kwargs.get("left_on")
                        params["right_on"] = merge_kwargs.get("right_on")
                        params["key_type"] = "different_columns"
                    elif merge_kwargs.get("left_index") or merge_kwargs.get(
                        "right_index"
                    ):
                        params["left_index"] = merge_kwargs.get("left_index", False)
                        params["right_index"] = merge_kwargs.get("right_index", False)
                        params["key_type"] = "index_based"
                    else:
                        params["key_type"] = "default"

                    # Suffixes for overlapping columns
                    params["suffixes"] = merge_kwargs.get("suffixes", ("_x", "_y"))

                    # Other merge parameters
                    params["sort"] = merge_kwargs.get("sort", False)
                    params["validate"] = merge_kwargs.get("validate", None)
                    params["indicator"] = merge_kwargs.get("indicator", False)

                # Add DataFrame context from args
                if kwargs.get("merge_args"):
                    merge_args = kwargs["merge_args"]
                    if merge_args:
                        params["right_df_provided"] = True
                        params["merge_with_external_df"] = True
                    else:
                        params["right_df_provided"] = False

                params["extraction_type"] = "enhanced"
                return params
            else:
                # Basic parameter extraction for other operations
                return {
                    "method": operation_method,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                    "extraction_type": "basic",
                }
        except Exception:
            return {"method": operation_method, "extraction_type": "error"}

    def _capture_dataframe_operation(
        self,
        input_df,
        output_df,
        operation_method: str,
        execution_context: dict,
        *args,
        **kwargs,
    ) -> None:
        """Capture DataFrame operation as DataFrame lineage event.

        Phase 2.2 implementation for DataFrame.__getitem__ operations.
        This is the main method that will be called from monkey patches.

        Args:
            input_df: Input DataFrame
            output_df: Output DataFrame
            operation_method: Method name
            execution_context: Current execution context
            *args: Operation arguments
            **kwargs: Operation keyword arguments
        """
        # Early return if DataFrame lineage is disabled
        if not self.dataframe_lineage_enabled:
            return

        try:
            # Phase 2.3: Enhanced variable tracking with meaningful names
            input_obj_id = id(input_df)
            output_obj_id = id(output_df)

            # Get operation type for semantic naming
            operation_type = self._detect_operation_type(operation_method)

            # Get input variable name (try to find existing meaningful name first)
            input_var = self._get_variable_name_from_object_id(input_obj_id, "input")

            # Get output variable name with operation-specific hint
            output_var = self._get_variable_name_from_object_id(
                output_obj_id, operation_type
            )

            # Track variable relationship for lineage
            self._track_variable_relationship(
                input_var, output_var, operation_type, operation_method
            )

            # Detect operation type and extract parameters
            operation_type = self._detect_operation_type(operation_method)
            operation_params = self._extract_operation_parameters(
                operation_method, *args, **kwargs
            )
            column_lineage = self._generate_basic_column_lineage(
                input_df, output_df, operation_method
            )

            # Create DataFrame lineage event
            event_data = {
                "event_type": "dataframe_lineage",
                "execution_id": execution_context.get("execution_id", "unknown"),
                "cell_id": execution_context.get("cell_id", "unknown"),
                "session_id": self.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "emitted_at": datetime.now(timezone.utc).isoformat(),
                # Operation details
                "operation_type": operation_type,
                "operation_method": operation_method,
                "operation_code": self._extract_operation_code(
                    operation_method, *args, **kwargs
                ),
                "operation_parameters": operation_params,
                # DataFrame information
                "input_dataframes": [self._analyze_dataframe(input_df, input_var)],
                "output_dataframes": [self._analyze_dataframe(output_df, output_var)],
                "column_lineage": column_lineage,
                # Performance monitoring
                "performance": {
                    "overhead_ms": 0.5,  # Minimal overhead for MVP
                    "df_size_mb": self._get_dataframe_memory_usage(input_df),
                    "sampled": False,
                },
            }

            # Emit DataFrame lineage event using Step 2.1 infrastructure
            self._emit_dataframe_lineage_event(event_data)

            unified_logger.debug(
                f"[CAPTURE] DataFrame {operation_method} operation captured: {input_var} -> {output_var}"
            )

        except Exception as e:
            unified_logger.error(f"[ERROR] Failed to capture DataFrame operation: {e}")

    def uninstall(self):
        """Remove all installed hooks and restore original pandas methods"""
        if not self.interceptor_active:
            return

        try:
            # Remove marimo hooks
            from marimo._runtime.runner.hooks import (
                ON_FINISH_HOOKS,
                POST_EXECUTION_HOOKS,
                PRE_EXECUTION_HOOKS,
            )

            unified_logger.info("[UNINSTALL] Uninstalling unified interceptor...")

            # Remove all installed hooks
            for hook_list_name, hook_func in self.installed_hooks:
                if hook_list_name == "PRE_EXECUTION_HOOKS":
                    if hook_func in PRE_EXECUTION_HOOKS:
                        PRE_EXECUTION_HOOKS.remove(hook_func)
                elif hook_list_name == "POST_EXECUTION_HOOKS":
                    if hook_func in POST_EXECUTION_HOOKS:
                        POST_EXECUTION_HOOKS.remove(hook_func)
                elif hook_list_name == "ON_FINISH_HOOKS":
                    if hook_func in ON_FINISH_HOOKS:
                        ON_FINISH_HOOKS.remove(hook_func)

            # Restore original pandas methods
            if self.original_pandas_methods:
                import pandas as pd

                for (
                    method_name,
                    original_method,
                ) in self.original_pandas_methods.items():
                    if method_name == "DataFrame.__init__":
                        pd.DataFrame.__init__ = original_method
                    elif method_name == "DataFrame.__setitem__":
                        pd.DataFrame.__setitem__ = original_method
                    elif method_name == "DataFrame.__getitem__":
                        pd.DataFrame.__getitem__ = original_method
                    elif method_name == "DataFrame.groupby":
                        pd.DataFrame.groupby = original_method
                    elif method_name == "DataFrame.merge":
                        pd.DataFrame.merge = original_method
                    elif method_name == "GroupBy.sum":
                        pd.core.groupby.DataFrameGroupBy.sum = original_method
                    elif method_name == "GroupBy.mean":
                        pd.core.groupby.DataFrameGroupBy.mean = original_method
                    elif method_name == "GroupBy.count":
                        pd.core.groupby.DataFrameGroupBy.count = original_method

            self.installed_hooks.clear()
            self.original_pandas_methods.clear()
            self._execution_contexts.clear()

            # NEW: Clear DataFrame lineage tracking state (Phase 2.1)
            self.object_to_variable.clear()
            self.performance_tracking.clear()

            self.interceptor_active = False

            unified_logger.success("[OK] Unified interceptor uninstalled")

        except Exception as e:
            unified_logger.warning(f"[WARN] Error during uninstall: {e}")

    def get_session_summary(self):
        """Get session summary"""
        return {
            "session_id": self.session_id,
            "interceptor_active": self.interceptor_active,
            "runtime_file": str(self.runtime_file),
            "lineage_file": str(self.lineage_file),
            "dataframe_lineage_file": str(
                self.dataframe_lineage_file
            ),  # NEW: Third file stream
            "hooks_installed": len(self.installed_hooks),
            "dataframes_tracked": len(_tracked_dataframes),
            "active_executions": len(self._execution_contexts),
            "dataframe_lineage_enabled": self.dataframe_lineage_enabled,  # NEW: Configuration status
            "variable_mappings": len(self.object_to_variable),  # NEW: Variable tracking
            "dataframe_lineage_config": self.dataframe_lineage_config,  # NEW: Full config
        }


def enable_unified_tracking(
    notebook_path: str | None = None,
    runtime_file: str | None = None,
    lineage_file: str | None = None,
    dataframe_lineage_file: str | None = None,
):
    """Enable unified marimo tracking for runtime, lineage, and DataFrame lineage events"""
    global _global_unified_interceptor  # noqa: PLW0603

    if _global_unified_interceptor is not None:
        unified_logger.warning(
            f"[WARN] Unified tracking already enabled (session: {_global_unified_interceptor.session_id})"
        )
        return _global_unified_interceptor

    unified_logger.info("[INIT] Creating unified interceptor instance...")
    _global_unified_interceptor = UnifiedMarimoInterceptor(
        notebook_path=notebook_path,
        runtime_file=runtime_file,
        lineage_file=lineage_file,
        dataframe_lineage_file=dataframe_lineage_file,
    )
    _global_unified_interceptor.install()
    return _global_unified_interceptor


def disable_unified_tracking():
    """Disable unified marimo tracking"""
    global _global_unified_interceptor  # noqa: PLW0603

    if _global_unified_interceptor is not None:
        _global_unified_interceptor.uninstall()
        _global_unified_interceptor = None
        unified_logger.info("[OK] Unified tracking disabled")
    else:
        unified_logger.warning("[WARN] Unified tracking was not active")


def is_unified_tracking_active():
    """Check if unified tracking is active"""
    return (
        _global_unified_interceptor is not None
        and _global_unified_interceptor.interceptor_active
    )


def get_unified_interceptor():
    """Get the current unified interceptor instance"""
    return _global_unified_interceptor
