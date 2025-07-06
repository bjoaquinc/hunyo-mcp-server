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
    ):
        # If specific file paths are provided, use them directly
        if runtime_file or lineage_file:
            self.runtime_file = Path(runtime_file or "marimo_runtime_events.jsonl")
            self.lineage_file = Path(lineage_file or "marimo_lineage_events.jsonl")
        elif notebook_path:
            # Generate proper file paths using naming convention
            from hunyo_capture import get_event_filenames, get_user_data_dir

            data_dir = get_user_data_dir()
            runtime_file_path, lineage_file_path = get_event_filenames(
                notebook_path, data_dir
            )
            self.runtime_file = Path(runtime_file_path)
            self.lineage_file = Path(lineage_file_path)
        else:
            # Fallback to defaults
            self.runtime_file = Path("marimo_runtime_events.jsonl")
            self.lineage_file = Path("marimo_lineage_events.jsonl")

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

        # Ensure both directories exist
        self.runtime_file.parent.mkdir(parents=True, exist_ok=True)
        self.lineage_file.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_file.touch()
        self.lineage_file.touch()

        unified_logger.status("Unified Marimo Interceptor v1.0")
        unified_logger.runtime(f"Runtime events: {self.runtime_file.name}")
        unified_logger.lineage(f"Lineage events: {self.lineage_file.name}")
        unified_logger.config(f"Session: {self.session_id}")

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

        except ImportError:
            unified_logger.warning("[WARN] pandas not available for DataFrame tracking")
        except Exception as e:
            unified_logger.error(f"[ERROR] Failed to install DataFrame patches: {e}")

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

        # Return the most recent execution context
        # In practice, there should only be one active execution at a time
        return next(iter(self._execution_contexts.values()))

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

            self.installed_hooks.clear()
            self.original_pandas_methods.clear()
            self._execution_contexts.clear()
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
            "hooks_installed": len(self.installed_hooks),
            "dataframes_tracked": len(_tracked_dataframes),
            "active_executions": len(self._execution_contexts),
        }


def enable_unified_tracking(
    notebook_path: str | None = None,
    runtime_file: str | None = None,
    lineage_file: str | None = None,
):
    """Enable unified marimo tracking for both runtime and lineage events"""
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
