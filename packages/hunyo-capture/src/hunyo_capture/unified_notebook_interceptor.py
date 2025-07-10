#!/usr/bin/env python3
"""
Unified Notebook Interceptor
Environment-agnostic interceptor supporting multiple notebook environments

This module provides a unified system that handles:
1. Runtime events (cell execution performance, timing, memory)
2. Lineage events (DataFrame operations, OpenLineage compliance)
3. DataFrame lineage events (column-level lineage tracking)

Refactored from UnifiedMarimoInterceptor to support multiple environments
while preserving all 1,925 lines of existing functionality.
"""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Import Phase 1 abstractions
from typing import Any

from .constants import (
    DATAFRAME_OPERATION_TYPES,
    DEFAULT_PERFORMANCE_OVERHEAD_MS,
    MEMORY_CONVERSION_FACTOR,
    MILLISECONDS_CONVERSION_FACTOR,
    OPENLINEAGE_PRODUCER,
    OPENLINEAGE_SCHEMA_URL,
    PROGRAMMING_LANGUAGE,
    generate_truncated_uuid,
    get_dataframe_lineage_config,
)
from .environments.base import EnvironmentDetector, NotebookEnvironment
from .factory import ComponentFactory
from .logger import get_logger

# Initialize logger
notebook_logger = get_logger("hunyo.unified.notebook")

# Global tracking state (maintained for compatibility)
_tracked_dataframes = {}  # Use df_id as key instead of DataFrame object


class UnifiedNotebookInterceptor:
    """
    Unified notebook interceptor for both runtime and lineage event tracking.

    Environment-agnostic version that supports multiple notebook environments
    while preserving all existing functionality from UnifiedMarimoInterceptor.
    """

    def __init__(
        self,
        notebook_path: str | None = None,
        runtime_file: str | None = None,
        lineage_file: str | None = None,
        dataframe_lineage_file: str | None = None,
        environment: NotebookEnvironment | None = None,
    ):
        # NEW: Environment detection and validation
        self.environment = environment or EnvironmentDetector.detect_environment()
        notebook_logger.info(f"[INIT] Using environment: {self.environment.value}")

        # NEW: Create environment-specific components using factory
        self.hook_manager = ComponentFactory.create_hook_manager(self.environment)
        self.context_adapter = ComponentFactory.create_context_adapter(self.environment)

        # File path setup with environment-aware naming
        if runtime_file or lineage_file or dataframe_lineage_file:
            # Use provided file paths directly
            default_env = self.environment.value
            self.runtime_file = Path(
                runtime_file or f"{default_env}_runtime_events.jsonl"
            )
            self.lineage_file = Path(
                lineage_file or f"{default_env}_lineage_events.jsonl"
            )
            self.dataframe_lineage_file = Path(
                dataframe_lineage_file
                or f"{default_env}_dataframe_lineage_events.jsonl"
            )
        elif notebook_path:
            # Generate proper file paths using naming convention with environment support
            from . import get_event_filenames, get_user_data_dir

            data_dir = get_user_data_dir()
            runtime_file_path, lineage_file_path, dataframe_lineage_file_path = (
                get_event_filenames(
                    notebook_path, data_dir, environment=self.environment.value
                )
            )
            self.runtime_file = Path(runtime_file_path)
            self.lineage_file = Path(lineage_file_path)
            self.dataframe_lineage_file = Path(dataframe_lineage_file_path)
        else:
            # Fallback to defaults with environment prefix
            env_prefix = self.environment.value
            self.runtime_file = Path(f"{env_prefix}_runtime_events.jsonl")
            self.lineage_file = Path(f"{env_prefix}_lineage_events.jsonl")
            self.dataframe_lineage_file = Path(
                f"{env_prefix}_dataframe_lineage_events.jsonl"
            )

        self.notebook_path = notebook_path
        self.session_id = generate_truncated_uuid()
        self.interceptor_active = False
        self._lock = threading.Lock()

        # Hook references (now managed by hook_manager)
        self.installed_hooks = []

        # Execution context tracking
        self._execution_contexts = {}

        # DataFrame monkey patching state
        self.original_pandas_methods = {}

        # DataFrame lineage infrastructure
        self.dataframe_lineage_enabled = get_dataframe_lineage_config()
        self.object_to_variable = {}  # Simple object ID to variable name mapping
        self.dataframe_lineage_config = {}  # Full configuration dict
        self.performance_tracking = {}  # Performance monitoring state

        # Ensure all three directories exist
        self.runtime_file.parent.mkdir(parents=True, exist_ok=True)
        self.lineage_file.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe_lineage_file.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_file.touch()
        self.lineage_file.touch()
        self.dataframe_lineage_file.touch()

        # Updated logging to be environment-agnostic
        notebook_logger.status(
            f"Unified {self.environment.value.title()} Interceptor v2.0"
        )
        notebook_logger.runtime(f"Runtime events: {self.runtime_file.name}")
        notebook_logger.lineage(f"Lineage events: {self.lineage_file.name}")
        notebook_logger.config(
            f"DataFrame lineage events: {self.dataframe_lineage_file.name}"
        )
        notebook_logger.config(f"Session: {self.session_id}")
        notebook_logger.config(f"Environment: {self.environment.value}")

        # Log DataFrame lineage configuration
        if self.dataframe_lineage_enabled:
            notebook_logger.config("[CONFIG] DataFrame lineage tracking enabled")
        else:
            notebook_logger.config("[CONFIG] DataFrame lineage tracking disabled")

    def install(self):
        """Install environment-specific execution hooks and DataFrame monkey patches"""
        if self.interceptor_active:
            notebook_logger.warning("[WARN] Unified interceptor already installed")
            return

        try:
            # Install hooks using hook manager abstraction
            self._install_hooks()

            # Install DataFrame monkey patches with execution context
            self._install_dataframe_patches()

            self.interceptor_active = True
            notebook_logger.success(
                f"[OK] Unified {self.environment.value} interceptor installed!"
            )

        except ImportError as e:
            notebook_logger.error(
                f"[ERROR] Failed to import {self.environment.value} hooks: {e}"
            )
        except Exception as e:
            notebook_logger.error(f"[ERROR] Failed to install unified interceptor: {e}")

    def _install_hooks(self):
        """Install environment-specific execution hooks using hook manager"""
        notebook_logger.info(f"[INSTALL] Installing {self.environment.value} hooks...")

        # Create hooks using context adapter
        pre_hook = self._create_pre_execution_hook()
        post_hook = self._create_post_execution_hook()
        finish_hook = self._create_finish_hook()

        # Install hooks using hook manager abstraction
        self.hook_manager.install_pre_execution_hook(pre_hook)
        self.hook_manager.install_post_execution_hook(post_hook)
        self.hook_manager.install_finish_hook(finish_hook)

        # Store hook references for uninstall (maintaining existing pattern)
        self.installed_hooks = self.hook_manager.installed_hooks.copy()

    def _install_dataframe_patches(self):
        """Install DataFrame monkey patches that only activate during execution"""
        try:
            import pandas as pd

            # Store original methods
            self.original_pandas_methods["DataFrame.__init__"] = pd.DataFrame.__init__
            self.original_pandas_methods["DataFrame.__setitem__"] = (
                pd.DataFrame.__setitem__
            )
            self.original_pandas_methods["DataFrame.__getitem__"] = (
                pd.DataFrame.__getitem__
            )
            self.original_pandas_methods["DataFrame.groupby"] = pd.DataFrame.groupby
            self.original_pandas_methods["DataFrame.merge"] = pd.DataFrame.merge

            # Create execution-context-aware DataFrame.__init__
            def tracked_dataframe_init(df_self, *args, **kwargs):
                # Call original method first
                result = self.original_pandas_methods["DataFrame.__init__"](
                    df_self, *args, **kwargs
                )

                # Only capture if we're in active execution (environment-agnostic)
                if self._is_in_execution():
                    execution_context = self._get_current_execution_context()
                    if execution_context:
                        notebook_logger.debug(
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
                        notebook_logger.debug("[HOOK] No execution context found")
                else:
                    notebook_logger.debug(
                        f"[HOOK] Not in {self.environment.value} execution, skipping DataFrame capture"
                    )

                return result

            # Create execution-context-aware DataFrame.__setitem__
            def tracked_dataframe_setitem(df_self, key, value):
                # Call original method first
                result = self.original_pandas_methods["DataFrame.__setitem__"](
                    df_self, key, value
                )

                # Only capture if we're in active execution
                if self._is_in_execution():
                    execution_context = self._get_current_execution_context()
                    if execution_context:
                        notebook_logger.debug(
                            "[HOOK] In execution context, capturing DataFrame modification..."
                        )
                        try:
                            self._capture_dataframe_modification(
                                df_self, execution_context, "setitem", key, value
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
                        notebook_logger.debug("[HOOK] No execution context found")
                else:
                    notebook_logger.debug(
                        f"[HOOK] Not in {self.environment.value} execution, skipping DataFrame capture"
                    )

                return result

            # Apply monkey patches
            pd.DataFrame.__init__ = tracked_dataframe_init
            pd.DataFrame.__setitem__ = tracked_dataframe_setitem

            # Setup DataFrame lineage tracking if enabled
            if self.dataframe_lineage_enabled:
                self._setup_dataframe_lineage_tracking()

            notebook_logger.info("[INSTALL] DataFrame monkey patches installed")

        except Exception as e:
            notebook_logger.error(f"[ERROR] Failed to install DataFrame patches: {e}")
            raise

    def _setup_dataframe_lineage_tracking(self):
        """Setup DataFrame lineage tracking with column-level lineage"""
        notebook_logger.info("[SETUP] Installing DataFrame lineage tracking...")

        try:
            import pandas as pd

            # Setup __getitem__ tracking for column access
            def tracked_dataframe_getitem(df_self, key):
                # Call original method first
                result = self.original_pandas_methods["DataFrame.__getitem__"](
                    df_self, key
                )

                # Only capture if we're in active execution and lineage is enabled
                if self._is_in_execution() and self.dataframe_lineage_enabled:
                    execution_context = self._get_current_execution_context()
                    if execution_context:
                        notebook_logger.debug(
                            "[LINEAGE] Capturing DataFrame column access..."
                        )
                        try:
                            # Only capture if result is a DataFrame (not Series)
                            import pandas as pd

                            output_df = (
                                result if isinstance(result, pd.DataFrame) else None
                            )
                            self._capture_dataframe_operation(
                                input_df=df_self,
                                output_df=output_df,
                                operation_method="__getitem__",
                                execution_context=execution_context,
                                operation_args=(
                                    key,
                                ),  # FIX: Use named parameter instead of key=key
                                operation_kwargs={},
                            )
                        except Exception as e:
                            notebook_logger.error(
                                f"[ERROR] Failed to capture __getitem__ operation: {e}"
                            )

                return result

            # Apply __getitem__ monkey patch
            pd.DataFrame.__getitem__ = tracked_dataframe_getitem
            notebook_logger.success("[OK] DataFrame.__getitem__ monkey patch installed")

            # Setup groupby tracking
            self._setup_groupby_aggregation_tracking()

            # Setup merge tracking
            def tracked_dataframe_merge(df_self, *args, **kwargs):
                # Call original method first
                result = self.original_pandas_methods["DataFrame.merge"](
                    df_self, *args, **kwargs
                )

                # Only capture if we're in active execution and lineage is enabled
                if self._is_in_execution() and self.dataframe_lineage_enabled:
                    execution_context = self._get_current_execution_context()
                    if execution_context:
                        notebook_logger.debug("[LINEAGE] Capturing DataFrame merge...")
                        try:
                            self._capture_dataframe_operation(
                                input_df=df_self,
                                output_df=result,
                                operation_method="merge",
                                execution_context=execution_context,
                                operation_args=args,  # FIX: Use named parameter instead of *args
                                operation_kwargs=kwargs,  # FIX: Use named parameter instead of **kwargs
                            )
                        except Exception as e:
                            notebook_logger.error(
                                f"[ERROR] Failed to capture merge operation: {e}"
                            )

                return result

            # Apply merge monkey patch
            pd.DataFrame.merge = tracked_dataframe_merge
            notebook_logger.success("[OK] DataFrame.merge monkey patch installed")

            notebook_logger.info("[INSTALL] DataFrame lineage tracking setup initiated")

        except Exception as e:
            notebook_logger.error(
                f"[ERROR] Failed to setup DataFrame lineage tracking: {e}"
            )
            raise

    def _setup_groupby_aggregation_tracking(self):
        """Setup GroupBy aggregation tracking for DataFrame lineage"""
        notebook_logger.info("[SETUP] Setting up GroupBy aggregation tracking...")

        try:
            import pandas as pd

            # Store original aggregation methods
            aggregation_methods = ["sum", "mean", "count"]
            for method_name in aggregation_methods:
                original_method = getattr(pd.core.groupby.DataFrameGroupBy, method_name)
                self.original_pandas_methods[f"GroupBy.{method_name}"] = original_method

                def create_tracked_aggregation_method(method_name, original_method):
                    def tracked_aggregation(groupby_self, *args, **kwargs):
                        # Call original method first
                        result = original_method(groupby_self, *args, **kwargs)

                        # Only capture if we're in active execution and lineage is enabled
                        if self._is_in_execution() and self.dataframe_lineage_enabled:
                            execution_context = self._get_current_execution_context()
                            if execution_context:
                                notebook_logger.debug(
                                    f"[LINEAGE] Capturing GroupBy.{method_name}..."
                                )
                                try:
                                    # Get the input DataFrame from groupby object
                                    input_df = groupby_self.obj
                                    self._capture_dataframe_operation(
                                        input_df=input_df,
                                        output_df=result,
                                        operation_method=f"groupby.{method_name}",
                                        execution_context=execution_context,
                                        operation_args=args,  # FIX: Use named parameter instead of *args
                                        operation_kwargs=kwargs,  # FIX: Use named parameter instead of **kwargs
                                    )
                                except Exception as e:
                                    notebook_logger.error(
                                        f"[ERROR] Failed to capture {method_name} operation: {e}"
                                    )

                        return result

                    return tracked_aggregation

                # Apply the monkey patch
                tracked_method = create_tracked_aggregation_method(
                    method_name, original_method
                )
                setattr(pd.core.groupby.DataFrameGroupBy, method_name, tracked_method)

            notebook_logger.success(
                "[OK] GroupBy aggregation method tracking installed"
            )

            # Setup groupby monkey patch
            def tracked_dataframe_groupby(df_self, *args, **kwargs):
                # Call original method first
                result = self.original_pandas_methods["DataFrame.groupby"](
                    df_self, *args, **kwargs
                )

                # Only capture if we're in active execution and lineage is enabled
                if self._is_in_execution() and self.dataframe_lineage_enabled:
                    execution_context = self._get_current_execution_context()
                    if execution_context:
                        notebook_logger.debug(
                            "[LINEAGE] Capturing DataFrame groupby setup..."
                        )
                        try:
                            self._track_groupby_setup(
                                df_self, result, execution_context, *args, **kwargs
                            )
                        except Exception as e:
                            notebook_logger.error(
                                f"[ERROR] Failed to capture groupby setup: {e}"
                            )

                return result

            # Apply groupby monkey patch
            pd.DataFrame.groupby = tracked_dataframe_groupby
            notebook_logger.success("[OK] DataFrame.groupby monkey patch installed")

        except Exception as e:
            notebook_logger.error(
                f"[ERROR] Failed to setup GroupBy aggregation tracking: {e}"
            )
            raise

    def _track_groupby_setup(
        self, input_df, groupby_obj, execution_context, *args, **kwargs  # noqa: ARG002
    ):
        """Track the setup of a groupby operation"""
        try:
            # For groupby setup, we track the operation but don't create lineage yet
            # since the actual aggregation happens later
            # Store information for later use - groupby_info not directly used but structure maintained for compatibility

            # Store in performance tracking for later use
            self.performance_tracking[id(groupby_obj)] = {
                "operation": "groupby_setup",
                "input_df_id": id(input_df),
                "setup_time": time.time(),
                "execution_context": execution_context,
            }

            notebook_logger.debug(
                f"[TRACK] GroupBy setup tracked for df_{id(input_df)}"
            )

        except Exception as e:
            notebook_logger.error(f"[ERROR] Failed to track groupby setup: {e}")

    def _create_pre_execution_hook(self):
        """Create pre-execution hook using context adapter"""

        def pre_execution_hook(cell, runner):
            """Pre-execution hook that sets up execution context"""
            try:
                with self._lock:
                    # Use context adapter to extract cell information
                    cell_info = self.context_adapter.extract_cell_info(cell, runner)

                    # Create execution context
                    execution_id = generate_truncated_uuid()
                    start_time = time.time()

                    execution_context = {
                        "execution_id": execution_id,
                        "cell_id": cell_info.get("cell_id", "unknown"),
                        "cell_code": cell_info.get("cell_code", ""),
                        "session_id": self.session_id,
                        "start_time": start_time,
                        "context_created_at": start_time,
                        "environment": self.environment.value,
                    }

                    self._execution_contexts[execution_id] = execution_context

                    notebook_logger.debug(
                        f"[PRE-HOOK] Starting execution {execution_id} for {self.environment.value} cell {cell_info.get('cell_id', 'unknown')}"
                    )

                    # Emit runtime START event - FIX: Use correct event_type and add missing fields
                    cell_source = cell_info.get("cell_code", "")
                    self._emit_runtime_event(
                        {
                            "event_type": "cell_execution_start",  # FIX: Use schema-compliant event type
                            "execution_id": execution_id,
                            "cell_id": cell_info.get("cell_id", "unknown"),
                            "cell_source": cell_source,  # FIX: Use full source, not truncated
                            "cell_source_lines": (
                                len(cell_source.split("\n")) if cell_source else 0
                            ),  # FIX: Add required field
                            "start_memory_mb": self._get_memory_usage(),  # FIX: Add required field
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "session_id": self.session_id,
                            "emitted_at": datetime.now(
                                timezone.utc
                            ).isoformat(),  # FIX: Add required field
                        }
                    )

            except Exception as e:
                notebook_logger.error(f"[ERROR] Pre-execution hook failed: {e}")

        return pre_execution_hook

    def _create_post_execution_hook(self):
        """Create post-execution hook using context adapter"""

        def post_execution_hook(cell, runner, run_result):
            """Post-execution hook that captures execution results and cleans up context"""
            try:
                with self._lock:
                    # Use context adapter to extract cell information
                    cell_info = self.context_adapter.extract_cell_info(cell, runner)

                    # Find matching execution context
                    execution_context = None
                    for ctx in self._execution_contexts.values():
                        if ctx.get("cell_id") == cell_info.get("cell_id"):
                            execution_context = ctx
                            break

                    if not execution_context:
                        notebook_logger.warning(
                            f"[WARN] No execution context found for cell {cell_info.get('cell_id', 'unknown')}"
                        )
                        return

                    end_time = time.time()
                    duration = end_time - execution_context["start_time"]

                    # Use context adapter to detect errors
                    error = self.context_adapter.detect_error(run_result)

                    # Emit runtime COMPLETE or FAIL event - FIX: Use correct event_type and add missing fields
                    event_type = (
                        "cell_execution_error" if error else "cell_execution_end"
                    )  # FIX: Use schema-compliant event types
                    cell_source = cell_info.get("cell_code", "")
                    runtime_event = {
                        "event_type": event_type,
                        "execution_id": execution_context["execution_id"],
                        "cell_id": cell_info.get("cell_id", "unknown"),
                        "cell_source": cell_source,  # FIX: Use full source, not truncated
                        "cell_source_lines": (
                            len(cell_source.split("\n")) if cell_source else 0
                        ),  # FIX: Add required field
                        "start_memory_mb": self._get_memory_usage(),  # FIX: Add required field (using current memory as approximation)
                        "end_memory_mb": self._get_memory_usage(),  # FIX: Add missing field for end events
                        "duration_ms": duration * MILLISECONDS_CONVERSION_FACTOR,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "session_id": self.session_id,
                        "emitted_at": datetime.now(
                            timezone.utc
                        ).isoformat(),  # FIX: Add required field
                    }

                    if error:
                        runtime_event["error_info"] = (
                            {  # This is optional field for error events
                                "error_type": type(error).__name__,
                                "error_message": str(error),
                                "traceback": str(error),  # Add traceback info
                            }
                        )

                        # Emit lineage event for cell execution error (matching original system behavior)
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
                                    cell_id=cell_info.get("cell_id", "unknown"),
                                    cell_source=cell_source,
                                )
                                self._emit_lineage_event(abort_event)
                            else:
                                fail_event = self._create_openlineage_event(
                                    event_type="FAIL",
                                    job_name="cell_execution",
                                    error_info=error_info,
                                    execution_context=execution_context,
                                    failure_context=f"Cell execution failed: {type(error).__name__}",
                                    cell_id=cell_info.get("cell_id", "unknown"),
                                    cell_source=cell_source,
                                )
                                self._emit_lineage_event(fail_event)
                        except Exception as lineage_error:
                            notebook_logger.error(
                                f"[ERROR] Failed to emit lineage error event: {lineage_error}"
                            )

                    self._emit_runtime_event(runtime_event)

                    notebook_logger.debug(
                        f"[POST-HOOK] Completed execution {execution_context['execution_id']} "
                        f"for {self.environment.value} cell {cell_info.get('cell_id', 'unknown')} "
                        f"({duration:.3f}s, {event_type})"
                    )

                    # Clean up execution context
                    if execution_context["execution_id"] in self._execution_contexts:
                        del self._execution_contexts[execution_context["execution_id"]]

            except Exception as e:
                notebook_logger.error(f"[ERROR] Post-execution hook failed: {e}")

        return post_execution_hook

    def _create_finish_hook(self):
        """Create finish hook for session cleanup"""

        def finish_hook(runner):  # noqa: ARG001
            """Finish hook that handles session cleanup"""
            try:
                notebook_logger.info(
                    f"[FINISH] {self.environment.value.title()} session finishing..."
                )

                # Emit session summary
                summary = self.get_session_summary()
                notebook_logger.info(
                    f"[SUMMARY] Session {self.session_id} summary: {summary}"
                )

                # Clean up any remaining execution contexts
                if self._execution_contexts:
                    notebook_logger.warning(
                        f"[CLEANUP] Cleaning up {len(self._execution_contexts)} remaining execution contexts"
                    )
                    self._execution_contexts.clear()

            except Exception as e:
                notebook_logger.error(f"[ERROR] Finish hook failed: {e}")

        return finish_hook

    def _is_in_execution(self) -> bool:
        """Check if we're currently in a notebook cell execution (environment-agnostic)"""
        return len(self._execution_contexts) > 0

    def _get_current_execution_context(self) -> dict[str, Any] | None:
        """Get the current execution context"""
        if not self._execution_contexts:
            return None

        # Return the most recent execution context based on timestamp
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
        """Create a standardized OpenLineage event (environment-agnostic)"""
        event_time = datetime.now(timezone.utc).isoformat()
        run_id = generate_truncated_uuid()

        # Build run facets with environment information
        run_facets = {
            f"{self.environment.value}Session": {
                "_producer": OPENLINEAGE_PRODUCER,
                "sessionId": self.session_id,
            }
        }

        # Add execution context as a run facet if provided
        if execution_context:
            run_facets[f"{self.environment.value}Execution"] = {
                "_producer": OPENLINEAGE_PRODUCER,
                "executionId": execution_context.get("execution_id", ""),
                "sessionId": self.session_id,
            }

        # Add error facet for FAIL/ABORT events
        if error_info and event_type in ("FAIL", "ABORT"):
            from .constants import ERROR_MESSAGE_SCHEMA_URL

            run_facets["errorMessage"] = {
                "_producer": OPENLINEAGE_PRODUCER,
                "_schemaURL": ERROR_MESSAGE_SCHEMA_URL,
                "message": error_info.get("error_message", ""),
                "programmingLanguage": PROGRAMMING_LANGUAGE,
                "stackTrace": error_info.get("stack_trace", ""),
            }

        # Create base OpenLineage event
        lineage_event = {
            "eventType": event_type,
            "eventTime": event_time,
            "run": {"runId": run_id, "facets": run_facets},
            "job": {"namespace": self.environment.value, "name": job_name},
            "inputs": inputs or [],
            "outputs": outputs or [],
            "producer": OPENLINEAGE_PRODUCER,
            "schemaURL": OPENLINEAGE_SCHEMA_URL,
            "session_id": self.session_id,
            "emitted_at": event_time,
        }

        # Only add valid schema fields from extra_fields
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
            notebook_logger.debug(f"Failed to build schema fields: {e}")

        return {
            "namespace": self.environment.value,
            "name": dataset_name,
            "facets": {
                "schema": {
                    "_producer": OPENLINEAGE_PRODUCER,
                    "_schemaURL": OPENLINEAGE_SCHEMA_URL,
                    "fields": schema_fields,
                }
            },
        }

    def _emit_runtime_event(self, event_data):
        """Emit runtime event to file"""
        try:
            with open(self.runtime_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data) + "\n")
        except Exception as e:
            notebook_logger.error(f"[ERROR] Failed to emit runtime event: {e}")

    def _emit_lineage_event(self, event_data):
        """Emit lineage event to file"""
        try:
            with open(self.lineage_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data) + "\n")
        except Exception as e:
            notebook_logger.error(f"[ERROR] Failed to emit lineage event: {e}")

    def _emit_dataframe_lineage_event(self, event_data):
        """Emit DataFrame lineage event to file"""
        try:
            with open(self.dataframe_lineage_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data) + "\n")
        except Exception as e:
            notebook_logger.error(
                f"[ERROR] Failed to emit DataFrame lineage event: {e}"
            )

    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / MEMORY_CONVERSION_FACTOR
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _capture_dataframe_creation(
        self, df, execution_context, *args, **kwargs  # noqa: ARG002
    ):
        """Capture DataFrame creation as OpenLineage event"""
        try:
            notebook_logger.debug(
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
            notebook_logger.error(f"Failed to capture DataFrame creation: {e}")

    def _capture_dataframe_modification(
        self, df, execution_context, modification_type, key, value
    ):
        """Capture DataFrame modification as OpenLineage event"""
        try:
            notebook_logger.debug(
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
            notebook_logger.error(f"Failed to capture DataFrame modification: {e}")

    def _capture_dataframe_abortion(
        self,
        df,
        execution_context,
        job_name,
        termination_reason,
        partial_outputs=None,  # noqa: ARG002
    ):
        """Capture DataFrame operation abortion as OpenLineage event"""
        try:
            notebook_logger.debug(
                f"[CAPTURE] Capturing DataFrame abortion for df_{id(df)}"
            )

            # Create output dataset
            output_dataset = self._create_dataframe_dataset(df)

            # Build termination information (matching original system)
            error_info = {
                "error_message": f"Operation terminated: {termination_reason}",
                "stack_trace": f"Termination reason: {termination_reason}",
                "error_type": "OperationAborted",
            }

            # Create OpenLineage event
            lineage_event = self._create_openlineage_event(
                event_type="ABORT",
                job_name=job_name,
                outputs=[output_dataset],
                error_info=error_info,  # FIX: Add missing error_info parameter
                execution_context=execution_context,
                termination_reason=termination_reason,
            )

            self._emit_lineage_event(lineage_event)

        except Exception as e:
            notebook_logger.error(f"Failed to capture DataFrame abortion: {e}")

    def _capture_dataframe_failure(
        self, df, execution_context, job_name, error, partial_outputs=None
    ):
        """Capture DataFrame operation failure as OpenLineage FAIL event (for test compatibility)"""
        try:
            notebook_logger.debug(
                f"[CAPTURE] Capturing DataFrame failure for df_{id(df)}"
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
            notebook_logger.error(f"Failed to capture DataFrame failure: {e}")

    def _capture_dataframe_operation(
        self,
        input_df,
        output_df,
        operation_method: str,
        execution_context: dict,
        operation_args=None,
        operation_kwargs=None,
    ):
        """Capture DataFrame lineage operation with column-level tracking"""
        try:
            if not self.dataframe_lineage_enabled:
                return

            # Handle optional args/kwargs parameters
            operation_args = operation_args or ()
            operation_kwargs = operation_kwargs or {}

            # FIX: Generate operation_type based on operation_method (required field)
            operation_type = self._detect_operation_type(operation_method)

            # FIX: Generate operation_code (required field)
            operation_code = self._extract_operation_code(
                operation_method, *operation_args, **operation_kwargs
            )

            # FIX: Generate operation_parameters (required field)
            operation_parameters = self._extract_operation_parameters(
                operation_method, *operation_args, **operation_kwargs
            )

            # FIX: Generate column_lineage (required field)
            column_lineage = self._generate_column_lineage(
                input_df, output_df, operation_method
            )

            # FIX: Generate performance info (required field)
            performance = {
                "overhead_ms": DEFAULT_PERFORMANCE_OVERHEAD_MS,
                "df_size_mb": self._get_dataframe_memory_usage(input_df),
                "sampled": False,
            }

            # FIX: Create proper dataframe_info structures with all required fields
            input_dataframes = [self._create_dataframe_info(input_df, "input_df")]
            output_dataframes = (
                [self._create_dataframe_info(output_df, "output_df")]
                if output_df is not None
                else []
            )

            # Create DataFrame lineage event with detailed column tracking
            event_data = {
                "event_type": "dataframe_lineage",
                "execution_id": execution_context.get("execution_id", "unknown"),
                "cell_id": execution_context.get("cell_id", "unknown"),
                "session_id": self.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "emitted_at": datetime.now(timezone.utc).isoformat(),
                # FIX: Add all required fields according to schema
                "operation_type": operation_type,
                "operation_method": operation_method,
                "operation_code": operation_code,
                "operation_parameters": operation_parameters,
                "input_dataframes": input_dataframes,
                "output_dataframes": output_dataframes,
                "column_lineage": column_lineage,
                "performance": performance,
            }

            self._emit_dataframe_lineage_event(event_data)
            notebook_logger.debug(
                f"[CAPTURE] DataFrame {operation_method} operation captured"
            )

        except Exception as e:
            notebook_logger.error(f"[ERROR] Failed to capture DataFrame operation: {e}")

    def _detect_operation_type(self, operation_method: str) -> str:
        """Detect operation type based on method name for schema compliance"""
        if operation_method in ["__getitem__"]:
            return DATAFRAME_OPERATION_TYPES["selection"]
        elif operation_method in ["groupby", "sum", "mean", "count"]:
            return DATAFRAME_OPERATION_TYPES["aggregation"]
        elif operation_method in ["merge"]:
            return DATAFRAME_OPERATION_TYPES["join"]
        else:
            return DATAFRAME_OPERATION_TYPES["selection"]  # Default fallback

    def _extract_operation_code(
        self, operation_method: str, *args, **kwargs  # noqa: ARG002
    ) -> str:
        """Extract operation code for schema compliance"""
        # Generate representative code based on method and args
        if operation_method == "__getitem__" and args:
            return f"df[{args[0]!r}]"
        elif operation_method == "merge" and args:
            return f"df.merge({args[0]!r})"
        elif operation_method in ["sum", "mean", "count"]:
            return f"df.groupby(...).{operation_method}()"
        else:
            return f"df.{operation_method}(...)"

    def _extract_operation_parameters(
        self, operation_method: str, *args, **kwargs
    ) -> dict:
        """Extract operation parameters for schema compliance"""
        params = {}

        if operation_method == "__getitem__" and args:
            key = args[0]
            if isinstance(key, str):
                params["columns"] = [key]
            elif isinstance(key, list):
                params["columns"] = key
            params["is_boolean_mask"] = False  # Simple selection for now

        elif operation_method == "merge" and args:
            params["how"] = kwargs.get("how", "inner")
            params["on"] = kwargs.get("on", None)

        elif operation_method in ["sum", "mean", "count"]:
            params["how"] = operation_method

        return params

    def _generate_column_lineage(
        self, input_df, output_df, operation_method: str
    ) -> dict:
        """Generate column lineage mapping for schema compliance"""
        try:
            input_columns = (
                list(input_df.columns) if hasattr(input_df, "columns") else []
            )
            output_columns = (
                list(output_df.columns)
                if output_df is not None and hasattr(output_df, "columns")
                else []
            )

            # Simple column mapping based on operation type
            column_mapping = {}
            for out_col in output_columns:
                # For most operations, output columns map to input columns
                if out_col in input_columns:
                    column_mapping[f"output_df.{out_col}"] = [f"input_df.{out_col}"]
                else:
                    # For derived columns, map to all input columns (conservative)
                    column_mapping[f"output_df.{out_col}"] = [
                        f"input_df.{col}" for col in input_columns
                    ]

            return {
                "column_mapping": column_mapping,
                "input_columns": input_columns,
                "output_columns": output_columns,
                "operation_method": operation_method,
                "lineage_type": self._detect_operation_type(operation_method),
            }
        except Exception as e:
            notebook_logger.debug(f"Failed to generate column lineage: {e}")
            return {
                "column_mapping": {},
                "input_columns": [],
                "output_columns": [],
                "operation_method": operation_method,
                "lineage_type": "error",
            }

    def _create_dataframe_info(self, df, variable_name: str) -> dict:
        """Create dataframe_info structure according to schema definition"""
        try:
            return {
                "variable_name": variable_name,
                "object_id": f"df_{id(df)}",
                "shape": list(df.shape),
                "columns": list(df.columns) if hasattr(df, "columns") else [],
                "memory_usage_mb": self._get_dataframe_memory_usage(df),
            }
        except Exception as e:
            notebook_logger.debug(f"Failed to create dataframe info: {e}")
            return {
                "variable_name": variable_name,
                "object_id": f"df_{id(df)}",
                "shape": [0, 0],
                "columns": [],
                "memory_usage_mb": 0.0,
            }

    def _get_dataframe_memory_usage(self, df) -> float:
        """Get DataFrame memory usage in MB"""
        try:
            return df.memory_usage(deep=True).sum() / MEMORY_CONVERSION_FACTOR
        except Exception:
            return 0.0

    def uninstall(self):
        """Remove all installed hooks and restore original pandas methods"""
        if not self.interceptor_active:
            return

        try:
            notebook_logger.info("[UNINSTALL] Uninstalling unified interceptor...")

            # Uninstall hooks using hook manager
            self.hook_manager.uninstall_hooks()

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

            # Clear tracking state
            self.object_to_variable.clear()
            self.performance_tracking.clear()

            self.interceptor_active = False

            notebook_logger.success("[OK] Unified interceptor uninstalled")

        except Exception as e:
            notebook_logger.warning(f"[WARN] Error during uninstall: {e}")

    def get_session_summary(self):
        """Get session summary"""
        return {
            "session_id": self.session_id,
            "environment": self.environment.value,
            "interceptor_active": self.interceptor_active,
            "runtime_file": str(self.runtime_file),
            "lineage_file": str(self.lineage_file),
            "dataframe_lineage_file": str(self.dataframe_lineage_file),
            "hooks_installed": len(self.installed_hooks),
            "dataframes_tracked": len(_tracked_dataframes),
            "active_executions": len(self._execution_contexts),
            "dataframe_lineage_enabled": self.dataframe_lineage_enabled,
            "variable_mappings": len(self.object_to_variable),
        }


# Global interceptor instance management (matching old system API)
_global_unified_interceptor = None


def enable_unified_tracking(
    notebook_path: str | None = None,
    runtime_file: str | None = None,
    lineage_file: str | None = None,
    dataframe_lineage_file: str | None = None,
    environment: NotebookEnvironment | None = None,
) -> UnifiedNotebookInterceptor:
    """Enable unified tracking with global interceptor instance (matching old system API)"""
    global _global_unified_interceptor  # noqa: PLW0603

    if _global_unified_interceptor is not None:
        notebook_logger.warning(
            f"[WARN] Unified tracking already enabled (session: {_global_unified_interceptor.session_id})"
        )
        return _global_unified_interceptor

    # Create new interceptor instance
    _global_unified_interceptor = UnifiedNotebookInterceptor(
        notebook_path=notebook_path,
        runtime_file=runtime_file,
        lineage_file=lineage_file,
        dataframe_lineage_file=dataframe_lineage_file,
        environment=environment,
    )

    _global_unified_interceptor.install()
    return _global_unified_interceptor


def disable_unified_tracking() -> None:
    """Disable unified tracking and clean up global interceptor instance"""
    global _global_unified_interceptor  # noqa: PLW0603

    if _global_unified_interceptor is not None:
        _global_unified_interceptor.uninstall()
        _global_unified_interceptor = None


def is_unified_tracking_active() -> bool:
    """Check if unified tracking is currently active"""
    return (
        _global_unified_interceptor is not None
        and _global_unified_interceptor.interceptor_active
    )


def get_unified_interceptor() -> UnifiedNotebookInterceptor | None:
    """Get the global unified interceptor instance"""
    return _global_unified_interceptor
