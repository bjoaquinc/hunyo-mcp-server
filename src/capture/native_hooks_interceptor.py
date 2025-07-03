#!/usr/bin/env python3
"""
Marimo Native Hooks Interceptor
Production-ready interceptor using marimo's native hook system

This module provides DataFrame lineage tracking by hooking into marimo's
execution lifecycle using PRE_EXECUTION_HOOKS, POST_EXECUTION_HOOKS, and ON_FINISH_HOOKS.
"""

import hashlib
import json
import os
import sys
import threading
import time
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Runtime tracking integration
try:
    from capture.lightweight_runtime_tracker import enable_runtime_tracking

    RUNTIME_TRACKING_AVAILABLE = True
except ImportError:
    RUNTIME_TRACKING_AVAILABLE = False

# Global tracking state
_tracked_dataframes = weakref.WeakKeyDictionary()
_global_native_interceptor = None
_pandas_intercepted = False

# Import the new logger
from capture.logger import get_logger

# Initialize logger
hooks_logger = get_logger("hunyo.hooks.native")

# Constants for data capture limits
MAX_OBJECT_SIZE = 1000  # Maximum object size in bytes to store
MAX_COLLECTION_SIZE = 10  # Maximum collection items to store
MAX_DICT_SIZE = 5  # Maximum dictionary items to store
MAX_HASH_LENGTH = 8  # Length of code hash


class MarimoNativeHooksInterceptor:
    """
    Native marimo hook-based interceptor for complete cell execution monitoring
    Uses marimo's built-in hook system instead of Python exec interception
    """

    def __init__(
        self,
        lineage_file: str = "marimo_lineage_events.jsonl",
        notebook_path: str | None = None,
    ):
        self.lineage_file = Path(lineage_file)
        self.session_id = str(uuid.uuid4())[:8]
        self.interceptor_active = False
        self._lock = threading.Lock()

        # Hook references
        self.installed_hooks = []

        # Execution context tracking
        self._execution_contexts = {}

        # Generate proper runtime and lineage file paths using naming convention
        if notebook_path:
            from capture import get_event_filenames, get_user_data_dir

            data_dir = get_user_data_dir()
            runtime_file, proper_lineage_file = get_event_filenames(
                notebook_path, data_dir
            )
            # Store both runtime and lineage file paths
            self.runtime_file = Path(runtime_file)
            self.lineage_file = Path(proper_lineage_file)
        else:
            # Fallback to proper lineage file name pattern
            if "_lineage_events.jsonl" in lineage_file:
                runtime_file = lineage_file.replace(
                    "_lineage_events.jsonl", "_runtime_events.jsonl"
                )
            else:
                runtime_file = "marimo_runtime_events.jsonl"
            self.runtime_file = Path(runtime_file)

        # Runtime tracking integration (handles cell execution events)
        self.runtime_tracker = None
        if RUNTIME_TRACKING_AVAILABLE:
            self.runtime_tracker = enable_runtime_tracking(str(self.runtime_file))

        # Ensure both runtime and lineage directories exist
        self.runtime_file.parent.mkdir(parents=True, exist_ok=True)
        self.lineage_file.parent.mkdir(parents=True, exist_ok=True)
        self.lineage_file.touch()

        hooks_logger.status("Marimo Native Hooks Interceptor v3.0 - OpenLineage")
        hooks_logger.runtime(f"Runtime log: {self.runtime_file.name}")
        hooks_logger.lineage(f"Lineage log: {self.lineage_file.name}")
        hooks_logger.config(f"Session: {self.session_id}")
        if self.runtime_tracker:
            hooks_logger.tracking("Runtime tracking enabled")

        # Install pandas interception
        self._install_pandas_interception()

    def install(self):
        """Install marimo's native execution hooks"""
        if self.interceptor_active:
            hooks_logger.warning("[WARN] Native hooks already installed")
            return

        try:
            # Import marimo's hook system
            from marimo._runtime.runner.hooks import (
                ON_FINISH_HOOKS,
                POST_EXECUTION_HOOKS,
                PRE_EXECUTION_HOOKS,
            )

            hooks_logger.info("🔧 Installing marimo native hooks...")

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

            self.interceptor_active = True
            hooks_logger.info("[OK] Marimo native hooks installed!")
            hooks_logger.info(f"   [DATA] Lineage log: {self.lineage_file.name}")

        except ImportError as e:
            hooks_logger.error(f"[ERROR] Failed to import marimo hooks: {e}")
        except Exception as e:
            hooks_logger.error(f"[ERROR] Failed to install native hooks: {e}")

    def _install_pandas_interception(self):
        """Install pandas method interception for OpenLineage tracking"""
        global _pandas_intercepted  # noqa: PLW0603

        if _pandas_intercepted:
            return

        try:
            import pandas as pd

            # Store original methods
            self._original_methods = {}

            # DataFrame data loading functions to intercept for digests
            read_functions = [
                "read_csv",
                "read_excel",
                "read_json",
                "read_parquet",
                "read_sql",
            ]
            for func_name in read_functions:
                if hasattr(pd, func_name):
                    original_func = getattr(pd, func_name)
                    self._original_methods[f"pandas.{func_name}"] = original_func
                    intercepted_func = self._create_pandas_reader_interceptor(
                        func_name, original_func
                    )
                    setattr(pd, func_name, intercepted_func)

            # Intercept DataFrame constructor for in-memory sources
            original_init = pd.DataFrame.__init__
            self._original_methods["DataFrame.__init__"] = original_init
            intercepted_init = self._create_pandas_constructor_interceptor(
                original_init
            )
            pd.DataFrame.__init__ = intercepted_init

            # DataFrame methods to intercept
            methods_to_intercept = [
                # Data manipulation
                "merge",
                "join",
                "concat",
                "groupby",
                "pivot",
                "pivot_table",
                "drop",
                "dropna",
                "fillna",
                "replace",
                "rename",
                # Data transformation
                "apply",
                "map",
                "transform",
                "aggregate",
                "agg",
                # Data selection/filtering
                "query",
                "filter",
                "sample",
                "head",
                "tail",
                # Data reshaping
                "melt",
                "stack",
                "unstack",
                "transpose",
                # Data combination
                "append",
                "assign",
            ]

            # Intercept DataFrame methods
            for method_name in methods_to_intercept:
                if hasattr(pd.DataFrame, method_name):
                    original_method = getattr(pd.DataFrame, method_name)
                    self._original_methods[f"DataFrame.{method_name}"] = original_method

                    # Create intercepted method
                    intercepted_method = self._create_pandas_interceptor(
                        method_name, original_method, "DataFrame"
                    )
                    setattr(pd.DataFrame, method_name, intercepted_method)

            # Intercept pandas module functions
            module_functions = ["merge", "concat", "pivot_table", "melt"]
            for func_name in module_functions:
                if hasattr(pd, func_name):
                    original_func = getattr(pd, func_name)
                    self._original_methods[f"pandas.{func_name}"] = original_func

                    # Create intercepted function
                    intercepted_func = self._create_pandas_interceptor(
                        func_name, original_func, "pandas"
                    )
                    setattr(pd, func_name, intercepted_func)

            _pandas_intercepted = True
            hooks_logger.success(
                "Pandas interception installed for OpenLineage tracking"
            )

        except ImportError:
            hooks_logger.warning(
                "Pandas not available - lineage tracking will be limited"
            )
        except Exception as e:
            hooks_logger.warning(f"Failed to install pandas interception: {e}")

    def _create_pandas_constructor_interceptor(self, original_init):
        """Create an interceptor for the pandas DataFrame constructor."""

        def intercepted_init(df_self, *args, **kwargs):
            # Execute the original constructor first
            original_init(df_self, *args, **kwargs)

            run_id = str(uuid.uuid4())
            job_name = "pandas_DataFrame"

            current_execution_id = None
            if self.runtime_tracker:
                current_execution_id = self.runtime_tracker.get_current_execution_id()

            # This is an origin point, so it has no inputs
            inputs = []
            outputs = self._extract_output_datasets(df_self)

            # Emit a single COMPLETE event for in-memory creation
            self._emit_openlineage_event(
                "COMPLETE", run_id, job_name, inputs, outputs, current_execution_id
            )

        return intercepted_init

    def _create_pandas_reader_interceptor(self, func_name: str, original_func):
        """Create an intercepted pandas reader function to capture file digests."""

        def intercepted_reader(*args, **kwargs):
            run_id = str(uuid.uuid4())
            job_name = f"pandas_{func_name}"

            # Extract file path or data source for hashing
            source_path = None
            if args:
                source_path = str(args[0])

            # Generate START event (no inputs, as this is a data source)
            self._emit_openlineage_event("START", run_id, job_name, [], [], None)

            try:
                start_time = time.time()
                result_df = original_func(*args, **kwargs)
                duration = time.time() - start_time

                # Create output dataset with digest
                output_datasets = self._extract_output_datasets(
                    result_df, source_path=source_path
                )

                self._emit_openlineage_event(
                    "COMPLETE", run_id, job_name, [], output_datasets, None, duration
                )

                return result_df

            except Exception as e:
                self._emit_openlineage_event(
                    "FAIL", run_id, job_name, [], [], None, error=str(e)
                )
                raise

        return intercepted_reader

    def _create_pandas_interceptor(
        self, method_name: str, original_method, context: str
    ):
        """Create an intercepted pandas method that generates OpenLineage events"""

        def intercepted_method(*args, **kwargs):
            # Generate unique run ID for this operation
            run_id = str(uuid.uuid4())

            # Get current execution context
            current_execution_id = None
            if self.runtime_tracker:
                current_execution_id = self.runtime_tracker.get_current_execution_id()

            # Extract input DataFrames
            input_datasets = self._extract_input_datasets(args, kwargs, context)

            # Generate OpenLineage START event
            job_name = f"{context.lower()}_{method_name}"
            self._emit_openlineage_event(
                "START", run_id, job_name, input_datasets, [], current_execution_id
            )

            try:
                # Execute original method
                start_time = time.time()
                result = original_method(*args, **kwargs)
                duration = time.time() - start_time

                # --- COLUMN LINEAGE LOGIC ---
                column_lineage = self._calculate_column_lineage(
                    method_name, input_datasets, result, args, kwargs
                )

                # Extract output datasets
                output_datasets = self._extract_output_datasets(
                    result, column_lineage=column_lineage
                )

                # Generate OpenLineage COMPLETE event
                self._emit_openlineage_event(
                    "COMPLETE",
                    run_id,
                    job_name,
                    input_datasets,
                    output_datasets,
                    current_execution_id,
                    duration,
                )

                return result

            except Exception as e:
                # Generate OpenLineage FAIL event
                self._emit_openlineage_event(
                    "FAIL",
                    run_id,
                    job_name,
                    input_datasets,
                    [],
                    current_execution_id,
                    error=str(e),
                )
                raise

        return intercepted_method

    def _get_input_df_from_args(self, args):
        """Find the first DataFrame in a tuple of arguments."""
        for arg in args:
            if hasattr(arg, "shape") and hasattr(arg, "columns"):
                return arg
        return None

    def _calculate_column_lineage(
        self, operation: str, inputs: list[dict], output_df: Any, args, kwargs
    ) -> dict | None:
        """
        Calculate column-level lineage for a given pandas operation.
        Returns a dictionary for the `columnLineage` facet.
        """
        if not hasattr(output_df, "columns"):
            return None

        output_fields = {}

        # Helper to get namespace/name for a DataFrame
        def get_dataset_identifier(df):
            # A simplified way to identify an input dataframe from the `inputs` list
            df_id = id(df)
            for i in inputs:
                if i["name"].startswith(f"dataframe_{df_id}"):
                    return {"namespace": i["namespace"], "name": i["name"]}
            return {"namespace": "marimo", "name": f"unknown_dataframe_{df_id}"}

        try:
            if operation in ["merge", "join"]:
                left_df = self._get_input_df_from_args(args)
                right_df = kwargs.get("right", args[1] if len(args) > 1 else None)

                if left_df is None or right_df is None:
                    return None

                left_id = get_dataset_identifier(left_df)
                right_id = get_dataset_identifier(right_df)

                for col in output_df.columns:
                    input_fields = []
                    if col in left_df.columns:
                        input_fields.append({**left_id, "field": str(col)})
                    if col in right_df.columns:
                        input_fields.append({**right_id, "field": str(col)})
                    output_fields[str(col)] = {"inputFields": input_fields}

            elif operation == "drop":
                input_df = self._get_input_df_from_args(args)
                if input_df is None:
                    return None
                input_id = get_dataset_identifier(input_df)
                for col in output_df.columns:
                    output_fields[str(col)] = {
                        "inputFields": [{**input_id, "field": str(col)}]
                    }

            elif operation == "rename":
                input_df = self._get_input_df_from_args(args)
                if input_df is None:
                    return None
                input_id = get_dataset_identifier(input_df)
                rename_mapping = kwargs.get("columns", args[0] if args else {})

                for out_col in output_df.columns:
                    in_col = next(
                        (k for k, v in rename_mapping.items() if v == out_col), out_col
                    )
                    output_fields[str(out_col)] = {
                        "inputFields": [{**input_id, "field": str(in_col)}]
                    }

            elif operation == "assign":
                input_df = self._get_input_df_from_args(args)
                if input_df is None:
                    return None
                input_id = get_dataset_identifier(input_df)

                for col in output_df.columns:
                    if col in input_df.columns:
                        output_fields[str(col)] = {
                            "inputFields": [{**input_id, "field": str(col)}]
                        }
                    else:
                        # New column from `assign` - assume all inputs contribute
                        all_input_fields = [
                            {**input_id, "field": str(in_col)}
                            for in_col in input_df.columns
                        ]
                        output_fields[str(col)] = {"inputFields": all_input_fields}

            else:
                # Default/Fallback: Assume all columns from all inputs contribute to all output columns
                input_df = self._get_input_df_from_args(args)
                if input_df is None:
                    return None
                input_id = get_dataset_identifier(input_df)

                all_input_fields = [
                    {**input_id, "field": str(in_col)} for in_col in input_df.columns
                ]
                for col in output_df.columns:
                    output_fields[str(col)] = {"inputFields": all_input_fields}

            return {"fields": output_fields}

        except Exception as e:
            hooks_logger.warning(
                f"[WARN] Could not calculate column lineage for '{operation}': {e}"
            )
            return None

    def _extract_input_datasets(
        self, args: tuple, kwargs: dict, _context: str
    ) -> list[dict]:
        """Extract input DataFrames and convert to OpenLineage dataset format"""
        datasets = []

        # Check args for DataFrames
        for i, arg in enumerate(args):
            if hasattr(arg, "shape") and hasattr(arg, "columns"):
                dataset = self._dataframe_to_openlineage_dataset(arg, f"input_arg_{i}")
                datasets.append(dataset)

        # Check kwargs for DataFrames
        for key, value in kwargs.items():
            if hasattr(value, "shape") and hasattr(value, "columns"):
                dataset = self._dataframe_to_openlineage_dataset(value, f"input_{key}")
                datasets.append(dataset)
            elif isinstance(value, list | tuple):
                # Check for DataFrames in collections
                for j, item in enumerate(value):
                    if hasattr(item, "shape") and hasattr(item, "columns"):
                        dataset = self._dataframe_to_openlineage_dataset(
                            item, f"input_{key}_{j}"
                        )
                        datasets.append(dataset)

        return datasets

    def _extract_output_datasets(
        self,
        result: Any,
        source_path: str | None = None,
        column_lineage: dict | None = None,
    ) -> list[dict]:
        """Extract output DataFrames and convert to OpenLineage dataset format"""
        datasets = []

        if hasattr(result, "shape") and hasattr(result, "columns"):
            # Single DataFrame result
            dataset = self._dataframe_to_openlineage_dataset(
                result, "output", source_path, column_lineage
            )
            datasets.append(dataset)
        elif isinstance(result, list | tuple):
            # Multiple DataFrames
            for i, item in enumerate(result):
                if hasattr(item, "shape") and hasattr(item, "columns"):
                    dataset = self._dataframe_to_openlineage_dataset(
                        item, f"output_{i}", source_path, column_lineage
                    )
                    datasets.append(dataset)
        elif isinstance(result, dict):
            # Dictionary of DataFrames
            for key, value in result.items():
                if hasattr(value, "shape") and hasattr(value, "columns"):
                    dataset = self._dataframe_to_openlineage_dataset(
                        value, f"output_{key}", source_path, column_lineage
                    )
                    datasets.append(dataset)

        return datasets

    def _dataframe_to_openlineage_dataset(
        self,
        df,
        name_suffix: str = "",
        source_path: str | None = None,
        column_lineage: dict | None = None,
    ) -> dict:
        """Convert a DataFrame to OpenLineage dataset format"""
        df_id = id(df)

        # Generate schema facet
        schema_fields = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            schema_fields.append(
                {
                    "name": str(col),
                    "type": dtype_str,
                    "description": f"Column {col} of type {dtype_str}",
                }
            )

        dataset = {
            "namespace": "marimo",
            "name": f"dataframe_{df_id}_{name_suffix}",
            "facets": {
                "schema": {
                    "_producer": "marimo-lineage-tracker",
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
                    "fields": schema_fields,
                },
                "dataSource": {
                    "_producer": "marimo-lineage-tracker",
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasourceDatasetFacet.json",
                    "name": "pandas-dataframe",
                    "uri": f"memory://dataframe_{df_id}",
                },
                "columnMetrics": {
                    "_producer": "marimo-lineage-tracker",
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ColumnMetricsDatasetFacet.json",
                    "columnMetrics": {
                        str(col): {
                            "nullCount": int(df[col].isnull().sum()),
                            "distinctCount": (
                                int(df[col].nunique())
                                if df[col].dtype != "object"
                                else None
                            ),
                            "min": (
                                float(df[col].min())
                                if df[col].dtype in ["int64", "float64"]
                                else None
                            ),
                            "max": (
                                float(df[col].max())
                                if df[col].dtype in ["int64", "float64"]
                                else None
                            ),
                        }
                        for col in df.columns
                    },
                },
            },
        }

        # Add digest/versioning facet if source path is available
        if source_path:
            try:
                # Use file path for name and URI
                p = Path(source_path)
                dataset["name"] = str(p.name)
                dataset["namespace"] = str(p.resolve().parent)
                dataset["facets"]["dataSource"]["name"] = str(p.resolve())
                dataset["facets"]["dataSource"]["uri"] = p.resolve().as_uri()

                # Add version facet with file hash
                file_hash = self._get_file_hash(source_path)
                dataset["facets"]["version"] = {
                    "_producer": "marimo-lineage-tracker",
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasetVersionDatasetFacet.json",
                    "datasetVersion": file_hash,
                }
            except Exception as e:
                hooks_logger.warning(
                    f"[WARN] Could not generate digest for {source_path}: {e}"
                )

        # Add column lineage facet if available
        if column_lineage:
            dataset["facets"]["columnLineage"] = {
                "_producer": "marimo-lineage-tracker",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ColumnLineageDatasetFacet.json",
                "fields": column_lineage["fields"],
            }

        return dataset

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate the SHA256 hash of a file."""
        if not os.path.exists(file_path):
            return None

        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _emit_openlineage_event(
        self,
        event_type: str,
        run_id: str,
        job_name: str,
        inputs: list[dict],
        outputs: list[dict],
        execution_id: str | None = None,
        duration: float | None = None,
        error: str | None = None,
    ):
        """Emit an OpenLineage-compliant event"""

        event = {
            "eventType": event_type,
            "eventTime": datetime.now(timezone.utc).isoformat(),
            "run": {"runId": run_id, "facets": {}},
            "job": {"namespace": "marimo", "name": job_name, "facets": {}},
            "inputs": inputs,
            "outputs": outputs,
            "producer": "marimo-lineage-tracker",
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
        }

        # Add execution context
        if execution_id:
            event["run"]["facets"]["marimoExecution"] = {
                "_producer": "marimo-lineage-tracker",
                "executionId": execution_id,
                "sessionId": self.session_id,
            }

        # Add performance metrics for COMPLETE events
        if event_type == "COMPLETE" and duration is not None:
            event["run"]["facets"]["performance"] = {
                "_producer": "marimo-lineage-tracker",
                "durationMs": round(duration * 1000, 2),
            }

        # Add error information for FAIL events
        if event_type == "FAIL" and error:
            event["run"]["facets"]["errorMessage"] = {
                "_producer": "marimo-lineage-tracker",
                "message": error,
                "programmingLanguage": "python",
            }

        # Emit the event
        self._emit_lineage_event(event)

        # Use marimo-aware logger that handles context properly
        hooks_logger.info(
            f"[LINK] OpenLineage {event_type}: {job_name} (run: {run_id[:8]})"
        )

    def _create_pre_execution_hook(self):
        """Create pre-execution hook function with correct signature"""

        def pre_execution_hook(cell, _runner):
            """Called before each cell execution"""
            try:
                # Extract REAL cell information from marimo
                cell_id = cell.cell_id  # REAL marimo cell ID
                cell_code = cell.code  # REAL cell source code

                # Generate execution ID for this cell run
                execution_id = str(uuid.uuid4())[:8]

                # Emit REAL cell execution start event
                self._emit_real_cell_event(
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

                # Start runtime tracking if available (non-blocking)
                if self.runtime_tracker:
                    try:
                        # DO NOT overwrite execution_id! Use the one we generated above
                        runtime_execution_id = (
                            self.runtime_tracker.track_cell_execution_start(
                                cell_id, cell_code
                            )
                        )
                        # Optionally log the runtime tracker's ID for debugging
                        hooks_logger.tracking(
                            f"Runtime tracker ID: {runtime_execution_id}"
                        )
                    except Exception as rt_error:
                        # Logger handles marimo context automatically
                        hooks_logger.warning(f"Runtime tracking error: {rt_error}")

                # Store execution context for post-hook using multiple keys for safety
                if not hasattr(self, "_execution_contexts"):
                    self._execution_contexts = {}

                context_data = {
                    "execution_id": execution_id,
                    "start_time": time.time(),
                    "cell_code": cell_code,
                }

                # Store by cell_id
                if cell_id:
                    self._execution_contexts[cell_id] = context_data

                # Also store by code hash as backup
                if cell_code:
                    try:
                        code_hash = str(hash(cell_code))[:MAX_HASH_LENGTH]
                        self._execution_contexts[f"code_{code_hash}"] = context_data
                    except Exception as e:
                        hooks_logger.error(f"Failed to store execution context: {e}")

                # Note: Runtime tracking handles execution events, we focus on lineage

            except Exception as e:
                # Non-blocking error handling - don't let hook errors break cell execution
                try:
                    # Logger handles marimo context automatically
                    hooks_logger.error(f"Error in pre-execution hook: {e}")
                except Exception as log_error:
                    # Even logging might fail in some contexts - use fallback
                    print(  # noqa: T201
                        f"Hook error (logging failed): {e}, {log_error}"
                    )

        return pre_execution_hook

    def _create_post_execution_hook(self):
        """Create post-execution hook function with correct signature"""

        def post_execution_hook(cell, _runner, run_result):
            """Called after each cell execution"""
            try:
                # Extract REAL cell information from marimo
                cell_id = cell.cell_id  # REAL marimo cell ID
                cell_code = cell.code  # REAL cell source code

                # Extract run result information safely
                error = None
                output = None

                try:
                    # Try to call success() method if it exists, otherwise check for exception
                    success_method = getattr(run_result, "success", None)
                    if callable(success_method):
                        success_method()
                    else:
                        pass  # Default to true

                    error = getattr(run_result, "exception", None)
                    output = getattr(run_result, "output", None)
                except AttributeError:
                    # Use defaults if attributes don't exist
                    hooks_logger.warning(
                        "run_result object missing expected attributes"
                    )

                # Get execution context
                execution_id = None
                start_time = time.time()
                if hasattr(self, "_execution_contexts"):
                    # Try direct cell_id lookup first
                    try:
                        if cell_id and cell_id in self._execution_contexts:
                            ctx = self._execution_contexts[cell_id]
                            execution_id = ctx.get("execution_id")
                            start_time = ctx.get("start_time", start_time)
                            del self._execution_contexts[cell_id]

                        # Try code hash lookup as backup
                        elif cell_code:
                            code_hash = str(hash(cell_code))[:MAX_HASH_LENGTH]
                            code_key = f"code_{code_hash}"
                            if code_key in self._execution_contexts:
                                ctx = self._execution_contexts[code_key]
                                execution_id = ctx.get("execution_id")
                                start_time = ctx.get("start_time", start_time)
                                del self._execution_contexts[code_key]
                    except (KeyError, AttributeError) as e:
                        # Context lookup failed, but continue
                        hooks_logger.warning(f"Context lookup failed: {e}")

                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Emit REAL cell execution end event
                if execution_id:
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

                    self._emit_real_cell_event(event_data)

                # Capture result safely (non-blocking)
                if output is not None:
                    try:
                        # Capture result info for potential future use
                        self._capture_result_safely(output)
                    except Exception as e:
                        # Capture failed, but don't break execution
                        hooks_logger.error(f"Result capture failed: {e}")

                # End runtime tracking if available (non-blocking)
                if self.runtime_tracker and execution_id:
                    try:
                        # Use our consistent execution_id for runtime tracking
                        if error:
                            self.runtime_tracker.track_cell_execution_error(
                                execution_id, cell_id, cell_code, start_time, error
                            )
                        else:
                            self.runtime_tracker.track_cell_execution_end(
                                execution_id, cell_id, cell_code, start_time
                            )
                    except Exception as e:
                        hooks_logger.error(f"Runtime tracker end failed: {e}")

                # Note: DataFrame operations are tracked via pandas interception

            except Exception as e:
                # Top-level error handling
                try:
                    hooks_logger.error(f"Error in post-execution hook: {e}")
                except Exception as log_error:
                    # Even logging might fail in some contexts - use fallback
                    print(  # noqa: T201
                        f"Post-hook error (logging failed): {e}, {log_error}"
                    )

        return post_execution_hook

    def _emit_real_cell_event(self, event_data):
        """Emit real cell execution events directly to runtime file (bypass logging)"""
        try:
            # Cell execution events are RUNTIME events - use runtime directory
            runtime_file = self.runtime_file

            # Ensure runtime directory exists
            runtime_file.parent.mkdir(parents=True, exist_ok=True)

            with open(runtime_file, "a") as f:
                f.write(json.dumps(event_data, default=str) + "\n")
        except Exception as e:
            # Fallback to logger if file write fails
            hooks_logger.error(f"Failed to emit real cell event: {e}")

    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _create_finish_hook(self):
        """Create finish hook for cleanup"""

        def finish_hook(_runner):
            """Called when marimo session ends"""
            try:
                # Emit session finish as OpenLineage event (avoid logging in marimo)
                self._emit_openlineage_event(
                    "COMPLETE",
                    f"session_{self.session_id}",
                    "marimo_session",
                    [],
                    [],
                    execution_id=None,
                )
            except Exception as e:
                # Use marimo-aware logger that handles context properly
                hooks_logger.warning(f"[WARN] Error in finish hook: {e}")

        return finish_hook

    def _capture_result_safely(self, result: Any) -> dict:
        """Safely capture information about cell execution result"""
        try:
            # Handle None result
            if result is None:
                return {"type": "NoneType", "value": None}

            result_type = type(result).__name__
            result_module = getattr(type(result), "__module__", "unknown")

            # Check if it's a DataFrame
            is_dataframe = hasattr(result, "shape") and hasattr(result, "columns")

            if is_dataframe:
                # DataFrame result
                dataframe_id = id(result)
                return {
                    "type": "DataFrame",
                    "module": result_module,
                    "is_dataframe": True,
                    "dataframe_id": dataframe_id,
                    "shape": list(result.shape),
                    "columns_count": len(result.columns),
                    "description": f"DataFrame with {result.shape[0]} rows and {result.shape[1]} columns",
                }

            # Get object size for memory management
            obj_size = sys.getsizeof(result)

            # Handle different result types with memory awareness
            if isinstance(result, str | int | float | bool):
                # Small primitive values - store completely
                if obj_size < MAX_OBJECT_SIZE:  # Less than 1KB
                    return {
                        "type": result_type,
                        "value": result,
                        "size_bytes": obj_size,
                    }
                else:
                    return {
                        "type": result_type,
                        "description": f"Large {result_type} ({obj_size:,} bytes)",
                        "size_bytes": obj_size,
                    }

            elif isinstance(result, list | tuple):
                # Collections - describe if large
                if len(result) <= MAX_COLLECTION_SIZE and obj_size < MAX_OBJECT_SIZE:
                    return {
                        "type": result_type,
                        "value": result,
                        "length": len(result),
                        "size_bytes": obj_size,
                    }
                else:
                    return {
                        "type": result_type,
                        "description": f"{result_type} with {len(result)} items ({obj_size:,} bytes)",
                        "length": len(result),
                        "size_bytes": obj_size,
                    }

            elif isinstance(result, dict):
                # Dictionaries - describe if large
                if len(result) <= MAX_DICT_SIZE and obj_size < MAX_OBJECT_SIZE:
                    return {
                        "type": result_type,
                        "value": result,
                        "keys_count": len(result),
                        "size_bytes": obj_size,
                    }
                else:
                    return {
                        "type": result_type,
                        "description": f"Dict with {len(result)} keys ({obj_size:,} bytes)",
                        "keys_count": len(result),
                        "size_bytes": obj_size,
                    }

            else:
                # Other object types
                return {
                    "type": result_type,
                    "module": result_module,
                    "description": f"{result_module}.{result_type} object ({obj_size:,} bytes)",
                    "size_bytes": obj_size,
                }

        except Exception as e:
            return {
                "type": "capture_error",
                "error": str(e),
                "fallback_type": str(type(result).__name__),
            }

    def _emit_lineage_event(self, event: dict):
        """Emit lineage event to lineage file"""
        try:
            event["session_id"] = self.session_id
            event["emitted_at"] = datetime.now(timezone.utc).isoformat()

            with self._lock:
                with open(self.lineage_file, "a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            # Use marimo-aware logger that handles context properly
            hooks_logger.warning(f"[WARN] Failed to emit lineage event: {e}")

    def uninstall(self):
        """Remove all installed hooks and restore original pandas methods"""
        if not self.interceptor_active:
            return

        try:
            # Import hook lists
            from marimo._runtime.runner.hooks import (
                ON_FINISH_HOOKS,
                POST_EXECUTION_HOOKS,
                PRE_EXECUTION_HOOKS,
            )

            hooks_logger.info("🔧 Uninstalling marimo native hooks...")

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
            self._restore_pandas_methods()

            self.installed_hooks.clear()
            self.interceptor_active = False
            hooks_logger.info("[OK] Native hooks uninstalled")

        except Exception as e:
            hooks_logger.warning(f"[WARN] Error during hook uninstall: {e}")

    def _restore_pandas_methods(self):
        """Restore original pandas methods"""
        global _pandas_intercepted  # noqa: PLW0603

        if not hasattr(self, "_original_methods") or not _pandas_intercepted:
            return

        try:
            import pandas as pd

            # Restore DataFrame methods
            for method_key, original_method in self._original_methods.items():
                if method_key.startswith("DataFrame."):
                    method_name = method_key.replace("DataFrame.", "")
                    setattr(pd.DataFrame, method_name, original_method)
                elif method_key.startswith("pandas."):
                    func_name = method_key.replace("pandas.", "")
                    setattr(pd, func_name, original_method)

            _pandas_intercepted = False
            hooks_logger.info("🐼 Pandas methods restored")

        except Exception as e:
            hooks_logger.warning(f"[WARN] Error restoring pandas methods: {e}")

    def get_session_summary(self):
        """Get session summary"""
        summary = {
            "session_id": self.session_id,
            "interceptor_active": self.interceptor_active,
            "lineage_file": str(self.lineage_file),
            "runtime_file": "marimo_runtime.jsonl",
            "hooks_installed": len(self.installed_hooks),
            "dataframes_tracked": len(_tracked_dataframes),
        }

        if self.runtime_tracker:
            summary["runtime_summary"] = self.runtime_tracker.get_session_summary()

        return summary


def enable_native_hook_tracking(
    lineage_file: str = "marimo_lineage_events.jsonl", notebook_path: str | None = None
):
    """Enable native marimo hook-based tracking"""
    global _global_native_interceptor  # noqa: PLW0603

    if _global_native_interceptor is not None:
        hooks_logger.warning(
            f"[WARN] Native hook tracking already enabled (session: {_global_native_interceptor.session_id})"
        )
        return _global_native_interceptor

    hooks_logger.info("🔧 Creating new interceptor instance...")
    _global_native_interceptor = MarimoNativeHooksInterceptor(
        lineage_file, notebook_path
    )
    _global_native_interceptor.install()
    return _global_native_interceptor


def disable_native_hook_tracking():
    """Disable native hook tracking"""
    global _global_native_interceptor  # noqa: PLW0603

    if _global_native_interceptor is not None:
        _global_native_interceptor.uninstall()
        _global_native_interceptor = None
        hooks_logger.info("🔇 Native hook tracking disabled")
    else:
        hooks_logger.warning("[WARN] Native hook tracking was not active")


def is_native_tracking_active():
    """Check if native hook tracking is active"""
    return (
        _global_native_interceptor is not None
        and _global_native_interceptor.interceptor_active
    )


def get_native_tracking_summary():
    """Get native tracking summary"""
    if _global_native_interceptor is not None:
        return _global_native_interceptor.get_session_summary()
    else:
        return {"status": "inactive", "message": "Native hook tracking is not enabled"}


# Auto-enable when imported (not in direct script execution)
if __name__ == "__main__":
    hooks_logger.info("Manual test mode - run enable_native_hook_tracking() manually")
else:
    # Don't auto-enable to prevent startup issues - user must call enable_native_hook_tracking()
    hooks_logger.info("🔧 Marimo hooks interceptor imported")
    hooks_logger.info(
        "🚨 IMPORTANT: Call enable_native_hook_tracking() to start tracking"
    )
    hooks_logger.info(
        "   Example: import marimo_native_hooks_interceptor; marimo_native_hooks_interceptor.enable_native_hook_tracking()"
    )
