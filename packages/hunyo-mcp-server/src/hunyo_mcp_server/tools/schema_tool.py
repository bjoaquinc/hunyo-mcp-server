#!/usr/bin/env python3
"""
Schema Tool - Database schema inspection for LLMs.

Provides information about database tables, columns, views, and relationships
to help LLMs understand the data structure for query construction.
"""

from typing import Any

# Import logging utility
from hunyo_mcp_server.logger import get_logger
from hunyo_mcp_server.orchestrator import get_global_orchestrator

tool_logger = get_logger("hunyo.tools.schema")


# Get the shared FastMCP instance
from hunyo_mcp_server.mcp_instance import mcp


@mcp.tool("inspect_schema")
def inspect_schema(
    table_name: str | None = None,
    *,
    include_views: bool = True,
    include_samples: bool = False,
) -> dict[str, Any]:
    """
    Inspect database schema to understand table structures and relationships.

    Provides detailed information about tables, columns, data types, and views
    to help with query construction and data analysis.

    Args:
        table_name: Specific table to inspect (None for all tables)
        include_views: Whether to include view definitions
        include_samples: Whether to include sample data (first few rows)

    Returns:
        Dictionary containing schema information for tables and views

    Available tables:
    - runtime_events: Cell execution performance and timing data
    - lineage_events: Data lineage and transformation tracking
    - dataframe_lineage_events: DataFrame computational operations and transformations
    - vw_lineage_io: View showing input/output relationships
    - vw_performance_metrics: View combining runtime and lineage metrics
    - vw_dataframe_lineage: View showing DataFrame operation analysis
    """

    try:
        # Get database manager from orchestrator
        orchestrator = get_global_orchestrator()
        db_manager = orchestrator.get_db_manager()

        tool_logger.info(f"Inspecting schema for table: {table_name or 'all tables'}")

        # Define known tables and views
        tables = ["runtime_events", "lineage_events", "dataframe_lineage_events"]
        views = (
            ["vw_lineage_io", "vw_performance_metrics", "vw_dataframe_lineage"]
            if include_views
            else []
        )

        schema_info = {
            "database_overview": {
                "total_tables": len(tables),
                "total_views": len(views),
                "available_tables": tables,
                "available_views": views,
            },
            "tables": {},
            "views": {} if include_views else None,
        }

        # Get information for specific table or all tables
        targets = [table_name] if table_name else tables + views

        for target in targets:
            try:
                # Get table structure
                table_info = db_manager.get_table_info(target)
                row_count = db_manager.get_table_count(target)

                target_info = {
                    "name": target,
                    "type": "view" if target.startswith("vw_") else "table",
                    "row_count": row_count,
                    "columns": [],
                }

                # Process column information
                for col_info in table_info:
                    column = {
                        "name": col_info.get(
                            "column_name", col_info.get("Field", "unknown")
                        ),
                        "type": col_info.get(
                            "column_type", col_info.get("Type", "unknown")
                        ),
                        "nullable": col_info.get(
                            "null", col_info.get("Null", "unknown")
                        ),
                        "default": col_info.get(
                            "column_default", col_info.get("Default", None)
                        ),
                    }
                    target_info["columns"].append(column)

                # Add sample data if requested
                if include_samples and row_count > 0:
                    try:
                        sample_query = f"SELECT * FROM {target} LIMIT 3"
                        sample_data = db_manager.execute_query(sample_query)
                        target_info["sample_data"] = sample_data
                    except Exception as e:
                        target_info["sample_data_error"] = str(e)

                # Add to appropriate section
                if target.startswith("vw_") and include_views:
                    schema_info["views"][target] = target_info
                else:
                    schema_info["tables"][target] = target_info

            except Exception as e:
                tool_logger.warning(f"Failed to get info for {target}: {e}")
                error_info = {"name": target, "error": str(e), "available": False}

                if target.startswith("vw_") and include_views:
                    schema_info["views"][target] = error_info
                else:
                    schema_info["tables"][target] = error_info

        # Add helpful metadata
        schema_info["usage_notes"] = get_schema_usage_notes()
        schema_info["relationships"] = get_table_relationships()

        tool_logger.info(
            f"Memory schema inspection complete for {len(targets)} objects"
        )
        return schema_info

    except Exception as e:
        error_msg = str(e)
        tool_logger.error(f"Memory schema inspection failed: {error_msg}")

        return {
            "success": False,
            "error": error_msg,
            "table_name": table_name,
            "available_tables": [
                "runtime_events",
                "lineage_events",
                "dataframe_lineage_events",
            ],
            "available_views": (
                ["vw_lineage_io", "vw_performance_metrics", "vw_dataframe_lineage"]
                if include_views
                else []
            ),
        }


def get_schema_usage_notes() -> dict[str, str]:
    """
    Get usage notes and explanations for each table/view.

    Returns:
        Dictionary with usage information for each schema object
    """

    return {
        "runtime_events": {
            "purpose": "Lightweight execution monitoring of cell execution within notebooks",
            "key_fields": "execution_id (links events), session_id (groups notebook runs), duration_ms, memory metrics",
            "common_queries": "Performance analysis, error tracking, execution timeline",
            "notes": "Contains start/end events for each cell execution with timing and memory data",
        },
        "lineage_events": {
            "purpose": "Comprehensive data lineage tracking following OpenLineage specification",
            "key_fields": "run_id (OpenLineage), execution_id (links to runtime), inputs_json, outputs_json",
            "common_queries": "Data lineage analysis, input/output tracking, transformation mapping",
            "notes": "JSON fields contain detailed lineage metadata and column-level transformations",
        },
        "dataframe_lineage_events": {
            "purpose": "DataFrame computational operations and transformations tracking",
            "key_fields": "execution_id, operation_type, operation_method, input_dataframes, output_dataframes, column_lineage (nested structure)",
            "common_queries": "DataFrame operation analysis, column-level lineage, computational performance tracking",
            "notes": "Tracks selection, aggregation, and join operations on DataFrames with rich column-level lineage metadata",
        },
        "vw_lineage_io": {
            "purpose": "Simplified view of input/output relationships from lineage events",
            "key_fields": "input_names, output_names, input_count, output_count, job_name",
            "common_queries": "Understanding data flow, finding dataset dependencies",
            "notes": "Convenience view that extracts and summarizes I/O from complex JSON fields",
        },
        "vw_performance_metrics": {
            "purpose": "Combined view linking runtime performance with lineage metadata",
            "key_fields": "execution_id (join key), duration_ms, memory_delta_mb, lineage_to_runtime_ratio",
            "common_queries": "Performance analysis with context, finding bottlenecks",
            "notes": "Joins runtime and lineage events to provide comprehensive execution metrics",
        },
        "vw_dataframe_lineage": {
            "purpose": "DataFrame operation analysis with extracted JSON fields and computed metrics",
            "key_fields": "operation_type, operation_method, input_variable, output_variable, row_retention_ratio",
            "common_queries": "DataFrame transformation analysis, operation performance, data shape changes",
            "notes": "Convenience view that extracts DataFrame info from JSON and computes transformation ratios",
        },
    }


def get_table_relationships() -> dict[str, Any]:
    """
    Get information about relationships between tables.

    Returns:
        Dictionary describing table relationships and join patterns
    """

    return {
        "primary_keys": {
            "runtime_events": "event_id (auto-increment)",
            "lineage_events": "ol_event_id (auto-increment)",
            "dataframe_lineage_events": "df_event_id (auto-increment)",
        },
        "foreign_keys": {"lineage_events.execution_id": "runtime_events.execution_id"},
        "common_joins": {
            "runtime_to_lineage": {
                "description": "Link runtime performance with lineage metadata",
                "join_condition": "runtime_events.execution_id = lineage_events.execution_id",
                "example": "SELECT r.duration_ms, l.job_name FROM runtime_events r JOIN lineage_events l ON r.execution_id = l.execution_id",
            },
            "session_analysis": {
                "description": "Group events by notebook session",
                "join_condition": "GROUP BY session_id",
                "example": "SELECT session_id, COUNT(*) FROM runtime_events GROUP BY session_id",
            },
        },
        "grouping_fields": {
            "session_id": "Groups all events from the same notebook session",
            "execution_id": "Links runtime and lineage events for the same cell execution",
            "cell_id": "Groups events by notebook cell",
        },
    }


schema_tool = inspect_schema
