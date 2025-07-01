#!/usr/bin/env python3
"""
Lineage Tool - Specialized data lineage analysis for LLMs.

Provides high-level functions for analyzing data lineage patterns,
tracking data flow, and understanding transformation relationships.
"""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from hunyo_mcp_server.orchestrator import get_global_orchestrator

# Import logging utility
try:
    from ...capture.logger import get_logger

    tool_logger = get_logger("hunyo.tools.lineage")
except ImportError:
    # Fallback for testing/standalone usage - use logging instead of print
    import logging

    logging.basicConfig(level=logging.INFO)
    tool_logger = logging.getLogger("hunyo.tools.lineage")


# Get the FastMCP instance from server.py
try:
    from hunyo_mcp_server.server import mcp
except ImportError:
    # Fallback for testing - create a minimal MCP instance
    mcp = FastMCP("lineage-tool-test")


# Constants for magic numbers
DEFAULT_QUERY_LIMIT = 100
MAX_PLACEHOLDERS = 1000


# Tool function
@mcp.tool("analyze_lineage")
def analyze_lineage_tool(
    dataset_name: str | None = None,
    analysis_type: str = "overview",
    *,
    include_metrics: bool = True,
) -> dict[str, Any]:
    """
    Analyze data lineage patterns from OpenLineage events.

    This tool provides comprehensive analysis of data lineage including:
    - Overview: High-level statistics and patterns
    - Dependencies: Upstream data dependencies for a dataset
    - Impact: Downstream impact analysis for a dataset
    - Flow: Complete data flow tracing through transformations
    - Patterns: Analysis of transformation patterns and bottlenecks

    Args:
        dataset_name: Specific dataset to analyze (optional)
        analysis_type: Type of analysis - 'overview', 'dependencies', 'impact', 'flow', or 'patterns'
        include_metrics: Whether to include performance metrics in the analysis

    Returns:
        Dict containing analysis results with metadata
    """
    tool_logger.info(f"Starting lineage analysis: {analysis_type}")

    try:
        # Get database manager from orchestrator
        orchestrator = get_global_orchestrator()
        if not orchestrator:
            return {
                "error": "Orchestrator not available",
                "suggestion": "Ensure the MCP server is properly initialized",
            }

        db_manager = orchestrator.get_duckdb_manager()
        if not db_manager:
            return {
                "error": "Database manager not available",
                "suggestion": "Check database initialization and file paths",
            }

        # Route to appropriate analysis function
        if analysis_type == "overview":
            result = _get_lineage_overview(db_manager, include_metrics=include_metrics)
        elif analysis_type == "dependencies":
            result = _analyze_dependencies(
                db_manager, dataset_name, include_metrics=include_metrics
            )
        elif analysis_type == "impact":
            result = _analyze_impact(
                db_manager, dataset_name, include_metrics=include_metrics
            )
        elif analysis_type == "flow":
            result = _trace_data_flow(
                db_manager, dataset_name, include_metrics=include_metrics
            )
        elif analysis_type == "patterns":
            result = _analyze_patterns(db_manager, include_metrics=include_metrics)
        else:
            return {
                "error": f"Unknown analysis type: {analysis_type}",
                "valid_types": [
                    "overview",
                    "dependencies",
                    "impact",
                    "flow",
                    "patterns",
                ],
            }

        # Add metadata to result
        result["metadata"] = {
            "analysis_type": analysis_type,
            "dataset_name": dataset_name,
            "include_metrics": include_metrics,
            "generated_at": json.dumps(
                {"timestamp": "now"}, default=str
            ),  # JSON serializable timestamp
        }

        tool_logger.info(f"Lineage analysis completed successfully: {analysis_type}")
        return result

    except Exception as e:
        error_msg = f"Failed to analyze lineage: {e!s}"
        tool_logger.error(error_msg)
        return {
            "error": error_msg,
            "suggestion": "Check database connectivity and data availability",
        }


def _get_lineage_overview(db_manager, *, include_metrics: bool) -> dict[str, Any]:
    """Get high-level overview of lineage patterns."""

    # Get basic lineage statistics
    stats_query = """
    SELECT
        COUNT(*) as total_lineage_events,
        COUNT(DISTINCT job_name) as unique_jobs,
        COUNT(DISTINCT execution_id) as unique_executions,
        COUNT(DISTINCT session_id) as unique_sessions,
        AVG(input_count) as avg_input_count,
        AVG(output_count) as avg_output_count,
        MAX(event_time) as latest_event,
        MIN(event_time) as earliest_event
    FROM vw_lineage_io
    """

    stats = db_manager.execute_query(stats_query)

    # Get most active jobs
    jobs_query = """
    SELECT
        job_name,
        COUNT(*) as execution_count,
        AVG(input_count) as avg_inputs,
        AVG(output_count) as avg_outputs,
        MAX(event_time) as last_execution
    FROM vw_lineage_io
    GROUP BY job_name
    ORDER BY execution_count DESC
    LIMIT 10
    """

    active_jobs = db_manager.execute_query(jobs_query)

    # Get dataset usage patterns
    datasets_query = """
    SELECT
        dataset_name,
        usage_type,
        COUNT(*) as usage_count
    FROM (
        SELECT
            TRIM('"' FROM json_extract(value, '$')) as dataset_name,
            'input' as usage_type
        FROM vw_lineage_io, json_each(input_names)
        WHERE input_names IS NOT NULL

        UNION ALL

        SELECT
            TRIM('"' FROM json_extract(value, '$')) as dataset_name,
            'output' as usage_type
        FROM vw_lineage_io, json_each(output_names)
        WHERE output_names IS NOT NULL
    )
    GROUP BY dataset_name, usage_type
    ORDER BY usage_count DESC
    LIMIT 20
    """

    try:
        dataset_usage = db_manager.execute_query(datasets_query)
    except Exception:
        # Fallback if JSON extraction doesn't work
        dataset_usage = []

    result = {
        "success": True,
        "overview": {
            "statistics": stats[0] if stats else {},
            "most_active_jobs": active_jobs,
            "dataset_usage_patterns": dataset_usage,
        },
    }

    # Add performance metrics if requested
    if include_metrics:
        metrics_query = """
        SELECT
            AVG(duration_ms) as avg_duration_ms,
            MAX(duration_ms) as max_duration_ms,
            AVG(lineage_to_runtime_ratio) as avg_lineage_overhead,
            COUNT(*) as total_measured_executions
        FROM vw_performance_metrics
        WHERE duration_ms IS NOT NULL
        """

        metrics = db_manager.execute_query(metrics_query)
        result["overview"]["performance_metrics"] = metrics[0] if metrics else {}

    return result


def _analyze_dependencies(
    db_manager, dataset_name: str | None, *, _include_metrics: bool
) -> dict[str, Any]:
    """Analyze upstream dependencies for datasets."""

    if dataset_name:
        # Find jobs that produce this dataset
        producer_query = """
        SELECT
            job_name,
            execution_id,
            input_names,
            output_names,
            event_time
        FROM vw_lineage_io
        WHERE json_extract(output_names, '$') LIKE '%' || ? || '%'
        ORDER BY event_time DESC
        LIMIT 20
        """

        try:
            producers = db_manager.execute_query(producer_query, [dataset_name])
            dependencies = set()

            # Extract input dependencies from each producer
            for row in producers:
                try:
                    inputs = json.loads(row.get("input_names", "[]"))
                    if isinstance(inputs, list):
                        dependencies.update(inputs)
                except Exception as e:
                    tool_logger.warning(f"Failed to parse input names: {e}")

            result = {
                "dataset": dataset_name,
                "producers": [dict(row) for row in producers],
                "upstream_dependencies": list(dependencies),
                "dependency_count": len(dependencies),
            }

        except Exception as e:
            tool_logger.error(f"Failed to analyze dependencies: {e}")
            return {"error": str(e), "dataset": dataset_name}

    else:
        # Overall dependency analysis
        dependency_query = """
        SELECT
            job_name,
            input_count,
            output_count,
            AVG(input_count) as avg_inputs,
            COUNT(*) as execution_count
        FROM vw_lineage_io
        WHERE input_count > 0
        GROUP BY job_name, input_count, output_count
        ORDER BY input_count DESC, execution_count DESC
        LIMIT 20
        """

        try:
            dependencies = db_manager.execute_query(dependency_query)
            result = {
                "dependency_patterns": [dict(row) for row in dependencies],
                "analysis_scope": "all_datasets",
            }
        except Exception as e:
            tool_logger.error(f"Failed to analyze general dependencies: {e}")
            return {"error": str(e)}

    return result


def _analyze_impact(
    db_manager, dataset_name: str | None, *, _include_metrics: bool
) -> dict[str, Any]:
    """Analyze downstream impact of dataset changes."""

    if dataset_name:
        # Find jobs that consume this dataset
        consumer_query = """
        SELECT
            job_name,
            execution_id,
            input_names,
            output_names,
            event_time
        FROM vw_lineage_io
        WHERE json_extract(input_names, '$') LIKE '%' || ? || '%'
        ORDER BY event_time DESC
        LIMIT 20
        """

        try:
            consumers = db_manager.execute_query(consumer_query, [dataset_name])
            downstream_outputs = set()

            # Extract downstream outputs from each consumer
            for row in consumers:
                try:
                    outputs = json.loads(row.get("output_names", "[]"))
                    if isinstance(outputs, list):
                        downstream_outputs.update(outputs)
                except Exception as e:
                    tool_logger.warning(f"Failed to parse output names: {e}")

            result = {
                "dataset": dataset_name,
                "consumers": [dict(row) for row in consumers],
                "downstream_outputs": list(downstream_outputs),
                "impact_scope": len(downstream_outputs),
            }

        except Exception as e:
            tool_logger.error(f"Failed to analyze impact: {e}")
            return {"error": str(e), "dataset": dataset_name}

    else:
        # Overall impact analysis
        impact_query = """
        SELECT
            job_name,
            output_count,
            input_count,
            AVG(output_count) as avg_outputs,
            COUNT(*) as execution_count
        FROM vw_lineage_io
        WHERE output_count > 0
        GROUP BY job_name, output_count, input_count
        ORDER BY output_count DESC, execution_count DESC
        LIMIT 20
        """

        try:
            impacts = db_manager.execute_query(impact_query)
            result = {
                "impact_patterns": [dict(row) for row in impacts],
                "analysis_scope": "all_datasets",
            }
        except Exception as e:
            tool_logger.error(f"Failed to analyze general impact: {e}")
            return {"error": str(e)}

    return result


def _trace_data_flow(
    db_manager, dataset_name: str | None, *, include_metrics: bool
) -> dict[str, Any]:
    """Trace complete data flow through transformations."""

    # Get chronological flow of data transformations
    flow_query = """
    SELECT
        execution_id,
        job_name,
        event_time,
        input_names,
        output_names,
        input_count,
        output_count,
        session_id
    FROM vw_lineage_io
    WHERE (? IS NULL OR
           json_extract(input_names, '$') LIKE '%' || ? || '%' OR
           json_extract(output_names, '$') LIKE '%' || ? || '%')
    ORDER BY event_time ASC
    """

    params = (
        [dataset_name, dataset_name, dataset_name]
        if dataset_name
        else [None, None, None]
    )
    flow_events = db_manager.execute_query(flow_query, params)

    # Group by session for flow analysis
    sessions = {}
    for event in flow_events:
        session_id = event.get("session_id")
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(event)

    # Build flow chains
    flow_chains = []
    for session_id, events in sessions.items():
        flow_chains.append(
            {
                "session_id": session_id,
                "event_count": len(events),
                "flow_events": events[:10],  # Limit for readability
                "start_time": events[0].get("event_time") if events else None,
                "end_time": events[-1].get("event_time") if events else None,
            }
        )

    result = {
        "success": True,
        "dataset_name": dataset_name,
        "data_flow": {
            "flow_chains": sorted(
                flow_chains, key=lambda x: x["start_time"] or "", reverse=True
            )[:5],
            "total_sessions": len(sessions),
            "total_events": len(flow_events),
        },
    }

    # Add metrics if requested
    if include_metrics and flow_events:
        execution_ids = [
            event["execution_id"] for event in flow_events if event.get("execution_id")
        ]

        if execution_ids:
            # Create safe parameterized query for metrics
            placeholders = ",".join("?" * len(execution_ids))
            metrics_query = f"""
            SELECT
                AVG(duration_ms) as avg_duration,
                MAX(duration_ms) as max_duration,
                SUM(duration_ms) as total_duration,
                COUNT(*) as measured_events
            FROM vw_performance_metrics
            WHERE execution_id IN ({placeholders})
            """

            try:
                metrics = db_manager.execute_query(metrics_query, execution_ids)
                result["data_flow"]["performance_metrics"] = (
                    dict(metrics[0]) if metrics else {}
                )
            except Exception as e:
                tool_logger.warning(f"Failed to fetch performance metrics: {e}")

    return result


@mcp.tool("find_lineage_patterns")
def find_lineage_patterns(
    pattern_type: str = "common_transforms", min_frequency: int = 2
) -> dict[str, Any]:
    """
    Find common patterns in data lineage and transformations.

    Args:
        pattern_type: Type of pattern to find - "common_transforms", "bottlenecks", "anomalies"
        min_frequency: Minimum frequency for pattern recognition

    Returns:
        Dictionary containing identified patterns and their characteristics
    """

    try:
        # Get database manager from orchestrator
        orchestrator = get_global_orchestrator()
        db_manager = orchestrator.get_db_manager()

        if pattern_type == "common_transforms":
            # Find frequently used transformation patterns
            pattern_query = """
            SELECT
                input_count,
                output_count,
                COUNT(*) as frequency,
                COUNT(DISTINCT job_name) as unique_jobs,
                GROUP_CONCAT(DISTINCT job_name) as example_jobs
            FROM vw_lineage_io
            GROUP BY input_count, output_count
            HAVING frequency >= ?
            ORDER BY frequency DESC
            """

            patterns = db_manager.execute_query(pattern_query, [min_frequency])

        elif pattern_type == "bottlenecks":
            # Find potential bottleneck datasets (high reuse)
            pattern_query = """
            SELECT
                dataset_name,
                COUNT(*) as usage_frequency,
                COUNT(DISTINCT job_name) as consuming_jobs
            FROM (
                SELECT
                    TRIM('"' FROM json_extract(value, '$')) as dataset_name,
                    job_name
                FROM vw_lineage_io, json_each(input_names)
                WHERE input_names IS NOT NULL
            )
            GROUP BY dataset_name
            HAVING usage_frequency >= ?
            ORDER BY usage_frequency DESC
            """

            try:
                patterns = db_manager.execute_query(pattern_query, [min_frequency])
            except Exception:
                patterns = []

        elif pattern_type == "anomalies":
            # Find unusual patterns (very high input/output ratios)
            pattern_query = """
            SELECT
                job_name,
                input_count,
                output_count,
                (input_count * 1.0 / NULLIF(output_count, 1)) as complexity_ratio,
                COUNT(*) as frequency,
                MAX(event_time) as last_seen
            FROM vw_lineage_io
            WHERE input_count > 5 OR output_count > 5
            GROUP BY job_name, input_count, output_count
            HAVING frequency >= ?
            ORDER BY complexity_ratio DESC
            """

            patterns = db_manager.execute_query(pattern_query, [min_frequency])

        else:
            return {
                "error": f"Unknown pattern type: {pattern_type}",
                "available_patterns": ["common_transforms", "bottlenecks", "anomalies"],
            }

        return {
            "success": True,
            "pattern_type": pattern_type,
            "min_frequency": min_frequency,
            "patterns": patterns,
            "pattern_count": len(patterns),
        }

    except Exception as e:
        return {"success": False, "error": str(e), "pattern_type": pattern_type}


def _analyze_patterns(db_manager, *, include_metrics: bool) -> dict[str, Any]:
    """Analyze transformation patterns and bottlenecks."""

    result = {
        "analysis_type": "patterns",
        "transformation_patterns": {},
        "bottleneck_analysis": {},
        "unusual_patterns": {},
    }

    try:
        # Find frequently used transformation patterns
        pattern_query = """
        SELECT
            input_count,
            output_count,
            COUNT(*) as pattern_frequency,
            AVG(CAST(input_count AS FLOAT) / NULLIF(output_count, 1)) as avg_ratio,
            MIN(event_time) as first_seen,
            MAX(event_time) as last_seen
        FROM vw_lineage_io
        WHERE input_count > 0 AND output_count > 0
        GROUP BY input_count, output_count
        HAVING COUNT(*) > 2
        ORDER BY pattern_frequency DESC
        LIMIT 15
        """

        patterns = db_manager.execute_query(pattern_query)
        result["transformation_patterns"] = {
            "common_patterns": [dict(row) for row in patterns],
            "description": "Most frequently used input/output transformation patterns",
        }

        # Find potential bottleneck datasets (high reuse)
        bottleneck_query = """
        SELECT
            dataset_name,
            COUNT(*) as usage_frequency,
            COUNT(DISTINCT job_name) as consuming_jobs,
            MIN(event_time) as first_usage,
            MAX(event_time) as last_usage
        FROM (
            SELECT
                TRIM('"' FROM json_extract(value, '$')) as dataset_name,
                job_name,
                event_time
            FROM vw_lineage_io, json_each(input_names)
            WHERE input_names IS NOT NULL
        ) dataset_usage
        WHERE dataset_name IS NOT NULL AND dataset_name != ''
        GROUP BY dataset_name
        HAVING COUNT(*) > 3
        ORDER BY usage_frequency DESC, consuming_jobs DESC
        LIMIT 10
        """

        bottlenecks = db_manager.execute_query(bottleneck_query)
        result["bottleneck_analysis"] = {
            "high_reuse_datasets": [dict(row) for row in bottlenecks],
            "description": "Datasets that are heavily reused across multiple jobs",
        }

        # Find unusual patterns (very high input/output ratios)
        unusual_query = """
        SELECT
            job_name,
            input_count,
            output_count,
            CAST(input_count AS FLOAT) / NULLIF(output_count, 1) as input_output_ratio,
            COUNT(*) as occurrence_count,
            MAX(event_time) as last_execution
        FROM vw_lineage_io
        WHERE input_count > 0 AND output_count > 0
        GROUP BY job_name, input_count, output_count
        HAVING input_output_ratio > 5 OR input_output_ratio < 0.2
        ORDER BY input_output_ratio DESC
        LIMIT 10
        """

        unusual = db_manager.execute_query(unusual_query)
        result["unusual_patterns"] = {
            "atypical_transformations": [dict(row) for row in unusual],
            "description": "Jobs with unusual input/output ratios (may indicate inefficiencies)",
        }

        if include_metrics:
            try:
                # Add performance context to patterns
                perf_query = """
                SELECT
                    AVG(duration_ms) as avg_job_duration,
                    MAX(duration_ms) as max_job_duration,
                    COUNT(DISTINCT execution_id) as measured_executions
                FROM vw_performance_metrics
                WHERE duration_ms IS NOT NULL
                """
                perf_data = db_manager.execute_query(perf_query)
                result["performance_context"] = dict(perf_data[0]) if perf_data else {}
            except Exception as e:
                tool_logger.warning(f"Failed to fetch performance context: {e}")

    except Exception as e:
        tool_logger.error(f"Failed to analyze patterns: {e}")
        return {"error": str(e), "analysis_type": "patterns"}

    return result


# Alias for backward compatibility
lineage_tool = analyze_lineage_tool
