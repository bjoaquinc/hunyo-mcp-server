#!/usr/bin/env python3
"""
Lineage Tool - Specialized data lineage analysis for LLMs.

Provides high-level functions for analyzing data lineage patterns,
tracking data flow, and understanding transformation relationships.
"""

import json
from typing import Any, Dict, List, Optional, Set

from mcp.server.fastmcp import FastMCP

from ..orchestrator import get_global_orchestrator

# Import logging utility
try:
    from ...capture.logger import get_logger

    tool_logger = get_logger("hunyo.tools.lineage")
except ImportError:
    # Fallback for testing
    class SimpleLogger:
        def info(self, msg):
            print(f"[INFO] {msg}")

        def warning(self, msg):
            print(f"[WARNING] {msg}")

        def error(self, msg):
            print(f"[ERROR] {msg}")

    tool_logger = SimpleLogger()


# Get the FastMCP instance from server.py
try:
    from ..server import mcp
except ImportError:
    # Fallback for testing - create a minimal MCP instance
    mcp = FastMCP("hunyo-test")


@mcp.tool("analyze_data_lineage")
def lineage_tool(
    dataset_name: str | None = None,
    analysis_type: str = "overview",
    include_metrics: bool = True,
) -> dict[str, Any]:
    """
    Analyze data lineage patterns and relationships in notebook executions.

    Provides specialized analysis of how data flows through notebook cells,
    including transformation tracking, dependency mapping, and impact analysis.

    Args:
        dataset_name: Focus analysis on specific dataset (None for all datasets)
        analysis_type: Type of analysis - "overview", "dependencies", "impact", "flow"
        include_metrics: Whether to include performance metrics in the analysis

    Returns:
        Dictionary containing lineage analysis results and insights

    Analysis types:
    - "overview": High-level summary of data lineage patterns
    - "dependencies": Map upstream dependencies for datasets
    - "impact": Show downstream impact of dataset changes
    - "flow": Trace complete data flow through transformations
    """

    try:
        # Get database manager from orchestrator
        orchestrator = get_global_orchestrator()
        db_manager = orchestrator.get_db_manager()

        tool_logger.info(
            f"Analyzing data lineage: {analysis_type} for dataset: {dataset_name or 'all'}"
        )

        # Route to specific analysis function
        if analysis_type == "overview":
            result = _get_lineage_overview(db_manager, include_metrics)
        elif analysis_type == "dependencies":
            result = _analyze_dependencies(db_manager, dataset_name, include_metrics)
        elif analysis_type == "impact":
            result = _analyze_impact(db_manager, dataset_name, include_metrics)
        elif analysis_type == "flow":
            result = _trace_data_flow(db_manager, dataset_name, include_metrics)
        else:
            return {
                "error": f"Unknown analysis type: {analysis_type}",
                "available_types": ["overview", "dependencies", "impact", "flow"],
            }

        # Add metadata
        result["analysis_metadata"] = {
            "analysis_type": analysis_type,
            "dataset_name": dataset_name,
            "include_metrics": include_metrics,
            "timestamp": "current",  # Could add actual timestamp
        }

        tool_logger.info(f"Lineage analysis complete: {analysis_type}")
        return result

    except Exception as e:
        error_msg = str(e)
        tool_logger.error(f"Lineage analysis failed: {error_msg}")

        return {
            "success": False,
            "error": error_msg,
            "analysis_type": analysis_type,
            "dataset_name": dataset_name,
        }


def _get_lineage_overview(db_manager, include_metrics: bool) -> dict[str, Any]:
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
    except:
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
    db_manager, dataset_name: str | None, include_metrics: bool
) -> dict[str, Any]:
    """Analyze upstream dependencies for datasets."""

    if dataset_name:
        # Focus on specific dataset
        # Find jobs that produce this dataset
        producer_query = """
        SELECT 
            job_name,
            execution_id,
            event_time,
            input_names,
            output_names
        FROM vw_lineage_io
        WHERE json_extract(output_names, '$') LIKE '%' || ? || '%'
        ORDER BY event_time DESC
        """

        producers = db_manager.execute_query(producer_query, [dataset_name])

        # Find dependencies (what inputs are needed)
        dependencies = set()
        for producer in producers:
            if producer.get("input_names"):
                try:
                    inputs = (
                        json.loads(producer["input_names"])
                        if isinstance(producer["input_names"], str)
                        else producer["input_names"]
                    )
                    if isinstance(inputs, list):
                        dependencies.update(inputs)
                except:
                    pass

        result = {
            "success": True,
            "dataset_name": dataset_name,
            "dependencies": {
                "direct_dependencies": list(dependencies),
                "producing_jobs": producers[:10],  # Limit results
                "dependency_count": len(dependencies),
            },
        }
    else:
        # Overall dependency analysis
        dependency_query = """
        SELECT 
            job_name,
            input_count,
            output_count,
            (input_count * 1.0 / NULLIF(output_count, 0)) as dependency_ratio,
            COUNT(*) as execution_count
        FROM vw_lineage_io
        GROUP BY job_name, input_count, output_count
        HAVING input_count > 0
        ORDER BY dependency_ratio DESC, execution_count DESC
        LIMIT 20
        """

        dependency_patterns = db_manager.execute_query(dependency_query)

        result = {
            "success": True,
            "dependencies": {
                "dependency_patterns": dependency_patterns,
                "analysis_note": "Shows jobs ordered by dependency complexity (input/output ratio)",
            },
        }

    return result


def _analyze_impact(
    db_manager, dataset_name: str | None, include_metrics: bool
) -> dict[str, Any]:
    """Analyze downstream impact of dataset changes."""

    if dataset_name:
        # Find jobs that consume this dataset
        consumer_query = """
        SELECT 
            job_name,
            execution_id,
            event_time,
            input_names,
            output_names,
            output_count
        FROM vw_lineage_io
        WHERE json_extract(input_names, '$') LIKE '%' || ? || '%'
        ORDER BY event_time DESC
        """

        consumers = db_manager.execute_query(consumer_query, [dataset_name])

        # Find downstream outputs
        downstream_outputs = set()
        for consumer in consumers:
            if consumer.get("output_names"):
                try:
                    outputs = (
                        json.loads(consumer["output_names"])
                        if isinstance(consumer["output_names"], str)
                        else consumer["output_names"]
                    )
                    if isinstance(outputs, list):
                        downstream_outputs.update(outputs)
                except:
                    pass

        result = {
            "success": True,
            "dataset_name": dataset_name,
            "impact": {
                "direct_consumers": consumers[:10],  # Limit results
                "downstream_datasets": list(downstream_outputs),
                "impact_scope": len(downstream_outputs),
            },
        }
    else:
        # Overall impact analysis
        impact_query = """
        SELECT 
            job_name,
            output_count,
            COUNT(*) as execution_count,
            MAX(event_time) as last_execution,
            AVG(input_count) as avg_input_complexity
        FROM vw_lineage_io
        GROUP BY job_name
        HAVING output_count > 0
        ORDER BY output_count DESC, execution_count DESC
        LIMIT 20
        """

        impact_patterns = db_manager.execute_query(impact_query)

        result = {
            "success": True,
            "impact": {
                "high_impact_jobs": impact_patterns,
                "analysis_note": "Shows jobs with highest output generation (potential impact)",
            },
        }

    return result


def _trace_data_flow(
    db_manager, dataset_name: str | None, include_metrics: bool
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
            metrics_query = """
            SELECT 
                AVG(duration_ms) as avg_duration,
                MAX(duration_ms) as max_duration,
                SUM(duration_ms) as total_duration,
                COUNT(*) as measured_events
            FROM vw_performance_metrics
            WHERE execution_id IN ({})
            """.format(
                ",".join(["?" for _ in execution_ids])
            )

            try:
                metrics = db_manager.execute_query(metrics_query, execution_ids)
                result["data_flow"]["performance_metrics"] = (
                    metrics[0] if metrics else {}
                )
            except:
                pass

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
            except:
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
