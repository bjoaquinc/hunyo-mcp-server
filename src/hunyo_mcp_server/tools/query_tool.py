#!/usr/bin/env python3
"""
Query Tool - General SQL querying interface for LLMs.

Provides a secure interface for LLMs to query the captured notebook data
using SQL with safety constraints and helpful query templates.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP

from hunyo_mcp_server.orchestrator import get_global_orchestrator

# Import logging utility
try:
    from ...capture.logger import get_logger

    tool_logger = get_logger("hunyo.tools.query")
except ImportError:
    # Fallback for testing
    class SimpleLogger:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            pass

    tool_logger = SimpleLogger()


# Constants
MAX_QUERY_LIMIT = 1000  # Maximum number of rows that can be returned by a query
QUERY_LOG_TRUNCATE_LENGTH = 100  # Maximum length for query logging

# Get the FastMCP instance from server.py
try:
    from hunyo_mcp_server.server import mcp
except ImportError:
    # Fallback for testing - create a minimal MCP instance
    mcp = FastMCP("hunyo-test")


@mcp.tool("query_database")
def query_tool(
    sql_query: str, limit: int | None = 100, *, safe_mode: bool = True
) -> dict[str, Any]:
    """
    Execute SQL queries on the captured notebook execution data.

    Allows querying of runtime events, lineage events, and derived views
    to analyze notebook execution patterns, performance, and data lineage.

    Args:
        sql_query: SQL query to execute (SELECT statements only in safe mode)
        limit: Maximum number of rows to return (default: 100, max: 1000)
        safe_mode: If True, only allow SELECT queries (default: True)

    Returns:
        Dictionary containing query results, metadata, and execution info

    Example queries:
    - "SELECT * FROM runtime_events ORDER BY timestamp DESC LIMIT 10"
    - "SELECT execution_id, duration_ms FROM vw_performance_metrics WHERE duration_ms > 1000"
    - "SELECT job_name, input_count, output_count FROM vw_lineage_io"
    """

    try:
        # Get database manager from orchestrator
        orchestrator = get_global_orchestrator()
        db_manager = orchestrator.get_db_manager()

        # Security: Validate query in safe mode
        if safe_mode:
            sql_query_lower = sql_query.lower().strip()

            # Only allow SELECT queries
            if not sql_query_lower.startswith("select"):
                return {
                    "error": "Only SELECT queries are allowed in safe mode",
                    "query": sql_query,
                    "safe_mode": safe_mode,
                }

            # Block potentially dangerous operations
            dangerous_keywords = [
                "delete",
                "update",
                "insert",
                "drop",
                "create",
                "alter",
                "truncate",
                "replace",
                "merge",
                "exec",
                "execute",
            ]

            for keyword in dangerous_keywords:
                if keyword in sql_query_lower:
                    return {
                        "error": f"Query contains potentially dangerous keyword: {keyword}",
                        "query": sql_query,
                        "safe_mode": safe_mode,
                    }

        # Apply limit constraints
        if limit is None:
            limit = 100
        elif limit > MAX_QUERY_LIMIT:
            limit = MAX_QUERY_LIMIT
            tool_logger.warning(f"Query limit capped at {MAX_QUERY_LIMIT} rows")

        # Add LIMIT clause if not present
        if "limit" not in sql_query.lower() and limit > 0:
            sql_query = f"{sql_query.rstrip(';')} LIMIT {limit}"

        tool_logger.info(
            f"Executing query: {sql_query[:QUERY_LOG_TRUNCATE_LENGTH]}{'...' if len(sql_query) > QUERY_LOG_TRUNCATE_LENGTH else ''}"
        )

        # Execute query
        results = db_manager.execute_query(sql_query)

        # Prepare response
        response = {
            "success": True,
            "query": sql_query,
            "row_count": len(results),
            "results": results,
            "limit_applied": limit,
            "safe_mode": safe_mode,
            "metadata": {
                "execution_time": "N/A",  # Could add timing if needed
                "columns": list(results[0].keys()) if results else [],
            },
        }

        tool_logger.info(f"Query executed successfully, returned {len(results)} rows")
        return response

    except Exception as e:
        error_msg = str(e)
        tool_logger.error(f"Query execution failed: {error_msg}")

        return {
            "success": False,
            "error": error_msg,
            "query": sql_query,
            "safe_mode": safe_mode,
            "results": [],
            "row_count": 0,
        }


def get_query_examples() -> list[dict[str, str]]:
    """
    Get example queries for different use cases.

    Returns:
        List of example queries with descriptions
    """

    examples = [
        {
            "name": "Recent Runtime Events",
            "description": "Get the most recent cell executions with performance metrics",
            "query": """
            SELECT
                execution_id,
                cell_id,
                event_type,
                duration_ms,
                memory_delta_mb,
                timestamp
            FROM vw_performance_metrics
            WHERE runtime_event_type IN ('cell_execution_start', 'cell_execution_end')
            ORDER BY runtime_timestamp DESC
            LIMIT 20
            """,
        },
        {
            "name": "Data Lineage Summary",
            "description": "Show input/output relationships for data processing jobs",
            "query": """
            SELECT
                job_name,
                input_names,
                output_names,
                input_count,
                output_count,
                event_time
            FROM vw_lineage_io
            WHERE event_type = 'COMPLETE'
            ORDER BY event_time DESC
            LIMIT 15
            """,
        },
        {
            "name": "Performance Analysis",
            "description": "Find slow-running cells and operations",
            "query": """
            SELECT
                execution_id,
                cell_id,
                duration_ms,
                memory_delta_mb,
                lineage_to_runtime_ratio
            FROM vw_performance_metrics
            WHERE duration_ms > 1000
            ORDER BY duration_ms DESC
            LIMIT 10
            """,
        },
        {
            "name": "Error Analysis",
            "description": "Find runtime errors and their details",
            "query": """
            SELECT
                execution_id,
                cell_id,
                event_type,
                error_info,
                timestamp
            FROM runtime_events
            WHERE error_info IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10
            """,
        },
        {
            "name": "Memory Usage Trends",
            "description": "Analyze memory usage patterns across executions",
            "query": """
            SELECT
                execution_id,
                start_memory_mb,
                end_memory_mb,
                (end_memory_mb - start_memory_mb) as memory_delta_mb,
                duration_ms,
                timestamp
            FROM runtime_events
            WHERE start_memory_mb IS NOT NULL
                AND end_memory_mb IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 20
            """,
        },
        {
            "name": "Session Overview",
            "description": "Get summary statistics by session",
            "query": """
            SELECT
                session_id,
                COUNT(*) as total_events,
                COUNT(DISTINCT execution_id) as unique_executions,
                AVG(duration_ms) as avg_duration_ms,
                MAX(duration_ms) as max_duration_ms,
                MIN(timestamp) as session_start,
                MAX(timestamp) as session_end
            FROM runtime_events
            WHERE duration_ms IS NOT NULL
            GROUP BY session_id
            ORDER BY session_start DESC
            """,
        },
    ]

    return examples


def validate_query_syntax(query: str) -> dict[str, Any]:
    """
    Basic validation of SQL query syntax.

    Args:
        query: SQL query string

    Returns:
        Validation result with success flag and any issues
    """

    issues = []
    query_lower = query.lower().strip()

    # Check basic structure
    if not query_lower.startswith("select"):
        issues.append("Query should start with SELECT")

    # Check for basic SQL injection patterns
    dangerous_patterns = [
        "--",
        "/*",
        "*/",
        "xp_",
        "sp_",
        "exec(",
        "execute(",
        "union select",
        "drop table",
        "delete from",
    ]

    for pattern in dangerous_patterns:
        if pattern in query_lower:
            issues.append(f"Potentially dangerous pattern detected: {pattern}")

    # Check for table references
    known_tables = [
        "runtime_events",
        "lineage_events",
        "vw_lineage_io",
        "vw_performance_metrics",
    ]
    has_known_table = any(table in query_lower for table in known_tables)

    if not has_known_table:
        issues.append(
            "Query should reference at least one known table: "
            + ", ".join(known_tables)
        )

    return {"valid": len(issues) == 0, "issues": issues, "query": query}
