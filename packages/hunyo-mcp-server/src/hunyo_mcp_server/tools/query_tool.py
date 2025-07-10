#!/usr/bin/env python3
"""
Query Tool - General SQL querying interface for LLMs.

Provides a secure interface for LLMs to query the captured notebook data
using SQL with safety constraints and helpful query templates.
"""

from typing import Annotated, Any, ClassVar

import sqlparse
from pydantic import Field
from sqlparse import tokens
from sqlparse.sql import Statement

from hunyo_mcp_server.orchestrator import get_global_orchestrator

# Import logging utility
try:
    from hunyo_mcp_server.logger import get_logger

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
QUERY_EXCERPT_LENGTH = 100  # Maximum length for query excerpts in error messages

# Get the shared FastMCP instance
from hunyo_mcp_server.mcp_instance import mcp


class SQLSecurityValidator:
    """Enhanced SQL security validator using sqlparse for AST-based validation."""

    FORBIDDEN_KEYWORDS: ClassVar[set[str]] = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "MERGE",
        "REPLACE",
        "EXEC",
        "EXECUTE",
        "CALL",
        "LOAD",
        "INSTALL",
        "SET",
        "PRAGMA",
        "ATTACH",
        "DETACH",
        "COPY",
    }

    ALLOWED_TABLES: ClassVar[set[str]] = {
        "runtime_events",
        "lineage_events",
        "dataframe_lineage_events",
        "vw_lineage_io",
        "vw_performance_metrics",
        "vw_dataframe_lineage",
    }

    # Common dangerous patterns for fast rejection
    DANGEROUS_PATTERNS: ClassVar[list[str]] = [
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
        "update set",
        "insert into",
        "create table",
        "alter table",
        "truncate table",
        "load_extension(",
        "system(",
        "load_file(",
        "file_read(",
        "file_write(",
        "shell(",
        "cmd(",
        "command(",
        "subprocess(",
        "into outfile",
        "into dumpfile",
        "load data",
        "backup",
        "restore",
        "copy to",
        "copy from",
        "bulk insert",
    ]

    def validate_readonly_query(self, sql: str) -> dict[str, Any]:
        """
        Fast path for common cases, comprehensive validation for edge cases.

        Uses hybrid approach:
        1. Fast rejection for obvious violations
        2. Quick keyword scan for performance
        3. Comprehensive AST parsing when needed
        """
        try:
            sql_clean = sql.strip().lower()

            # Fast rejection for obvious violations
            if not sql_clean.startswith("select"):
                return {
                    "valid": False,
                    "error": "Only SELECT statements allowed",
                    "suggestion": "Start your query with SELECT to retrieve data",
                    "query_excerpt": self._get_query_excerpt(sql),
                }

            # Quick dangerous pattern scan
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern in sql_clean:
                    return {
                        "valid": False,
                        "error": f"Query contains dangerous pattern: {pattern}",
                        "suggestion": "Remove dangerous SQL operations and use only SELECT statements",
                        "query_excerpt": self._get_query_excerpt(sql),
                    }

            # Quick keyword scan for performance
            for keyword in self.FORBIDDEN_KEYWORDS:
                if keyword.lower() in sql_clean:
                    # Use AST parsing to confirm (avoid false positives)
                    return self._comprehensive_validation(sql)

            # For clean queries, still do basic AST validation
            return self._ast_validation(sql)

        except Exception as e:
            return {
                "valid": False,
                "error": f"SQL parsing failed: {e!s}",
                "query_excerpt": self._get_query_excerpt(sql),
                "suggestion": "Check your SQL syntax and try again",
            }

    def _comprehensive_validation(self, sql: str) -> dict[str, Any]:
        """Comprehensive AST-based validation for suspicious queries."""
        try:
            parsed = sqlparse.parse(sql)

            if not parsed:
                return {
                    "valid": False,
                    "error": "Empty or invalid SQL",
                    "query_excerpt": self._get_query_excerpt(sql),
                }

            # Check each statement
            for statement in parsed:
                # Verify it's a SELECT statement
                if not self._is_select_statement(statement):
                    return {
                        "valid": False,
                        "error": "Only SELECT statements allowed",
                        "suggestion": "Use SELECT to query data, not modify it",
                        "query_excerpt": self._get_query_excerpt(sql),
                    }

                # Check for forbidden operations
                forbidden_op = self._get_forbidden_operation(statement)
                if forbidden_op:
                    return {
                        "valid": False,
                        "error": f"Query contains forbidden operation: {forbidden_op}",
                        "suggestion": "Remove data modification operations and use only SELECT",
                        "query_excerpt": self._get_query_excerpt(sql),
                    }

                # Validate table access
                unauthorized_tables = self._get_unauthorized_tables(statement)
                if unauthorized_tables:
                    return {
                        "valid": False,
                        "error": f"Query accesses unauthorized tables: {', '.join(unauthorized_tables)}",
                        "suggestion": f"Use only authorized tables: {', '.join(self.ALLOWED_TABLES)}",
                        "query_excerpt": self._get_query_excerpt(sql),
                    }

            return {"valid": True, "error": None}

        except Exception as e:
            return {
                "valid": False,
                "error": f"Comprehensive validation failed: {e!s}",
                "query_excerpt": self._get_query_excerpt(sql),
                "suggestion": "Check your SQL syntax and try again",
            }

    def _ast_validation(self, sql: str) -> dict[str, Any]:
        """Basic AST validation for clean queries."""
        try:
            parsed = sqlparse.parse(sql)

            if not parsed:
                return {
                    "valid": False,
                    "error": "Empty or invalid SQL",
                    "query_excerpt": self._get_query_excerpt(sql),
                }

            # Basic checks
            for statement in parsed:
                if not self._is_select_statement(statement):
                    return {
                        "valid": False,
                        "error": "Only SELECT statements allowed",
                        "suggestion": "Use SELECT to query notebook memories",
                        "query_excerpt": self._get_query_excerpt(sql),
                    }

                # Basic table validation
                unauthorized_tables = self._get_unauthorized_tables(statement)
                if unauthorized_tables:
                    return {
                        "valid": False,
                        "error": f"Query accesses unauthorized tables: {', '.join(unauthorized_tables)}",
                        "suggestion": f"Use only authorized tables: {', '.join(self.ALLOWED_TABLES)}",
                        "query_excerpt": self._get_query_excerpt(sql),
                    }

            return {"valid": True, "error": None}

        except Exception as e:
            return {
                "valid": False,
                "error": f"AST validation failed: {e!s}",
                "query_excerpt": self._get_query_excerpt(sql),
                "suggestion": "Check your SQL syntax and try again",
            }

    def _is_select_statement(self, statement: Statement) -> bool:
        """Check if statement is a SELECT query."""
        first_token = statement.token_first(skip_ws=True, skip_cm=True)
        return (
            first_token
            and first_token.ttype is tokens.Keyword.DML
            and first_token.normalized == "SELECT"
        )

    def _get_forbidden_operation(self, statement: Statement) -> str | None:
        """Get the first forbidden operation found, or None."""
        for token in statement.flatten():
            if (
                token.ttype is tokens.Keyword
                and token.normalized in self.FORBIDDEN_KEYWORDS
            ):
                return token.normalized
        return None

    def _get_unauthorized_tables(self, statement: Statement) -> set[str]:
        """Get set of unauthorized tables referenced in the query."""
        all_tables = self._extract_table_names(statement)
        unauthorized = all_tables - self.ALLOWED_TABLES
        return unauthorized

    def _extract_table_names(self, statement: Statement) -> set[str]:
        """Extract table names using enhanced FROM/JOIN clause detection."""
        tables = set()

        # Convert to flat token list for easier processing
        token_list = list(statement.flatten())

        for i, token in enumerate(token_list):
            if token.ttype is tokens.Keyword and (
                "FROM" in token.normalized or "JOIN" in token.normalized
            ):
                # Look for the next identifier token
                for j in range(i + 1, len(token_list)):
                    next_token = token_list[j]

                    # Skip whitespace and comments
                    if next_token.ttype in (
                        None,
                        tokens.Whitespace,
                        tokens.Comment.Single,
                        tokens.Comment.Multiline,
                    ):
                        continue

                    # Found an identifier (name, quoted string, or untyped)
                    if (
                        next_token.ttype is tokens.Name
                        or next_token.ttype is None
                        or next_token.ttype
                        in (
                            tokens.Literal.String.Symbol,
                            tokens.Literal.String.Single,
                            tokens.Literal.String.Double,
                        )
                    ):
                        table_name = next_token.value.lower().strip()
                        # Remove quotes if present
                        table_name = table_name.strip('"`\'')
                        if table_name:
                            tables.add(table_name)
                        break

        return tables

    def _get_query_excerpt(self, sql: str) -> str:
        """Get a safe excerpt of the query for error reporting."""
        if len(sql) <= QUERY_EXCERPT_LENGTH:
            return sql
        return sql[:QUERY_EXCERPT_LENGTH] + "..."

    def _generate_suggestion(self, sql: str, error: str) -> str:
        """Generate helpful suggestions based on the error."""
        sql_lower = sql.lower()

        if "select" not in sql_lower:
            return (
                "Start your query with SELECT to retrieve data from notebook memories"
            )

        if any(keyword.lower() in sql_lower for keyword in self.FORBIDDEN_KEYWORDS):
            return "Remove data modification operations and use only SELECT statements"

        if "unauthorized tables" in error:
            return f"Use only authorized tables: {', '.join(self.ALLOWED_TABLES)}"

        return "Check your SQL syntax and ensure it's a valid SELECT query"


@mcp.tool(
    "query_memories",
    description="Execute SQL queries against notebook execution memories with enhanced security",
)
def query_memories(
    sql_query: Annotated[
        str, Field(description="SQL query to execute (SELECT statements only)")
    ],
    limit: Annotated[
        int,
        Field(description="Maximum number of rows to return (default: 100, max: 1000)"),
    ] = 100,
) -> dict[str, Any]:
    """
    Execute SQL queries against notebook execution memories with enhanced security.

    Uses sqlparse for AST-based validation to prevent SQL injection and
    unauthorized operations. Only SELECT queries on authorized tables are allowed.

    Security Features:
    - AST-based SQL parsing prevents injection attacks
    - Whitelist-based table access control
    - Hybrid validation for performance and security
    - Enhanced error reporting with suggestions
    - Always-on security (no unsafe mode)

    Args:
        sql_query: SQL query to execute (SELECT statements only)
        limit: Maximum number of rows to return (default: 100, max: 1000)

    Returns:
        Dictionary containing query results, metadata, and execution info

    Example queries:
    - "SELECT * FROM runtime_events ORDER BY timestamp DESC LIMIT 10"
    - "SELECT execution_id, duration_ms FROM vw_performance_metrics WHERE duration_ms > 1000"
    - "SELECT operation_type, COUNT(*) FROM dataframe_lineage_events GROUP BY operation_type"
    - "SELECT job_name, input_count, output_count FROM vw_lineage_io"
    """

    try:
        # Get database manager from orchestrator
        orchestrator = get_global_orchestrator()
        db_manager = orchestrator.get_db_manager()

        # Security: Always-on enhanced validation
        validator = SQLSecurityValidator()
        validation_result = validator.validate_readonly_query(sql_query)

        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["error"],
                "query": sql_query,
                "suggestion": validation_result.get("suggestion"),
                "query_excerpt": validation_result.get("query_excerpt"),
                "results": [],
                "row_count": 0,
            }

        # Apply limit constraints
        if limit > MAX_QUERY_LIMIT:
            limit = MAX_QUERY_LIMIT
            tool_logger.warning(f"Query limit capped at {MAX_QUERY_LIMIT} rows")

        # Add LIMIT clause if not present
        if "limit" not in sql_query.lower() and limit > 0:
            sql_query = f"{sql_query.rstrip(';')} LIMIT {limit}"

        tool_logger.info(
            f"Querying memories: {sql_query[:QUERY_LOG_TRUNCATE_LENGTH]}{'...' if len(sql_query) > QUERY_LOG_TRUNCATE_LENGTH else ''}"
        )

        # Execute query against notebook memories
        results = db_manager.execute_query(sql_query)

        # Prepare response
        response = {
            "success": True,
            "query": sql_query,
            "row_count": len(results),
            "results": results,
            "limit_applied": limit,
            "metadata": {
                "execution_time": "N/A",  # Could add timing if needed
                "columns": list(results[0].keys()) if results else [],
            },
        }

        tool_logger.info(
            f"Memory query executed successfully, returned {len(results)} rows"
        )
        return response

    except Exception as e:
        error_msg = str(e)
        tool_logger.error(f"Memory query execution failed: {error_msg}")

        return {
            "success": False,
            "error": error_msg,
            "query": sql_query,
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
    Enhanced validation using sqlparse AST parsing with helpful error reporting.

    Args:
        query: SQL query string

    Returns:
        Validation result with success flag, issues, and suggestions
    """
    validator = SQLSecurityValidator()
    result = validator.validate_readonly_query(query)

    return {
        "valid": result["valid"],
        "issues": [result["error"]] if result["error"] else [],
        "suggestions": [result.get("suggestion")] if result.get("suggestion") else [],
        "query": query,
        "query_excerpt": result.get("query_excerpt", query),
    }


query_tool = query_memories
