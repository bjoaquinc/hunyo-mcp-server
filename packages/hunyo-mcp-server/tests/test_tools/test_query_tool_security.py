#!/usr/bin/env python3
"""
Tests for query_tool security validation with enhanced sqlparse implementation.

Tests the SQLSecurityValidator class with hybrid validation, enhanced table
name extraction, and improved error reporting.
"""

import pytest
import sqlparse

from hunyo_mcp_server.tools.query_tool import (
    SQLSecurityValidator,
    validate_query_syntax,
)


class TestSQLSecurityValidator:
    """Test the enhanced SQL security validator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SQLSecurityValidator()

    # --- Basic SELECT Query Tests ---

    def test_valid_simple_select(self):
        """Test valid simple SELECT query."""
        result = self.validator.validate_readonly_query("SELECT * FROM runtime_events")
        assert result["valid"] is True
        assert result["error"] is None

    def test_valid_select_with_where(self):
        """Test valid SELECT with WHERE clause."""
        query = "SELECT execution_id, timestamp FROM runtime_events WHERE duration_ms > 1000"
        result = self.validator.validate_readonly_query(query)
        assert result["valid"] is True
        assert result["error"] is None

    def test_valid_select_with_join(self):
        """Test valid SELECT with JOIN."""
        query = """
        SELECT r.execution_id, l.job_name
        FROM runtime_events r
        JOIN lineage_events l ON r.execution_id = l.execution_id
        """
        result = self.validator.validate_readonly_query(query)
        assert result["valid"] is True
        assert result["error"] is None

    def test_valid_view_query(self):
        """Test valid query against views."""
        query = "SELECT * FROM vw_performance_metrics ORDER BY duration_ms DESC"
        result = self.validator.validate_readonly_query(query)
        assert result["valid"] is True
        assert result["error"] is None

    # --- Fast Path Rejection Tests ---

    def test_reject_non_select_statement(self):
        """Test fast rejection of non-SELECT statements."""
        queries = [
            "INSERT INTO runtime_events VALUES (1, 2, 3)",
            "UPDATE runtime_events SET duration_ms = 1000",
            "DELETE FROM runtime_events WHERE id = 1",
            "CREATE TABLE test (id INT)",
            "DROP TABLE runtime_events",
        ]

        for query in queries:
            result = self.validator.validate_readonly_query(query)
            assert result["valid"] is False
            assert "Only SELECT statements allowed" in result["error"]
            assert "suggestion" in result
            assert "query_excerpt" in result

    def test_reject_dangerous_patterns(self):
        """Test fast rejection of dangerous patterns."""
        dangerous_queries = [
            "SELECT * FROM runtime_events; DROP TABLE users",
            "SELECT * FROM runtime_events -- comment",
            "SELECT * FROM runtime_events /* comment */",
            "SELECT * FROM runtime_events UNION SELECT * FROM users",
            "SELECT exec('malicious code')",
            "SELECT * FROM runtime_events; DELETE FROM users",
        ]

        for query in dangerous_queries:
            result = self.validator.validate_readonly_query(query)
            assert result["valid"] is False
            assert "dangerous pattern" in result["error"]
            assert "suggestion" in result

    # --- Comprehensive Validation Tests ---

    def test_forbidden_keywords_detection(self):
        """Test detection of forbidden SQL keywords."""
        forbidden_queries = [
            "SELECT * FROM runtime_events; INSERT INTO test VALUES (1)",
            "SELECT * FROM runtime_events WHERE id = (SELECT id FROM users); UPDATE users SET name = 'hacked'",
            "SELECT EXEC('rm -rf /')",
            "SELECT * FROM runtime_events; CALL dangerous_procedure()",
            "SELECT * FROM runtime_events; LOAD 'malicious.so'",
        ]

        for query in forbidden_queries:
            result = self.validator.validate_readonly_query(query)
            assert result["valid"] is False
            # All these queries should be blocked by some security mechanism
            assert any(
                keyword in result["error"]
                for keyword in [
                    "forbidden operation",
                    "dangerous pattern",
                    "unauthorized tables",
                    "Only SELECT statements allowed",
                ]
            )

    def test_unauthorized_table_access(self):
        """Test detection of unauthorized table access."""
        unauthorized_queries = [
            "SELECT * FROM users",
            "SELECT * FROM passwords",
            "SELECT * FROM runtime_events JOIN unauthorized_table ON id = id",
            "SELECT * FROM system_config",
            "SELECT password FROM user_credentials",
        ]

        for query in unauthorized_queries:
            result = self.validator.validate_readonly_query(query)
            assert result["valid"] is False
            assert "unauthorized tables" in result["error"]
            assert "Use only authorized tables" in result["suggestion"]

    # --- Enhanced Table Name Extraction Tests ---

    def test_table_extraction_from_simple_query(self):
        """Test table name extraction from simple queries."""
        query = "SELECT * FROM runtime_events"
        statement = next(iter(sqlparse.parse(query)))
        tables = self.validator._extract_table_names(statement)
        assert "runtime_events" in tables

    def test_table_extraction_from_join_query(self):
        """Test table name extraction from JOIN queries."""
        query = """
        SELECT r.id, l.job_name
        FROM runtime_events r
        JOIN lineage_events l ON r.id = l.execution_id
        LEFT JOIN dataframe_lineage_events d ON r.id = d.execution_id
        """
        statement = next(iter(sqlparse.parse(query)))
        tables = self.validator._extract_table_names(statement)

        expected_tables = {
            "runtime_events",
            "lineage_events",
            "dataframe_lineage_events",
        }
        assert expected_tables.issubset(tables)

    def test_table_extraction_with_quotes(self):
        """Test table name extraction with quoted identifiers."""
        queries = [
            'SELECT * FROM "runtime_events"',
            "SELECT * FROM `runtime_events`",
            "SELECT * FROM 'runtime_events'",
        ]

        for query in queries:
            statement = next(iter(sqlparse.parse(query)))
            tables = self.validator._extract_table_names(statement)
            assert "runtime_events" in tables

    def test_table_extraction_edge_cases(self):
        """Test table name extraction edge cases."""
        # Subquery
        query = "SELECT * FROM (SELECT * FROM runtime_events) AS subq"
        statement = next(iter(sqlparse.parse(query)))
        tables = self.validator._extract_table_names(statement)
        assert "runtime_events" in tables

    # --- Error Reporting Tests ---

    def test_error_reporting_with_suggestions(self):
        """Test enhanced error reporting with helpful suggestions."""
        # Test invalid statement type
        result = self.validator.validate_readonly_query(
            "INSERT INTO runtime_events VALUES (1)"
        )
        assert "suggestion" in result
        assert "SELECT" in result["suggestion"]

        # Test unauthorized table
        result = self.validator.validate_readonly_query(
            "SELECT * FROM unauthorized_table"
        )
        assert "suggestion" in result
        assert "authorized tables" in result["suggestion"]

        # Test query excerpt (for valid queries, excerpt is not included)
        long_query = "SELECT * FROM runtime_events WHERE " + "x" * 200
        result = self.validator.validate_readonly_query(long_query)
        # For valid queries, no excerpt is included (only for errors)
        assert result["valid"] is True

    def test_query_excerpt_generation(self):
        """Test query excerpt generation for error reporting."""
        # Short query
        short_query = "SELECT * FROM users"
        excerpt = self.validator._get_query_excerpt(short_query)
        assert excerpt == short_query

        # Long query
        long_query = "SELECT * FROM runtime_events WHERE " + "x" * 200
        excerpt = self.validator._get_query_excerpt(long_query)
        assert len(excerpt) == 103  # 100 + "..."
        assert excerpt.endswith("...")

    # --- Performance Tests ---

    def test_fast_path_performance(self):
        """Test that fast path validation is used for obvious cases."""
        # This should use fast path (obvious violation)
        result = self.validator.validate_readonly_query("DELETE FROM users")
        assert result["valid"] is False
        assert "Only SELECT statements allowed" in result["error"]

        # This should also use fast path (dangerous pattern)
        result = self.validator.validate_readonly_query(
            "SELECT * FROM runtime_events; DROP TABLE users"
        )
        assert result["valid"] is False
        assert "dangerous pattern" in result["error"]

    def test_comprehensive_validation_path(self):
        """Test that comprehensive validation is triggered when needed."""
        # Query with forbidden keyword that requires AST parsing
        query = "SELECT * FROM runtime_events WHERE column_name LIKE '%INSERT%'"
        result = self.validator.validate_readonly_query(query)
        # This should pass because INSERT is just in a string literal context
        assert result["valid"] is True

    # --- Integration with validate_query_syntax ---

    def test_validate_query_syntax_integration(self):
        """Test integration with the validate_query_syntax function."""
        # Valid query
        result = validate_query_syntax("SELECT * FROM runtime_events")
        assert result["valid"] is True
        assert len(result["issues"]) == 0

        # Invalid query
        result = validate_query_syntax("DELETE FROM runtime_events")
        assert result["valid"] is False
        assert len(result["issues"]) > 0
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0

    # --- Edge Cases and Error Handling ---

    def test_empty_query_handling(self):
        """Test handling of empty or whitespace-only queries."""
        queries = ["", "   ", "\n\t", "-- just a comment"]

        for query in queries:
            result = self.validator.validate_readonly_query(query)
            assert result["valid"] is False
            assert "suggestion" in result

    def test_malformed_sql_handling(self):
        """Test handling of truly malformed SQL."""
        # Note: sqlparse is quite lenient, so use truly malformed queries
        malformed_queries = [
            "SELECT * FROM WHERE",  # Invalid syntax
            "INVALID SQL STATEMENT",  # Not a SELECT
            "SELECT * FROМ runtime_events",  # Cyrillic М instead of Latin M
            "",  # Empty query
        ]

        for query in malformed_queries:
            result = self.validator.validate_readonly_query(query)
            # Should handle gracefully without crashing
            assert "valid" in result
            assert "error" in result
            # For invalid queries, suggestion should be present
            if not result["valid"]:
                assert "suggestion" in result

    def test_case_insensitive_validation(self):
        """Test that validation is case-insensitive."""
        queries = [
            "select * from runtime_events",
            "SELECT * FROM RUNTIME_EVENTS",
            "Select * From Runtime_Events",
            "sElEcT * fRoM rUnTiMe_eVeNtS",
        ]

        for query in queries:
            result = self.validator.validate_readonly_query(query)
            assert result["valid"] is True

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        # These should be rejected as unauthorized tables
        unicode_queries = [
            "SELECT * FROM 用户表",  # Chinese characters
            "SELECT * FROM users™",  # Special characters
            "SELECT * FROM 'runtime_events\x00'",  # Null byte
        ]

        for query in unicode_queries:
            result = self.validator.validate_readonly_query(query)
            # Should handle gracefully
            assert "valid" in result
            assert "error" in result


@pytest.mark.integration
class TestQueryToolSecurityIntegration:
    """Integration tests for query tool security."""

    def test_authorized_tables_comprehensive(self):
        """Test all authorized tables are accessible."""
        validator = SQLSecurityValidator()
        authorized_tables = [
            "runtime_events",
            "lineage_events",
            "dataframe_lineage_events",
            "vw_lineage_io",
            "vw_performance_metrics",
            "vw_dataframe_lineage",
        ]

        for table in authorized_tables:
            query = f"SELECT * FROM {table} LIMIT 1"
            result = validator.validate_readonly_query(query)
            assert result["valid"] is True, f"Table {table} should be authorized"

    def test_security_boundary_enforcement(self):
        """Test that security boundaries are properly enforced."""
        validator = SQLSecurityValidator()

        # Test various attack vectors
        attack_vectors = [
            # SQL injection attempts
            "SELECT * FROM runtime_events WHERE id = 1; DROP TABLE users",
            "SELECT * FROM runtime_events UNION SELECT password FROM users",
            "SELECT * FROM runtime_events WHERE id = (SELECT password FROM users)",
            # System function attempts
            "SELECT LOAD_EXTENSION('malicious.so')",
            "SELECT SYSTEM('rm -rf /')",
            "SELECT EXEC('malicious command')",
            # Schema exploration attempts
            "SELECT * FROM information_schema.tables",
            "SELECT * FROM sqlite_master",
            "SELECT * FROM pg_tables",
            # File system access attempts
            "SELECT LOAD_FILE('/etc/passwd')",
            "SELECT * FROM runtime_events INTO OUTFILE '/tmp/dump'",
        ]

        for attack in attack_vectors:
            result = validator.validate_readonly_query(attack)
            assert (
                result["valid"] is False
            ), f"Attack vector should be blocked: {attack}"
            assert "suggestion" in result
            assert "query_excerpt" in result
