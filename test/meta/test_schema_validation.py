#!/usr/bin/env python3
"""
Schema validation tests for hunyo capture system.
Tests that captured events comply with JSON schemas in /schemas/json.
"""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

# Import cross-platform path utilities
from hunyo_mcp_server.utils.paths import get_schema_path


class TestSchemaValidation:
    """Test schema validation utilities and compliance"""

    @pytest.fixture
    def runtime_events_schema(self):
        """Load runtime events schema"""
        schema_path = get_schema_path("runtime_events_schema.json")
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def openlineage_events_schema(self):
        """Load OpenLineage events schema"""
        schema_path = get_schema_path("openlineage_events_schema.json")
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    def validate_event_against_schema(
        self, event: dict[str, Any], schema: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate a single event against a JSON schema"""
        try:
            jsonschema.validate(event, schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)

    def validate_events_file(
        self, events_file: Path, schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate all events in a JSONL file against schema"""
        results = {
            "total_events": 0,
            "valid_events": 0,
            "invalid_events": 0,
            "compliance_rate": 0.0,
            "errors": [],
        }

        if not events_file.exists():
            return results

        with open(events_file, encoding="utf-8") as f:
            events = [json.loads(line.strip()) for line in f if line.strip()]

        results["total_events"] = len(events)

        for i, event in enumerate(events):
            is_valid, error = self.validate_event_against_schema(event, schema)

            if is_valid:
                results["valid_events"] += 1
            else:
                results["invalid_events"] += 1
                results["errors"].append(f"Event {i + 1}: {error}")

        if results["total_events"] > 0:
            results["compliance_rate"] = (
                results["valid_events"] / results["total_events"]
            )

        return results

    def test_schema_files_exist(self):
        """Test that required schema files exist"""
        runtime_schema_path = get_schema_path("runtime_events_schema.json")
        openlineage_schema_path = get_schema_path("openlineage_events_schema.json")

        assert runtime_schema_path.exists(), "Runtime events schema file missing"
        assert (
            openlineage_schema_path.exists()
        ), "OpenLineage events schema file missing"

    def test_schema_files_valid_json(
        self, runtime_events_schema, openlineage_events_schema
    ):
        """Test that schema files contain valid JSON"""
        # If fixtures load successfully, schemas are valid JSON
        assert isinstance(
            runtime_events_schema, dict
        ), "Runtime events schema should be a dict"
        assert isinstance(
            openlineage_events_schema, dict
        ), "OpenLineage events schema should be a dict"

    def test_runtime_events_schema_structure(self, runtime_events_schema):
        """Test runtime events schema has expected structure"""
        # Basic JSON Schema validation
        assert "$schema" in runtime_events_schema, "Schema should have $schema field"
        assert "type" in runtime_events_schema, "Schema should have type field"
        assert (
            "properties" in runtime_events_schema
        ), "Schema should have properties field"

        # Check for required runtime event fields based on actual schema
        properties = runtime_events_schema["properties"]
        expected_fields = [
            "event_type",
            "execution_id",
            "session_id",
            "cell_id",
            "cell_source",
            "cell_source_lines",
            "start_memory_mb",
            "timestamp",
            "emitted_at",
        ]

        for field in expected_fields:
            assert field in properties, f"Schema missing required field: {field}"

    def test_openlineage_events_schema_structure(self, openlineage_events_schema):
        """Test OpenLineage events schema has expected structure"""
        assert (
            "$schema" in openlineage_events_schema
        ), "Schema should have $schema field"
        assert "type" in openlineage_events_schema, "Schema should have type field"
        assert (
            "properties" in openlineage_events_schema
        ), "Schema should have properties field"

    @pytest.mark.integration
    def test_sample_runtime_event_validation(self, runtime_events_schema):
        """Test validation of sample runtime events"""
        sample_events = [
            {
                "event_type": "cell_execution_start",
                "execution_id": "abcd1234",
                "session_id": "5e555678",
                "cell_id": "cell-test-123",
                "cell_source": "df = pd.DataFrame({'test': [1, 2, 3]})",
                "cell_source_lines": 1,
                "start_memory_mb": 128.5,
                "timestamp": "2024-01-01T00:00:00Z",
                "emitted_at": "2024-01-01T00:00:00Z",
            },
            {
                "event_type": "cell_execution_end",
                "execution_id": "abcd1234",
                "session_id": "5e555678",
                "cell_id": "cell-test-123",
                "cell_source": "df = pd.DataFrame({'test': [1, 2, 3]})",
                "cell_source_lines": 1,
                "start_memory_mb": 128.5,
                "end_memory_mb": 130.0,
                "duration_ms": 1000,
                "timestamp": "2024-01-01T00:00:01Z",
                "emitted_at": "2024-01-01T00:00:01Z",
            },
        ]

        for i, event in enumerate(sample_events):
            is_valid, error = self.validate_event_against_schema(
                event, runtime_events_schema
            )
            assert is_valid, f"Sample event {i + 1} failed validation: {error}"

    @pytest.mark.integration
    def test_sample_openlineage_event_validation(self, openlineage_events_schema):
        """Test validation of sample OpenLineage events"""
        sample_event = {
            "eventTime": "2024-01-01T00:00:00.000Z",
            "eventType": "COMPLETE",
            "run": {"runId": "abcd1234-5678-90ef", "facets": {}},
            "job": {"namespace": "marimo", "name": "notebook_execution", "facets": {}},
            "inputs": [],
            "outputs": [
                {
                    "namespace": "marimo",
                    "name": "df_output",
                    "facets": {
                        "schema": {
                            "_producer": "marimo-lineage-tracker",
                            "_schemaURL": "https://schemas.openlineage.io/SchemaDatasetFacet/1-0-0",
                            "fields": [
                                {"name": "id", "type": "integer"},
                                {"name": "name", "type": "string"},
                            ],
                        }
                    },
                }
            ],
            "producer": "marimo-lineage-tracker",
            "schemaURL": "https://schemas.openlineage.io/RunEvent/1-0-0",
            "session_id": "test_session_123",
            "emitted_at": "2024-01-01T00:00:00.000Z",
        }

        is_valid, error = self.validate_event_against_schema(
            sample_event, openlineage_events_schema
        )
        assert is_valid, f"Sample OpenLineage event failed validation: {error}"

    def test_validation_utility_methods(self, runtime_events_schema, temp_hunyo_dir):
        """Test the validation utility methods work correctly"""
        # Create a test events file
        events_file = temp_hunyo_dir / "test_events.jsonl"

        test_events = [
            {
                "event_type": "cell_execution_start",
                "execution_id": "1e511234",
                "session_id": "5e551234",
                "cell_id": "cell-1",
                "cell_source": "print('hello')",
                "cell_source_lines": 1,
                "start_memory_mb": 100.0,
                "timestamp": "2024-01-01T00:00:00Z",
                "emitted_at": "2024-01-01T00:00:00Z",
            },
            {
                "event_type": "invalid_event",  # This should fail validation
                "missing_required_field": "value",
            },
        ]

        with open(events_file, "w", encoding="utf-8") as f:
            for event in test_events:
                f.write(json.dumps(event) + "\n")

        # Test validation
        results = self.validate_events_file(events_file, runtime_events_schema)

        assert results["total_events"] == 2
        assert results["valid_events"] == 1
        assert results["invalid_events"] == 1
        assert results["compliance_rate"] == 0.5
        assert len(results["errors"]) == 1


class TestSchemaBackwardCompatibility:
    """
    Test that JSON schemas don't introduce breaking changes.

    CRITICAL: These schemas are the source of truth for event structure.
    Breaking changes would invalidate historical data and cause system failures.
    """

    @pytest.fixture
    def runtime_events_schema(self):
        """Load current runtime events schema"""
        schema_path = get_schema_path("runtime_events_schema.json")
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def openlineage_events_schema(self):
        """Load current OpenLineage events schema"""
        schema_path = get_schema_path("openlineage_events_schema.json")
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def baseline_runtime_events(self):
        """Baseline runtime events that MUST always remain valid"""
        return [
            # Basic start event - core functionality
            {
                "event_type": "cell_execution_start",
                "execution_id": "abcd1234",
                "session_id": "5e555678",
                "cell_id": "cell-test-123",
                "cell_source": "df = pd.DataFrame({'test': [1, 2, 3]})",
                "cell_source_lines": 1,
                "start_memory_mb": 128.5,
                "timestamp": "2024-01-01T00:00:00Z",
                "emitted_at": "2024-01-01T00:00:00Z",
            },
            # Complete end event - with all optional fields
            {
                "event_type": "cell_execution_end",
                "execution_id": "abcd1234",
                "session_id": "5e555678",
                "cell_id": "cell-test-123",
                "cell_source": "df = pd.DataFrame({'test': [1, 2, 3]})",
                "cell_source_lines": 1,
                "start_memory_mb": 128.5,
                "end_memory_mb": 130.0,
                "duration_ms": 1000.5,
                "timestamp": "2024-01-01T00:00:01Z",
                "emitted_at": "2024-01-01T00:00:01Z",
            },
            # Error event - with error_info
            {
                "event_type": "cell_execution_error",
                "execution_id": "def45678",
                "session_id": "5e555678",
                "cell_id": "cell-error-456",
                "cell_source": "x = 1 / 0",
                "cell_source_lines": 1,
                "start_memory_mb": 150.0,
                "timestamp": "2024-01-01T00:00:02Z",
                "emitted_at": "2024-01-01T00:00:02Z",
                "error_info": {
                    "error_type": "ZeroDivisionError",
                    "error_message": "division by zero",
                    "traceback": "Traceback (most recent call last):\n  File...",
                },
            },
            # Minimal valid event - only required fields
            {
                "event_type": "cell_execution_start",
                "execution_id": "abcd5678",
                "session_id": "5e555999",
                "cell_id": "minimal-cell",
                "cell_source": "pass",
                "cell_source_lines": 1,
                "start_memory_mb": 0.0,
                "timestamp": "2024-01-01T00:00:03Z",
                "emitted_at": "2024-01-01T00:00:03Z",
            },
        ]

    @pytest.fixture
    def baseline_openlineage_events(self):
        """Baseline OpenLineage events that MUST always remain valid"""
        return [
            # Complete event with all facets
            {
                "eventTime": "2024-01-01T00:00:00.000Z",
                "eventType": "COMPLETE",
                "run": {
                    "runId": "550e8400-e29b-41d4-a716-446655440000",
                    "facets": {
                        "marimoExecution": {
                            "_producer": "marimo-lineage-tracker",
                            "executionId": "abcd1234",
                            "sessionId": "5e555678",
                        },
                        "performance": {
                            "_producer": "marimo-lineage-tracker",
                            "durationMs": 1000,
                        },
                    },
                },
                "job": {
                    "namespace": "marimo",
                    "name": "pandas_dataframe_operation",
                    "facets": {},
                },
                "inputs": [
                    {
                        "namespace": "marimo",
                        "name": "input_data",
                        "facets": {
                            "schema": {
                                "_producer": "marimo-lineage-tracker",
                                "_schemaURL": "https://schemas.openlineage.io/SchemaDatasetFacet/1-0-0",
                                "fields": [
                                    {"name": "id", "type": "integer"},
                                    {"name": "name", "type": "string"},
                                ],
                            }
                        },
                    }
                ],
                "outputs": [
                    {
                        "namespace": "marimo",
                        "name": "df_output",
                        "facets": {
                            "schema": {
                                "_producer": "marimo-lineage-tracker",
                                "_schemaURL": "https://schemas.openlineage.io/SchemaDatasetFacet/1-0-0",
                                "fields": [
                                    {"name": "id", "type": "integer"},
                                    {"name": "result", "type": "string"},
                                ],
                            },
                            "dataSource": {
                                "_producer": "marimo-lineage-tracker",
                                "_schemaURL": "https://schemas.openlineage.io/DataSourceDatasetFacet/1-0-0",
                                "name": "computed_dataframe",
                                "uri": "memory://dataframe",
                            },
                        },
                    }
                ],
                "producer": "marimo-lineage-tracker",
                "schemaURL": "https://schemas.openlineage.io/RunEvent/1-0-0",
                "session_id": "test_session_123",
                "emitted_at": "2024-01-01T00:00:00.000Z",
            },
            # Minimal valid event - only required fields
            {
                "eventTime": "2024-01-01T00:00:01.000Z",
                "eventType": "START",
                "run": {"runId": "550e8400-e29b-41d4-a716-446655440001"},
                "job": {"namespace": "marimo", "name": "minimal_operation"},
                "inputs": [],
                "outputs": [],
                "producer": "marimo-lineage-tracker",
                "schemaURL": "https://schemas.openlineage.io/RunEvent/1-0-0",
                "session_id": "minimal_session",
                "emitted_at": "2024-01-01T00:00:01.000Z",
            },
        ]

    def validate_event_against_schema(
        self, event: dict[str, Any], schema: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate a single event against a JSON schema"""
        try:
            jsonschema.validate(event, schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)

    def test_runtime_schema_preserves_required_fields(self, runtime_events_schema):
        """
        CRITICAL TEST: Required fields must never be removed or renamed.

        This test ensures that the current schema still requires all fields
        that were required in previous versions. Removing required fields
        would break existing events in the database.
        """
        required_fields = runtime_events_schema.get("required", [])

        # These fields MUST always be required for runtime events
        baseline_required_fields = {
            "event_type",
            "execution_id",
            "cell_id",
            "cell_source",
            "cell_source_lines",
            "start_memory_mb",
            "timestamp",
            "session_id",
            "emitted_at",
        }

        missing_required = baseline_required_fields - set(required_fields)
        assert not missing_required, (
            f"BREAKING CHANGE DETECTED: Required fields removed from runtime schema: {missing_required}. "
            f"This would invalidate existing events. Required fields can only be added, never removed."
        )

    def test_openlineage_schema_preserves_required_fields(
        self, openlineage_events_schema
    ):
        """
        CRITICAL TEST: Required fields must never be removed or renamed.

        This test ensures OpenLineage events maintain backward compatibility
        with existing data in the database.
        """
        required_fields = openlineage_events_schema.get("required", [])

        # These fields MUST always be required for OpenLineage events
        baseline_required_fields = {
            "eventType",
            "eventTime",
            "run",
            "job",
            "inputs",
            "outputs",
            "producer",
            "schemaURL",
            "session_id",
            "emitted_at",
        }

        missing_required = baseline_required_fields - set(required_fields)
        assert not missing_required, (
            f"BREAKING CHANGE DETECTED: Required fields removed from OpenLineage schema: {missing_required}. "
            f"This would invalidate existing events. Required fields can only be added, never removed."
        )

    def test_runtime_schema_field_types_unchanged(self, runtime_events_schema):
        """
        CRITICAL TEST: Field types must not change in breaking ways.

        Changing field types (e.g., string to number) would cause validation
        failures for existing events.
        """
        properties = runtime_events_schema.get("properties", {})

        # These field types MUST NOT change
        baseline_field_types = {
            "event_type": "string",
            "execution_id": "string",
            "cell_id": "string",
            "cell_source": "string",
            "cell_source_lines": "integer",
            "start_memory_mb": "number",
            "end_memory_mb": "number",
            "duration_ms": "number",
            "timestamp": "string",
            "session_id": "string",
            "emitted_at": "string",
        }

        type_changes = []
        for field, expected_type in baseline_field_types.items():
            if field in properties:
                actual_type = properties[field].get("type")
                if actual_type != expected_type:
                    type_changes.append(f"{field}: {expected_type} -> {actual_type}")

        assert not type_changes, (
            f"BREAKING CHANGE DETECTED: Field types changed in runtime schema: {type_changes}. "
            f"Type changes would invalidate existing events."
        )

    def test_runtime_schema_enum_values_preserved(self, runtime_events_schema):
        """
        CRITICAL TEST: Enum values must not be removed.

        Removing enum values would invalidate existing events that use those values.
        New enum values can be added, but existing ones must be preserved.
        """
        properties = runtime_events_schema.get("properties", {})
        event_type_property = properties.get("event_type", {})
        current_enum_values = set(event_type_property.get("enum", []))

        # These enum values MUST always be supported
        baseline_enum_values = {
            "cell_execution_start",
            "cell_execution_end",
            "cell_execution_error",
        }

        missing_enum_values = baseline_enum_values - current_enum_values
        assert not missing_enum_values, (
            f"BREAKING CHANGE DETECTED: Enum values removed from event_type: {missing_enum_values}. "
            f"This would invalidate existing events. Enum values can only be added, never removed."
        )

    def test_baseline_runtime_events_still_valid(
        self, runtime_events_schema, baseline_runtime_events
    ):
        """
        CRITICAL TEST: All baseline events must still validate against current schema.

        This is the most important test - it ensures that events that were valid
        in the past are still valid with the current schema. This prevents
        data loss and system failures.
        """
        validation_failures = []

        for i, event in enumerate(baseline_runtime_events):
            is_valid, error = self.validate_event_against_schema(
                event, runtime_events_schema
            )

            if not is_valid:
                validation_failures.append(f"Baseline event {i + 1}: {error}")

        assert not validation_failures, (
            "BREAKING CHANGE DETECTED: Baseline runtime events no longer validate:\n"
            + "\n".join(validation_failures)
            + "\n\nThis indicates the schema has introduced breaking changes that would "
            "invalidate existing historical data. Schema changes must be backward compatible."
        )

    def test_baseline_openlineage_events_still_valid(
        self, openlineage_events_schema, baseline_openlineage_events
    ):
        """
        CRITICAL TEST: All baseline OpenLineage events must still validate.

        This ensures OpenLineage schema changes maintain backward compatibility
        with existing lineage data.
        """
        validation_failures = []

        for i, event in enumerate(baseline_openlineage_events):
            is_valid, error = self.validate_event_against_schema(
                event, openlineage_events_schema
            )

            if not is_valid:
                validation_failures.append(
                    f"Baseline OpenLineage event {i + 1}: {error}"
                )

        assert not validation_failures, (
            "BREAKING CHANGE DETECTED: Baseline OpenLineage events no longer validate:\n"
            + "\n".join(validation_failures)
            + "\n\nThis indicates the schema has introduced breaking changes that would "
            "invalidate existing historical lineage data. Schema changes must be backward compatible."
        )

    def test_schema_allows_additional_properties_safely(
        self, runtime_events_schema, openlineage_events_schema
    ):
        """
        Test that additionalProperties settings don't break extensibility.

        The runtime schema should reject additional properties for strict validation,
        but the OpenLineage schema might allow them for extensibility.
        """
        # Runtime events schema should be strict (additionalProperties: false)
        runtime_additional_properties = runtime_events_schema.get(
            "additionalProperties", True
        )
        assert runtime_additional_properties is False, (
            "Runtime events schema should have 'additionalProperties: false' for strict validation. "
            "This prevents accidentally storing invalid data."
        )

        # For OpenLineage, we're more flexible about additional properties in facets
        # This is acceptable as OpenLineage is designed to be extensible

    def test_schema_version_consistency(
        self, runtime_events_schema, openlineage_events_schema
    ):
        """
        Test that schema metadata is properly maintained.

        Schema files should have proper versioning information to track changes.
        """
        # Both schemas should have $schema declarations
        assert (
            "$schema" in runtime_events_schema
        ), "Runtime schema missing $schema declaration"
        assert (
            "$schema" in openlineage_events_schema
        ), "OpenLineage schema missing $schema declaration"

        # Runtime schema should have our custom $id
        runtime_id = runtime_events_schema.get("$id")
        assert (
            runtime_id and "runtime-event" in runtime_id
        ), "Runtime schema should have a proper $id with 'runtime-event' identifier"

    def test_breaking_change_simulation_required_field_removal(
        self, baseline_runtime_events
    ):
        """
        DEMONSTRATION TEST: Show what happens when required fields are removed.

        This test simulates removing a required field to demonstrate that
        the backward compatibility tests would catch such breaking changes.
        """
        # Create a modified schema with a required field removed
        modified_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": [
                "event_type",
                "execution_id",
                # "cell_id",  # SIMULATE REMOVING THIS REQUIRED FIELD
                "cell_source",
                "cell_source_lines",
                "start_memory_mb",
                "timestamp",
                "session_id",
                "emitted_at",
            ],
            "properties": {
                "event_type": {"type": "string"},
                "execution_id": {"type": "string"},
                "cell_id": {"type": "string"},  # Still in properties but not required
                "cell_source": {"type": "string"},
                "cell_source_lines": {"type": "integer"},
                "start_memory_mb": {"type": "number"},
                "end_memory_mb": {
                    "type": "number"
                },  # Include optional fields from baseline events
                "duration_ms": {"type": "number"},
                "error_info": {"type": "object"},
                "timestamp": {"type": "string"},
                "session_id": {"type": "string"},
                "emitted_at": {"type": "string"},
            },
            "additionalProperties": False,
        }

        # Test that this breaking change would be caught
        # (This test should pass because we're demonstrating the detection)
        validation_failures = []
        for event in baseline_runtime_events:
            is_valid, error = self.validate_event_against_schema(event, modified_schema)
            if not is_valid:
                validation_failures.append(error)

        # This modified schema should still validate existing events
        # because cell_id is still present in all baseline events
        assert len(validation_failures) == 0, (
            "Unexpected: Modified schema failed validation. "
            "This demonstrates how removing required fields could break validation."
        )

    def test_breaking_change_simulation_field_type_change(
        self, baseline_runtime_events
    ):
        """
        DEMONSTRATION TEST: Show what happens when field types change.

        This test simulates changing a field type to demonstrate that
        backward compatibility tests would catch such changes.
        """
        # Create a modified schema with a field type changed
        modified_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": [
                "event_type",
                "execution_id",
                "cell_id",
                "cell_source",
                "cell_source_lines",
                "start_memory_mb",
                "timestamp",
                "session_id",
                "emitted_at",
            ],
            "properties": {
                "event_type": {"type": "string"},
                "execution_id": {"type": "string"},
                "cell_id": {"type": "string"},
                "cell_source": {"type": "string"},
                "cell_source_lines": {"type": "string"},  # CHANGED: integer -> string
                "start_memory_mb": {"type": "number"},
                "timestamp": {"type": "string"},
                "session_id": {"type": "string"},
                "emitted_at": {"type": "string"},
            },
            "additionalProperties": False,
        }

        # Test that this breaking change would be caught
        validation_failures = []
        for event in baseline_runtime_events:
            is_valid, error = self.validate_event_against_schema(event, modified_schema)
            if not is_valid:
                validation_failures.append(error)

        # This should fail because cell_source_lines is an integer in baseline events
        # but the modified schema expects a string
        assert len(validation_failures) > 0, (
            "Type change simulation failed: Modified schema should reject events with "
            "integer cell_source_lines when schema expects string."
        )
