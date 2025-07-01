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


class TestSchemaValidation:
    """Test schema validation utilities and compliance"""

    @pytest.fixture
    def runtime_events_schema(self):
        """Load runtime events schema"""
        schema_path = Path("schemas/json/runtime_events_schema.json")
        with open(schema_path) as f:
            return json.load(f)

    @pytest.fixture
    def openlineage_events_schema(self):
        """Load OpenLineage events schema"""
        schema_path = Path("schemas/json/openlineage_events_schema.json")
        with open(schema_path) as f:
            return json.load(f)

    def validate_event_against_schema(self, event: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate a single event against a JSON schema"""
        try:
            jsonschema.validate(event, schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)

    def validate_events_file(self, events_file: Path, schema: dict[str, Any]) -> dict[str, Any]:
        """Validate all events in a JSONL file against schema"""
        results = {
            "total_events": 0,
            "valid_events": 0,
            "invalid_events": 0,
            "compliance_rate": 0.0,
            "errors": []
        }

        if not events_file.exists():
            return results

        with open(events_file) as f:
            events = [json.loads(line.strip()) for line in f if line.strip()]

        results["total_events"] = len(events)

        for i, event in enumerate(events):
            is_valid, error = self.validate_event_against_schema(event, schema)

            if is_valid:
                results["valid_events"] += 1
            else:
                results["invalid_events"] += 1
                results["errors"].append(f"Event {i+1}: {error}")

        if results["total_events"] > 0:
            results["compliance_rate"] = results["valid_events"] / results["total_events"]

        return results

    def test_schema_files_exist(self):
        """Test that required schema files exist"""
        runtime_schema_path = Path("schemas/json/runtime_events_schema.json")
        openlineage_schema_path = Path("schemas/json/openlineage_events_schema.json")

        assert runtime_schema_path.exists(), "Runtime events schema file missing"
        assert openlineage_schema_path.exists(), "OpenLineage events schema file missing"

    def test_schema_files_valid_json(self, runtime_events_schema, openlineage_events_schema):
        """Test that schema files contain valid JSON"""
        # If fixtures load successfully, schemas are valid JSON
        assert isinstance(runtime_events_schema, dict), "Runtime events schema should be a dict"
        assert isinstance(openlineage_events_schema, dict), "OpenLineage events schema should be a dict"

    def test_runtime_events_schema_structure(self, runtime_events_schema):
        """Test runtime events schema has expected structure"""
        # Basic JSON Schema validation
        assert "$schema" in runtime_events_schema, "Schema should have $schema field"
        assert "type" in runtime_events_schema, "Schema should have type field"
        assert "properties" in runtime_events_schema, "Schema should have properties field"

        # Check for required runtime event fields based on actual schema
        properties = runtime_events_schema["properties"]
        expected_fields = ["event_type", "execution_id", "session_id", "cell_id", "cell_source", "cell_source_lines", "start_memory_mb", "timestamp", "emitted_at"]

        for field in expected_fields:
            assert field in properties, f"Schema missing required field: {field}"

    def test_openlineage_events_schema_structure(self, openlineage_events_schema):
        """Test OpenLineage events schema has expected structure"""
        assert "$schema" in openlineage_events_schema, "Schema should have $schema field"
        assert "type" in openlineage_events_schema, "Schema should have type field"
        assert "properties" in openlineage_events_schema, "Schema should have properties field"

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
                "emitted_at": "2024-01-01T00:00:00Z"
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
                "emitted_at": "2024-01-01T00:00:01Z"
            }
        ]

        for i, event in enumerate(sample_events):
            is_valid, error = self.validate_event_against_schema(event, runtime_events_schema)
            assert is_valid, f"Sample event {i+1} failed validation: {error}"

    @pytest.mark.integration
    def test_sample_openlineage_event_validation(self, openlineage_events_schema):
        """Test validation of sample OpenLineage events"""
        sample_event = {
            "eventTime": "2024-01-01T00:00:00.000Z",
            "eventType": "COMPLETE",
            "run": {
                "runId": "abcd1234-5678-90ef",
                "facets": {}
            },
            "job": {
                "namespace": "marimo",
                "name": "notebook_execution",
                "facets": {}
            },
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
                                {"name": "name", "type": "string"}
                            ]
                        }
                    }
                }
            ],
            "producer": "marimo-lineage-tracker",
            "schemaURL": "https://schemas.openlineage.io/RunEvent/1-0-0",
            "session_id": "test_session_123",
            "emitted_at": "2024-01-01T00:00:00.000Z"
        }

        is_valid, error = self.validate_event_against_schema(sample_event, openlineage_events_schema)
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
                "emitted_at": "2024-01-01T00:00:00Z"
            },
            {
                "event_type": "invalid_event",  # This should fail validation
                "missing_required_field": "value"
            }
        ]

        with open(events_file, "w") as f:
            for event in test_events:
                f.write(json.dumps(event) + "\n")

        # Test validation
        results = self.validate_events_file(events_file, runtime_events_schema)

        assert results["total_events"] == 2
        assert results["valid_events"] == 1
        assert results["invalid_events"] == 1
        assert results["compliance_rate"] == 0.5
        assert len(results["errors"]) == 1
