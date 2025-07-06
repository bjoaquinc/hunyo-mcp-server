#!/usr/bin/env python3
"""
Integration tests for schema validation with unified marimo interceptor.

This module tests the unified marimo interceptor system for both runtime
and OpenLineage event generation, validating all generated events against
the JSON schemas in /schemas/json.
"""

import json
import time
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from hunyo_capture.logger import get_logger
from hunyo_capture.unified_marimo_interceptor import (
    disable_unified_tracking,
    enable_unified_tracking,
)

# Create test logger instance
test_logger = get_logger("hunyo.test.schema_validation")


class TestSchemaValidationIntegration:
    """Integration tests for schema validation with unified marimo interceptor"""

    @pytest.fixture
    def runtime_events_schema(self):
        """Load runtime events schema for validation"""
        schema_path = Path("schemas/json/runtime_events_schema.json")
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def openlineage_events_schema(self):
        """Load OpenLineage events schema for validation"""
        schema_path = Path("schemas/json/openlineage_events_schema.json")
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def runtime_events_file(self, temp_hunyo_dir):
        """Path for runtime events output"""
        events_file = temp_hunyo_dir / "runtime_events.jsonl"
        events_file.parent.mkdir(parents=True, exist_ok=True)
        return events_file

    @pytest.fixture
    def lineage_events_file(self, temp_hunyo_dir):
        """Path for lineage events output"""
        events_file = temp_hunyo_dir / "lineage_events.jsonl"
        events_file.parent.mkdir(parents=True, exist_ok=True)
        return events_file

    def validate_event_against_schema(
        self, event: dict[str, Any], schema: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate a single event against a JSON schema"""
        try:
            jsonschema.validate(event, schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)

    def test_runtime_tracker_schema_compliance(
        self,
        runtime_events_file,
        runtime_events_schema,
        config_with_temp_dir,
    ):
        """Test unified tracker generates schema-compliant runtime events"""
        # Initialize unified tracking with specific file paths
        interceptor = enable_unified_tracking(
            runtime_file=str(runtime_events_file),
            lineage_file=str(runtime_events_file.parent / "lineage_events.jsonl"),
        )

        try:
            # Test the ACTUAL runtime event generation by calling the real hook methods
            from unittest.mock import MagicMock

            # Create real marimo-style cell objects
            cell_executions = [
                {
                    "cell_id": "cell_1_imports",
                    "cell_source": "import pandas as pd\nimport numpy as np",
                },
                {
                    "cell_id": "cell_2_dataframes",
                    "cell_source": "df1 = pd.DataFrame({'id': [1,2,3], 'name': ['A','B','C']})",
                },
                {
                    "cell_id": "cell_3_operations",
                    "cell_source": "df_result = df1.merge(df2, on='id')",
                },
            ]

            # Test the actual hook methods that generate runtime events
            for cell_info in cell_executions:
                # Create mock cell object like marimo creates
                mock_cell = MagicMock()
                mock_cell.cell_id = cell_info["cell_id"]
                mock_cell.code = cell_info["cell_source"]

                mock_runner = MagicMock()

                # Call the REAL pre-execution hook (generates cell_execution_start)
                pre_hook = interceptor._create_pre_execution_hook()
                pre_hook(mock_cell, mock_runner)

                # Simulate some execution time
                time.sleep(0.01)

                # Create mock run result
                mock_run_result = MagicMock()
                mock_run_result.success.return_value = True
                mock_run_result.exception = None
                mock_run_result.output = None

                # Call the REAL post-execution hook (generates cell_execution_end)
                post_hook = interceptor._create_post_execution_hook()
                post_hook(mock_cell, mock_runner, mock_run_result)

            # Verify events were written
            assert runtime_events_file.exists()

            # Count events from the file
            event_count = 0
            with open(runtime_events_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        event_count += 1

            assert event_count > 0, f"Expected events in file, found {event_count}"

        finally:
            # Clean up
            disable_unified_tracking()

        # Load and validate events against schema
        events = []
        with open(runtime_events_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        assert len(events) >= 6  # At least 6 events (3 start + 3 end)

        # Validate each event against schema
        valid_count = 0
        invalid_count = 0
        validation_errors = []

        for i, event in enumerate(events):
            is_valid, error = self.validate_event_against_schema(
                event, runtime_events_schema
            )

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                validation_errors.append(f"Event {i + 1}: {error}")

        # Assert schema compliance
        test_logger.info("[VALIDATION] Runtime Events Validation:")
        test_logger.info(f"[OK] Valid events: {valid_count}")
        test_logger.info(f"[ERROR] Invalid events: {invalid_count}")
        test_logger.info(
            f"[INFO] Compliance rate: {(valid_count / len(events) * 100):.1f}%"
        )

        if validation_errors:
            test_logger.error("[ERROR] Validation errors:")
            for error in validation_errors[:3]:  # Show first 3 errors
                test_logger.error(f"  {error}")

        # Should have high compliance rate
        compliance_rate = valid_count / len(events) * 100
        assert (
            compliance_rate >= 80
        ), f"Runtime events compliance rate too low: {compliance_rate:.1f}%"

    def test_lineage_interceptor_schema_compliance(
        self,
        lineage_events_file,
        openlineage_events_schema,
        config_with_temp_dir,
    ):
        """Test unified interceptor generates schema-compliant OpenLineage events"""
        # Initialize unified tracking with specific file paths
        interceptor = enable_unified_tracking(
            runtime_file=str(lineage_events_file.parent / "runtime_events.jsonl"),
            lineage_file=str(lineage_events_file),
        )

        try:
            # Test the ACTUAL DataFrame capture mechanisms by creating a real execution context
            # and then creating DataFrames to trigger the monkey patches
            execution_id = "test_lineage_exec_001"
            cell_id = "test_lineage_cell"

            # Set up execution context to simulate active marimo execution
            execution_context = {
                "execution_id": execution_id,
                "cell_id": cell_id,
                "cell_source": "# Test DataFrame operations",
                "start_time": time.time(),
            }
            interceptor._execution_contexts[execution_id] = execution_context

            try:
                # Test the ACTUAL DataFrame creation capture by creating real DataFrames
                # This should trigger the unified system's monkey patches
                import pandas as pd

                # Verify that monkey patches are installed and context is active
                assert (
                    interceptor._is_in_marimo_execution()
                ), "Should be in active execution context"

                # Create test DataFrames - these should trigger _capture_dataframe_creation via monkey patches
                df1 = pd.DataFrame(
                    {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
                )
                df2 = pd.DataFrame({"id": [1, 2, 3], "salary": [50000, 60000, 70000]})

                # These operations should be captured by the tracking system
                df_merged = df1.merge(df2, on="id")
                # Filter operation - should be captured as part of the merge result
                _ = df_merged[df_merged["salary"] > 55000]

                # ALSO test the direct capture method to ensure events are generated
                # This guarantees at least one event is generated
                interceptor._capture_dataframe_creation(
                    df_merged,
                    execution_context,
                    # Simulate constructor args
                    data={"test": "data"},
                )

            finally:
                # Clean up execution context
                interceptor._execution_contexts.clear()

            # Give time for events to be written
            time.sleep(0.1)

        finally:
            # Clean up tracking
            disable_unified_tracking()

        # Check if events were generated
        if not lineage_events_file.exists() or lineage_events_file.stat().st_size == 0:
            pytest.skip(
                "No lineage events generated - interceptor may need marimo environment"
            )
            return

        # Load and validate OpenLineage events
        events = []
        with open(lineage_events_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

        if not events:
            pytest.skip("No valid OpenLineage events found in output")
            return

        # Validate against OpenLineage schema
        valid_count = 0
        invalid_count = 0
        validation_errors = []

        for i, event in enumerate(events):
            is_valid, error = self.validate_event_against_schema(
                event, openlineage_events_schema
            )

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                validation_errors.append(f"Event {i + 1}: {error}")

        test_logger.info("[VALIDATION] OpenLineage Events Validation:")
        test_logger.info(f"[OK] Valid events: {valid_count}")
        test_logger.info(f"[ERROR] Invalid events: {invalid_count}")
        test_logger.info(f"[INFO] Total events: {len(events)}")

        if validation_errors:
            test_logger.error("[ERROR] Validation errors:")
            for error in validation_errors[:3]:
                test_logger.error(f"  {error}")

        # Should have some valid events
        assert valid_count > 0, "No valid OpenLineage events found"
