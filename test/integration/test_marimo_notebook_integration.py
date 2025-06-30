from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# This test creates a marimo notebook that can be tested with `marimo edit`
# It follows marimo's testing patterns but creates an actual notebook file


class TestMarimoNotebookIntegration:
    """Integration tests for marimo notebook capture functionality"""

    @pytest.fixture
    def marimo_notebook_content(self):
        """Marimo notebook content for testing capture functionality"""
        return '''import marimo

__generated_with = "0.8.20"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# DataFrame Tracking Test Notebook

    This notebook tests the capture system integration with marimo.
    """)
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    return np, pd


@app.cell
def __(pd):
    # Create initial DataFrames
    df1 = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales']
    })

    df2 = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'salary': [75000, 65000, 85000, 70000, 80000],
        'bonus': [5000, 3000, 7000, 4000, 6000]
    })

    print(f"Created df1 with shape: {df1.shape}")
    print(f"Created df2 with shape: {df2.shape}")
    return df1, df2


@app.cell
def __(df1, df2):
    # Merge operations
    df_merged = df1.merge(df2, on='id', how='inner')
    print(f"Merged DataFrame shape: {df_merged.shape}")
    print("Merged DataFrame columns:", list(df_merged.columns))
    return (df_merged,)


@app.cell
def __(df_merged):
    # Filtering and transformation
    engineering_df = df_merged[df_merged['department'] == 'Engineering']
    high_earners = df_merged[df_merged['salary'] > 70000]

    print(f"Engineering team size: {len(engineering_df)}")
    print(f"High earners count: {len(high_earners)}")
    return engineering_df, high_earners


@app.cell
def __(df_merged):
    # Aggregation operations
    dept_summary = df_merged.groupby('department').agg({
        'salary': ['mean', 'max', 'min'],
        'bonus': 'sum',
        'age': 'mean'
    }).round(2)

    print("Department Summary:")
    print(dept_summary)
    return (dept_summary,)


@app.cell
def __(df_merged, pd):
    # More complex transformations
    df_transformed = df_merged.copy()
    df_transformed['total_compensation'] = df_transformed['salary'] + df_transformed['bonus']
    df_transformed['age_group'] = pd.cut(df_transformed['age'], 
                                       bins=[0, 30, 40, 100], 
                                       labels=['Young', 'Middle', 'Senior'])

    print("Transformation complete:")
    print(df_transformed[['name', 'total_compensation', 'age_group']].head())
    return (df_transformed,)


@app.cell
def __(df_transformed):
    # Final analysis
    age_group_analysis = df_transformed.groupby('age_group').agg({
        'total_compensation': ['mean', 'count'],
        'salary': 'mean'
    }).round(2)

    print("Age Group Analysis:")
    print(age_group_analysis)

    # Test capture system integration point
    print("\\n=== CAPTURE SYSTEM TEST RESULTS ===")
    print("If capture system is working, this notebook execution should generate:")
    print("1. DataFrame creation events for df1, df2")
    print("2. Merge operation tracking for df_merged")
    print("3. Filter operation tracking for engineering_df, high_earners")
    print("4. Aggregation tracking for dept_summary, age_group_analysis")
    print("5. Transformation tracking for df_transformed")

    return (age_group_analysis,)


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
'''

    def test_marimo_notebook_creation(self, marimo_notebook_content, temp_hunyo_dir):
        """Test creation of marimo notebook for capture testing"""
        # Create notebook file
        notebook_path = temp_hunyo_dir / "test_capture_notebook.py"

        with open(notebook_path, "w") as f:
            f.write(marimo_notebook_content)

        assert notebook_path.exists()
        assert notebook_path.stat().st_size > 0

        # Verify notebook content
        with open(notebook_path) as f:
            content = f.read()

        assert "import pandas as pd" in content
        assert "df1.merge(df2" in content
        assert "groupby" in content
        assert "CAPTURE SYSTEM TEST RESULTS" in content

    def test_notebook_dataframe_operations_parsing(self, marimo_notebook_content):
        """Test parsing of DataFrame operations from notebook content"""
        # Expected operations that should be tracked
        expected_operations = [
            "create",  # df1, df2 creation
            "merge",  # df1.merge(df2)
            "filter",  # department filtering, salary filtering
            "group",  # groupby operations
            "transform",  # copy, assignment operations
        ]

        # Parse operations from notebook content
        detected_operations = []

        if "pd.DataFrame" in marimo_notebook_content:
            detected_operations.append("create")
        if ".merge(" in marimo_notebook_content:
            detected_operations.append("merge")
        if "[df_" in marimo_notebook_content or "[df." in marimo_notebook_content:
            detected_operations.append("filter")
        if ".groupby(" in marimo_notebook_content:
            detected_operations.append("group")
        if ".copy()" in marimo_notebook_content or "= df" in marimo_notebook_content:
            detected_operations.append("transform")

        # Verify expected operations are present
        for operation in expected_operations:
            assert (
                operation in detected_operations
            ), f"Operation {operation} not found in notebook"

    def test_notebook_capture_integration_points(self, marimo_notebook_content):
        """Test that notebook has proper integration points for capture system"""
        # Should have multiple DataFrame variables
        df_variables = []
        lines = marimo_notebook_content.split("\n")

        for line in lines:
            if "= pd.DataFrame(" in line or ("= df" in line and ".merge" in line):
                # Extract variable name
                if "=" in line:
                    var_name = line.split("=")[0].strip()
                    if var_name.startswith("df"):
                        df_variables.append(var_name)

        assert (
            len(df_variables) >= 3
        ), "Should have multiple DataFrame variables for tracking"

    def test_notebook_validation_output(self, marimo_notebook_content, temp_hunyo_dir):
        """Test that notebook would produce expected validation output"""
        notebook_path = temp_hunyo_dir / "validation_notebook.py"

        with open(notebook_path, "w") as f:
            f.write(marimo_notebook_content)

        # Mock marimo execution environment
        with patch("pandas.DataFrame") as mock_df_class:
            mock_df = MagicMock()
            mock_df.shape = (5, 4)
            mock_df.columns = ["id", "name", "age", "department"]
            mock_df.merge.return_value = mock_df
            mock_df.__getitem__.return_value = mock_df
            mock_df.groupby.return_value.agg.return_value = mock_df

            mock_df_class.return_value = mock_df

            # Simulate execution context
            exec_globals = {
                "pd": type("MockPandas", (), {"DataFrame": mock_df_class})(),
                "print": lambda *args: None,  # Suppress output
            }

            # Should be able to execute key sections without errors
            try:
                # Extract and execute DataFrame creation section
                creation_code = """
df1 = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45],
    'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales']
})
df2 = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'salary': [75000, 65000, 85000, 70000, 80000],
    'bonus': [5000, 3000, 7000, 4000, 6000]
})
"""
                exec(creation_code, exec_globals)

                # Should have created DataFrame variables
                assert "df1" in exec_globals
                assert "df2" in exec_globals

            except Exception as e:
                pytest.fail(f"Notebook execution simulation failed: {e}")

    def test_notebook_marimo_compatibility(self, marimo_notebook_content):
        """Test that notebook follows marimo app structure"""
        # Should have marimo app structure
        assert "import marimo" in marimo_notebook_content
        assert "app = marimo.App(" in marimo_notebook_content
        assert "@app.cell" in marimo_notebook_content
        assert 'if __name__ == "__main__":' in marimo_notebook_content
        assert "app.run()" in marimo_notebook_content

        # Should have proper cell structure
        cell_count = marimo_notebook_content.count("@app.cell")
        assert cell_count >= 5, "Should have multiple cells for comprehensive testing"

    @pytest.mark.integration
    def test_full_notebook_workflow_simulation(
        self, marimo_notebook_content, temp_hunyo_dir, config_with_temp_dir
    ):
        """Integration test simulating full notebook workflow with capture"""
        from test.mocks import (
            MockLightweightRuntimeTracker as LightweightRuntimeTracker,
        )

        # Create notebook
        notebook_path = temp_hunyo_dir / "full_workflow_test.py"
        with open(notebook_path, "w") as f:
            f.write(marimo_notebook_content)

        # Initialize capture system
        tracker = LightweightRuntimeTracker(config_with_temp_dir)

        # Simulate DataFrame operations from notebook
        test_operations = [
            {
                "df_name": "df1",
                "operation": "create",
                "code": "df1 = pd.DataFrame({'id': [1,2,3,4,5], ...})",
                "shape": (5, 4),
                "columns": ["id", "name", "age", "department"],
            },
            {
                "df_name": "df2",
                "operation": "create",
                "code": "df2 = pd.DataFrame({'id': [1,2,3,4,5], ...})",
                "shape": (5, 3),
                "columns": ["id", "salary", "bonus"],
            },
            {
                "df_name": "df_merged",
                "operation": "merge",
                "code": "df_merged = df1.merge(df2, on='id')",
                "shape": (5, 6),
                "columns": ["id", "name", "age", "department", "salary", "bonus"],
            },
        ]

        # Track operations
        for operation in test_operations:
            tracker.track_dataframe_operation(**operation)

        # Verify tracking occurred
        events_file = config_with_temp_dir.events_dir / "runtime_events.jsonl"
        assert events_file.exists()

        # Should have recorded all operations
        with open(events_file) as f:
            import json

            events = [
                json.loads(line.strip()) for line in f.readlines() if line.strip()
            ]

        assert len(events) >= len(test_operations)

        # Verify operation types were captured
        captured_operations = [event.get("operation") for event in events]
        assert "create" in captured_operations
        assert "merge" in captured_operations
