# ruff: noqa
# Prevent pytest from collecting this fixture file as a test module
__test__ = False

import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def setup_tracking():
    """Set up unified tracking system - this must run first"""
    import marimo as mo
    import os
    import sys
    from pathlib import Path

    # Find project root by looking for pyproject.toml (more robust than __file__)
    current_path = Path.cwd()
    project_root = current_path

    # Walk up directories to find pyproject.toml (project root indicator)
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent

    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Import and enable unified tracking
    from hunyo_capture.unified_marimo_interceptor import enable_unified_tracking

    # Enable unified tracking (creates and installs interceptor)
    tracker = enable_unified_tracking(
        runtime_file="test_marimo_runtime_events.jsonl",
        lineage_file="test_marimo_lineage_events.jsonl",
    )

    print(f"[SETUP] Unified tracker installed: {tracker.session_id}")
    print(f"[SETUP] Tracker active: {tracker.interceptor_active}")

    return mo, tracker


@app.cell
def create_dataframe(tracker):
    """Create DataFrame operations - should trigger lineage events via hooks"""
    import pandas as pd

    print(f"[CREATE] Tracker active: {tracker.interceptor_active}")
    print("[CREATE] Creating DataFrame - should trigger lineage events...")

    # This DataFrame creation should trigger our pandas monkey patch
    df1 = pd.DataFrame(
        {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
        }
    )

    print(f"[CREATE] Created df1 with shape: {df1.shape}")

    # Another DataFrame operation
    df2 = pd.DataFrame(
        {'id': [1, 2, 3], 'department': ['Engineering', 'Sales', 'Marketing']}
    )

    print(f"[CREATE] Created df2 with shape: {df2.shape}")

    return df1, df2, pd


@app.cell
def process_dataframe(df1, df2, pd, tracker):
    """Process DataFrames - should trigger more lineage events"""

    print(f"[PROCESS] Tracker active: {tracker.interceptor_active}")
    print("[PROCESS] Processing DataFrames - should trigger more lineage events...")

    # Merge operation - should create lineage event with 2 inputs, 1 output
    merged_df = pd.merge(df1, df2, on='id', how='left')
    print(f"[PROCESS] Merged DataFrames, result shape: {merged_df.shape}")

    # Filter operation - should create lineage event
    filtered_df = merged_df[merged_df['age'] > 30]
    print(f"[PROCESS] Filtered DataFrame, result shape: {filtered_df.shape}")

    # Group by operation - should create lineage event
    grouped_stats = filtered_df.groupby('department')['age'].mean()
    print(f"[PROCESS] Grouped stats: {grouped_stats.to_dict()}")

    return merged_df, filtered_df, grouped_stats


if __name__ == "__main__":
    app.run()
