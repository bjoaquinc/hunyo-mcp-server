#!/usr/bin/env python3
# ruff: noqa
"""
Test notebook for the updated Marimo Lineage MCP capture layer
This file tests that the new file naming strategy works correctly
"""

import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __():
    import os
    import sys
    from pathlib import Path

    import pandas as pd

    # Add the src directory to the path so we can import our capture modules
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))

    # Set the notebook path for testing
    os.environ["MARIMO_NOTEBOOK_PATH"] = __file__

    print(f"🧪 Testing notebook: {__file__}")
    return os, sys, pd, Path


@app.cell
def __():
    # Test the basic utilities
    from capture import (
        get_event_filenames,
        get_notebook_file_hash,
        get_notebook_name,
        get_user_data_dir,
    )

    notebook_path = __file__
    print(f"📝 Notebook path: {notebook_path}")
    print(f"🔗 Hash: {get_notebook_file_hash(notebook_path)}")
    print(f"📛 Name: {get_notebook_name(notebook_path)}")
    print(f"📁 Data dir: {get_user_data_dir()}")

    runtime_file, lineage_file = get_event_filenames(notebook_path, get_user_data_dir())
    print(f"📊 Runtime events: {runtime_file}")
    print(f"🔄 Lineage events: {lineage_file}")
    return (
        get_notebook_file_hash,
        get_notebook_name,
        get_event_filenames,
        get_user_data_dir,
        notebook_path,
        runtime_file,
        lineage_file,
    )


@app.cell
def __():
    # Test the runtime tracker with the new naming
    from capture.lightweight_runtime_tracker import enable_runtime_tracking

    tracker = enable_runtime_tracking(notebook_path=__file__)
    print("✅ Runtime tracker enabled with unique file naming")
    return (tracker,)


@app.cell
def __():
    # Test some basic operations that should be tracked
    df1 = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    print(f"📊 Created df1: {df1.shape}")

    df2 = pd.DataFrame({"x": [1, 2], "z": [7, 8]})
    print(f"📊 Created df2: {df2.shape}")

    df_merged = df1.merge(df2, on="x")
    print(f"🔗 Merged result: {df_merged.shape}")
    return df1, df2, df_merged


@app.cell
def __():
    # Test the live lineage interceptor with the new naming
    from capture.live_lineage_interceptor import enable_live_tracking

    interceptor = enable_live_tracking(notebook_path=__file__)
    print("✅ Live lineage tracking enabled with unique file naming")
    return (interceptor,)


@app.cell
def __():
    # Check the session summary
    summary = tracker.get_session_summary()
    print("📋 Runtime Tracker Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    tracking_summary = interceptor.get_session_summary()
    print("\n📋 Lineage Tracker Summary:")
    for key, value in tracking_summary.items():
        print(f"  {key}: {value}")


@app.cell
def __():
    # Verify files were created with the correct naming pattern
    data_dir = get_user_data_dir()
    runtime_dir = Path(data_dir) / "events" / "runtime"
    lineage_dir = Path(data_dir) / "events" / "lineage"

    print("🔍 Checking for event files...")
    print(f"📁 Runtime dir: {runtime_dir}")
    print(f"📁 Lineage dir: {lineage_dir}")

    if runtime_dir.exists():
        runtime_files = list(runtime_dir.glob("*test_capture_integration*"))
        print(f"📊 Runtime files: {[f.name for f in runtime_files]}")

    if lineage_dir.exists():
        lineage_files = list(lineage_dir.glob("*test_capture_integration*"))
        print(f"🔄 Lineage files: {[f.name for f in lineage_files]}")

    print("✅ File naming verification complete")
    return data_dir, runtime_dir, lineage_dir


if __name__ == "__main__":
    app.run()
