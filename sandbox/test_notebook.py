# ruff: noqa
# This is a marimo notebook - ignore linting rules

import marimo

__generated_with = "0.14.9"
app = marimo.App()


@app.cell
def _():
    # Setup path for imports
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).parent.parent / "packages" / "hunyo-capture" / "src")
    )

    # UNIFIED APPROACH: Use unified marimo interceptor
    try:
        from hunyo_capture.unified_marimo_interceptor import enable_unified_tracking

        # Enable unified tracking with proper naming convention
        notebook_path = str(Path(__file__).resolve())

        # Enable unified tracking for both runtime and lineage events
        interceptor = enable_unified_tracking(notebook_path=notebook_path)

    except ImportError:
        print(
            "[ERROR] hunyo-capture not installed. Install with: pip install hunyo-capture"
        )
        interceptor = None

    if interceptor:
        print(f"[TARGET] Unified interceptor initialized: {interceptor.session_id}")
        print(f"[FILE] Lineage file: {interceptor.lineage_file}")
        print(f"[FILE] Runtime file: {interceptor.runtime_file}")
        print(f"[CONFIG] Hooks installed: {len(interceptor.installed_hooks)}")
        print(f"[OK] Interceptor active: {interceptor.interceptor_active}")

        # Verify directory structure
        if ".hunyo/events/lineage" in str(interceptor.lineage_file):
            print("[OK] Lineage events will be saved to .hunyo/events/lineage/")
        else:
            print(
                f"[ERROR] Lineage events in wrong location: {interceptor.lineage_file}"
            )

        if ".hunyo/events/runtime" in str(interceptor.runtime_file):
            print("[OK] Runtime events will be saved to .hunyo/events/runtime/")
        else:
            print(
                f"[ERROR] Runtime events in wrong location: {interceptor.runtime_file}"
            )
    else:
        print(
            "[INFO] Running without unified tracking - install hunyo-capture to enable tracking"
        )

    # Now just run normal code - the hooks will capture everything automatically (if interceptor is available)!
    import numpy as np
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    # Create some sample data
    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "salary": [50000, 60000, 70000, 55000],
    }
    df = pd.DataFrame(data)
    print(f"Created dataframe with {len(df)} rows")
    return (df,)


@app.cell
def _(df):
    # Some data transformations
    df_filtered = df[df["age"] > 25]
    avg_salary = df["salary"].mean()
    print(f"Average salary: ${avg_salary:,.2f}")
    print(f"Filtered dataframe has {len(df_filtered)} rows")
    return (df_filtered,)


@app.cell
def _(df_filtered):
    # Final analysis
    senior_employees = df_filtered[df_filtered["age"] >= 25]
    result = {
        "total_senior_employees": len(senior_employees),
        "senior_avg_salary": senior_employees["salary"].mean(),
    }
    print(f"Analysis complete: {result}")
    print("test")
    return


if __name__ == "__main__":
    app.run()
