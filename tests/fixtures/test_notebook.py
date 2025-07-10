import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def cell_1():
    """Cell 1: Create and display DataFrame with marimo UI"""

    import marimo as mo
    import pandas as pd

    # Import hunyo capture functionality
    from hunyo_capture.unified_notebook_interceptor import (
        enable_unified_tracking,
        is_unified_tracking_active,
    )

    # Enable unified tracking for this test notebook
    if not is_unified_tracking_active():
        enable_unified_tracking(notebook_path=__file__)

    df = pd.DataFrame({"person": ["Alice", "Bob", "Charlie"], "age": [20, 30, 40]})
    transformed_df = mo.ui.dataframe(df)
    return df, mo, transformed_df


@app.cell
def cell_2(df, mo):
    """Cell 2: Access the transformed dataframe value"""

    # Add a new column 'age_category' based on each person’s age
    df["age_category"] = df["age"].apply(lambda age: "young" if age < 30 else "adult")

    # Display the updated DataFrame interactively
    mo.ui.dataframe(df)


if __name__ == "__main__":
    app.run()
