import marimo

__generated_with = "0.14.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os
    import sys

    # Add project root to path to allow importing marimo_native_hooks_interceptor
    # This is necessary because the test is in a subdirectory.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    # Now the import should work
    import marimo_native_hooks_interceptor
    marimo_native_hooks_interceptor.enable_native_hook_tracking()
    return mo, os, pd


@app.cell
def _(pd):
    # Test Case 1: Create a DataFrame from a dictionary
    # This should generate an OpenLineage event for a new dataset.
    df1 = pd.read_csv("test_data.csv")
    return df1,


@app.cell
def _(pd):
    # Test Case 2: Create another DataFrame
    data2 = {'colA': [1, 2], 'colB': [5, 6]}
    df2 = pd.DataFrame(data2)
    df2
    return (df2,)


@app.cell
def _(df1, df2, pd):
    # Test Case 3: Merge two DataFrames
    # This should generate an OpenLineage event with two inputs (df1, df2)
    # and one output (merged_df).
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True)
    merged_df
    return (merged_df,)


@app.cell
def _(merged_df):
    # Test Case 4: Transform the merged DataFrame (e.g., drop a column)
    # This should generate an event with merged_df as input and transformed_df as output.
    transformed_df = merged_df.drop(columns=['colA'])
    transformed_df
    return


@app.cell
def _(os, pd):
    # Test Case 5: Read from a CSV
    # This should generate an event for a new dataset from a file source.
    csv_content = "id,value\n1,A\n2,B\n3,C"
    csv_path = "test_data.csv"
    with open(csv_path, "w") as f:
        f.write(csv_content)

    df_from_csv = pd.read_csv(csv_path)
    os.remove(csv_path) # Clean up
    df_from_csv
    return


@app.cell
def _(mo):
    # Verify the generated lineage events
    lineage_file = "marimo_lineage_events.jsonl"
    mo.ui.h3(f"Contents of {lineage_file}")
    return (lineage_file,)


@app.cell
def _(lineage_file, mo):
    try:
        with open(lineage_file, "r") as f:
            lines = f.readlines()

        if not lines:
            mo.md(f"**⚠️  The lineage file `{lineage_file}` is empty.**")
        else:
            # Display the last few events for clarity
            num_events = len(lines)
            mo.md(f"**Found {num_events} events. Displaying the last 5:**")
            for line in lines[-5:]:
                mo.ui.code(line.strip(), language="json")

    except FileNotFoundError:
        mo.md(f"**❌ Error: The lineage file `{lineage_file}` was not found.**")
    return


if __name__ == "__main__":
    app.run()
