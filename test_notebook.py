import marimo

__generated_with = "0.8.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import numpy as np
    import pandas as pd

    # Enable Hunyo tracking
    from capture.lightweight_runtime_tracker import enable_runtime_tracking
    from capture.live_lineage_interceptor import enable_live_tracking
    enable_runtime_tracking()
    enable_live_tracking()

    print("🎯 Hunyo tracking enabled!")

    return np, pd


@app.cell
def __(np, pd):
    # Create some test data
    data = {
        "A": np.random.randn(100),
        "B": np.random.randn(100),
        "C": np.random.randn(100),
    }
    df = pd.DataFrame(data)
    print(f"Created dataframe with shape: {df.shape}")
    return data, df


@app.cell
def __(df):
    # Simple transformation
    df_filtered = df[df["A"] > 0]
    print(f"Filtered dataframe shape: {df_filtered.shape}")
    return (df_filtered,)


@app.cell
def __(df_filtered):
    # Another transformation
    df_final = df_filtered.copy()
    df_final["D"] = df_final["A"] + df_final["B"]
    print(f"Final dataframe shape: {df_final.shape}")
    return (df_final,)


if __name__ == "__main__":
    app.run()
