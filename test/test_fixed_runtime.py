# ruff: noqa
import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    print("🧪 Testing FIXED runtime tracking...")


@app.cell
def _():
    # Enable runtime tracking with the fixed system
    print("Enabling fixed runtime tracking...")
    import marimo_native_hooks_interceptor

    interceptor = marimo_native_hooks_interceptor.enable_native_hook_tracking()
    print(f"✅ Interceptor enabled: {interceptor.session_id}")


@app.cell
def _():
    # Simple test operations
    print("Running test operations...")
    x = 10 + 5
    print(f"Math result: {x}")

    text = "Hello World"
    print(f"String: {text}")

    return (text,)


@app.cell
def _(text):
    # DataFrame test
    print("Testing DataFrame operations...")
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.sum()
    print(f"DataFrame sum: {result.to_dict()}")
    print(text)


@app.cell
def _():
    # Check if files were created
    print("Checking log files...")
    import os

    runtime_exists = os.path.exists("marimo_runtime.jsonl")
    lineage_exists = os.path.exists("marimo_lineage_events.jsonl")

    print(f"Runtime log exists: {runtime_exists}")
    print(f"Lineage log exists: {lineage_exists}")

    if runtime_exists:
        with open("marimo_runtime.jsonl") as f:
            runtime_events = len(f.readlines())
        print(f"Runtime events: {runtime_events}")
    else:
        runtime_events = 0
        print("❌ No runtime events")

    if lineage_exists:
        with open("marimo_lineage_events.jsonl") as f:
            lineage_events = len(f.readlines())
        print(f"Lineage events: {lineage_events}")
    else:
        lineage_events = 0
        print("❌ No lineage events")

    print("\n🏁 Results:")
    print(
        f"   Runtime tracking: {'✅ WORKING' if runtime_events > 0 else '❌ NOT WORKING'}"
    )
    print(
        f"   Lineage tracking: {'✅ WORKING' if lineage_events > 0 else '❌ NOT WORKING'}"
    )


if __name__ == "__main__":
    app.run()
