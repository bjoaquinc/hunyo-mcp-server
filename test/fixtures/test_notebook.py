#!/usr/bin/env python3
"""
Test notebook for hunyo-mcp-server end-to-end testing.

This is a simple Python file that represents a marimo notebook for testing purposes.
The MCP server will process this file to set up infrastructure and validate the pipeline.
"""

import pandas as pd


# Simple data operations that could be tracked
def create_test_dataframe():
    """Create a simple test DataFrame"""
    data = {
        "id": range(1, 6),
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [85, 92, 78, 96, 88],
        "category": ["A", "B", "A", "B", "A"],
    }
    return pd.DataFrame(data)


def transform_data(df):
    """Simple data transformation"""
    return (
        df.groupby("category")
        .agg({"score": ["mean", "max", "min"], "id": "count"})
        .round(2)
    )


# Test notebook execution
if __name__ == "__main__":
    print("🧪 Test notebook execution started")

    # Create test data
    df = create_test_dataframe()
    print(f"📊 Created DataFrame with {len(df)} rows")

    # Transform data
    summary = transform_data(df)
    print(f"📈 Generated summary with {len(summary)} rows")

    print("✅ Test notebook execution completed")
