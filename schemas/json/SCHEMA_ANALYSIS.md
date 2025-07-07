# Schema Analysis: Runtime Events vs OpenLineage Events

## Overview

This document analyzes the JSON data structures found in two distinct event tracking systems within the Marimo lineage tracking project:

1. **Runtime Events** (`marimo_runtime.jsonl`) - Lightweight execution monitoring
2. **OpenLineage Events** (`marimo_lineage_events.jsonl`) - Comprehensive data lineage tracking

## Runtime Events Analysis

### Purpose
Runtime events provide lightweight monitoring of cell execution within Marimo notebooks, focusing on performance metrics and execution metadata.

### Structure
Runtime events follow a simple, flat JSON structure with the following characteristics:

- **Event Count**: 6 events (all `cell_execution_start` type)
- **Size**: ~200-300 bytes per event
- **Focus**: Execution monitoring and performance tracking

### Key Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `event_type` | string | Type of event | `"cell_execution_start"` |
| `execution_id` | string | 8-char hex execution ID | `"3731cb9b"` |
| `cell_id` | string | Cell identifier | `"MJUe"` |
| `cell_source` | string | Source code of the cell | `"df1 = pd.read_csv(\"test_data.csv\")"` |
| `cell_source_lines` | integer | Number of source lines | `3` |
| `start_memory_mb` | number | Memory usage at start (MB) | `101.09` |
| `timestamp` | string | Event occurrence time | `"2025-06-26T22:55:40.932379+00:00"` |
| `session_id` | string | Session identifier | `"6ac33987"` |
| `emitted_at` | string | Event emission time | `"2025-06-26T22:55:40.932495+00:00"` |

### Schema Extensions
The runtime schema includes optional fields for future functionality:
- `end_memory_mb` - Memory usage at execution end
- `duration_ms` - Execution duration
- `error_info` - Error details if execution fails

## OpenLineage Events Analysis

### Purpose
OpenLineage events provide comprehensive data lineage tracking following the OpenLineage 1.0.5 specification, enabling detailed data governance and lineage analysis.

### Structure
OpenLineage events follow a complex, nested structure with rich facet-based metadata:

- **Event Count**: 23 events (START/COMPLETE pairs)
- **Size**: ~2-10KB per event
- **Focus**: Data lineage, schema tracking, column-level transformations

### Top-Level Structure

| Field | Type | Description |
|-------|------|-------------|
| `eventType` | string | `START`, `COMPLETE`, `ABORT`, `FAIL` |
| `eventTime` | string | ISO 8601 timestamp |
| `run` | object | Run information with facets |
| `job` | object | Job/operation information |
| `inputs` | array | Input datasets |
| `outputs` | array | Output datasets |
| `producer` | string | Always `"marimo-lineage-tracker"` |
| `schemaURL` | string | OpenLineage schema URL |
| `session_id` | string | Marimo session ID |
| `emitted_at` | string | Emission timestamp |

### Facet Types

#### 1. Schema Facet
Defines the structure of datasets with column information:
```json
{
  "_producer": "marimo-lineage-tracker",
  "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
  "fields": [
    {
      "name": "col_x",
      "type": "object",
      "description": "Column col_x of type object"
    }
  ]
}
```

#### 2. Data Source Facet
Identifies the source of the data:
```json
{
  "_producer": "marimo-lineage-tracker",
  "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasourceDatasetFacet.json",
  "name": "pandas-dataframe",
  "uri": "memory://dataframe_4395437424"
}
```

#### 3. Column Metrics Facet
Provides statistical information about columns:
```json
{
  "columnMetrics": {
    "col_y": {
      "nullCount": 0,
      "distinctCount": 3,
      "min": 1.0,
      "max": 3.0
    }
  }
}
```

#### 4. Column Lineage Facet
Tracks column-level data flow:
```json
{
  "fields": {
    "col_x": {
      "inputFields": [
        {
          "namespace": "marimo",
          "name": "dataframe_4395437424_input_arg_0",
          "field": "col_x"
        }
      ]
    }
  }
}
```

#### 5. Performance Facet
Captures execution performance metrics:
```json
{
  "_producer": "marimo-lineage-tracker",
  "durationMs": 4.13
}
```

#### 6. Marimo Execution Facet
Links events to Marimo execution context:
```json
{
  "_producer": "marimo-lineage-tracker",
  "executionId": "3731cb9b",
  "sessionId": "60908ddd"
}
```

### Tracked Operations

The OpenLineage events capture various pandas operations:

1. **pandas_read_csv** - Reading CSV files
2. **pandas_DataFrame** - DataFrame creation
3. **pandas_merge** - DataFrame merging
4. **pandas_concat** - DataFrame concatenation
5. **dataframe_head** - DataFrame head operations
6. **dataframe_drop** - Column dropping operations

## Key Differences Comparison

| Aspect | Runtime Events | OpenLineage Events |
|--------|----------------|-------------------|
| **Purpose** | Execution monitoring | Data lineage tracking |
| **Complexity** | Simple, flat structure | Complex, nested with facets |
| **Size** | ~200-300 bytes | ~2-10KB |
| **Standards** | Custom format | OpenLineage 1.0.5 compliant |
| **Metadata Depth** | Basic execution info | Rich schema & lineage data |
| **Column Tracking** | None | Column-level lineage |
| **Performance Data** | Memory usage | Execution duration |
| **Event Pairing** | Single events | START/COMPLETE pairs |
| **Schema Evolution** | Simple extensions | Facet-based extensibility |

## Event Flow Examples

### DataFrame Creation Flow
1. **Runtime Event**: `cell_execution_start` with cell source
2. **OpenLineage START**: Job begins (`pandas_DataFrame`)
3. **OpenLineage COMPLETE**: Job completes with output dataset and facets

### DataFrame Transformation Flow
1. **Runtime Event**: Cell execution starts
2. **OpenLineage START**: Operation begins (e.g., `pandas_merge`)
3. **OpenLineage COMPLETE**: Operation completes with:
   - Input datasets with schemas and metrics
   - Output dataset with new schema
   - Column lineage showing data flow
   - Performance metrics

## Use Cases

### Runtime Events
- **Performance Monitoring**: Track memory usage and execution times
- **Debugging**: Identify slow or problematic cells
- **Audit Trails**: Basic execution logging
- **Resource Management**: Monitor system resource usage

### OpenLineage Events
- **Data Governance**: Full lineage tracking for compliance
- **Schema Discovery**: Automatic detection of data structures
- **Impact Analysis**: Understanding downstream effects of changes
- **Quality Monitoring**: Column-level metrics and validation

## DataFrame Lineage Events Analysis (NEW)

### Purpose
DataFrame lineage events provide computational lineage tracking specifically for pandas DataFrame operations, filling the gap between basic OpenLineage events and runtime monitoring by capturing detailed operation-level transformations.

### Structure
DataFrame lineage events follow a structured format designed to track specific DataFrame transformations:

- **Event Type**: Always `"dataframe_lineage"`
- **Size**: ~500-1500 bytes per event  
- **Focus**: Computational operations, column lineage, variable tracking

### Key Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `event_type` | string | Always "dataframe_lineage" | `"dataframe_lineage"` |
| `execution_id` | string | Links to runtime execution | `"cac41f13"` |
| `cell_id` | string | Marimo cell identifier | `"vblA"` |
| `session_id` | string | Session identifier | `"12c3604f"` |
| `operation_type` | string | Operation category | `"selection"`, `"aggregation"`, `"join"` |
| `operation_method` | string | Pandas method called | `"__getitem__"`, `"groupby"`, `"merge"` |
| `operation_code` | string | Source code | `"df[df['age'] > 25]"` |
| `input_dataframes` | array | Input DataFrame info | Variable names, shapes, columns |
| `output_dataframes` | array | Output DataFrame info | Variable names, shapes, columns |
| `column_lineage` | object | Column-level lineage | `{"output.col": ["input.col"]}` |
| `performance` | object | Operation monitoring | Overhead, size, sampling info |

### MVP Operation Types

The DataFrame lineage schema supports three core operation categories:

1. **Selection** (`"selection"`)
   - Methods: `__getitem__`
   - Captures: Filtering, column selection, slicing
   - Example: `df[df['age'] > 25]`, `df['name']`

2. **Aggregation** (`"aggregation"`)
   - Methods: `groupby`, `sum`, `mean`, `count`
   - Captures: GroupBy operations and aggregations
   - Example: `df.groupby('dept').sum()`

3. **Join** (`"join"`)
   - Methods: `merge`
   - Captures: DataFrame merging operations
   - Example: `df1.merge(df2, on='id')`

### DataFrame Info Schema

Each DataFrame reference includes:
```json
{
  "variable_name": "df_filtered_123",
  "object_id": "140234567891",
  "shape": [3, 3],
  "columns": ["name", "age", "salary"],
  "memory_usage_mb": 0.001
}
```

### Column Lineage Mapping

Column lineage uses dot notation to map output columns to input columns:
```json
{
  "column_lineage": {
    "df_filtered.name": ["df.name"],
    "df_filtered.age": ["df.age"],
    "df_filtered.salary": ["df.salary"]
  }
}
```

## Comprehensive Schema Comparison

| Aspect | Runtime Events | OpenLineage Events | DataFrame Lineage Events |
|--------|----------------|-------------------|-------------------------|
| **Purpose** | Execution monitoring | Data lineage tracking | Computational lineage |
| **Granularity** | Cell-level | Dataset/Job-level | Operation-level |
| **DataFrame Focus** | None | Basic creation/modification | Detailed transformations |
| **Variable Tracking** | None | Object IDs only | Variable names + object IDs |
| **Column Lineage** | None | Complex facet-based | Simple dot notation |
| **Operation Details** | Source code only | Job metadata | Method + parameters |
| **Performance** | Memory usage | Duration | Overhead + sampling |
| **Primary Gap Addressed** | N/A | N/A | **Missing DataFrame transformations** |

### Schema Integration Strategy

The three schemas work together to provide comprehensive coverage:

1. **Runtime Events** → Execution context and performance baseline
2. **OpenLineage Events** → Standard data lineage and external I/O operations  
3. **DataFrame Lineage Events** → **NEW** Computational transformations and column-level lineage

This addresses the **critical gap** where filtering operations like `df[df['age'] > 25]` were completely missing from lineage tracking.
- **Impact Analysis**: Understand data flow dependencies
- **Data Quality**: Track schema changes and data metrics
- **Debugging**: Detailed column-level data flow analysis
- **Documentation**: Automatic data pipeline documentation

## Integration Considerations

Both event systems share common identifiers (`session_id`, `execution_id`) allowing for correlation:

1. **Performance Analysis**: Combine runtime metrics with lineage complexity
2. **Complete Audit**: Runtime events for execution, OpenLineage for data flow
3. **Optimization**: Use runtime events to identify bottlenecks, OpenLineage to understand data flow
4. **Compliance**: Runtime events for execution logs, OpenLineage for data governance

## Schema Validation

Both schemas are provided as JSON Schema documents:

- `runtime_events_schema.json` - Validates runtime event structure
- `openlineage_events_schema.json` - Validates OpenLineage event structure with all facets

These schemas ensure data consistency and enable automated validation of event streams. 