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