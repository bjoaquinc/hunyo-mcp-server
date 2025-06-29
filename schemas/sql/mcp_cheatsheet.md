# DuckDB Schema Cheat Sheet for MCP Tools

## Tables Overview

| Table | Purpose | Size | Key Columns |
|-------|---------|------|-------------|
| `runtime_events` | Per-cell execution telemetry | ~300 bytes/event | `execution_id`, `cell_id`, `duration_ms` |
| `lineage_events` | OpenLineage data lineage events | ~2-10KB/event | `execution_id`, `run_id`, `inputs_json`, `outputs_json` |

## Key Relationships

- **Link runtime ↔ lineage**: `runtime_events.execution_id = lineage_events.execution_id`
- **Session grouping**: Both tables have `session_id` for notebook session tracking

## Essential Columns

### runtime_events
```sql
event_id            -- Auto-increment primary key
execution_id        -- Links to lineage_events (CHAR(8))
cell_id             -- Marimo cell identifier
event_type          -- 'start' | 'end' | 'error'
duration_ms         -- Cell execution time
start_memory_mb     -- Memory at start
end_memory_mb       -- Memory at end
timestamp           -- When execution occurred
session_id          -- Notebook session (CHAR(8))
error_info          -- JSON: {error_type, error_message, traceback}
```

### lineage_events
```sql
ol_event_id         -- Auto-increment primary key
execution_id        -- Links to runtime_events (CHAR(8))
run_id              -- OpenLineage run UUID
event_type          -- 'START' | 'COMPLETE' | 'ABORT' | 'FAIL'
job_name            -- Operation name (e.g., 'pandas_read_csv')
duration_ms         -- Operation duration
inputs_json         -- JSON array of input datasets
outputs_json        -- JSON array of output datasets
column_lineage_json -- Column-level data flow
column_metrics_json -- Statistical metrics per column
```

## Common Query Patterns

### 💡 Example Queries for LLM

```sql
-- 1️⃣ What datasets fed an execution?
SELECT json_extract(inputs_json, '$[*].name') AS input_datasets
FROM lineage_events
WHERE execution_id = '3731cb9b';

-- 2️⃣ Runtime vs. lineage timing comparison
SELECT 
    r.cell_id,
    r.duration_ms AS runtime_ms,
    l.duration_ms AS lineage_ms,
    l.job_name
FROM runtime_events r
JOIN lineage_events l USING (execution_id)
WHERE r.cell_id = 'MJUe';

-- 3️⃣ Memory usage by session
SELECT 
    session_id,
    AVG(end_memory_mb - start_memory_mb) AS avg_memory_delta,
    COUNT(*) AS cell_count
FROM runtime_events
WHERE event_type = 'end'
GROUP BY session_id;

-- 4️⃣ Data flow for a specific dataset
SELECT 
    execution_id,
    job_name,
    json_extract(inputs_json, '$[*].name') AS inputs,
    json_extract(outputs_json, '$[*].name') AS outputs
FROM lineage_events
WHERE json_extract(inputs_json, '$[*].name') LIKE '%dataframe_123%';

-- 5️⃣ Column lineage tracing
SELECT 
    execution_id,
    job_name,
    json_extract(column_lineage_json, '$.fields') AS column_mappings
FROM lineage_events
WHERE column_lineage_json IS NOT NULL;
```

## Views Available

### vw_lineage_io
Quick I/O dataset lookup with counts:
```sql
SELECT * FROM vw_lineage_io WHERE execution_id = '3731cb9b';
```

### vw_performance_metrics  
Combined runtime + lineage performance:
```sql
SELECT * FROM vw_performance_metrics ORDER BY runtime_duration_ms DESC LIMIT 10;
```

## JSON Extraction Helpers

```sql
-- Extract all input dataset names
json_extract(inputs_json, '$[*].name')

-- Extract specific facet data
json_extract(column_metrics_json, '$.columnMetrics.col_x.nullCount')

-- Check if dataset exists in inputs
json_extract(inputs_json, '$[*].name') LIKE '%dataset_name%'

-- Count items in JSON array
json_array_length(inputs_json)

-- Extract nested schema information
json_extract(inputs_json, '$[0].facets.schema.fields[*].name')
```

## Performance Tips

- Use `DESCRIBE table_name;` to see current schema
- DuckDB auto-creates zone maps; explicit indexes rarely needed
- JSON functions are SIMD-accelerated and very fast
- Use `EXPLAIN` to see query plans for optimization

## Schema Evolution Path

| When | What | Action |
|------|------|--------|
| JSON facet in >20% queries | Hot path optimization | `ALTER TABLE ADD COLUMN facet_x AS (json_extract(...))` |
| Dataset names repeat >1M times | Normalization | Create `dim_dataset` reference table |
| Regulatory compliance needed | Full normalization | Migrate to FK/PK constrained schema | 