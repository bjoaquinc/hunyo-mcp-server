-- Marimo Lineage MCP Database Initialization
-- This script sets up the complete DuckDB schema for the hybrid design

-- Enable JSON support
INSTALL json;
LOAD json;

------------------------------------------------------------
-- 1. Runtime execution telemetry
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runtime_events (
    event_id            BIGINT      PRIMARY KEY,
    event_type          VARCHAR,        -- start | end | error
    execution_id        CHAR(8),
    cell_id             VARCHAR,
    cell_source         VARCHAR,
    cell_source_lines   INT,
    start_memory_mb     DOUBLE,
    end_memory_mb       DOUBLE,
    duration_ms         DOUBLE,
    timestamp           TIMESTAMPTZ,
    session_id          CHAR(8),
    emitted_at          TIMESTAMPTZ,
    error_info          JSON           -- {error_type, error_message, traceback}
);

-- Indexes for runtime_events
CREATE INDEX IF NOT EXISTS idx_runtime_execution_id ON runtime_events(execution_id);
CREATE INDEX IF NOT EXISTS idx_runtime_session_id ON runtime_events(session_id);
CREATE INDEX IF NOT EXISTS idx_runtime_timestamp ON runtime_events(timestamp);

------------------------------------------------------------
-- 2. OpenLineage events with semi-structured facets
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lineage_events (
    ol_event_id         BIGINT      PRIMARY KEY,
    run_id              UUID,
    execution_id        CHAR(8),        -- link to runtime_events.execution_id
    event_type          VARCHAR,        -- START | COMPLETE | ABORT | FAIL
    job_name            VARCHAR,
    event_time          TIMESTAMPTZ,
    duration_ms         DOUBLE,
    session_id          CHAR(8),
    emitted_at          TIMESTAMPTZ,

    -- Facets kept as JSON for flexibility
    inputs_json             JSON,
    outputs_json            JSON,
    column_lineage_json     JSON,
    column_metrics_json     JSON,
    other_facets_json       JSON
);

-- Indexes for lineage_events
CREATE INDEX IF NOT EXISTS idx_lineage_execution_id ON lineage_events(execution_id);
CREATE INDEX IF NOT EXISTS idx_lineage_run_id ON lineage_events(run_id);
CREATE INDEX IF NOT EXISTS idx_lineage_session_id ON lineage_events(session_id);
CREATE INDEX IF NOT EXISTS idx_lineage_event_time ON lineage_events(event_time);
CREATE INDEX IF NOT EXISTS idx_lineage_job_name ON lineage_events(job_name);

------------------------------------------------------------
-- 3. DataFrame computational lineage events (MVP)
------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS dataframe_lineage_events_seq START 1;

CREATE TABLE IF NOT EXISTS dataframe_lineage_events (
    df_event_id         BIGINT      PRIMARY KEY DEFAULT nextval('dataframe_lineage_events_seq'),
    event_type          VARCHAR     NOT NULL DEFAULT 'dataframe_lineage',
    execution_id        CHAR(8)     NOT NULL,
    cell_id             VARCHAR     NOT NULL,
    session_id          CHAR(8)     NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL,
    emitted_at          TIMESTAMPTZ NOT NULL,
    
    -- Operation Information (MVP Core)
    operation_type      VARCHAR     NOT NULL,           -- selection, aggregation, join
    operation_method    VARCHAR     NOT NULL,           -- __getitem__, groupby, sum, mean, count, merge
    operation_code      VARCHAR,                        -- Source code
    operation_parameters JSON,                          -- Operation-specific parameters
    
    -- DataFrame Information (MVP)
    input_dataframes    JSON        NOT NULL,           -- Array of input DataFrame info
    output_dataframes   JSON        NOT NULL,           -- Array of output DataFrame info
    column_lineage      JSON,                           -- Column-level lineage mapping
    
    -- Performance Monitoring (MVP - Simplified)
    overhead_ms         DOUBLE,                         -- Time spent in interception logic (ms)
    df_size_mb          DOUBLE,                         -- Size of input DataFrame in MB
    sampled             BOOLEAN                         -- Whether sampling was used
);

-- Indexes for dataframe_lineage_events
CREATE INDEX IF NOT EXISTS idx_df_lineage_execution_id ON dataframe_lineage_events(execution_id);
CREATE INDEX IF NOT EXISTS idx_df_lineage_session_time ON dataframe_lineage_events(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_df_lineage_operation ON dataframe_lineage_events(operation_type, operation_method);
CREATE INDEX IF NOT EXISTS idx_df_lineage_timestamp ON dataframe_lineage_events(timestamp);

------------------------------------------------------------
-- 4. Convenience views
------------------------------------------------------------

-- Lineage I/O Summary View
CREATE OR REPLACE VIEW vw_lineage_io AS
SELECT
    ol_event_id,
    run_id,
    execution_id,
    event_type,
    job_name,
    event_time,
    session_id,
    json_extract(inputs_json , '$[*].name') AS input_names,
    json_extract(outputs_json, '$[*].name') AS output_names,
    json_array_length(inputs_json) AS input_count,
    json_array_length(outputs_json) AS output_count
FROM lineage_events;

-- Performance Metrics View
CREATE OR REPLACE VIEW vw_performance_metrics AS
SELECT
    r.execution_id,
    r.session_id,
    r.cell_id,
    r.event_type AS runtime_event_type,
    r.start_memory_mb,
    r.end_memory_mb,
    (r.end_memory_mb - r.start_memory_mb) AS memory_delta_mb,
    r.duration_ms AS runtime_duration_ms,
    r.timestamp AS runtime_timestamp,
    
    l.ol_event_id,
    l.run_id,
    l.event_type AS lineage_event_type,
    l.job_name,
    l.duration_ms AS lineage_duration_ms,
    l.event_time AS lineage_timestamp,
    
    json_array_length(l.inputs_json) AS input_dataset_count,
    json_array_length(l.outputs_json) AS output_dataset_count,
    
    -- Performance ratios
    CASE 
        WHEN r.duration_ms > 0 THEN (l.duration_ms / r.duration_ms) 
        ELSE NULL 
    END AS lineage_to_runtime_ratio
    
FROM runtime_events r
LEFT JOIN lineage_events l ON r.execution_id = l.execution_id
WHERE r.event_type IN ('cell_execution_start', 'cell_execution_end');

-- DataFrame Lineage Summary View
CREATE OR REPLACE VIEW vw_dataframe_lineage AS
SELECT
    df_event_id,
    execution_id,
    cell_id,
    session_id,
    timestamp,
    operation_type,
    operation_method,
    operation_code,
    
    -- Extract DataFrame information from JSON
    json_array_length(input_dataframes) AS input_df_count,
    json_array_length(output_dataframes) AS output_df_count,
    json_extract(input_dataframes, '$[0].variable_name') AS input_variable,
    json_extract(output_dataframes, '$[0].variable_name') AS output_variable,
    json_extract(input_dataframes, '$[0].shape[0]') AS input_rows,
    json_extract(input_dataframes, '$[0].shape[1]') AS input_cols,
    json_extract(output_dataframes, '$[0].shape[0]') AS output_rows,
    json_extract(output_dataframes, '$[0].shape[1]') AS output_cols,
    
    -- Column lineage summary
    json_keys(column_lineage) AS output_columns,
    
    -- Performance metrics
    overhead_ms,
    df_size_mb,
    sampled,
    
    -- Calculate transformation ratios
    CASE 
        WHEN json_extract(input_dataframes, '$[0].shape[0]') > 0 
        THEN CAST(json_extract(output_dataframes, '$[0].shape[0]') AS DOUBLE) / 
             CAST(json_extract(input_dataframes, '$[0].shape[0]') AS DOUBLE)
        ELSE NULL 
    END AS row_retention_ratio

FROM dataframe_lineage_events; 