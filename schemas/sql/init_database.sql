-- Marimo Lineage MCP Database Initialization
-- This script sets up the complete DuckDB schema for the hybrid design

-- Enable JSON support
INSTALL json;
LOAD json;

------------------------------------------------------------
-- 1. Runtime execution telemetry
------------------------------------------------------------
CREATE TABLE runtime_events (
    event_id            BIGINT      AUTO_INCREMENT PRIMARY KEY,
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
CREATE INDEX idx_runtime_execution_id ON runtime_events(execution_id);
CREATE INDEX idx_runtime_session_id ON runtime_events(session_id);
CREATE INDEX idx_runtime_timestamp ON runtime_events(timestamp);

------------------------------------------------------------
-- 2. OpenLineage events with semi-structured facets
------------------------------------------------------------
CREATE TABLE lineage_events (
    ol_event_id         BIGINT      AUTO_INCREMENT PRIMARY KEY,
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
CREATE INDEX idx_lineage_execution_id ON lineage_events(execution_id);
CREATE INDEX idx_lineage_run_id ON lineage_events(run_id);
CREATE INDEX idx_lineage_session_id ON lineage_events(session_id);
CREATE INDEX idx_lineage_event_time ON lineage_events(event_time);
CREATE INDEX idx_lineage_job_name ON lineage_events(job_name);

------------------------------------------------------------
-- 3. Convenience views
------------------------------------------------------------

-- Lineage I/O Summary View
CREATE VIEW vw_lineage_io AS
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
CREATE VIEW vw_performance_metrics AS
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