-- DataFrame Computational Lineage Events Table
-- Tracks computational operations on DataFrames (filtering, grouping, joining)

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