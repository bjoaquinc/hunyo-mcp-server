-- OpenLineage Events Table
-- Purpose: Comprehensive data lineage tracking following OpenLineage 1.0.5 specification
-- Focus: Data lineage, schema tracking, column-level transformations

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

-- Indexes for common query patterns
CREATE INDEX idx_lineage_execution_id ON lineage_events(execution_id);
CREATE INDEX idx_lineage_run_id ON lineage_events(run_id);
CREATE INDEX idx_lineage_session_id ON lineage_events(session_id);
CREATE INDEX idx_lineage_event_time ON lineage_events(event_time);
CREATE INDEX idx_lineage_job_name ON lineage_events(job_name); 