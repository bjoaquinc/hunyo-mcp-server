-- Runtime Events Table
-- Purpose: Lightweight execution monitoring of cell execution within Marimo notebooks
-- Focus: Performance metrics and execution metadata

CREATE TABLE IF NOT EXISTS runtime_events (
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

-- Index for common query patterns
CREATE INDEX IF NOT EXISTS idx_runtime_execution_id ON runtime_events(execution_id);
CREATE INDEX IF NOT EXISTS idx_runtime_session_id ON runtime_events(session_id);
CREATE INDEX IF NOT EXISTS idx_runtime_timestamp ON runtime_events(timestamp); 