-- Lineage I/O Summary View
-- Purpose: Quick lookup of input/output dataset names for lineage analysis
-- Usage: Simplifies common queries about data flow

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