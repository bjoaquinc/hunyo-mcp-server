-- Performance Metrics View
-- Purpose: Combined runtime and lineage performance analysis
-- Usage: Compare execution times, memory usage, and data processing metrics

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