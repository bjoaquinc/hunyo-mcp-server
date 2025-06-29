# Schemas Directory

This directory contains all schema definitions and data models for the Marimo Lineage MCP project.

## Structure

```
schemas/
├── json/                           # JSON Schema definitions
│   ├── openlineage_events_schema.json    # OpenLineage event validation schema
│   ├── runtime_events_schema.json        # Runtime event validation schema
│   └── SCHEMA_ANALYSIS.md                # Detailed analysis of both schemas
│
└── sql/                            # SQL schema definitions
    ├── init_database.sql                  # Complete database initialization script
    ├── runtime_events_table.sql           # DuckDB table for runtime events
    ├── openlineage_events_table.sql       # DuckDB table for lineage events
    ├── mcp_cheatsheet.md                  # LLM query assistance guide
    └── views/                             # SQL views for common queries
        ├── lineage_summary.sql
        └── performance_metrics.sql
```

## JSON Schemas

### Runtime Events Schema
- **File**: `json/runtime_events_schema.json`
- **Purpose**: Validates lightweight execution monitoring events
- **Event Types**: `cell_execution_start`, `cell_execution_end`, `cell_execution_error`
- **Size**: ~200-300 bytes per event

### OpenLineage Events Schema
- **File**: `json/openlineage_events_schema.json`
- **Purpose**: Validates comprehensive data lineage events following OpenLineage 1.0.5 spec
- **Event Types**: `START`, `COMPLETE`, `ABORT`, `FAIL`
- **Size**: ~2-10KB per event
- **Features**: Column-level lineage, schema tracking, performance metrics

## SQL Schemas

The SQL directory contains a **Hybrid DuckDB Schema** design that balances flexibility with performance:

### Design Philosophy
- **Hybrid Structure**: Structured columns for hot queries + JSON columns for flexible facets
- **LLM-Optimized**: Minimal join depth and self-describing schema for better Text-to-SQL performance
- **Evolution-Ready**: Start semi-structured, flatten high-frequency facets later

### Key Files
- **`init_database.sql`**: Complete setup script with tables, indexes, and views
- **`mcp_cheatsheet.md`**: LLM query assistance with common patterns and examples
- **Individual table files**: Modular DDL for specific components

### Tables
- **`runtime_events`**: Lightweight execution telemetry (~300B/event)
- **`lineage_events`**: OpenLineage data with JSON facets (~2-10KB/event)
- **Views**: Pre-built joins for I/O analysis and performance metrics

## Usage

These schemas are used by:
1. **Capture Layer**: Event validation before emission
2. **Ingestion Layer**: Data validation before database insertion  
3. **MCP Tools**: Schema introspection and query assistance
4. **Testing**: Fixture generation and validation 