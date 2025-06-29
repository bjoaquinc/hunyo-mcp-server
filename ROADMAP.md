# Hunyo MCP Server - Project Roadmap

## 🎯 Project Purpose

**Hunyo MCP Server** is a single-command orchestrator that provides zero-touch notebook instrumentation for comprehensive DataFrame lineage tracking and runtime analysis. One `pipx run` command automatically instruments marimo notebooks, captures execution events, ingests them into DuckDB, and exposes rich querying capabilities to LLMs via MCP tools.

### Core Value Proposition
- **Zero manual setup**: `pipx run hunyo-mcp-server --notebook file.py` just works
- **No notebook modifications**: Auto-injection handles all instrumentation
- **Real-time insights**: Background ingestion keeps data current
- **LLM-friendly**: Rich MCP tools for analysis and querying

## 🏗️ Overarching Architecture

### Complete Workflow
```bash
pipx run hunyo-mcp-server --notebook /path/to/analysis.py
```

**Startup Sequence:**
1. **Auto-inject capture code** → Parse notebook AST, inject import statements
2. **Initialize DuckDB** → Create tables, indexes, views from existing schema
3. **Start background watcher** → Monitor JSONL files for real-time ingestion
4. **Expose MCP tools** → Provide LLM query interface
5. **User executes notebook** → Capture layer generates events → Real-time analysis available

### Data Flow
```
Notebook Execution → Capture Layer → JSONL Events → File Watcher → 
DuckDB Tables → MCP Query Tools → LLM Analysis
```

### Data Storage Strategy
```
# Production mode
~/.hunyo/
├── events/
│   ├── runtime_events.jsonl
│   └── openlineage_events.jsonl
├── database/
│   └── lineage.duckdb
└── config/
    └── settings.yaml

# Development mode (when running from repo)
hunyo-mcp-server/.hunyo/
├── events/
│   ├── runtime_events.jsonl
│   └── openlineage_events.jsonl
├── database/
│   └── lineage.duckdb
└── config/
    └── settings.yaml
```

## 📊 Current Implementation Status

### ✅ **Fully Implemented (Excellent Quality)**

**1. Data Capture Layer**
- `src/capture/live_lineage_interceptor.py` (30KB, 710 lines) - Sophisticated pandas monkey-patching
- `src/capture/lightweight_runtime_tracker.py` (21KB, 518 lines) - Runtime execution tracking
- `src/capture/native_hooks_interceptor.py` (39KB, 934 lines) - Advanced hooking mechanisms
- `src/capture/websocket_interceptor.py` (15KB, 388 lines) - WebSocket integration
- **Status**: Production-ready, sophisticated event linking, DataFrame ID tracking

**2. Database Schema Design**
- `schemas/sql/init_database.sql` - Complete DuckDB schema with runtime_events + lineage_events tables
- `schemas/sql/openlineage_events_table.sql` - OpenLineage-compliant event structure
- `schemas/sql/runtime_events_table.sql` - Execution telemetry schema
- `schemas/sql/views/` - Performance metrics and lineage summary views
- **Status**: Excellent hybrid design, proper indexing, JSON facet support

**3. Event Validation & Schemas**
- `schemas/json/openlineage_events_schema.json` (365 lines) - Comprehensive OpenLineage schema
- `schemas/json/runtime_events_schema.json` - Runtime event validation
- `schemas/json/SCHEMA_ANALYSIS.md` - Documentation and analysis
- **Status**: Well-structured, compliant with OpenLineage standards

**4. Testing Infrastructure**
- `test/test_capture_integration.py` - Integration testing framework
- `test/test_fixed_runtime.py` - Runtime tracking tests
- `test/test_openlineage_generation.py` - Lineage event validation
- **Status**: Comprehensive testing coverage for capture layer

**5. Package Management & Build System**
- `pyproject.toml` - Complete modern Python packaging configuration
- CLI entry points configured for `pipx run hunyo-mcp-server`
- Development dependencies and tool configuration (Black, Ruff, MyPy, Pytest)
- **Status**: Ready for distribution and development workflows

### 🔄 **Currently Missing (Need Implementation)**

**1. MCP Server Architecture**
- No `server.py` - CLI entry point and MCP server
- No `orchestrator.py` - Component coordination
- No `config.py` - Data path management (dev vs prod)
- **Impact**: No single-command orchestration capability

**2. Database Ingestion Pipeline**
- No `ingestion/file_watcher.py` - JSONL file monitoring
- No `ingestion/duckdb_manager.py` - Database initialization and management
- No `ingestion/event_processor.py` - Event validation and insertion
- **Impact**: JSONL files exist but never get ingested into DuckDB

**3. Notebook Auto-Injection**
- No `notebook_injector.py` - AST parsing and import injection
- **Impact**: Requires manual import statements in notebooks

**4. MCP Query Tools**
- No `tools/query_tool.py` - SQL execution interface
- No `tools/schema_tool.py` - Database schema inspection
- No `tools/lineage_tool.py` - DataFrame lineage analysis
- **Impact**: No LLM query capabilities

**6. Remaining Package Integration**
- Need project structure setup (`src/hunyo_mcp_server/` directories)
- Need `__init__.py` files for proper module imports
- **Impact**: Package structure exists but modules not yet importable

## 🗺️ Target Project Structure

```
hunyo-mcp-server/
├── src/hunyo_mcp_server/
│   ├── __init__.py
│   ├── server.py                    # 🚧 MISSING - CLI entry + MCP server
│   ├── orchestrator.py              # 🚧 MISSING - Component coordination
│   ├── config.py                    # 🚧 MISSING - Data path management
│   │
│   ├── capture/                     # ✅ COMPLETE - Excellent quality
│   │   ├── __init__.py
│   │   ├── live_lineage_interceptor.py
│   │   ├── lightweight_runtime_tracker.py
│   │   ├── native_hooks_interceptor.py
│   │   ├── websocket_interceptor.py
│   │   └── notebook_injector.py     # 🚧 MISSING - AST injection
│   │
│   ├── ingestion/                   # 🚧 MISSING - JSONL → DuckDB pipeline
│   │   ├── __init__.py
│   │   ├── file_watcher.py
│   │   ├── duckdb_manager.py
│   │   └── event_processor.py
│   │
│   └── tools/                       # 🚧 MISSING - MCP interface
│       ├── __init__.py
│       ├── query_tool.py
│       ├── schema_tool.py
│       └── lineage_tool.py
│
├── schemas/                         # ✅ COMPLETE - Excellent design
│   ├── sql/
│   │   ├── init_database.sql
│   │   ├── openlineage_events_table.sql
│   │   ├── runtime_events_table.sql
│   │   └── views/
│   └── json/
│       ├── openlineage_events_schema.json
│       └── runtime_events_schema.json
│
├── test/                           # ✅ COMPLETE - Good coverage
│   ├── test_capture_integration.py
│   ├── test_fixed_runtime.py
│   └── test_openlineage_generation.py
│
├── .hunyo/                         # 🚧 MISSING - Dev mode data directory
│   ├── events/
│   ├── database/
│   └── config/
│
├── pyproject.toml                  # ✅ COMPLETE - Package configuration
├── README.md                       # ✅ EXISTS - Needs update for MCP
└── ROADMAP.md                      # 📄 THIS FILE
```

## 🚀 Implementation Roadmap

### **Phase 1: Foundation (High Priority)**
**Goal**: Basic CLI orchestration and data path management

1. ✅ **Create `pyproject.toml`** 
   - ✅ Define package metadata and CLI entry points
   - ✅ Set up dependencies (marimo, duckdb, mcp, click)
   - ✅ Enable `pipx run hunyo-mcp-server`
   - ✅ Remove legacy `requirements.txt`

2. **Implement `src/hunyo_mcp_server/config.py`**
   - Data directory resolution (dev vs prod)
   - Environment detection logic
   - Path management utilities

3. **Create `src/hunyo_mcp_server/server.py`**
   - CLI interface with `--notebook` option
   - Basic MCP server setup
   - Integration with orchestrator

4. **Implement `src/hunyo_mcp_server/orchestrator.py`**
   - Component coordination logic
   - Startup sequence management
   - Graceful shutdown handling

### **Phase 2: Auto-Instrumentation (High Priority)**
**Goal**: Zero-touch notebook setup

5. **Create `src/hunyo_mcp_server/capture/notebook_injector.py`**
   - AST parsing for marimo notebooks
   - Import injection at appropriate locations
   - Backup and restoration mechanisms
   - Error handling for malformed notebooks

6. **Implement `src/hunyo_mcp_server/ingestion/duckdb_manager.py`**
   - Execute existing SQL schemas
   - Database initialization and connection management
   - Schema migration support

### **Phase 3: Background Processing (Medium Priority)**
**Goal**: Real-time data ingestion

7. **Create `src/hunyo_mcp_server/ingestion/file_watcher.py`**
   - Monitor JSONL files for changes
   - Handle file rotation and cleanup
   - Efficient incremental processing

8. **Implement `src/hunyo_mcp_server/ingestion/event_processor.py`**
   - Event validation against schemas
   - Data transformation and enrichment
   - Batch insertion optimization

### **Phase 4: MCP Query Interface (Medium Priority)**
**Goal**: LLM-accessible analysis tools

9. **Create `src/hunyo_mcp_server/tools/query_tool.py`**
   - Direct SQL execution against DuckDB
   - Query result formatting
   - Security and validation

10. **Implement `src/hunyo_mcp_server/tools/schema_tool.py`**
    - Database schema inspection
    - Table and column metadata
    - Example query generation

11. **Create `src/hunyo_mcp_server/tools/lineage_tool.py`**
    - DataFrame lineage chain analysis
    - Performance metrics aggregation
    - Specialized lineage queries

### **Phase 5: Polish & Production (Low Priority)**
**Goal**: Production readiness and documentation

12. **Enhanced Error Handling**
    - Comprehensive error recovery
    - Logging and monitoring
    - Graceful degradation

13. **Documentation Updates**
    - README.md refresh for MCP usage
    - API documentation
    - Troubleshooting guides

14. **Advanced Features**
    - Configuration file support
    - Multiple notebook support
    - Export/import capabilities

## 🤔 Technical Considerations

### **Architecture Decisions**

**1. Import Injection Strategy**
- **Approach**: AST manipulation to inject import statements
- **Alternative**: Manual imports (current approach)
- **Trade-off**: Zero-touch vs. reliability
- **Decision**: Proceed with AST injection for true zero-touch experience

**2. Database Choice**
- **Current**: DuckDB (excellent choice)
- **Rationale**: Embedded, fast analytics, excellent JSON support
- **Benefits**: No external dependencies, perfect for MCP context

**3. Event Format**
- **Current**: JSONL files + DuckDB tables
- **Benefits**: Durability, debuggability, flexible schema evolution
- **Architecture**: Keep JSONL as durability layer, DuckDB for querying

**4. Capture Layer Integration**
- **Current**: Sophisticated monkey-patching (keep as-is)
- **Quality**: Excellent, production-ready
- **Decision**: Minimal changes, focus on orchestration around it

### **Risk Mitigation**

**1. AST Injection Risks**
- **Risk**: Malformed notebooks, syntax errors
- **Mitigation**: Comprehensive backup, validation, error recovery
- **Fallback**: Manual import mode

**2. File Watching Complexity**
- **Risk**: Race conditions, file locking, performance
- **Mitigation**: Use proven libraries (watchdog), proper locking
- **Testing**: Comprehensive concurrency testing

**3. MCP Server Reliability**
- **Risk**: Memory leaks, connection issues, crashes
- **Mitigation**: Proper resource management, monitoring, restart logic
- **Monitoring**: Health checks and metrics

### **Performance Considerations**

**1. Event Ingestion**
- **Target**: <100ms latency from JSONL write to DuckDB availability
- **Optimization**: Batch processing, prepared statements
- **Monitoring**: Track ingestion lag and throughput

**2. Query Performance**
- **Target**: <1s response time for typical lineage queries
- **Optimization**: Proper indexing, query optimization
- **Caching**: Intelligent caching for repeated queries

**3. Memory Usage**
- **Target**: <50MB baseline memory usage
- **Optimization**: Efficient data structures, garbage collection
- **Monitoring**: Memory profiling and leak detection

## 🎯 Success Criteria

### **Functional Requirements**
- [ ] Single command startup: `pipx run hunyo-mcp-server --notebook file.py`
- [ ] Zero notebook modifications required
- [ ] Real-time event ingestion (<100ms latency)
- [ ] Comprehensive MCP query interface
- [ ] Proper error handling and recovery

### **Quality Requirements**
- [ ] <5% performance overhead on notebook execution
- [ ] <50MB memory usage baseline
- [ ] Comprehensive test coverage (>80%)
- [ ] Production-ready error handling
- [ ] Clear documentation and examples

### **Integration Requirements**
- [ ] Compatible with existing marimo workflows
- [ ] Preserves all current capture functionality
- [ ] Maintains OpenLineage compliance
- [ ] Supports both dev and prod environments

## 📋 Next Immediate Steps

1. ✅ **Create `pyproject.toml`** with proper packaging and CLI entry points
2. **Implement `config.py`** for data path management  
3. **Create basic project structure** with missing directories and `__init__.py` files
4. **Create basic `server.py`** CLI that accepts --notebook parameter
5. **Test end-to-end**: Ensure CLI can start and basic orchestration works

## 📝 Notes for Future Development

- This roadmap should be updated after each major milestone
- Current capture layer quality is excellent - preserve it
- Focus on orchestration and MCP integration
- Maintain backward compatibility with existing functionality
- Consider future multi-notebook support in architectural decisions

---

*Last Updated: 2024-12-27*  
*Phase 1 Progress: 1/4 items complete (pyproject.toml ✅)*  
*Next Review: After Phase 1 completion* 