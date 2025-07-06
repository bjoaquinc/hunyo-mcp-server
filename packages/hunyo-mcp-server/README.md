# Hunyo MCP Server

**Zero-configuration DataFrame tracking and runtime debugging for Marimo notebooks via MCP**

A single-command orchestrator that provides automatic marimo notebook instrumentation, real-time event capture, DuckDB ingestion, and LLM-accessible query tools via the Model Context Protocol (MCP).

## 🚀 Quick Start

```bash
# Install and run in one command
pipx run hunyo-mcp-server --notebook analysis.py

# Or install globally  
pipx install hunyo-mcp-server
hunyo-mcp-server --notebook /path/to/your/notebook.py
```

**That's it!** No notebook modifications required. Your LLM assistant can now analyze DataFrame operations, performance metrics, and data lineage in real-time.

## 🎯 What It Does

**Hunyo MCP Server** automatically:

1. **📝 Instruments your notebook** - Zero-touch capture layer injection
2. **🔍 Captures execution events** - DataFrame operations, runtime metrics, errors
3. **💾 Stores in DuckDB** - Real-time ingestion with OpenLineage compliance  
4. **🤖 Exposes MCP tools** - Rich querying interface for LLM analysis

### Example Workflow

```bash
# Start the MCP server
hunyo-mcp-server --notebook my_analysis.py

# Your LLM can now ask questions like:
# "What DataFrames were created in the last run?"
# "Show me the performance metrics for the merge operations"
# "Trace the lineage of the final results DataFrame"
# "Which operations took the most memory?"
```

## 🏗️ Architecture

### Data Flow
```
Marimo Notebook → Capture Layer → JSONL Events → File Watcher → 
DuckDB Database → MCP Query Tools → LLM Analysis
```

### Storage Structure
```
# Production: ~/.hunyo/
# Development: {repo}/.hunyo/
├── events/
│   ├── runtime/           # Cell execution metrics, timing, memory
│   └── lineage/          # DataFrame operations, OpenLineage events
├── database/
│   └── lineage.duckdb    # Queryable database with rich schema
└── config/
    └── settings.yaml     # Configuration and preferences
```

## 🛠️ MCP Tools Available to LLMs

Your LLM assistant gets access to these powerful analysis tools:

### 📊 **Query Tool** - Direct SQL Analysis
```sql
-- Your LLM can run queries like:
SELECT operation, AVG(duration_ms) as avg_time 
FROM runtime_events 
WHERE success = true 
GROUP BY operation;
```

### 🔍 **Schema Tool** - Database Inspection  
- Table structures and column definitions
- Data type information and constraints
- Example queries and usage patterns
- Statistics and metadata

### 🔗 **Lineage Tool** - DataFrame Tracking
- Complete DataFrame transformation chains
- Performance metrics per operation
- Memory usage and optimization insights
- Data flow visualization

## 📋 Features

### ✅ **Zero Configuration**
- **No notebook modifications** - Automatic instrumentation
- **Smart environment detection** - Dev vs production modes
- **Automatic directory management** - Creates `.hunyo/` structure
- **One-command startup** - `pipx run hunyo-mcp-server --notebook file.py`

### ✅ **Comprehensive Tracking**
- **DataFrame operations** - Create, transform, merge, filter, groupby
- **Runtime metrics** - Execution time, memory usage, success/failure
- **OpenLineage compliance** - Standard lineage format for interoperability
- **Smart output handling** - Large objects described, not stored

### ✅ **Real-Time Analysis**
- **Background ingestion** - File watching with <100ms latency
- **Live querying** - Database updates in real-time
- **Performance monitoring** - Track operations as they happen
- **Error context** - Link DataFrame issues to execution environment

### ✅ **LLM-Friendly Interface**
- **Rich MCP tools** - Structured data access for AI assistants
- **Natural language queries** - Ask questions about your data pipeline
- **Contextual analysis** - Link performance to specific operations
- **Historical tracking** - Analyze patterns across multiple runs

## 🔧 Installation & Usage

### Prerequisites
- **Python 3.10+** (3.11+ recommended) 
- **Marimo notebooks** - Works with `.py` marimo notebook files
- **Cross-platform** - Fully compatible with Windows, macOS, and Linux

### Installation Options

```bash
# Option 1: Run directly without installation (recommended)
pipx run hunyo-mcp-server --notebook analysis.py

# Option 2: Install globally
pipx install hunyo-mcp-server
hunyo-mcp-server --notebook analysis.py

# Option 3: Development installation
git clone https://github.com/hunyo-dev/hunyo-notebook-memories-mcp
cd hunyo-notebook-memories-mcp
hatch shell
hunyo-mcp-server --notebook examples/demo.py
```

### Command-Line Options

```bash
hunyo-mcp-server --help

Options:
  --notebook PATH     Path to marimo notebook file [required]
  --dev-mode         Force development mode (.hunyo in repo root)
  --verbose, -v      Enable verbose logging
  --standalone       Run standalone (for testing/development)
  --help             Show this message and exit
```

### Usage Examples

```bash
# Basic usage
hunyo-mcp-server --notebook data_analysis.py

# Development mode with verbose logging
hunyo-mcp-server --notebook ml_pipeline.py --dev-mode --verbose

# Standalone mode (for testing)
hunyo-mcp-server --notebook test.py --standalone
```

## 📊 Example LLM Interactions

Once running, your LLM assistant can analyze your notebook with natural language:

### **Performance Analysis**
> *"Which DataFrame operations in my notebook are the slowest?"*

```sql
SELECT operation, AVG(duration_ms) as avg_time, COUNT(*) as count
FROM runtime_events 
WHERE event_type = 'dataframe_operation'
GROUP BY operation 
ORDER BY avg_time DESC;
```

### **Memory Usage Tracking**
> *"Show me memory usage patterns for large DataFrames"*

```sql
SELECT operation, input_shape, output_shape, memory_delta_mb
FROM lineage_events 
WHERE memory_delta_mb > 10
ORDER BY memory_delta_mb DESC;
```

### **Data Lineage Analysis**
> *"Trace the transformation chain for my final results DataFrame"*

The lineage tool provides complete DataFrame ancestry and transformation history with visual representation.

## 🎯 Use Cases

### **🔬 Data Science Workflows**
- Track DataFrame transformations across complex analysis pipelines
- Monitor memory usage and performance bottlenecks
- Debug data quality issues with execution context
- Analyze patterns in iterative model development

### **📈 Performance Optimization**
- Identify slow operations and memory-intensive transformations
- Compare execution metrics across different implementations
- Track improvements after optimization changes
- Monitor resource usage in production notebooks

### **🐛 Debugging & Troubleshooting**
- Link DataFrame errors to specific execution context
- Trace data flow through complex transformation chains
- Identify where data quality issues are introduced
- Understand the impact of individual operations

### **📚 Documentation & Knowledge Sharing**
- Automatic documentation of data transformation logic
- Share lineage analysis with team members
- Understand inherited notebooks and data pipelines
- Maintain data governance and compliance records

## 🔧 Development

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/hunyo-dev/hunyo-notebook-memories-mcp
cd hunyo-notebook-memories-mcp

# Set up development environment
hatch shell

# Run tests
hatch run test          # (or: hatch run test:pytest)

# Check code quality
hatch run style
hatch run typing

# Run with development notebook
hunyo-mcp-server --notebook test/fixtures/openlineage_demo_notebook.py --dev-mode
```

### Project Structure

```
src/
├── hunyo_mcp_server/           # Main MCP server package
│   ├── server.py              # CLI entry point and FastMCP setup
│   ├── orchestrator.py        # Component coordination
│   ├── config.py              # Environment detection and paths
│   ├── ingestion/             # Data pipeline components
│   │   ├── duckdb_manager.py  # Database operations
│   │   ├── event_processor.py # Event validation and transformation
│   │   └── file_watcher.py    # Real-time file monitoring
│   └── tools/                 # MCP tools for LLM access
│       ├── query_tool.py      # Direct SQL querying
│       ├── schema_tool.py     # Database inspection
│       └── lineage_tool.py    # DataFrame lineage analysis
└── capture/                   # Instrumentation layer
    ├── live_lineage_interceptor.py    # DataFrame operation capture
    ├── lightweight_runtime_tracker.py # Execution metrics tracking
    ├── native_hooks_interceptor.py    # Advanced hooking system
    └── websocket_interceptor.py       # Marimo integration
```

## 🤝 Contributing

We welcome contributions! See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development setup and guidelines.

### Quick Contribution Setup
```bash
git clone https://github.com/hunyo-dev/hunyo-notebook-memories-mcp
cd hunyo-notebook-memories-mcp
hatch shell
hatch run test  # Run test suite
```

## 📝 Requirements

```
Python >= 3.10
mcp >= 1.0.0
click >= 8.0.0
duckdb >= 0.9.0
pandas >= 2.0.0
marimo >= 0.8.0
```

## 🔗 Links

- **Documentation**: [GitHub README](https://github.com/hunyo-dev/hunyo-notebook-memories-mcp#readme)
- **Issues**: [GitHub Issues](https://github.com/hunyo-dev/hunyo-notebook-memories-mcp/issues)
- **Source Code**: [GitHub Repository](https://github.com/hunyo-dev/hunyo-notebook-memories-mcp)
- **Model Context Protocol**: [MCP Specification](https://spec.modelcontextprotocol.io/)
- **OpenLineage**: [OpenLineage.io](https://openlineage.io/)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Ready to supercharge your notebook analysis?**

```bash
pipx run hunyo-mcp-server --notebook your_notebook.py
```

Your LLM assistant is now equipped with powerful DataFrame lineage and performance analysis capabilities! 🚀 