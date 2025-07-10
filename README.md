# Hunyo MCP Server

**Zero-configuration DataFrame tracking and runtime debugging for multiple notebook environments via MCP**

A single-command orchestrator that provides automatic notebook instrumentation, real-time event capture, DuckDB ingestion, and LLM-accessible query tools via the Model Context Protocol (MCP). Supports Marimo notebooks with extensible architecture for Jupyter and other environments.

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

## ��️ Architecture

### Environment-Agnostic Dual-Package System

**Hunyo MCP Server** uses a **dual-package architecture** with **environment-agnostic design** for optimal deployment flexibility across multiple notebook environments:

#### **Package 1: `hunyo-mcp-server` (pipx installable)**
- **Purpose**: Global CLI tool for orchestration and data analysis
- **Installation**: `pipx install hunyo-mcp-server`
- **Dependencies**: Full MCP stack (DuckDB, OpenLineage, WebSockets)
- **Features**: Database management, MCP tools, file watching, graceful fallback

#### **Package 2: `hunyo-capture` (pip installable)**
- **Purpose**: Lightweight DataFrame instrumentation layer with multi-environment support
- **Installation**: `pip install hunyo-capture` (in notebook environment)
- **Dependencies**: Minimal (pandas only)
- **Features**: DataFrame tracking, event generation, **environment-agnostic architecture**
  - **🔧 Marimo support**: Full integration with Marimo runtime hooks
  - **📝 Jupyter support**: Extensible design for future Jupyter integration  
  - **🤖 Auto-detection**: Automatically detects and adapts to notebook environment
  - **🔄 Unified API**: Same tracking functions work across all supported environments

### Data Flow
```
Notebook Environment → hunyo-capture → JSONL Events → File Watcher → 
(Marimo/Jupyter/etc.)     ↓           (Environment-aware)
                   Auto-detects environment
                         ↓
              Unified tracking interface → DuckDB Database → MCP Query Tools → LLM Analysis
                         ↑                                          ↑
                   (pip install)                            (pipx install)
```

### Environment Isolation Benefits

**Production Setup (Recommended)**:
```bash
# Global MCP server installation
pipx install hunyo-mcp-server

# Capture layer in notebook environment  
pip install hunyo-capture
```

**Benefits**:
- ✅ **Clean separation**: MCP server isolated from notebook dependencies
- ✅ **Minimal notebook overhead**: Only lightweight capture layer installed
- ✅ **Graceful fallback**: MCP server works without capture layer
- ✅ **Easy management**: Global server, environment-specific capture
- ✅ **Environment flexibility**: Same capture layer works across Marimo, Jupyter, and future environments
- ✅ **Auto-detection**: Automatically adapts to detected notebook environment without configuration

### Graceful Fallback System

When capture layer is not available, the MCP server provides helpful guidance:

```bash
hunyo-mcp-server --notebook analysis.py
# Output: 
# [INFO] To enable notebook tracking, add this to your notebook:
# [INFO]   # Install capture layer: pip install hunyo-capture
# [INFO]   from hunyo_capture import enable_unified_tracking
# [INFO]   enable_unified_tracking()  # Auto-detects environment
```

### Environment-Agnostic Architecture

The capture layer automatically detects and adapts to different notebook environments:

```python
# Same API works across all supported environments
from hunyo_capture import enable_unified_tracking

# Auto-detects environment (Marimo, Jupyter, etc.)
enable_unified_tracking()

# Or specify environment explicitly  
enable_unified_tracking(environment='marimo')
```

**Architecture Components:**
- **Environment Detection**: Auto-identifies notebook type (Marimo, Jupyter, etc.)
- **Hook Abstractions**: Environment-specific hook management (MarimoHooks, JupyterHooks)  
- **Context Adapters**: Normalize cell execution context across environments
- **Component Factory**: Creates appropriate components for detected environment
- **Unified API**: Same tracking functions work across all environments

### Storage Structure
```
# Production: ~/.hunyo/
# Development: {repo}/.hunyo/
├── events/
│   ├── runtime/           # Cell execution metrics, timing, memory
│   ├── lineage/          # DataFrame operations, OpenLineage events  
│   └── dataframe_lineage/ # Column-level lineage and transformations
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
- **Notebook environments** - Supports multiple notebook types:
  - **Marimo notebooks** - Full support for `.py` marimo notebook files
  - **Jupyter notebooks** - Extensible architecture for future integration
  - **Auto-detection** - Automatically detects and adapts to environment
- **Cross-platform** - Fully compatible with Windows, macOS, and Linux

### Installation Options

```bash
# Option 1: MCP Server Only (Recommended)
# Install MCP server globally via pipx
pipx install hunyo-mcp-server

# Install capture layer in your notebook environment
pip install hunyo-capture

# Then in your notebook, add one line:
# from hunyo_capture import enable_unified_tracking
# enable_unified_tracking()  # Auto-detects environment (Marimo/Jupyter/etc.)

# Option 2: Quick Start (Run without installing)
pipx run hunyo-mcp-server --notebook analysis.py
# Note: Capture layer must be installed separately in notebook environment

# Option 3: All-in-One Installation (Same environment)
pip install hunyo-mcp-server hunyo-capture

# Option 4: Development installation
git clone https://github.com/hunyo-dev/hunyo-notebook-memories-mcp
cd hunyo-notebook-memories-mcp
hatch run install-packages
hunyo-mcp-server --notebook examples/demo.py
```

### Installation Scenarios

#### **🏭 Production Setup (Recommended)**
```bash
# Install MCP server in isolated environment
pipx install hunyo-mcp-server

# In your notebook environment (conda, venv, etc.)
pip install hunyo-capture

# Start MCP server
hunyo-mcp-server --notebook your_analysis.py
```

#### **🔬 Development/Testing Setup**  
```bash
# Install both packages in same environment
pip install hunyo-mcp-server hunyo-capture
hunyo-mcp-server --notebook your_analysis.py
```

#### **⚡ Graceful Fallback (MCP server only)**
```bash
# If capture layer not available, MCP server provides helpful instructions
pipx install hunyo-mcp-server
hunyo-mcp-server --notebook your_analysis.py
# Shows: "To enable tracking, install: pip install hunyo-capture"
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

# Set up development environment (installs both packages)
hatch run install-packages

# Run tests (both commands work - they use the test environment)
hatch run test          # Shorter command (delegates to test environment)
hatch run test:pytest   # Explicit command (direct test environment)

# Check code quality
hatch run style
hatch run typing

# Run with development notebook
hunyo-mcp-server --notebook test/fixtures/openlineage_demo_notebook.py --dev-mode
```

### Monorepo Package Structure

```
hunyo-notebook-memories-mcp/
├── packages/
│   ├── hunyo-mcp-server/              # MCP server package (pipx installable)
│   │   ├── pyproject.toml
│   │   ├── src/hunyo_mcp_server/
│   │   │   ├── server.py              # CLI entry point
│   │   │   ├── orchestrator.py        # Component coordination
│   │   │   ├── config.py              # Environment detection and paths
│   │   │   ├── ingestion/             # Data pipeline components
│   │   │   │   ├── duckdb_manager.py  # Database operations
│   │   │   │   ├── event_processor.py # Event validation and transformation
│   │   │   │   └── file_watcher.py    # Real-time file monitoring
│   │   │   └── tools/                 # MCP tools for LLM access
│   │   │       ├── query_tool.py      # Direct SQL querying
│   │   │       ├── schema_tool.py     # Database inspection
│   │   │       └── lineage_tool.py    # DataFrame lineage analysis
│   │   └── tests/                     # MCP server-specific tests
│   └── hunyo-capture/                 # Capture layer package (pip installable)
│       ├── pyproject.toml
│       ├── src/hunyo_capture/         # Instrumentation layer
│       │   ├── __init__.py
│       │   ├── logger.py              # Logging utilities
│       │   └── unified_marimo_interceptor.py  # DataFrame capture
│       └── tests/                     # Capture layer-specific tests
├── tests/integration/                 # Cross-package integration tests
├── schemas/                           # Shared database schemas
├── .github/workflows/
│   ├── test.yml                       # Per-package testing
│   └── test-integration.yml           # Package separation testing
└── pyproject.toml                     # Workspace configuration
```

### Package Independence

**MCP Server (`hunyo-mcp-server`)**:
- Zero dependencies on capture package
- Graceful fallback when capture not available
- Standalone CLI tool for data analysis
- Installable via pipx for global access

**Capture Layer (`hunyo-capture`)**:
- Lightweight DataFrame instrumentation
- No dependencies on MCP server
- Installable in any notebook environment
- Works with existing marimo workflows

## 🤝 Contributing

We welcome contributions! See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development setup and guidelines.

### Quick Contribution Setup
```bash
git clone https://github.com/hunyo-dev/hunyo-notebook-memories-mcp
cd hunyo-notebook-memories-mcp
hatch shell
hatch run test  # Run test suite (or: hatch run test:pytest)
```

## 📝 Requirements

### Core Dependencies

**MCP Server (`hunyo-mcp-server`)**:
```toml
# Runtime requirements
mcp >= 1.0.0              # MCP protocol implementation
click >= 8.0.0             # CLI framework
duckdb >= 0.9.0            # Database engine
pandas >= 2.0.0            # Data processing
pydantic >= 2.0.0          # Data validation
watchdog >= 3.0.0          # File monitoring
websockets >= 11.0.0       # WebSocket support
openlineage-python >= 0.28.0  # Data lineage specification
```

**Capture Layer (`hunyo-capture`)**:
```toml
# Lightweight requirements
pandas >= 2.0.0            # DataFrame operations (only dependency)
```

### Optional Dependencies

**Development & Testing**:
```toml
pytest >= 7.0.0           # Testing framework
pytest-cov >= 4.0.0       # Coverage reporting
pytest-timeout >= 2.1.0   # Test timeout management
marimo >= 0.8.0            # Marimo notebook support
```

**Development Tools**:
```toml
black >= 23.0.0            # Code formatting
ruff >= 0.1.0              # Fast linting
mypy >= 1.0.0              # Type checking
```

### Installation Dependencies

**Production Setup**:
- `pipx` for global MCP server installation
- `pip` for capture layer in notebook environments
- Python 3.10+ with virtual environment support

**Development Setup**:
- `hatch` for workspace management
- `git` for version control
- IDE with Python support (VS Code, PyCharm, etc.)

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