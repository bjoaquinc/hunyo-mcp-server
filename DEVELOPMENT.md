# Development Guide

This guide explains how to set up the **Hunyo MCP Server** project for development using modern Python tooling.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd hunyo-notebook-memories-mcp

# 2. Enter development environment 
hatch shell

# 3. You're ready to develop!
```

That's it! Hatch automatically reads `pyproject.toml` and sets up everything you need.

## 📋 Prerequisites

- **Python 3.8+** (3.11+ recommended)
- **Hatch** - Modern Python project manager
- **Git** - Version control

### Installing Hatch

```bash
# macOS (Homebrew)
brew install hatch

# pip (any platform)
pip install hatch

# pipx (recommended)
pipx install hatch
```

## 🏗️ Project Structure

```
hunyo-notebook-memories-mcp/
├── src/hunyo_mcp_server/          # Main package source
├── test/                          # Test files
├── schemas/                       # Database schemas and JSON schemas
├── pyproject.toml                 # Project configuration (deps, tools, metadata)
├── ROADMAP.md                     # Project roadmap and status
└── DEVELOPMENT.md                 # This file
```

## 🔧 Development Environment Setup

### 1. Enter Development Shell

```bash
# Automatically creates venv and installs all dependencies
hatch shell
```

**What this does:**
- ✅ Creates isolated virtual environment
- ✅ Installs runtime dependencies from `pyproject.toml`
- ✅ Installs development dependencies (`[dev]` extras)
- ✅ Installs the package in editable mode
- ✅ Activates the environment

### 2. Verify Installation

```bash
# Check that CLI entry point works
hunyo-mcp-server --help

# Check that package imports work
python -c "import hunyo_mcp_server; print('✅ Package imported successfully')"
```

**Note**: Some imports may fail initially since we're still implementing core modules.

## 🧪 Development Workflow

### Running Tests

```bash
# Run all tests
hatch run test

# Run with coverage
hatch run cov

# Run specific test file
hatch run pytest test/test_capture_integration.py

# Run with verbose output
hatch run pytest -v
```

### Code Quality

```bash
# Format code with Black
hatch run fmt

# Check linting with Ruff  
hatch run lint

# Fix auto-fixable lint issues
hatch run lint --fix

# Type checking with MyPy
hatch run types

# Run all quality checks
hatch run quality
```

### Development Commands

```bash
# Install development dependencies only
hatch run pip install -e .[dev]

# Run a command in the environment
hatch run python scripts/generate_schemas.py

# Open Python REPL in environment
hatch run python
```

## 🛠️ Hatch Scripts (Defined in pyproject.toml)

Add these to your `pyproject.toml` under `[tool.hatch.envs.default.scripts]`:

```toml
[tool.hatch.envs.default.scripts]
test = "pytest {args}"
cov = "pytest --cov=hunyo_mcp_server --cov-report=html --cov-report=term"
fmt = "black src/ test/"
lint = "ruff check src/ test/ {args}"
types = "mypy src/"
quality = ["fmt", "lint", "types"]
```

## 🎯 Testing Your Changes

### 1. Test Core Functionality

```bash
# Test existing capture layer
hatch run python test/test_capture_integration.py

# Test OpenLineage generation
hatch run python test/test_openlineage_generation.py
```

### 2. Test CLI Entry Point

```bash
# Test help command (should work once server.py is implemented)
hunyo-mcp-server --help

# Test with notebook (once implemented)
hunyo-mcp-server --notebook test/sample_notebook.py
```

### 3. Test Package Import

```bash
hatch run python -c "
from hunyo_mcp_server.capture.live_lineage_interceptor import enable_live_tracking
print('✅ Capture layer imports working')
"
```

## 🔄 Managing Dependencies

### Adding Dependencies

```bash
# Edit pyproject.toml directly, then:
hatch env prune    # Remove old environment
hatch shell        # Recreate with new deps
```

Example `pyproject.toml` addition:
```toml
dependencies = [
    "existing-dep>=1.0.0",
    "new-dependency>=2.0.0",  # Add this line
]
```

### Development Dependencies

```bash
# Development-only dependencies go in [project.optional-dependencies]
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "new-dev-tool>=1.0.0",  # Add development tools here
]
```

## 🐛 Troubleshooting

### Environment Issues

```bash
# Clean slate - remove environment and recreate
hatch env prune
hatch shell

# Check environment location
hatch env find

# Show environment info
hatch env show
```

### Import Errors

```bash
# Make sure package is installed in editable mode
hatch run pip install -e .

# Check that modules exist
ls -la src/hunyo_mcp_server/
```

### CLI Not Working

```bash
# Verify entry point configuration in pyproject.toml
[project.scripts]
hunyo-mcp-server = "hunyo_mcp_server.server:main"

# Reinstall in editable mode
hatch run pip install -e .
```

## 📚 Development Tips

### 1. Use Hatch Scripts for Common Tasks

Instead of remembering long commands, use the scripts defined in `pyproject.toml`:

```bash
hatch run test      # Instead of: pytest
hatch run fmt       # Instead of: black src/ test/
hatch run lint      # Instead of: ruff check src/ test/
```

### 2. Environment Management

```bash
# List all environments
hatch env show

# Use specific environment
hatch -e test run pytest    # Use test environment

# Create named environment
hatch env create docs
```

### 3. Building and Distribution

```bash
# Build wheel and sdist
hatch build

# Clean build artifacts
hatch clean

# Publish to PyPI (when ready)
hatch publish
```

## 🎯 Next Steps for New Contributors

1. **Explore the codebase**: Start with `src/capture/` to understand the existing capture layer
2. **Read ROADMAP.md**: Understand the project architecture and current status
3. **Run existing tests**: `hatch run test` to see what's working
4. **Pick a task**: Check ROADMAP.md for "Next Immediate Steps"
5. **Set up your environment**: `hatch shell` and you're ready to code!

## 🔗 Useful Links

- **Hatch Documentation**: https://hatch.pypa.io
- **PyProject.toml Guide**: https://packaging.python.org/specifications/pyproject-toml/
- **MCP Protocol**: https://spec.modelcontextprotocol.io/
- **OpenLineage**: https://openlineage.io/

---

**Questions?** Check the existing issues or create a new one in the repository. 