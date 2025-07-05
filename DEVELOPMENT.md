# Development Guide

This guide explains how to set up the **Hunyo MCP Server** project for development using modern Python tooling with **hatch-only architecture**.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd hunyo-notebook-memories-mcp

# 2. Enter development environment 
hatch shell

# 3. You're ready to develop!
```

That's it! Our modern **hatch-only** setup automatically reads `pyproject.toml` and sets up everything you need with perfect CI/local consistency.

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

**✅ Quick Verification Checklist:**

```bash
# 1. Test basic package import
hatch run python -c "import hunyo_mcp_server; print('✅ Package imported successfully')"

# 2. Test capture layer (main functionality) - UPDATED to use unified system
hatch run python -c "from capture.unified_marimo_interceptor import enable_unified_tracking; print('✅ Unified capture layer working')"

# 3. Run test suite (most important verification)
hatch run test

# 4. Test code quality tools
hatch run style

# 5. (Optional) Fix code style issues 
hatch run fmt
```

**Expected Results:**
- ✅ **Package import**: Should work
- ✅ **Capture layer**: Should work (with verbose output)  
- ✅ **Test suite**: All 70 tests should pass
- ✅ **Code style**: 405 style issues remaining (88.4% improvement from ~3500!)

**💡 Style Status**: 
- ✅ **3,095 issues automatically fixed** with `hatch run fmt` and manual cleanup
- ⚠️ **405 remaining issues** need manual attention (mostly test files and print statements)
- 🎯 **Safe to develop** - all critical style issues resolved

**⚠️ Expected to fail (not implemented yet):**
- ❌ `hunyo-mcp-server --help` (CLI not implemented)
- ❌ `import hunyo_mcp_server.tools` (MCP tools not implemented)

### 3. Getting Started Summary

**🎉 You're ready to develop when:**
- ✅ All 70 tests pass
- ✅ Capture layer imports without errors  
- ✅ Code quality checks pass
- ✅ You can create/modify test files and run them

**🔨 What you can work on:**
- ✅ **Capture system improvements** (`src/capture/`)
- ✅ **Database schemas** (`schemas/`)
- ✅ **Configuration system** (`src/hunyo_mcp_server/config.py`)
- ✅ **Test additions** (`test/`)
- 🚧 **MCP server implementation** (`src/hunyo_mcp_server/server.py` - missing)
- 🚧 **Database ingestion** (`src/hunyo_mcp_server/ingestion/` - missing)

**🎯 Quick Development Workflow:**
```bash
# 1. Make changes to code
# 2. Run tests to verify nothing broke
hatch run test

# 3. Check code quality
hatch run style

# 4. Run specific tests for your changes
hatch run test test/test_capture/test_your_module.py
```

## 🏗️ Modern CI/CD Architecture

This project uses a **streamlined hatch-only approach** that eliminates common CI/CD problems:

### ✅ **What We Fixed**
- **Eliminated tox conflicts** - No more dual environment management 
- **Fixed test isolation** - Tests use temporary directories, not polluting development
- **Resolved dependency timing** - All dependencies available during pytest collection
- **Cross-platform compatibility** - Works on Linux, macOS, Windows
- **Simplified CI workflow** - Single tool (hatch) for consistency

### 🚀 **CI Pipeline Features**
- **Multi-Python matrix** (3.10, 3.11, 3.12, 3.13) on Ubuntu, macOS, Windows
- **Comprehensive testing** - Unit, integration, style, typing checks
- **Automated dependency management** - Hatch handles everything
- **Performance optimized** - Cached dependencies, parallel execution
- **Zero configuration drift** - Local environment matches CI exactly

### 📋 **Development = CI**
Your local commands work identically in CI:
```bash
hatch run test      # Same command used in CI
hatch run style     # Same style checks as CI  
hatch run typing    # Same type checking as CI
```

## 🧪 Development Workflow

### Running Tests

The project has a comprehensive test suite with **200+ tests** covering both unit and integration testing:

```bash
# Run all tests (70 tests total)
hatch run test

# Run with coverage reporting
hatch run test-cov

# Run with verbose output  
hatch run test -v

# Run specific test file
hatch run test test/test_capture_integration.py

# Stop on first failure
hatch run test -x
```

**Test Categories:**

```bash
# Unit tests - Test individual modules
hatch run test test/test_capture/
hatch run test test/test_hunyo_mcp_server/

# Integration tests - Test component interactions  
hatch run test test/integration/

# End-to-end tests - Test complete user workflows
hatch run test test/e2e/

# Current integration tests
hatch run test test/integration/test_schema_validation_integration.py
hatch run test test/integration/test_unified_system_integration.py
hatch run test test/integration/test_real_marimo_cell_execution.py

# Meta tests (schema validation, roadmap compliance)
hatch run test test/meta/
```

### Code Quality

```bash
# Format code with Black and Ruff
hatch run fmt

# Check code style and linting (includes import ban enforcement)
hatch run style

# Type checking with MyPy
hatch run typing

# All quality checks at once
hatch run style && hatch run typing

# Example: Test import ban enforcement
echo "from src.capture.logger import get_logger" > bad_import.py
hatch run ruff check bad_import.py  # Shows: TID253 `src` is banned at the module level
rm bad_import.py
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

Current scripts available in the project:

```toml
[tool.hatch.envs.default.scripts]
test = "pytest {args:test}"
test-cov = "pytest --cov-report=term-missing --cov-config=pyproject.toml --cov=src/hunyo_mcp_server --cov=src/capture {args:test}"
typing = "mypy --install-types --non-interactive {args:src/hunyo_mcp_server src/capture test}"
style = [
    "ruff check {args:.}",
    "black --check --diff {args:.}",
]
fmt = [
    "black {args:.}",
    "ruff check --fix {args:.}",
    "style",
]
```

**Available Commands:**
- `hatch run test` - Run all tests
- `hatch run test-cov` - Run tests with coverage reporting
- `hatch run typing` - Type checking with MyPy  
- `hatch run style` - Check code style (Black + Ruff)
- `hatch run fmt` - Format code and fix style issues

## 🎯 Testing Your Changes

### 1. Test Core Functionality

**Current Status: ✅ All 70 tests passing (100% success rate)**

```bash
# Test all capture layer functionality (53 unit tests)
hatch run test test/test_capture/

# Test core integration (3 tests)  
hatch run test test/test_capture_integration.py

# Test OpenLineage generation (4 tests)
hatch run test test/test_marimo_notebook_fixtures.py -k openlineage

# Test runtime tracking (4 tests)
hatch run test test/test_marimo_notebook_fixtures.py -k runtime
```

### 2. Test Marimo Integration (Critical)

**⚠️ IMPORTANT**: Before any capture system changes, run marimo integration tests to ensure compatibility:

```bash
# Run all integration tests
hatch run test test/integration/

# Run end-to-end tests (complete user workflows)
hatch run test test/e2e/

# Run specific integration tests
hatch run test test/integration/test_schema_validation_integration.py -v
hatch run test test/integration/test_unified_system_integration.py -v
hatch run test test/integration/test_real_marimo_cell_execution.py -v

# Run with coverage for integration and e2e tests
hatch run test-cov test/integration/ test/e2e/
```

**What these tests verify**:
- ✅ Marimo notebook creation and content parsing
- ✅ DataFrame operation detection in notebook content
- ✅ Capture system integration points with marimo cells
- ✅ Notebook validation and marimo compatibility
- ✅ Complete capture pipeline functionality
- ✅ Multi-component integration workflows
- ✅ WebSocket integration with marimo communication
- ✅ Native hooks integration with marimo execution
- ✅ Performance optimization with large DataFrames
- ✅ Error recovery and graceful degradation

**When to run these tests**:
- Before committing changes to `src/capture/`
- After updating marimo integration points
- Before releasing new versions
- When debugging marimo compatibility issues
- When modifying DataFrame tracking logic

### 3. Test CLI Entry Point

**⚠️ CLI Not Yet Implemented** - The main CLI is not available until `server.py` is implemented.

```bash
# This will fail (expected) - CLI not implemented yet
hunyo-mcp-server --help  # ❌ command not found

# Alternative: Test capture modules directly in development
hatch run python -c "
from capture.live_lineage_interceptor import MarimoLiveInterceptor
print('✅ Core functionality available for development')
"
```

**When CLI is implemented, these commands will work:**
```bash
# Future: Test help command
hunyo-mcp-server --help

# Future: Test with notebook
hunyo-mcp-server --notebook test/sample_notebook.py
```

### 4. Test Package Import

```bash
# Test basic package structure
hatch run python -c "import hunyo_mcp_server; print('✅ Basic package working')"

# Test capture layer functionality - UPDATED to use unified system
hatch run python -c "from capture.unified_marimo_interceptor import enable_unified_tracking; print('✅ Unified capture layer working')"

# Test configuration system
hatch run python -c "from hunyo_mcp_server.config import get_hunyo_data_dir; print('✅ Config system working')"
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

## 🛡️ **Import Standards & Enforcement**

### ✅ **Proper Package Imports**
This project enforces **strict import standards** to prevent common CI/development issues:

```python
# ✅ CORRECT - Use proper package imports
from capture.live_lineage_interceptor import MarimoLiveInterceptor
from hunyo_mcp_server.config import get_hunyo_data_dir

# ❌ BANNED - src. imports are automatically flagged
from src.capture.live_lineage_interceptor import MarimoLiveInterceptor  # TID253 error
from src.hunyo_mcp_server.config import get_hunyo_data_dir            # TID253 error
```

### 🚫 **Why `src.` Imports Are Banned**

**Root Cause of Major Issues**: Using `from src.module` imports caused:
- **Test isolation failures** - Wrong module resolution during pytest collection 
- **CI/CD dependency conflicts** - Modules not found during test discovery
- **Development environment inconsistencies** - Different behavior locally vs CI
- **Package installation problems** - Broke when project installed in development mode

**Technical Details**: The `src.` import pattern assumes the `src/` directory is in `PYTHONPATH`, but modern Python packaging (hatch, pip install -e) creates proper package namespaces that don't include `src/` in the module path.

### 🔧 **Automatic Enforcement**

Our **ruff configuration** automatically prevents these problematic imports:

```toml
[tool.ruff.lint.flake8-tidy-imports]
banned-module-level-imports = [
    "src",          # Ban any import starting with "src."
    "src.*",        # Ban all submodules of src package  
]
```

**Error Code**: `TID253` - Shows whenever someone tries to use `src.` imports

**Resolution**: Use proper package imports based on your `pyproject.toml` package structure

## 🧪 Development Workflow

### 🚀 **CI/CD & Quality Assurance**

**Full Cross-Platform Testing Matrix**: Our CI/CD pipeline runs comprehensive tests across:
- **Python versions**: 3.10, 3.11, 3.12, 3.13
- **Operating systems**: Ubuntu, Windows, macOS  
- **Total matrix**: 12 test jobs ensuring universal compatibility

```bash
# Check all quality gates locally before pushing
hatch run style    # Black + Ruff checks (includes Windows Unicode detection)
hatch run typing   # MyPy type checking
hatch run test     # Full test suite
hatch run test-cov # Tests with coverage report
```

### 🛡️ **Windows Compatibility**

**Unicode Safety**: Automatic prevention of Windows CI failures:
- **Ruff rules** - `RUF001`, `RUF002`, `RUF003` detect problematic Unicode characters
- **CI environment** - UTF-8 encoding enforcement with `PYTHONIOENCODING=utf-8`
- **ASCII-safe output** - All CI messages use `[OK]`, `[INFO]` instead of emojis
- **Strategic exceptions** - Unicode allowed in tests, docs, and appropriate files

**Testing Windows Locally**:
```bash
# Test Unicode detection (should show warnings)
echo 'test = "✅ Unicode emoji"' > test_unicode.py
hatch run style test_unicode.py

# Current codebase should be clean
hatch run style --select RUF001,RUF002,RUF003
```

### 🔒 **Pre-commit Hooks**

**Automated Quality Enforcement**:
```bash
# Install pre-commit hooks (one-time setup)
hatch run pre-commit install

# Manual run on all files
hatch run pre-commit run --all-files
```

**Hooks Active:**
- **Auto-fixing**: Black formatting, Ruff auto-fixes, import sorting
- **Quality checks**: Ruff linting, MyPy type checking, test execution
- **Standards enforcement**: Prevents `src.` imports, ensures import structure