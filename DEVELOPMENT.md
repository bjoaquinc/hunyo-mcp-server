# Contributing Guide

We welcome all kinds of contributions to the **Hunyo MCP Server** project. _You don't need to be an expert in MCP protocol or marimo development to help out._

## Checklist

Contributions are made through pull requests. Before sending a pull request, make sure to do the following:

- [Lint, typecheck, and format](#lint-typecheck-format) your code
- [Write tests](#tests) for your changes
- [Run tests](#tests) and check that they pass (all 107 tests should pass)
- Test [marimo integration](#marimo-integration) if you modified capture functionality
- Verify [package separation](#package-structure) works correctly

_Please reach out to the maintainers before starting work on a large contribution._ Get in touch through GitHub issues.

## Setup

Install [hatch](https://hatch.pypa.io) to manage your development environment. This project uses a **hatch-only architecture** with separated packages for optimal development experience.

### Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **Hatch** - Modern Python project manager
- **Git** - Version control

### Installation

```bash
# Install hatch
pip install hatch
# or: pipx install hatch
# or: brew install hatch

# Clone and enter development environment
git clone <repository-url>
cd hunyo-notebook-memories-mcp
hatch shell
```

### Quick Verification

```bash
# Test basic functionality (should all pass)
hatch run test:python -c "import hunyo_capture; import hunyo_mcp_server; print('✅ Packages imported')"
hatch run test:python -c "from hunyo_capture.unified_marimo_interceptor import enable_unified_tracking; print('✅ Capture system working')"
hatch run test  # Should pass all 107 tests
```

## Package Structure

This project uses separated packages for modularity:

```
hunyo-notebook-memories-mcp/
├── packages/
│   ├── hunyo-capture/              # Data capture system
│   │   ├── src/hunyo_capture/      # Capture source code
│   │   ├── tests/                  # Capture tests (24 tests)
│   │   └── pyproject.toml          # Capture package config
│   └── hunyo-mcp-server/           # MCP server implementation
│       ├── src/hunyo_mcp_server/   # Server source code
│       ├── tests/                  # Server tests (60 tests)
│       └── pyproject.toml          # Server package config
├── tests/                          # Integration tests (23 tests)
├── schemas/                        # Database and JSON schemas
└── pyproject.toml                  # Workspace configuration
```

## `hatch` commands

> [!NOTE]  
> All commands use the test environment which includes marimo and all dependencies

| Command                    | Category    | Description                                              |
| -------------------------- | ----------- | -------------------------------------------------------- |
| `hatch run test`          | Test        | 🧪 Run all tests (107 total: 24+60+23)                  |
| `hatch run test:test-capture` | Test    | 🧪 Run capture package tests (24 tests)                 |
| `hatch run test:test-mcp` | Test        | 🧪 Run MCP server tests (60 tests)                      |
| `hatch run test tests/integration/` | Test | 🧪 Run integration tests (23 tests)             |
| `hatch run test:test-cov` | Test        | 🧪 Run tests with coverage reporting                    |
| `hatch run style`         | Lint        | 🔍 Check code style (Black + Ruff)                      |
| `hatch run fmt`           | Format      | 🔧 Format code and fix auto-fixable issues              |
| `hatch run typing`        | Lint        | 🔍 Type checking with MyPy                              |
| `hatch env prune`         | Setup       | 🗑️ Remove unused environments                            |
| `hatch env show`          | Setup       | 📋 Show environment information                          |
| `hatch shell`             | Setup       | 🐚 Enter development shell                               |

## Lint, Typecheck, Format

**All quality checks.**

```bash
hatch run style && hatch run typing
```

**Code formatting and auto-fixes.**

```bash
hatch run fmt
```

**Individual checks.**

<table>
  <tr>
    <th>Check Type</th>
    <th>Command</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Style</td>
    <td><code>hatch run style</code></td>
    <td>Black formatting + Ruff linting</td>
  </tr>
  <tr>
    <td>Types</td>
    <td><code>hatch run typing</code></td>
    <td>MyPy type checking</td>
  </tr>
  <tr>
    <td>Format</td>
    <td><code>hatch run fmt</code></td>
    <td>Auto-fix formatting issues</td>
  </tr>
</table>

## Tests

We have package unit tests, integration tests, and marimo compatibility tests. All changes should be accompanied by appropriate tests.

**Current Status: ✅ All 107 tests passing**
- **Capture package**: 24/24 tests
- **MCP server package**: 60/60 tests  
- **Integration tests**: 23/23 tests

### Run All Tests

```bash
hatch run test
```

This runs the complete test suite. For faster development cycles, run specific test categories:

### Package Tests

<table>
  <tr>
    <th>Package</th>
    <th>Command</th>
    <th>Tests</th>
  </tr>
  <tr>
    <td>Capture</td>
    <td><code>hatch run test:test-capture</code></td>
    <td>24 tests</td>
  </tr>
  <tr>
    <td>MCP Server</td>
    <td><code>hatch run test:test-mcp</code></td>
    <td>60 tests</td>
  </tr>
  <tr>
    <td>Integration</td>
    <td><code>hatch run test tests/integration/</code></td>
    <td>23 tests</td>
  </tr>
</table>

### Specific Test Patterns

```bash
# Run specific test file
hatch run test:pytest packages/hunyo-capture/tests/test_unified_marimo_interceptor.py

# Run tests matching pattern
hatch run test:pytest -k "test_marimo"        # All marimo-related tests
hatch run test:pytest -k "test_dataframe"     # All DataFrame tests
hatch run test:pytest -k "integration"        # All integration tests

# Run with coverage
hatch run test:test-cov packages/hunyo-capture/tests/
```

### Debugging Tests

<table>
  <tr>
    <th>Debugging Need</th>
    <th>Command</th>
  </tr>
  <tr>
    <td>Stop on first failure</td>
    <td><code>hatch run test -x</code></td>
  </tr>
  <tr>
    <td>Verbose output</td>
    <td><code>hatch run test -v</code></td>
  </tr>
  <tr>
    <td>Show print statements</td>
    <td><code>hatch run test:pytest -s</code></td>
  </tr>
  <tr>
    <td>Drop into debugger</td>
    <td><code>hatch run test:pytest --pdb</code></td>
  </tr>
  <tr>
    <td>Short traceback</td>
    <td><code>hatch run test --tb=short</code></td>
  </tr>
</table>

### Marimo Integration

**✅ Marimo Always Available**: All marimo integration tests run without conditional skips.

Critical integration tests to run when modifying capture functionality:

```bash
# Core system integration (7 tests)
hatch run test tests/integration/test_unified_system_integration.py

# Real marimo execution with Playwright (1 test)
hatch run test tests/integration/test_real_marimo_cell_execution.py

# Schema validation (2 tests)
hatch run test tests/integration/test_schema_validation_integration.py
```

**What these verify:**
- ✅ Marimo hook installation and execution
- ✅ DataFrame operation detection in real marimo cells  
- ✅ WebSocket communication with marimo server
- ✅ Complete capture pipeline functionality

## Environment Management

### When Dependencies Change

```bash
# Recommended: Clean all environments
hatch env prune
hatch run test    # Recreates environment with new dependencies
```

### Alternative: Target Specific Environment

```bash
# Remove just test environment
hatch env remove test
hatch run test    # Recreates automatically
```

### Environment Information

```bash
# Show all environments
hatch env show

# Find environment location
hatch env find

# List environments in ASCII table
hatch env show --ascii
```

## Package Development

### Adding Dependencies

**For Package-Specific Dependencies:**

```bash
# Edit the specific package's pyproject.toml
cd packages/hunyo-capture          # or packages/hunyo-mcp-server
# Edit pyproject.toml dependencies, then:
hatch run test:pip install -e .
```

**For Test Environment Dependencies:**

```bash
# Edit main pyproject.toml [tool.hatch.envs.test] section
# Then recreate environment:
hatch env prune
hatch run test    # Recreates with new dependencies
```

### Package Import Standards

**✅ Correct Package Imports:**

```python
# Between packages
from hunyo_capture.unified_marimo_interceptor import enable_unified_tracking
from hunyo_mcp_server.config import get_hunyo_data_dir

# Within packages  
from hunyo_capture.logger import get_logger
from hunyo_mcp_server.ingestion.event_processor import EventProcessor
```

**❌ Banned Imports:**

```python
# These will trigger TID253 error
from src.capture.interceptor import MarimoInterceptor
from src.hunyo_mcp_server.config import get_config
```

## Development Workflow

### Quick Development Cycle

```bash
# 1. Make changes to packages/hunyo-capture/ or packages/hunyo-mcp-server/

# 2. Run relevant tests
hatch run test:test-capture        # If you changed capture code
hatch run test:test-mcp           # If you changed MCP server code  
hatch run test tests/integration/ # If you changed integration points

# 3. Run quality checks
hatch run style && hatch run typing

# 4. Full test suite before commit
hatch run test
```

### Testing Specific Functionality

<table>
  <tr>
    <th>Functionality</th>
    <th>Command</th>
  </tr>
  <tr>
    <td>Marimo integration</td>
    <td><code>hatch run test:pytest -k "marimo"</code></td>
  </tr>
  <tr>
    <td>DataFrame tracking</td>
    <td><code>hatch run test:pytest -k "dataframe"</code></td>
  </tr>
  <tr>
    <td>Database operations</td>
    <td><code>hatch run test:pytest -k "database"</code></td>
  </tr>
  <tr>
    <td>Schema validation</td>
    <td><code>hatch run test:pytest -k "schema"</code></td>
  </tr>
</table>

## Troubleshooting

### Environment Issues

```bash
# Clean slate - remove all environments and recreate
hatch env prune
hatch run test

# Check what's installed in test environment
hatch run test:pip list | grep hunyo
```

### Import Errors

```bash
# Reinstall packages in development mode
hatch run test:pip install -e ./packages/hunyo-mcp-server
hatch run test:pip install -e ./packages/hunyo-capture
```

### Marimo Integration Issues

```bash
# Verify marimo is available
hatch run test:python -c "import marimo; print(f'Marimo version: {marimo.__version__}')"

# Test hooks directly
hatch run test:python -c "
from hunyo_capture.unified_marimo_interceptor import enable_unified_tracking
tracker = enable_unified_tracking()
print(f'Tracker active: {tracker.interceptor_active}')
"
```

### Test Failures

```bash
# Debug specific failing test
hatch run test:pytest path/to/test.py::test_name -v -s

# Run only failed tests from last run
hatch run test:pytest --lf

# Get minimal traceback for cleaner output
hatch run test --tb=short
```

## CI/CD Compatibility

Your local commands work identically in CI:

```bash
# These are the exact commands used in CI
hatch run test      # Full test matrix
hatch run style     # Style checks  
hatch run typing    # Type checking
```

**CI Matrix:**
- **Python versions**: 3.10, 3.11, 3.12, 3.13
- **Operating systems**: Ubuntu, Windows, macOS
- **Package separation**: Both packages tested independently and together
- **Total**: 12 test jobs ensuring universal compatibility

## Your First PR

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the package structure
3. **Add tests** for your changes
4. **Run the checklist**:
   ```bash
   hatch run test           # All 107 tests should pass
   hatch run style          # Style checks should pass
   hatch run typing         # Type checks should pass
   ```
5. **Test marimo integration** if you modified capture functionality:
   ```bash
   hatch run test tests/integration/test_unified_system_integration.py
   ```
6. **Submit your PR** with a clear description of changes

## Useful Links

- **Hatch Documentation**: https://hatch.pypa.io
- **MCP Protocol**: https://spec.modelcontextprotocol.io/
- **OpenLineage**: https://openlineage.io/
- **Marimo**: https://marimo.io/

---

**Questions?** Open an issue or start a discussion in the repository.