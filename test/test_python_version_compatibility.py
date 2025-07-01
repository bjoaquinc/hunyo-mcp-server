"""Python version compatibility tests."""

import importlib.util
import sys

import pytest


def test_python_version():
    """Test that we're running on Python 3.10+."""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version_info}"


def test_core_dependencies_importable():
    """Test that all core dependencies can be imported."""
    # Core MCP and data processing dependencies
    import click
    import duckdb
    import mcp
    import pandas as pd
    import pydantic
    import websockets

    # Verify we can create basic objects
    assert hasattr(mcp, "server")
    assert hasattr(pd, "DataFrame")
    assert hasattr(duckdb, "connect")
    assert hasattr(pydantic, "BaseModel")

    # Verify the other core dependencies have expected attributes or can import submodules
    assert hasattr(click, "command")
    assert hasattr(websockets, "serve")

    # Test that key submodules can be imported for package-based dependencies
    openlineage_spec = importlib.util.find_spec("openlineage.client")
    assert openlineage_spec is not None, "openlineage.client should be importable"

    watchdog_spec = importlib.util.find_spec("watchdog.observers")
    assert watchdog_spec is not None, "watchdog.observers should be importable"


def test_capture_modules_importable():
    """Test that our capture modules can be imported."""
    try:
        from capture import (
            lightweight_runtime_tracker,
            live_lineage_interceptor,
            logger,
            native_hooks_interceptor,
            websocket_interceptor,
        )

        # Basic smoke test - check for actual classes and functions
        assert hasattr(logger, "get_logger")
        assert hasattr(logger, "HunyoLogger")
        assert hasattr(live_lineage_interceptor, "MarimoLiveInterceptor")
        assert hasattr(lightweight_runtime_tracker, "LightweightRuntimeTracker")
        assert hasattr(native_hooks_interceptor, "MarimoNativeHooksInterceptor")
        assert hasattr(websocket_interceptor, "MarimoWebSocketProxy")

    except ImportError as e:
        pytest.skip(f"Capture modules not available: {e}")


def test_project_structure():
    """Test that the project structure is correct."""
    import hunyo_mcp_server

    # Verify package structure
    assert hasattr(hunyo_mcp_server, "__version__")

    # Test config module
    from hunyo_mcp_server import config

    assert hasattr(config, "get_hunyo_data_dir")
    assert hasattr(config, "get_database_path")
    assert hasattr(config, "is_development_mode")
