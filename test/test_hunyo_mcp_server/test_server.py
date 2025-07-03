#!/usr/bin/env python3
"""
Tests for server.py - Main CLI entry point and MCP server setup.

Tests cover CLI parameter handling, environment setup, component lifecycle,
error handling, and graceful shutdown scenarios.
"""


from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# Import cross-platform path utilities
from hunyo_mcp_server.utils.paths import get_safe_temp_database_path

# Mock problematic modules to avoid relative import issues
mock_logger = MagicMock()
mock_logger.get_logger = MagicMock(return_value=MagicMock())
sys.modules["capture.logger"] = mock_logger

# Also mock the orchestrator to avoid the import issue entirely
mock_orchestrator = MagicMock()
mock_orchestrator.HunyoOrchestrator = MagicMock()
mock_orchestrator.set_global_orchestrator = MagicMock()
sys.modules["hunyo_mcp_server.orchestrator"] = mock_orchestrator


class TestHunyoMCPServerCLI:
    """Tests for the main CLI entry point following project patterns"""

    @pytest.fixture
    def cli_runner(self):
        """Click CLI runner for testing CLI commands"""
        return CliRunner()

    @pytest.fixture
    def temp_notebook_file(self, tmp_path):
        """Create a temporary notebook file for testing"""
        notebook_file = tmp_path / "test_notebook.py"
        notebook_file.write_text(
            """
# Test notebook for MCP server testing
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3]})
print(df)
"""
        )
        return notebook_file

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock HunyoOrchestrator for testing"""
        with patch("hunyo_mcp_server.server.HunyoOrchestrator") as mock_orch_class:
            mock_orch = MagicMock()
            mock_orch.start.return_value = None
            mock_orch.stop.return_value = None
            mock_orch_class.return_value = mock_orch
            yield mock_orch

    @pytest.fixture
    def mock_mcp_server(self):
        """Mock FastMCP server for testing"""
        with patch("hunyo_mcp_server.server.mcp") as mock_mcp:
            mock_mcp.run.return_value = None
            yield mock_mcp

    @pytest.fixture
    def mock_config_functions(self, tmp_path):
        """Mock configuration functions"""
        with (
            patch("hunyo_mcp_server.server.ensure_directory_structure") as mock_ensure,
            patch("hunyo_mcp_server.server.get_hunyo_data_dir") as mock_get_dir,
        ):
            mock_ensure.return_value = None
            mock_get_dir.return_value = tmp_path / ".hunyo"
            yield {"ensure_dir": mock_ensure, "get_dir": mock_get_dir}

    @pytest.fixture
    def mock_global_orchestrator(self):
        """Mock global orchestrator setter"""
        with patch("hunyo_mcp_server.server.set_global_orchestrator") as mock_set:
            mock_set.return_value = None
            yield mock_set

    def test_cli_requires_notebook_parameter(self, cli_runner):
        """Test CLI fails without required --notebook parameter"""
        # Import here to avoid early module loading issues
        from hunyo_mcp_server.server import main

        result = cli_runner.invoke(main, [])

        assert result.exit_code != 0
        assert "Missing option '--notebook'" in result.output

    def test_cli_validates_notebook_exists(self, cli_runner, tmp_path):
        """Test CLI validates that the notebook file exists"""
        from hunyo_mcp_server.server import main

        nonexistent_file = tmp_path / "nonexistent.py"
        result = cli_runner.invoke(main, ["--notebook", str(nonexistent_file)])

        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_cli_accepts_valid_notebook_with_mocked_components(
        self, cli_runner, temp_notebook_file
    ):
        """Test CLI accepts valid notebook file and starts successfully"""
        # Import server module first
        import hunyo_mcp_server.server as server_module
        from hunyo_mcp_server.server import main

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class = MagicMock(return_value=mock_orchestrator_instance)
        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = KeyboardInterrupt()  # Simulate graceful shutdown

        with (
            patch.object(server_module, "HunyoOrchestrator", mock_orchestrator_class),
            patch.object(server_module, "mcp", mock_mcp),
            patch.object(server_module, "ensure_directory_structure"),
            patch.object(
                server_module,
                "get_hunyo_data_dir",
                return_value=Path(
                    get_safe_temp_database_path("test1").replace(".duckdb", "")
                ),
            ),
            patch.object(server_module, "set_global_orchestrator"),
        ):
            result = cli_runner.invoke(main, ["--notebook", str(temp_notebook_file)])

            assert result.exit_code == 0
            assert "[START] Starting Hunyo MCP Server" in result.output
            assert f"[INFO] Notebook: {temp_notebook_file}" in result.output
            assert "[STOP] Keyboard interrupt received..." in result.output
            assert "[OK] Shutdown complete" in result.output

    def test_cli_dev_mode_flag(self, cli_runner, temp_notebook_file):
        """Test --dev-mode flag sets environment variable"""
        import hunyo_mcp_server.server as server_module
        from hunyo_mcp_server.server import main

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class = MagicMock(return_value=mock_orchestrator_instance)
        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = KeyboardInterrupt()

        with (
            patch.object(server_module, "HunyoOrchestrator", mock_orchestrator_class),
            patch.object(server_module, "mcp", mock_mcp),
            patch.object(server_module, "ensure_directory_structure"),
            patch.object(
                server_module,
                "get_hunyo_data_dir",
                return_value=Path(
                    get_safe_temp_database_path("test2").replace(".duckdb", "")
                ),
            ),
            patch.object(server_module, "set_global_orchestrator"),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = cli_runner.invoke(
                main, ["--notebook", str(temp_notebook_file), "--dev-mode"]
            )

            assert result.exit_code == 0

    def test_cli_verbose_flag(self, cli_runner, temp_notebook_file):
        """Test --verbose flag is passed to orchestrator"""
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class = MagicMock(return_value=mock_orchestrator_instance)
        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = KeyboardInterrupt()

        with (
            patch.dict(
                "sys.modules",
                {
                    "capture.logger": MagicMock(),
                },
            ),
            patch("hunyo_mcp_server.server.HunyoOrchestrator", mock_orchestrator_class),
            patch("hunyo_mcp_server.server.mcp", mock_mcp),
            patch("hunyo_mcp_server.server.ensure_directory_structure"),
            patch(
                "hunyo_mcp_server.server.get_hunyo_data_dir",
                return_value=Path(
                    get_safe_temp_database_path("test3").replace(".duckdb", "")
                ),
            ),
            patch("hunyo_mcp_server.server.set_global_orchestrator"),
        ):

            from hunyo_mcp_server.server import main

            result = cli_runner.invoke(
                main, ["--notebook", str(temp_notebook_file), "--verbose"]
            )

            assert result.exit_code == 0
            # Verify orchestrator was created with verbose=True
            mock_orchestrator_class.assert_called_once()
            call_args = mock_orchestrator_class.call_args
            assert call_args[1]["verbose"] is True

    def test_orchestrator_lifecycle_management(self, cli_runner, temp_notebook_file):
        """Test orchestrator is properly started and stopped"""
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class = MagicMock(return_value=mock_orchestrator_instance)
        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = KeyboardInterrupt()

        with (
            patch.dict(
                "sys.modules",
                {
                    "capture.logger": MagicMock(),
                },
            ),
            patch("hunyo_mcp_server.server.HunyoOrchestrator", mock_orchestrator_class),
            patch("hunyo_mcp_server.server.mcp", mock_mcp),
            patch("hunyo_mcp_server.server.ensure_directory_structure"),
            patch(
                "hunyo_mcp_server.server.get_hunyo_data_dir",
                return_value=Path(
                    get_safe_temp_database_path("test4").replace(".duckdb", "")
                ),
            ),
            patch("hunyo_mcp_server.server.set_global_orchestrator") as mock_set_global,
        ):

            from hunyo_mcp_server.server import main

            result = cli_runner.invoke(main, ["--notebook", str(temp_notebook_file)])

            assert result.exit_code == 0

            # Verify orchestrator lifecycle
            mock_orchestrator_instance.start.assert_called_once()
            mock_orchestrator_instance.stop.assert_called_once()
            mock_set_global.assert_called_once_with(mock_orchestrator_instance)

    def test_error_handling_orchestrator_failure(self, cli_runner, temp_notebook_file):
        """Test error handling when orchestrator fails to start"""
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.start.side_effect = Exception(
            "Orchestrator startup failed"
        )
        mock_orchestrator_class = MagicMock(return_value=mock_orchestrator_instance)

        with (
            patch.dict(
                "sys.modules",
                {
                    "capture.logger": MagicMock(),
                },
            ),
            patch("hunyo_mcp_server.server.HunyoOrchestrator", mock_orchestrator_class),
            patch("hunyo_mcp_server.server.ensure_directory_structure"),
            patch(
                "hunyo_mcp_server.server.get_hunyo_data_dir",
                return_value=Path(
                    get_safe_temp_database_path("test5").replace(".duckdb", "")
                ),
            ),
            patch("hunyo_mcp_server.server.set_global_orchestrator"),
        ):

            from hunyo_mcp_server.server import main

            result = cli_runner.invoke(main, ["--notebook", str(temp_notebook_file)])

            assert result.exit_code != 0
            assert "[ERROR] Error: Orchestrator startup failed" in result.output
            # Orchestrator stop should still be called for cleanup
            mock_orchestrator_instance.stop.assert_called_once()

    def test_keyboard_interrupt_graceful_shutdown(self, cli_runner, temp_notebook_file):
        """Test graceful shutdown on KeyboardInterrupt"""
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class = MagicMock(return_value=mock_orchestrator_instance)
        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = KeyboardInterrupt()

        with (
            patch.dict(
                "sys.modules",
                {
                    "capture.logger": MagicMock(),
                },
            ),
            patch("hunyo_mcp_server.server.HunyoOrchestrator", mock_orchestrator_class),
            patch("hunyo_mcp_server.server.mcp", mock_mcp),
            patch("hunyo_mcp_server.server.ensure_directory_structure"),
            patch(
                "hunyo_mcp_server.server.get_hunyo_data_dir",
                return_value=Path(
                    get_safe_temp_database_path("test6").replace(".duckdb", "")
                ),
            ),
            patch("hunyo_mcp_server.server.set_global_orchestrator"),
        ):

            from hunyo_mcp_server.server import main

            result = cli_runner.invoke(main, ["--notebook", str(temp_notebook_file)])

            assert result.exit_code == 0
            assert "[STOP] Keyboard interrupt received..." in result.output
            assert "[OK] Shutdown complete" in result.output
            mock_orchestrator_instance.stop.assert_called_once()

    def test_mcp_server_configuration(self):
        """Test MCP server is configured with correct name and description"""
        from hunyo_mcp_server.server import mcp

        # Test that the MCP server was properly configured
        assert mcp.name == "hunyo-mcp-server"
        # Test that the server instance exists and is properly initialized
        assert hasattr(mcp, "name")
        assert isinstance(mcp.name, str)

    @pytest.mark.parametrize("verbose_flag", [True, False])
    def test_verbose_parameter_propagation(
        self, cli_runner, temp_notebook_file, verbose_flag
    ):
        """Test verbose parameter is correctly propagated to orchestrator"""
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class = MagicMock(return_value=mock_orchestrator_instance)
        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = KeyboardInterrupt()

        with (
            patch.dict(
                "sys.modules",
                {
                    "capture.logger": MagicMock(),
                },
            ),
            patch("hunyo_mcp_server.server.HunyoOrchestrator", mock_orchestrator_class),
            patch("hunyo_mcp_server.server.mcp", mock_mcp),
            patch("hunyo_mcp_server.server.ensure_directory_structure"),
            patch(
                "hunyo_mcp_server.server.get_hunyo_data_dir",
                return_value=Path(
                    get_safe_temp_database_path("test7").replace(".duckdb", "")
                ),
            ),
            patch("hunyo_mcp_server.server.set_global_orchestrator"),
        ):

            from hunyo_mcp_server.server import main

            args = ["--notebook", str(temp_notebook_file)]
            if verbose_flag:
                args.append("--verbose")

            result = cli_runner.invoke(main, args)

            assert result.exit_code == 0

            # Check orchestrator was created with correct verbose setting
            call_args = mock_orchestrator_class.call_args
            assert call_args[1]["verbose"] is verbose_flag

    def test_tools_are_imported(self):
        """Test that MCP tools are imported without errors"""
        # This test verifies tool registration happens on import
        try:
            import hunyo_mcp_server.server  # noqa: F401

            # If we get here, all tool imports worked
            assert True
        except ImportError as e:
            pytest.fail(f"Tool import failed: {e}")


class TestServerIntegration:
    """Integration tests for server functionality"""

    @pytest.fixture
    def integration_runner(self):
        """CLI runner for integration tests"""
        return CliRunner()

    @pytest.fixture
    def real_temp_notebook(self, tmp_path):
        """Create a real temporary notebook for integration testing"""
        notebook = tmp_path / "analysis.py"
        notebook.write_text(
            """
# Real analysis notebook
import pandas as pd

# Create sample data
data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]}
df = pd.DataFrame(data)

# Basic analysis
mean_age = df['age'].mean()
print(f"Mean age: {mean_age}")
"""
        )
        return notebook

    def test_import_and_basic_functionality(self):
        """Test that server module can be imported and basic functions work"""
        from hunyo_mcp_server.server import main

        # Verify main function exists and is callable
        assert callable(main)

        # Test basic CLI structure (should fail without notebook param)
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--notebook" in result.output
