#!/usr/bin/env python3
"""
Hunyo MCP Server - Main CLI entry point and MCP server setup.

Provides the main command-line interface for the hunyo-mcp-server that:
1. Accepts --notebook parameter
2. Sets up MCP server with tools for LLM querying
3. Orchestrates capture, ingestion, and query components
"""

import signal
import sys
from pathlib import Path

import click
from mcp.server.fastmcp import FastMCP

from hunyo_mcp_server.config import ensure_directory_structure, get_hunyo_data_dir
from hunyo_mcp_server.orchestrator import HunyoOrchestrator, set_global_orchestrator

# Import tools so they get registered with the MCP server
from hunyo_mcp_server.tools import lineage_tool, query_tool, schema_tool  # noqa: F401

# Create the FastMCP server instance
mcp = FastMCP(
    name="hunyo-mcp-server",
    description="Zero-touch notebook instrumentation for DataFrame lineage tracking",
)


@click.command()
@click.option(
    "--notebook",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the marimo notebook file to instrument",
)
@click.option(
    "--dev-mode", is_flag=True, help="Force development mode (use .hunyo in repo root)"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--standalone",
    is_flag=True,
    help="Run in standalone mode (for testing/development)",
)
def main(notebook: Path, *, dev_mode: bool, verbose: bool, standalone: bool):
    """
    Hunyo MCP Server - Zero-touch notebook instrumentation for DataFrame lineage tracking.

    Instruments the specified marimo notebook, captures execution events,
    ingests them into DuckDB, and exposes MCP tools for LLM analysis.

    Examples:
        hunyo-mcp-server --notebook analysis.py
        hunyo-mcp-server --notebook /path/to/notebook.py --verbose
    """

    # Override environment detection if requested
    if dev_mode:
        import os

        os.environ["HUNYO_DEV_MODE"] = "1"

    # Ensure data directory structure exists
    ensure_directory_structure()

    click.echo("🎯 Starting Hunyo MCP Server")
    click.echo(f"📝 Notebook: {notebook}")
    click.echo(f"📁 Data directory: {get_hunyo_data_dir()}")

    orchestrator = None

    def signal_handler(signum, _frame):
        """Handle termination signals gracefully."""
        click.echo(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        if orchestrator:
            orchestrator.stop()
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Create and start the orchestrator
        orchestrator = HunyoOrchestrator(notebook_path=notebook, verbose=verbose)

        # Set global orchestrator for MCP tools
        set_global_orchestrator(orchestrator)

        # Start all components
        orchestrator.start()

        click.echo("🚀 MCP server starting...")

        # Check if we're running in standalone mode
        if standalone:
            click.echo("📡 Running in standalone mode - waiting for connections...")
            # Keep-alive loop for standalone operation (testing/development)
            try:
                while True:
                    import time

                    time.sleep(1)
            except KeyboardInterrupt:
                raise
        else:
            # Normal MCP protocol mode (with client via stdin/stdout)
            click.echo("💬 Running in MCP protocol mode...")
            mcp.run()

    except KeyboardInterrupt:
        try:
            click.echo("\n🛑 Keyboard interrupt received...")
        except (OSError, ValueError):
            pass
    except Exception as e:
        try:
            click.echo(f"❌ Error: {e}")
        except (OSError, ValueError):
            pass
        raise
    finally:
        # Clean shutdown
        if orchestrator:
            orchestrator.stop()

        # Safe echo during shutdown (may fail if stdout is closed)
        try:
            click.echo("✅ Shutdown complete")
        except (OSError, ValueError):
            # stdout/stderr may be closed during process termination
            pass


if __name__ == "__main__":
    main()
