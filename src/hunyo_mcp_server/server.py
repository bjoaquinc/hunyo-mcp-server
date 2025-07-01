#!/usr/bin/env python3
"""
Hunyo MCP Server - Main CLI entry point and MCP server setup.

Provides the main command-line interface for the hunyo-mcp-server that:
1. Accepts --notebook parameter
2. Sets up MCP server with tools for LLM querying
3. Orchestrates capture, ingestion, and query components
"""

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
def main(notebook: Path, dev_mode: bool, verbose: bool):
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

    try:
        # Create and start the orchestrator
        orchestrator = HunyoOrchestrator(notebook_path=notebook, verbose=verbose)

        # Set global orchestrator for MCP tools
        set_global_orchestrator(orchestrator)

        # Start all components
        orchestrator.start()

        click.echo("🚀 MCP server starting...")

        # Run the MCP server (this blocks until shutdown)
        mcp.run()

    except KeyboardInterrupt:
        click.echo("\n🛑 Shutting down...")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        raise
    finally:
        # Clean shutdown
        if "orchestrator" in locals():
            orchestrator.stop()
        click.echo("✅ Shutdown complete")


if __name__ == "__main__":
    main()
