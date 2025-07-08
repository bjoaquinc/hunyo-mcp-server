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

from hunyo_mcp_server.config import ensure_directory_structure, get_hunyo_data_dir
from hunyo_mcp_server.mcp_instance import mcp
from hunyo_mcp_server.orchestrator import HunyoOrchestrator, set_global_orchestrator

# Import tools so they get registered with the MCP server
from hunyo_mcp_server.tools import query_tool, schema_tool  # noqa: F401


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

    click.echo("[START] Starting Hunyo MCP Server", err=True)
    click.echo(f"[INFO] Notebook: {notebook}", err=True)
    click.echo(f"[INFO] Data directory: {get_hunyo_data_dir()}", err=True)

    orchestrator = None

    def signal_handler(signum, _frame):
        """Handle termination signals gracefully."""
        click.echo(
            f"\n[STOP] Received signal {signum}, shutting down gracefully...", err=True
        )
        if orchestrator:
            orchestrator.stop()
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Create and start the orchestrator
        click.echo(f"[HASH] Server.py received notebook path: {notebook}", err=True)
        click.echo(f"[HASH] Server.py notebook path type: {type(notebook)}", err=True)
        click.echo(
            f"[HASH] Server.py notebook path exists: {notebook.exists()}", err=True
        )
        click.echo(
            f"[HASH] Server.py notebook path absolute: {notebook.is_absolute()}",
            err=True,
        )
        click.echo(
            f"[HASH] Server.py notebook path resolved: {notebook.resolve()}", err=True
        )

        orchestrator = HunyoOrchestrator(notebook_path=notebook, verbose=verbose)

        # Set global orchestrator for MCP tools
        set_global_orchestrator(orchestrator)

        # Start all components
        orchestrator.start()

        click.echo("[START] MCP server starting...", err=True)

        # Simple marker for test parsing (bypasses rich logger formatting)
        print(  # noqa: T201
            "HUNYO_READY_MARKER: MCP_SERVER_STARTING", file=sys.stderr, flush=True
        )

        # Check if we're running in standalone mode
        if standalone:
            click.echo(
                "[INFO] Running in standalone mode - waiting for connections...",
                err=True,
            )
            # Keep-alive loop for standalone operation (testing/development)
            try:
                while True:
                    import time

                    time.sleep(1)
            except KeyboardInterrupt:
                raise
        else:
            # Normal MCP protocol mode (with client via stdin/stdout)
            click.echo("[INFO] Running in MCP protocol mode...", err=True)
            mcp.run()

    except KeyboardInterrupt:
        try:
            click.echo("\n[STOP] Keyboard interrupt received...", err=True)
        except (OSError, ValueError):
            pass
    except Exception as e:
        try:
            click.echo(f"[ERROR] Error: {e}", err=True)
        except (OSError, ValueError):
            pass
        raise
    finally:
        # Clean shutdown
        if orchestrator:
            orchestrator.stop()

        # Safe echo during shutdown (may fail if stdout is closed)
        try:
            click.echo("[OK] Shutdown complete", err=True)
        except (OSError, ValueError):
            # stdout/stderr may be closed during process termination
            pass


if __name__ == "__main__":
    main()
