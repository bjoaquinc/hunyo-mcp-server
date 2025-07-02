#!/usr/bin/env python3
"""
MCP Instance - Centralized FastMCP instance to avoid circular imports.

This module provides the single FastMCP instance that tools can import
without creating circular dependencies with server.py.
"""

from mcp.server.fastmcp import FastMCP

# Create the single FastMCP server instance
mcp = FastMCP(
    name="hunyo-mcp-server",
    description="Zero-touch notebook instrumentation for DataFrame lineage tracking",
)
