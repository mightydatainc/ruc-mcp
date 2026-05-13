"""Render Unto Caesar MCP server built with FastMCP."""

from __future__ import annotations

import fastmcp

mcp: fastmcp.FastMCP = fastmcp.FastMCP(
    name="ruc-mcp",
    instructions=(
        "Scaffold MCP server for splitting semantic interpretation from "
        "deterministic execution."
    ),
)


@mcp.tool()
def hello_world(name: str = "World") -> str:
    """Return a Hello World greeting.

    Args:
        name: The name to greet.
    """
    return f"Hello, {name}!"


@mcp.resource("ruc://description")
def server_description() -> str:
    """Expose the server's purpose as a readable resource."""
    return (
        "Scaffold MCP server for splitting semantic interpretation from "
        "deterministic execution."
    )


def main() -> None:
    """Entrypoint for local development."""
    mcp.run(transport="stdio")
