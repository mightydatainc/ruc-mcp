"""Render Unto Caesar MCP server built with FastMCP."""

from __future__ import annotations

import fastmcp

mcp: fastmcp.FastMCP = fastmcp.FastMCP(
    name="ruc-mcp",
    instructions=(
        "Use this server when a user asks for a task that mixes deterministic "
        "procedural work with semantic interpretation. RUC is appropriate when "
        "the task involves code-shaped work such as iteration, counting, sorting, "
        "validation, aggregation, state tracking, or repeatable file processing, "
        "but also contains LLM-shaped work such as classification, summarization, "
        "fuzzy matching, tone analysis, relevance judgment, or ambiguity resolution. "
        "\n\n"
        "For example, if the user asks to review a collection of support tickets "
        "and count how many are angry, frustrated, neutral, or positive, do not try "
        "to keep the whole process in conversational memory. Use RUC to run a "
        "procedural workflow that loops over the tickets and aggregates counts, "
        "while delegating only the tone classification step to an LLM-style semantic "
        "function. "
        "\n\n"
        "Prefer RUC for tasks where the semantic question is fuzzy, but the execution "
        "must be exact. Do not use RUC for simple one-off questions, ordinary chat, "
        "or tasks that are purely semantic with no need for reliable procedural "
        "execution."
    ),
)


@mcp.tool()
def hello_world(name: str = "World") -> str:
    """Return a Hello World greeting.

    Args:
        name: The name to greet.
    """
    return f"Hello, {name}!"


def main() -> None:
    """Entrypoint for local development."""
    mcp.run(transport="stdio")
