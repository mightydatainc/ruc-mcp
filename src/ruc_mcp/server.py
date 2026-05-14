"""Render Unto Caesar MCP server built with FastMCP."""

from __future__ import annotations

from typing import Annotated, Any

import fastmcp
from pydantic import Field

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
        "must be exact."
    ),
)


@mcp.tool()
def hello_world(name: str = "World") -> str:
    """Return a Hello World greeting.

    Args:
        name: The name to greet.
    """
    return f"Hello, {name}!"


@mcp.tool(
    description=(
        "Perform a task that mixes deterministic procedural work (primarily the "
        "domain of traditional programming languages) with semantic "
        "interpretation (primarily the domain of LLMs). Use this when a "
        "user describes a fuzzy task that must be executed rigorously "
        "across records, files, or other repeatable inputs."
    ),
)
def execute_semantic_code_workflow(
    task_description: Annotated[
        str,
        Field(
            description=(
                "Plain-English description of the task the user wants performed."
            )
        ),
    ],
    context_explanation: Annotated[
        str | None,
        Field(
            description=(
                "Optional background information needed to interpret the task correctly. "
                "Use this for domain context, terminology, business rules, assumptions, etc."
            )
        ),
    ] = None,
    data_source_uris: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional indicators of one or more sources of data upon which to perform the task. "
                "Currently only accepts file URIs with absolute paths, e.g. `file:///c/users/mvol/Documents/client_list.csv`"
            )
        ),
    ] = None,
    expected_result_schema: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Optional JSON Schema describing the expected shape of the final result. "
            )
        ),
    ] = None,
    behavioral_requirements: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional non-negotiable requirements the workflow must obey, such as "
                "'process every row exactly once', 'preserve source row IDs', "
                "'do not modify input files', or 'include low-confidence cases separately'."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Perform a RUC task."""
    raise NotImplementedError


def main() -> None:
    """Entrypoint for local development."""
    mcp.run(transport="stdio")
