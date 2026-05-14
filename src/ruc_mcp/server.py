"""Render Unto Caesar MCP server built with FastMCP."""

from __future__ import annotations

import logging
import os
from pathlib import Path
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
        "function."
        "\n\n"
        "RUC is ideal for tasks where the semantic question is fuzzy, but the execution "
        "must be exact."
        "\n\n"
        'Note that tasks suitable to RUC might sometimes "sneak up" on you. That is, '
        "a user might ask for what seems like a straightforward LLM task, but once you "
        "start to implement it, you realize that there are hidden complexities that make "
        "it a better fit for RUC. That happens often! RUC is very flexible, and is designed "
        "to accommodate a wide range of use cases."
    ),
)


def configure_logging() -> None:
    """Write runtime logs to a fresh file on each launch and echo to stderr."""
    log_level_name = os.getenv("RUC_MCP_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    log_file_path = Path(os.getenv("RUC_MCP_LOG_FILE", "ruc-mcp.log"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


@mcp.tool()
def ruc_hello_world(name: str = "World") -> str:
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
def ruc_execute_semantic_code_workflow(
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
                "Optional indicators of one or more sources of data upon which to perform "
                "the task. Currently only accepts file URIs with absolute paths, "
                "e.g. `file:///c/users/mvol/Documents/client_list.csv`"
            )
        ),
    ] = None,
    expected_result_schema: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Optional JSON Schema describing the expected shape of the final result. "
                "If not provided, RUC will decide for itself what an appropriate result schema "
                "should be, based on the task description."
            )
        ),
    ] = None,
    behavioral_requirements: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional non-negotiable requirements the workflow must obey, and which might "
                "not be immediately obvious from the task description. Use this to specify "
                'hard constraints or stipulations, such as "Don\'t process repeated records", '
                'or "Scrub personally identifiable information from the output", or '
                '"Interpret dates as eight-digit numerical sequences in DDMMYYYY format". '
                "Not all tasks will require this field, so feel free to leave it blank if "
                "there are no special requirements."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Perform a RUC task."""
    logging.getLogger(__name__).info(
        "execute_semantic_code_workflow started for task: %s", task_description
    )
    return {
        "status": "not_implemented",
        "message": "execute_semantic_code_workflow is not implemented yet.",
    }


def main() -> None:
    """Entrypoint for local development."""
    configure_logging()
    logging.getLogger(__name__).info("Starting RUC MCP server over stdio")
    mcp.run(transport="stdio")
