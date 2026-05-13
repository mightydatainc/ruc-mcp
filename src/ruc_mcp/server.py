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


@mcp.tool()
def classify_request(request_text: str) -> dict[str, object]:
    """Classify user intent into LLM-shaped vs code-shaped work.

    This is a Hello World stub that always returns a placeholder classification.

    Args:
        request_text: Plain-English description of the task to classify.
    """
    return {
        "request_text": request_text,
        "llm_steps": ["Hello World: semantic interpretation placeholder"],
        "procedural_steps": ["Hello World: deterministic execution placeholder"],
    }


@mcp.tool()
def execute_procedural_plan(steps: list[str]) -> dict[str, object]:
    """Execute deterministic workflow steps and validation loops.

    This is a Hello World stub that echoes the provided steps as completed.

    Args:
        steps: Ordered list of procedural step descriptions to execute.
    """
    return {
        "status": "ok",
        "completed_steps": steps,
        "message": "Hello World: procedural execution placeholder",
    }


@mcp.resource("ruc://description")
def server_description() -> str:
    """Expose the server's purpose as a readable resource."""
    return (
        "Scaffold MCP server for splitting semantic interpretation from "
        "deterministic execution."
    )


@mcp.prompt()
def decompose_task(task: str) -> str:
    """Generate a prompt that asks an LLM to decompose a plain-English task.

    Args:
        task: The plain-English task description to decompose.
    """
    return (
        f"You are Render Unto Caesar. Decompose the following task into steps "
        f"that should be handled by an LLM and steps that should be handled by "
        f"deterministic code.\n\nTask: {task}\n\n"
        f"Hello World: this is a placeholder decomposition prompt."
    )


class RucMcpServer:
    """Thin wrapper around the FastMCP instance for backward compatibility."""

    def describe(self) -> str:
        """Return a static scaffold description."""
        return (
            "Scaffold MCP server for splitting semantic interpretation from "
            "deterministic execution."
        )

    def classify_request(self, request_text: str) -> dict[str, object]:
        """Delegate to the FastMCP classify_request tool."""
        return classify_request(request_text)

    def execute_procedural_plan(self, plan: dict[str, object]) -> dict[str, object]:
        """Delegate to the FastMCP execute_procedural_plan tool."""
        steps: list[str] = list(plan.get("steps", []))
        return execute_procedural_plan(steps)

    def run(self) -> None:
        """Start the FastMCP server using stdio transport."""
        mcp.run(transport="stdio")


def main() -> None:
    """Entrypoint for local development."""
    server = RucMcpServer()
    server.run()
