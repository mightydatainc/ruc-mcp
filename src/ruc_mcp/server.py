"""Bare-bones MCP scaffold for Render Unto Caesar."""

from __future__ import annotations


class RucMcpServer:
    """Scaffold for an MCP server that separates LLM and procedural work."""

    def describe(self) -> str:
        """Return a static scaffold description."""
        return (
            "Scaffold MCP server for splitting semantic interpretation from "
            "deterministic execution."
        )

    def classify_request(self, request_text: str) -> dict[str, str]:
        """TODO: Classify user intent into LLM-shaped vs code-shaped work."""
        raise NotImplementedError("TODO: implement request classification")

    def execute_procedural_plan(self, plan: dict[str, object]) -> dict[str, object]:
        """TODO: Execute deterministic workflow steps and validation loops."""
        raise NotImplementedError("TODO: implement procedural execution")

    def run(self) -> None:
        """TODO: Start MCP transport and route incoming tool calls."""
        raise NotImplementedError("TODO: implement MCP runtime wiring")


def main() -> None:
    """Entrypoint scaffold for local development."""
    server = RucMcpServer()
    server.run()
