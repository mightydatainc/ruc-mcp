"""Tests for the FastMCP-backed RUC MCP server."""

import asyncio
import unittest
from unittest.mock import patch

import fastmcp

from src.ruc_mcp.server import hello_world, main, mcp


class FastMcpInstanceTests(unittest.TestCase):
    def test_mcp_is_fastmcp_instance(self) -> None:
        self.assertIsInstance(mcp, fastmcp.FastMCP)

    def test_mcp_name(self) -> None:
        self.assertEqual(mcp.name, "ruc-mcp")

    def test_mcp_instructions(self) -> None:
        self.assertEqual(
            mcp.instructions,
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
            "execution.",
        )

    def test_only_hello_world_tool_is_registered(self) -> None:
        async def get_tool_names() -> list[str]:
            return [tool.name for tool in await mcp.list_tools()]

        self.assertEqual(asyncio.run(get_tool_names()), ["hello_world"])

    def test_no_prompts_are_registered(self) -> None:
        async def get_prompt_names() -> list[str]:
            return [prompt.name for prompt in await mcp.list_prompts()]

        self.assertEqual(asyncio.run(get_prompt_names()), [])


class HelloWorldToolTests(unittest.TestCase):
    def test_hello_world_default(self) -> None:
        result = hello_world()
        self.assertEqual(result, "Hello, World!")

    def test_hello_world_custom_name(self) -> None:
        result = hello_world(name="Caesar")
        self.assertEqual(result, "Hello, Caesar!")

class MainEntrypointTests(unittest.TestCase):
    def test_main_runs_fastmcp_over_stdio(self) -> None:
        with patch.object(mcp, "run") as run:
            main()

        run.assert_called_once_with(transport="stdio")


if __name__ == "__main__":
    unittest.main()
