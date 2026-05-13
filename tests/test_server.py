"""Tests for the FastMCP-backed RUC MCP server."""

import asyncio
import unittest
from unittest.mock import patch

import fastmcp

from src.ruc_mcp.server import hello_world, main, mcp, server_description


class FastMcpInstanceTests(unittest.TestCase):
    def test_mcp_is_fastmcp_instance(self) -> None:
        self.assertIsInstance(mcp, fastmcp.FastMCP)

    def test_mcp_name(self) -> None:
        self.assertEqual(mcp.name, "ruc-mcp")

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


class ServerDescriptionResourceTests(unittest.TestCase):
    def test_server_description_mentions_scaffold_intent(self) -> None:
        description = server_description()

        self.assertIn("Scaffold", description)
        self.assertIn("deterministic execution", description)


class MainEntrypointTests(unittest.TestCase):
    def test_main_runs_fastmcp_over_stdio(self) -> None:
        with patch.object(mcp, "run") as run:
            main()

        run.assert_called_once_with(transport="stdio")


if __name__ == "__main__":
    unittest.main()
