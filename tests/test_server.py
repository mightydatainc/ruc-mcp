"""Tests for the FastMCP-backed RUC MCP server."""

import asyncio
import logging
import unittest
from unittest.mock import patch
from pathlib import Path

import fastmcp

from src.ruc_mcp.server import (
    ruc_execute_semantic_code_workflow,
    ruc_hello_world,
    main,
    mcp,
)


class FastMcpInstanceTests(unittest.TestCase):
    def test_mcp_is_fastmcp_instance(self) -> None:
        self.assertIsInstance(mcp, fastmcp.FastMCP)

    def test_mcp_name(self) -> None:
        self.assertEqual(mcp.name, "ruc-mcp")

    def test_mcp_instructions(self) -> None:
        instructions = mcp.instructions

        self.assertIn("deterministic procedural work", instructions)
        self.assertIn("semantic interpretation", instructions)
        self.assertIn("support tickets", instructions)
        self.assertIn("tone classification", instructions)
        self.assertIn("must be exact", instructions)

    def test_registered_tools(self) -> None:
        async def get_tool_names() -> list[str]:
            return sorted(tool.name for tool in await mcp.list_tools())

        self.assertEqual(
            asyncio.run(get_tool_names()),
            sorted(["ruc_hello_world", "ruc_execute_semantic_code_workflow"]),
        )

    def test_no_prompts_are_registered(self) -> None:
        async def get_prompt_names() -> list[str]:
            return [prompt.name for prompt in await mcp.list_prompts()]

        self.assertEqual(asyncio.run(get_prompt_names()), [])


class HelloWorldToolTests(unittest.TestCase):
    def test_hello_world_default(self) -> None:
        result = ruc_hello_world()
        self.assertEqual(result, "Hello, World!")

    def test_hello_world_custom_name(self) -> None:
        result = ruc_hello_world(name="Caesar")
        self.assertEqual(result, "Hello, Caesar!")


class ExecuteSemanticCodeWorkflowToolTests(unittest.TestCase):
    def test_returns_not_implemented_payload(self) -> None:
        with patch("src.ruc_mcp.server.logging.getLogger") as get_logger:
            result = ruc_execute_semantic_code_workflow(
                task_description="Classify support tickets by sentiment"
            )

        self.assertEqual(
            result,
            {
                "status": "not_implemented",
                "message": "execute_semantic_code_workflow is not implemented yet.",
            },
        )
        get_logger.return_value.info.assert_called_once_with(
            "execute_semantic_code_workflow started for task: %s",
            "Classify support tickets by sentiment",
        )

    def test_returns_not_implemented_payload_with_all_args(self) -> None:
        result = ruc_execute_semantic_code_workflow(
            task_description="Classify support tickets by sentiment",
            context_explanation="Tickets are from a SaaS product help desk.",
            data_source_uris=["file:///data/tickets.csv"],
            expected_result_schema={
                "type": "object",
                "properties": {"sentiment": {"type": "string"}},
            },
            behavioral_requirements=["process every row exactly once"],
        )

        self.assertEqual(result["status"], "not_implemented")


class MainEntrypointTests(unittest.TestCase):
    def test_configure_logging_sets_up_basic_logging(self) -> None:
        with (
            patch("src.ruc_mcp.server.logging.FileHandler") as file_handler,
            patch("src.ruc_mcp.server.logging.StreamHandler") as stream_handler,
            patch("src.ruc_mcp.server.logging.basicConfig") as basic_config,
        ):
            from src.ruc_mcp.server import configure_logging

            configure_logging()

        basic_config.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[file_handler.return_value, stream_handler.return_value],
            force=True,
        )

        file_handler.assert_called_once_with(
            Path("ruc-mcp.log"), mode="w", encoding="utf-8"
        )
        stream_handler.assert_called_once_with()

    def test_configure_logging_uses_env_var_for_log_file(self) -> None:
        with (
            patch.dict("os.environ", {"RUC_MCP_LOG_FILE": "custom-debug.log"}),
            patch("src.ruc_mcp.server.logging.FileHandler") as file_handler,
            patch("src.ruc_mcp.server.logging.StreamHandler"),
            patch("src.ruc_mcp.server.logging.basicConfig"),
        ):
            from src.ruc_mcp.server import configure_logging

            configure_logging()

        file_handler.assert_called_once_with(
            Path("custom-debug.log"), mode="w", encoding="utf-8"
        )

    def test_main_runs_fastmcp_over_stdio(self) -> None:
        with (
            patch("src.ruc_mcp.server.configure_logging") as configure_logging,
            patch.object(mcp, "run") as run,
        ):
            main()

        configure_logging.assert_called_once_with()
        run.assert_called_once_with(transport="stdio")


if __name__ == "__main__":
    unittest.main()
