"""Minimal smoke tests for the FastMCP-backed RUC MCP server."""

import asyncio
import logging
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import fastmcp

from src.ruc_mcp import server


class FastMcpInstanceTests(unittest.TestCase):
    def test_mcp_is_fastmcp_instance(self) -> None:
        self.assertIsInstance(server.mcp, fastmcp.FastMCP)

    def test_mcp_name(self) -> None:
        self.assertEqual(server.mcp.name, "ruc-mcp")

    def test_registered_tools(self) -> None:
        async def get_tool_names() -> list[str]:
            return sorted(tool.name for tool in await server.mcp.list_tools())

        self.assertEqual(
            asyncio.run(get_tool_names()),
            ["ruc_execute_semantic_code_workflow"],
        )


class ExecuteSemanticCodeWorkflowSmokeTests(unittest.TestCase):
    def test_success_path_without_data_sources(self) -> None:
        mock_ctx = AsyncMock()

        with (
            patch(
                "src.ruc_mcp.server._write_workflow",
                new=AsyncMock(
                    return_value={
                        "pycode": "async def execute_workflow(ctx):\n    return {'ok': True}",
                        "implementation_strategy": "smoke strategy",
                    }
                ),
            ),
            patch(
                "src.ruc_mcp.server._replace_all_stubs_with_implementations",
                new=AsyncMock(side_effect=lambda ctx, pycode, convo: pycode),
            ),
            patch(
                "src.ruc_mcp.server._execute_workflow_code",
                new=AsyncMock(return_value={"ok": True}),
            ),
        ):
            result = asyncio.run(
                server.ruc_execute_semantic_code_workflow(
                    task_description="Do a simple task",
                    ctx=mock_ctx,
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], {"ok": True})
        self.assertEqual(result["implementation_strategy"], "smoke strategy")

    def test_data_sources_trigger_exploration(self) -> None:
        mock_ctx = AsyncMock()

        with (
            patch(
                "src.ruc_mcp.server._explore_data",
                new=AsyncMock(return_value="exploration report"),
            ) as explore_data,
            patch(
                "src.ruc_mcp.server._write_workflow",
                new=AsyncMock(
                    return_value={
                        "pycode": "async def execute_workflow(ctx):\n    return {'ok': True}",
                        "implementation_strategy": "with data",
                    }
                ),
            ),
            patch(
                "src.ruc_mcp.server._replace_all_stubs_with_implementations",
                new=AsyncMock(side_effect=lambda ctx, pycode, convo: pycode),
            ),
            patch(
                "src.ruc_mcp.server._execute_workflow_code",
                new=AsyncMock(return_value={"ok": True}),
            ),
        ):
            result = asyncio.run(
                server.ruc_execute_semantic_code_workflow(
                    task_description="Process a data source",
                    ctx=mock_ctx,
                    data_sources="/workspace/data.csv",
                )
            )

        self.assertEqual(result["status"], "success")
        explore_data.assert_awaited_once()

    def test_repair_path_retries_after_failure(self) -> None:
        mock_ctx = AsyncMock()

        with (
            patch(
                "src.ruc_mcp.server._write_workflow",
                new=AsyncMock(
                    return_value={
                        "pycode": "async def execute_workflow(ctx):\n    return {'ok': True}",
                        "implementation_strategy": "repair strategy",
                    }
                ),
            ),
            patch(
                "src.ruc_mcp.server._replace_all_stubs_with_implementations",
                new=AsyncMock(side_effect=lambda ctx, pycode, convo: pycode),
            ),
            patch(
                "src.ruc_mcp.server._execute_workflow_code",
                new=AsyncMock(side_effect=[RuntimeError("boom"), {"ok": True}]),
            ),
            patch(
                "src.ruc_mcp.server._repair_workflow_code",
                new=AsyncMock(
                    return_value="async def execute_workflow(ctx):\n    return {'ok': True}"
                ),
            ) as repair,
        ):
            result = asyncio.run(
                server.ruc_execute_semantic_code_workflow(
                    task_description="Task that requires one repair",
                    ctx=mock_ctx,
                )
            )

        self.assertEqual(result["status"], "success")
        repair.assert_awaited_once()


class WorkflowGenerationParsingTests(unittest.TestCase):
    def test_write_workflow_parses_python_block_and_strategy(self) -> None:
        mock_ctx = AsyncMock()
        mock_ctx.sample = AsyncMock(
            side_effect=[
                SimpleNamespace(text="READY TO PROCEED"),
                SimpleNamespace(
                    text=(
                        "Plan first.\n"
                        "```python\n"
                        "async def execute_workflow(ctx):\n"
                        "    return {'status': 'ok'}\n"
                        "```"
                    )
                ),
                SimpleNamespace(
                    text="Uses procedural orchestration with semantic callbacks."
                ),
            ]
        )

        result = asyncio.run(server._write_workflow(mock_ctx, ["TASK:\n\nTest"]))

        self.assertIn("import fastmcp", result["pycode"])
        self.assertIn("async def execute_workflow(ctx)", result["pycode"])
        self.assertEqual(
            result["implementation_strategy"],
            "Uses procedural orchestration with semantic callbacks.",
        )

    def test_write_workflow_retries_and_fails_for_missing_python_block(self) -> None:
        mock_ctx = AsyncMock()

        side_effects = [SimpleNamespace(text="READY TO PROCEED")]
        for _ in range(5):
            side_effects.extend(
                [
                    SimpleNamespace(text="Here is analysis but no code block."),
                    SimpleNamespace(text="Short strategy text."),
                ]
            )
        mock_ctx.sample = AsyncMock(side_effect=side_effects)

        with self.assertRaisesRegex(
            ValueError,
            "failed to write any initial workflow code after 5 attempts",
        ):
            asyncio.run(server._write_workflow(mock_ctx, ["TASK:\n\nTest"]))


class MainEntrypointTests(unittest.TestCase):
    def test_configure_logging_sets_up_basic_logging(self) -> None:
        with (
            patch("src.ruc_mcp.server.logging.FileHandler") as file_handler,
            patch("src.ruc_mcp.server.logging.StreamHandler") as stream_handler,
            patch("src.ruc_mcp.server.logging.basicConfig") as basic_config,
        ):
            server.configure_logging()

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

    def test_main_runs_fastmcp_over_stdio(self) -> None:
        with (
            patch("src.ruc_mcp.server.configure_logging") as configure_logging,
            patch.object(server.mcp, "run") as run,
        ):
            server.main()

        configure_logging.assert_called_once_with()
        run.assert_called_once_with(transport="stdio")


if __name__ == "__main__":
    unittest.main()
