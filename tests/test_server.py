"""Tests for the FastMCP-backed RUC MCP server."""

import unittest

import fastmcp

from src.ruc_mcp.server import RucMcpServer, classify_request, execute_procedural_plan, hello_world, mcp


class RucMcpServerDescribeTests(unittest.TestCase):
    def test_describe_mentions_scaffold_intent(self) -> None:
        server = RucMcpServer()
        description = server.describe()

        self.assertIn("Scaffold", description)
        self.assertIn("deterministic execution", description)


class FastMcpInstanceTests(unittest.TestCase):
    def test_mcp_is_fastmcp_instance(self) -> None:
        self.assertIsInstance(mcp, fastmcp.FastMCP)

    def test_mcp_name(self) -> None:
        self.assertEqual(mcp.name, "ruc-mcp")


class HelloWorldToolTests(unittest.TestCase):
    def test_hello_world_default(self) -> None:
        result = hello_world()
        self.assertEqual(result, "Hello, World!")

    def test_hello_world_custom_name(self) -> None:
        result = hello_world(name="Caesar")
        self.assertEqual(result, "Hello, Caesar!")


class ClassifyRequestToolTests(unittest.TestCase):
    def test_returns_dict_with_expected_keys(self) -> None:
        result = classify_request("review support tickets")
        self.assertIn("request_text", result)
        self.assertIn("llm_steps", result)
        self.assertIn("procedural_steps", result)

    def test_echoes_request_text(self) -> None:
        result = classify_request("compare contracts")
        self.assertEqual(result["request_text"], "compare contracts")


class ExecuteProceduralPlanToolTests(unittest.TestCase):
    def test_returns_ok_status(self) -> None:
        result = execute_procedural_plan(["step one", "step two"])
        self.assertEqual(result["status"], "ok")

    def test_echoes_steps_as_completed(self) -> None:
        steps = ["step one", "step two"]
        result = execute_procedural_plan(steps)
        self.assertEqual(result["completed_steps"], steps)


class RucMcpServerDelegationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = RucMcpServer()

    def test_classify_request_returns_dict(self) -> None:
        result = self.server.classify_request("clean a spreadsheet")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["request_text"], "clean a spreadsheet")

    def test_execute_procedural_plan_returns_dict(self) -> None:
        result = self.server.execute_procedural_plan({"steps": ["a", "b"]})
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["completed_steps"], ["a", "b"])

    def test_execute_procedural_plan_empty_steps(self) -> None:
        result = self.server.execute_procedural_plan({"steps": []})
        self.assertEqual(result["completed_steps"], [])


if __name__ == "__main__":
    unittest.main()
