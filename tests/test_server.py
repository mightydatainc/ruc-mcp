"""Tests for scaffolded RUC MCP server."""

import unittest

from src.ruc_mcp.server import RucMcpServer


class RucMcpServerScaffoldTests(unittest.TestCase):
    def test_describe_mentions_scaffold_intent(self) -> None:
        server = RucMcpServer()
        description = server.describe()

        self.assertIn("Scaffold", description)
        self.assertIn("deterministic execution", description)

    def test_todo_methods_are_explicitly_unimplemented(self) -> None:
        server = RucMcpServer()

        with self.assertRaises(NotImplementedError):
            server.classify_request("review support tickets")

        with self.assertRaises(NotImplementedError):
            server.execute_procedural_plan({"steps": []})

        with self.assertRaises(NotImplementedError):
            server.run()


if __name__ == "__main__":
    unittest.main()
