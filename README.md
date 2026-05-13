# ruc-mcp

An MCP server scaffold that separates LLM semantic judgment from deterministic procedural execution.

## Status

This repository currently contains a bare-bones scaffold only. Core functionality is intentionally left as TODOs.

## Scaffold layout

- `src/ruc_mcp/server.py`: `RucMcpServer` skeleton and TODO method stubs.
- `tests/test_server.py`: scaffold tests that verify placeholders and explicit TODO behavior.

## Run tests

```bash
python -m unittest discover -s tests -p "test*.py"
```
