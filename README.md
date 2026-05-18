# Render Unto Caesar (`ruc-mcp`)

LLMs make a great interface for requesting tasks, but a poor engine for executing them. They are good at assessing messy, ambiguous information, but poor at carrying out long, exact procedures. When an operation requires both the fuzziness of LLMs and the methodical rigor of traditional code, **Render Unto Caesar** separates the work into the parts that need interpretation and the parts that need machinery, and makes them interoperate to give you the best of both worlds.

Modern LLM apps make it natural to ask for complex work in plain English. The problem is that plain-English requests often mix together things LLMs are good at with things they are famously bad at. A user might ask ChatGPT, Claude, or Cursor to review support tickets, compare contracts, or clean a messy spreadsheet. Those tasks require semantic judgment, but they also require loops, counts, state, validation, consistency, and auditability — the stuff traditional code is built for.

Most users, even engineers, don’t naturally separate those two layers when they ask for the task. They describe the outcome they want, not the architecture needed to produce it reliably. RUC does that separation for them. It identifies which parts require deterministic execution — iteration, arithmetic, validation, state, and auditability — and which parts require classification, summarization, or ambiguity resolution. It defines the interfaces between those parts, specifying how code-shaped work should pass data into LLM-shaped work, and how LLM-shaped results should flow back into procedural execution. That lets both kinds of work operate in concert: code provides structure, continuity, and rigor, while the LLM handles nuance where rigid rules would break down.

The result is a system that can do LLM-shaped work with the control, structure, and repeatability of procedural software.

## Project status

`ruc-mcp` is currently in the concept/prototype stage.

The repository now includes an end-to-end prototype flow in which RUC generates workflow code, fills semantic stubs, and executes the workflow. The current execution path is still an in-process proof of concept and is not yet production-hardened.

## Core idea

Render Unto Caesar is built around a simple split:

- **Code handles** loops, state, arithmetic, validation, retries, files, progress tracking, and aggregation.
- **LLMs handle** classification, summarization, ambiguity resolution, fuzzy matching, and other semantic decisions.

The important part is not merely that both are available. The important part is that RUC defines the boundary between them, so procedural code and LLM judgment can safely interoperate inside one workflow.

## Why this matters

As LLMs become the interface to more software, users increasingly ask them to perform tasks that are partly semantic and partly procedural. Without a system like RUC, the LLM is often asked to do everything itself: reason, loop, count, remember progress, validate outputs, and produce final results.

That is exactly where LLMs are weakest.

RUC exists to let the LLM serve as the interface and semantic engine, while giving the procedural work back to code.


## Status

This repository contains a working prototype, not just a scaffold. It demonstrates dynamic workflow generation, LLM-assisted semantic steps, and execution orchestration through FastMCP.

Hardening work is still pending. In particular, generated code execution should move from the current in-process path to an isolated Docker-based sandbox with strict resource and timeout controls.

## Scaffold layout

- `src/ruc_mcp/server.py`: FastMCP server implementation for `ruc_execute_semantic_code_workflow`, including data loading, workflow generation, stub expansion, and execution orchestration.
- `logs/temp_auto_generated_workflow.py`: debug artifact written at runtime with generated workflow code.
- `tests/test_server.py`: tests covering server behavior in the current prototype scope.

## Run tests

```bash
python -m unittest discover -s tests -p "test*.py"
```

## Docker image

This repository now carries the Docker build recipe for the MCP server. The intended workflow is to build the image locally or in CI, then have VS Code launch the prebuilt image. Do not check a built image into the repository.

Build the image with:

```bash
docker build -t mightydatainc/ruc-mcp:local .
```

The workspace MCP configuration in `.vscode/mcp.json` expects that local tag and starts the server with `docker run --rm -i mightydatainc/ruc-mcp:local`.

This keeps MCP startup fast and predictable because VS Code only launches the container; it does not rebuild the image each time the server starts.
