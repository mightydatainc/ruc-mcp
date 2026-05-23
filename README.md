# Render Unto Caesar (RUC) (`ruc-mcp`)

> **If you've built a machine whose whole purpose is to be less machine-like, but you still ask it to do machine-like things, you're going to have a bad time.**
>
> Foundation model developers keep training LLMs to follow user instructions more faithfully. They expend untold gigajoules of energy and eons of CPU/GPU time trying to get neural networks -- fundamentally unpredictable probabilistic beasts -- to count in sequence, execute steps in the order they're given, and refrain from getting distracted or prematurely declaring completion or outright inventing completely new un-asked-for operations.
>
> This is a strange thing to optimize for.
>
> **We already have machines that follow instructions exactly, repeatably, and tirelessly. They are called computers.**
>
> Neural networks make bad computers. Yes, neural networks *run on* computers, but that fact alone doesn't magically grant them the ability to work *as* computers -- any more so than the fact that you're made of cells automatically makes you a microbiologist.
>
> Neural nets are not bad computers because they need more reinforcement learning. They are bad computers because *they are not computers*. They are pattern-completion engines: powerful, flexible, fuzzy, associative, and astonishingly useful -- but fundamentally ill-suited to long procedural execution.
>
> This is not even unique to *artificial* neural nets. Human brains are biological neural nets, and we are terrible at executing long procedural workflows from memory. That is why we use checklists, calendars, recipe cards, cockpit procedures, laminated flowcharts, and sticky notes on refrigerators. It's called "externalized cognition". It's a superpower that's mostly unique to our species: we're smart enough to know what we're not smart about, and to use tools to patch the holes in our smarts. Externalized cognition exists precisely because neural cognition is not reliable procedure.
>
> RUC is built around that distinction: Let the neural network interpret the task. Let the computer execute it.

LLMs make a great interface for requesting tasks, but a poor engine for executing them. They are good at writing verbiage and assessing messy or ambiguous information, but poor at carrying out long, exact procedures. When an operation requires both the fuzziness of LLMs and the methodical rigor of traditional code, **Render Unto Caesar** (RUC) bridges the gap. RUC separates the work into the parts that need interpretation and the parts that need machinery, and makes them interoperate to give you the best of both worlds.

Modern LLM apps make it natural to ask for complex work in plain English. The problem is that plain-English requests often mix together things LLMs are good at with things they are famously bad at. A user might ask ChatGPT, Claude, or Cursor to review support tickets, impute missing data in a spreadsheet, or brainstorm a series of ads. Those tasks require judgment and creativity, but they also require loops, counts, state, validation, consistency, and auditability — the stuff traditional code is built for.

Most users, even engineers, don’t naturally separate those two layers when they ask for the task. They describe the outcome they want, not the architecture needed to produce it reliably. RUC does that separation for them. It identifies which parts require deterministic execution — iteration, arithmetic, validation, state, and auditability — and which parts require classification, summarization, ambiguity resolution, or freeform writing. It defines the interfaces between those parts, specifying how code-shaped work should pass data into LLM-shaped work, and how LLM-shaped results should flow back into procedural execution. That lets both kinds of work operate in concert: code provides structure, continuity, and rigor, while the LLM handles nuance where rigid rules would break down.

The result is a system that can do LLM-shaped work with the control, structure, and repeatability of procedural software.

## Core idea

Render Unto Caesar is built around a simple split:

- **Code handles** loops, state, arithmetic, validation, retries, files, progress tracking, and aggregation.
- **LLMs handle** creative writing, classification, summarization, ambiguity resolution, fuzzy matching, and other semantic decisions.

The important part is not merely that both are available. The important part is that RUC defines the boundary between them, so procedural code and LLM judgment can safely interoperate inside one workflow.

## Why this matters

As LLMs become the interface to more software, users increasingly ask them to perform tasks that are partly semantic and partly procedural. Without a system like RUC, the LLM is often asked to do everything itself: reason, loop, count, remember progress, validate outputs, and produce final results.

That is exactly where LLMs are weakest.

RUC exists to let the LLM serve as the interface and semantic engine, while giving the procedural work back to code.


## Status

This repository contains a working implementation. It demonstrates dynamic workflow generation, LLM-assisted semantic steps, and execution orchestration through FastMCP.

Baseline hardening for execution isolation is in place: the MCP server runs in a Docker container via the workspace MCP configuration.

Current development focus is on expanding reliability, test coverage, and operational maturity of the workflow pipeline.

## Project layout

- `src/ruc_mcp/server.py`: FastMCP server implementation for `ruc_execute_semantic_code_workflow`, including data loading, workflow generation, stub expansion, and execution orchestration.
- `tests/test_server.py`: tests covering current server behavior.

## Run tests

```bash
python -m unittest discover -s tests -p "test*.py"
```

## Docker image

This repository carries the Docker build recipe for the MCP server. The intended workflow is to build the image locally or in CI, then have VS Code launch the prebuilt image. This repository is not intended to contain pre-built images.

Build the image with:

```bash
docker build -t mightydatainc/ruc-mcp:local . --no-cache
```

(The --no-cache flag is optional. When specified, Docker will rebuild the image from scratch. If omitted, Docker will build atop an existing image if one is present.)

The workspace MCP configuration in `.vscode/mcp.json` expects that local tag and starts the server with `docker run --rm -i mightydatainc/ruc-mcp:local`.

This keeps MCP startup fast and predictable because VS Code only launches the container; it does not rebuild the image each time the server starts.

## Future expansions

- Add resumable execution behavior so interrupted runs can pick up from saved progress.
- Add caching for frequently requested workflows to reduce repeated generation overhead.
