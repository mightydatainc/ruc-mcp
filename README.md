# Render Unto Caesar (RUC) (`ruc-mcp`)

*Render Unto Caesar (RUC)* plugs into your AI agent and gives it the ability to write and run snippets of code on an as-needed basis &mdash; snippets of code that call right back to the AI agent during their execution. It effectively melds LLMs with traditional software, allowing each part of a task to be handled by the architecture that suits it best. The result is inference, judgment, and creativity that nonetheless executes methodically and reliably across long operations, large datasets, and complex processes.

## Deterministic execution + LLM judgment in a single integrated workflow

> **If you've built a machine whose whole purpose is to be less machine-like, but you still ask it to do machine-like things, you're going to have a bad time.**
>
> Foundation model developers keep training LLMs to follow user instructions more faithfully. They expend untold gigajoules of energy and eons of CPU/GPU time trying to get neural networks &mdash; fundamentally fickle probabilistic beasts by their very nature &mdash; to execute steps in the order they're given, and to refrain from getting distracted, prematurely declaring completion, or outright inventing completely new un-asked-for operations.
>
> This is a silly thing to optimize for.
>
> **We already have machines that follow instructions exactly, repeatably, and tirelessly. They are called computers.**
>
> Neural networks make bad computers. Yes, neural networks *run on* computers, but that fact alone doesn't magically grant them the ability to work *as* computers &mdash; any more so than the fact that you're made of cells automatically makes you a microbiologist.
>
> Neural nets are not bad computers because they need more reinforcement learning. They are bad computers because *they are not computers*. They are pattern-completion engines: powerful, flexible, fuzzy, associative, and astonishingly useful &mdash; but fundamentally ill-suited to long procedural execution.
>
> This is not even unique to *artificial* neural nets. Human brains are bad at it too. That's why we have to use a pencil to perform long division. That is why we use checklists, calendars, recipe cards, laminated flowcharts, and sticky notes. It's called &ldquo;externalized cognition&rdquo;. It's a superpower that's mostly unique to our species: we're smart enough to know what we're not smart about, and to use tools to patch the holes in our smarts. Externalized cognition exists precisely because neural cognition is not reliable procedure.
>
> RUC is built around that distinction: **Let the neural network handle interpretation, judgment, and creativity. Let the computer handle execution.**

LLMs make a great interface for requesting tasks, but a poor engine for executing them. They are good at writing verbiage and assessing messy or ambiguous information, but poor at carrying out long, exact procedures. When an operation requires both the fuzziness of LLMs and the methodical rigor of traditional code, **Render Unto Caesar (RUC)** bridges the gap. RUC separates the work into the parts that need interpretation and the parts that need machinery, and makes them interoperate to give you the best of both worlds.

Modern LLM apps make it natural to ask for complex work in plain English. The problem is that plain-English requests often mix together things LLMs are good at with things they are famously bad at. A user might ask ChatGPT, Claude, or Cursor to review support tickets, impute missing data in a spreadsheet, or brainstorm a series of ads. Those tasks require judgment and creativity, but they also require loops, counts, state, validation, consistency, and auditability &mdash; the stuff traditional code is built for.

Most users, even engineers, don’t naturally separate those two layers when they ask for the task. They describe the outcome they want, not the architecture needed to produce it reliably. RUC does that separation for them. It identifies which parts require deterministic execution — iteration, arithmetic, validation, state, and auditability — and which parts require classification, summarization, ambiguity resolution, or freeform writing. It defines the interfaces between those parts, specifying how code-shaped work should pass data into LLM-shaped work, and how LLM-shaped results should flow back into procedural execution. That lets both kinds of work operate in concert: code provides structure, continuity, and rigor, while the LLM handles nuance where rigid rules would break down.

The result is a system that can do LLM-shaped work with the control, structure, and repeatability of procedural software.

### Core idea

Render Unto Caesar is built around a simple split:

- **Code handles** loops, arithmetic, validation, retries, files, progress tracking, aggregation, and state management.
- **LLMs handle** language comprehension, creative writing, classification, summarization, ambiguity resolution, fuzzy matching, and other semantic decisions.

The important part is not merely that both are available. The important part is that RUC defines the boundary between them, so procedural code and LLM judgment can safely interoperate inside one workflow.

## Who this is for

RUC is built for people who are comfortable asking for work in plain language, but still need reliable execution on real files.

In practice, that usually means:

- Product leaders and PMs working with large CSV or spreadsheet exports.
- Analysts and operations teams who run repeated cleanup, classification, and normalization tasks.
- Domain owners and data stewards who are accountable for data quality outcomes.
- Engineers and AI power users supporting non-engineering teammates in VS Code.

### Example situations in which RUC would be really handy

1. A PM needs to triage 1,700 support feedback tickets before a roadmap review.
2. A financial analyst has the transcripts of hundreds of earnings calls. The analyst needs to populate a spreadsheet detailing which company had the call, what date the call occurred on, who was on it, what key metrics were discussed, and what the values of those metrics were.
3. A regional sales manager needs to sort 7,000 free-text customer survey comments into bins based on what features of the product each customer described most prominently.
4. A national advertising executive wants to create custom copy for an ad to run on a county-by-county basis for all 3,144 counties in the United States.
5. An engineer is helping non-engineering teammates run recurring data correction workflows, involving imputation of missing data, cleanup of messy fields, etc.

## Installation Guide

### Prerequisites

You will need Docker installed to run RUC.

RUC is an [MCP](https://modelcontextprotocol.io/) server that runs inside a Docker container. If these concepts are new, these introductions can help:

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/): a communication standard that allows "hosts" (like VS Code, or your Claude Desktop executable) to load third-party tool systems (like RUC) and inform their LLMs (either cloud-based or running locally) about the capabilities that these tools offer them, and to invoke said tools upon the LLM's request.
- [Docker](https://www.docker.com/): a lightweight virtualization system that allows you to run mini-computers as simulations inside your real computer (more or less). Great for things like security, dependency management, and portability.

Install Docker using the guide for your operating system:

- [Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
- [Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
- [Linux](https://docs.docker.com/engine/install/)

### Download

Download the Docker image from:

- `ghcr.io/mighty-data-inc/ruc-mcp`

Using Docker, pull the image with:

```bash
docker pull ghcr.io/mighty-data-inc/ruc-mcp:latest
```

### Integrating With Your AI Agent

#### VS Code

Add a `.vscode/mcp.json` file to your workspace folder (or open the one already present in this repository) with the following contents:

```json
{
  "servers": {
    "Render Unto Caesar": {
      "type": "stdio",
      "command": "docker",
      "cwd": "${workspaceFolder}",
      "args": [
        "run",
        "--rm",
        "-i",
        "-e",
        "RUC_MCP_LOG_LEVEL=DEBUG",
        "-v",
        "${workspaceFolder}:/workspace",
        "mightydatainc/ruc-mcp:local"
      ]
    }
  }
}
```

VS Code will automatically detect this file and make the "Render Unto Caesar" MCP server available to GitHub Copilot and other MCP-aware extensions. The server runs via Docker, mounting your workspace folder into the container at `/workspace` so that RUC can read and write your local files.

Make sure you have [built the Docker image locally](#docker-image) before starting the server.

#### Cursor

#### Claude Desktop

#### Codex

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

## CI and release automation

- CI runs on pushes to `main` and on pull requests.
- Docker publishing to GHCR is automated on version tags matching `v*`.

To publish a release image, push a tag like:

```bash
git tag v0.1.0
git push origin v0.1.0
```

This publishes the following tags to GHCR:

- `ghcr.io/mighty-data-inc/ruc-mcp:v0.1.0`
- `ghcr.io/mighty-data-inc/ruc-mcp:0.1.0`
- `ghcr.io/mighty-data-inc/ruc-mcp:latest`

### Release checklist

Before tagging:

1. Ensure CI is green on `main`.
2. Confirm GitHub Actions has permission to write packages.
3. Confirm the package visibility plan (public or private) for GHCR.
4. Update any release notes/changelog text you want published with the tag.

Release steps:

1. Pull latest `main`.
2. Create and push a version tag (`vX.Y.Z`).
3. Wait for `Publish Docker image to GHCR` to complete.
4. Verify image tags exist in GHCR (`vX.Y.Z`, `X.Y.Z`, and `latest`).

Post-release verification:

1. Pull the new image locally.
2. Run the container with a sample MCP invocation.
3. Confirm your client config references the intended tag.

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
- Add network access
  - Needs a way to constrain volume of page loads
  - Needs a way to present pages to the LLM in a textually meaningful way
  - Needs a way to run a real browser
    - Leverage JavaScript
    - Leverage semantically significant layout details (e.g. elements with `display: none` or `opacity: 0` don't count)
    - Possibly leverage user's cookies, existing login states, etc.
- Add support for MCP sampling tool calls
  - VS Code doesn't support sampling tool calls anyway
  - Sampling tool support should be conditional
    - Structured JSON is currently conditional anyway
- Possibly use WebAssembly instead of Docker?
