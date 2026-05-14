"""Render Unto Caesar MCP server built with FastMCP."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import traceback
from typing import Annotated, Any, cast
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import fastmcp
from mcp.types import SamplingMessage, TextContent
from pydantic import Field

mcp: fastmcp.FastMCP = fastmcp.FastMCP(
    name="ruc-mcp",
    instructions=(
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
        "function."
        "\n\n"
        "RUC is ideal for tasks where the semantic question is fuzzy, but the execution "
        "must be exact."
        "\n\n"
        'Note that tasks suitable to RUC might sometimes "sneak up" on you. That is, '
        "a user might ask for what seems like a straightforward LLM task, but once you "
        "start to implement it, you realize that there are hidden complexities that make "
        "it a better fit for RUC. That happens often! RUC is very flexible, and is designed "
        "to accommodate a wide range of use cases."
    ),
)


def configure_logging() -> None:
    """Write runtime logs to a fresh file on each launch and echo to stderr."""
    log_level_name = os.getenv("RUC_MCP_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    log_file_path = Path(os.getenv("RUC_MCP_LOG_FILE", "ruc-mcp.log"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _extract_python_code_block(response_text: str) -> str:
    """Extract the last python fenced code block from a model response."""
    python_blocks = re.findall(
        r"```python\s*(.*?)```", response_text, flags=re.IGNORECASE | re.DOTALL
    )
    if python_blocks:
        return python_blocks[-1].strip()

    generic_blocks = re.findall(r"```\s*(.*?)```", response_text, flags=re.DOTALL)
    if generic_blocks:
        return generic_blocks[-1].strip()

    raise ValueError("Sampling response did not include an executable code block.")


async def _load_data_from_uri(uri: str, ctx: fastmcp.Context) -> list[dict[str, Any]]:
    """Load data from a given URI and return it as a list of records."""
    logger = logging.getLogger(__name__)
    logger.info("Loading data from URI: %s", uri)

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme != "file":
        logger.error("Unsupported URI scheme in %s", uri)
        raise ValueError(f"Unsupported URI scheme in {uri}")

    data_str = ""

    try:
        # Canonical Windows URI: file:///C:/path/to/file.csv
        # Also preserves compatibility with file://C:/path style during transition.
        if parsed_uri.netloc and parsed_uri.netloc != "localhost":
            if len(parsed_uri.netloc) == 2 and parsed_uri.netloc[1] == ":":
                file_uri_path = f"/{parsed_uri.netloc}{parsed_uri.path}"
            else:
                file_uri_path = f"//{parsed_uri.netloc}{parsed_uri.path}"
        else:
            file_uri_path = parsed_uri.path

        file_path = Path(url2pathname(unquote(file_uri_path)))

        with open(file_path, "r", encoding="utf-8") as f:
            data_str = f.read()

    except Exception as e:
        logger.error("Failed to read data from %s: %s", uri, e)
        raise ValueError(f"Could not read data from {uri}: {e}")

    SNIPPET_LENGTH = 10000
    data_str_preview = (
        data_str[:SNIPPET_LENGTH] + "..."
        if len(data_str) > SNIPPET_LENGTH
        else data_str
    )

    system_prompt = """
The user will copy-paste a snippet from the beginning of some kind of data file. We don't know 
a priori what format it's in or what its structure or contents are; hopefully this will become
obvious once the client presents the data.

I want this data restructured into a JSON list of dicts.

Here's the catch, though: I don't want you to restructure it yourself. It's way too much data,
and I don't want you to lose your place or get confused. Or to burn too many tokens trying to
analyze it as an LLM, for that matter.

Instead, I want you to write a Python function called `restructure_data`. This function takes
a string (the full contents of the user's data file), and returns a list of dicts. 

The Python environment you'll be running in is a default Python 3.11 install. You can import
default packages like `json` or `csv`, but you don't have access to fancy advanced tools like
dataframes. Also, it's running in a restricted environment, so you can't access the network or
the filesystem (not that you should need such things for a data restructuring operation).

You should talk through your planned implementation first, to organize your plan and to
strategize for the best approach to this problem.

When you're ready to write the function, do it inside of a triple-backticked block labeled
"```python". The contents of this block will be parsed and executed in the Python sandbox.
In order to make it easy for me to find this block in your response, please make this block
be the last thing you say, with no further requests or commentary after its closing 
triple-backtick delimiter.
"""

    MAX_ATTEMPTS = 5
    MAX_TOKENS = 15000
    last_error = ""
    last_stage = ""
    last_traceback = ""

    convo = [
        SamplingMessage(
            role="user",
            content=TextContent(type="text", text=data_str_preview),
        )
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        error_stage = "sampling"
        try:
            sample_result = await ctx.sample(
                messages=convo,
                system_prompt=system_prompt,
                max_tokens=MAX_TOKENS,
            )

            error_stage = "sampling_response"
            if not sample_result.text:
                raise ValueError("Sampling returned no text to execute.")

            convo.append(
                SamplingMessage(
                    role="assistant",
                    content=TextContent(type="text", text=sample_result.text),
                )
            )

            error_stage = "extract"
            generated_code = _extract_python_code_block(sample_result.text)

            # POC path: run generated code directly in-process.
            # Production hardening plan: execute this inside an ephemeral Docker container
            # with strict CPU/memory/time limits, then destroy the container immediately.
            namespace: dict[str, Any] = {}
            error_stage = "exec"
            exec(generated_code, namespace, namespace)

            error_stage = "function_lookup"
            restructure_data = namespace.get("restructure_data")
            if not callable(restructure_data):
                raise ValueError(
                    "Generated code did not define callable restructure_data(data_str)."
                )

            error_stage = "runtime"
            # Note: client timeouts protect the caller's wait time, but they do not
            # guarantee in-process generated code stops running immediately.
            # For this POC we accept that risk; the Docker sandbox plan above is the
            # intended safety boundary for reliable execution timeouts/cancellation.
            records = restructure_data(data_str)

            error_stage = "shape_validation"
            if not isinstance(records, list) or not all(
                isinstance(item, dict) for item in records
            ):
                raise ValueError("restructure_data must return a list of dictionaries.")

            logger.info(
                "Restructured %d records from %s on attempt %d",
                len(records),
                uri,
                attempt,
            )
            return cast(list[dict[str, Any]], records)
        except Exception as e:
            last_stage = error_stage
            last_error = f"{type(e).__name__}: {e}"
            last_traceback = traceback.format_exc()
            logger.warning(
                "Attempt %d/%d failed while restructuring %s at stage %s: %s",
                attempt,
                MAX_ATTEMPTS,
                uri,
                last_stage,
                last_error,
            )

            if last_stage in ("sampling", "sampling_response"):
                continue

            convo.append(
                SamplingMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Your previous code attempt resulted in this error when I tried to "
                            f"run it.\n\n"
                            f"Failure stage: {last_stage}\n"
                            f"Error: {last_error}\n\n"
                            f"Traceback:\n{last_traceback}\n"
                            "Please revise your implementation to fix this error. Remember, I "
                            "just want a function that takes the full data string and returns "
                            "a list of dicts. Don't try to do anything fancy with it, and don't "
                            "worry about performance - just get it working correctly."
                            "\n\n"
                            "As before, write your revised function in a triple-backticked block "
                            "labeled ```python, and make it the last thing you say."
                        ),
                    ),
                )
            )

    raise ValueError(
        f"Failed to generate working restructure_data after {MAX_ATTEMPTS} attempts for {uri}. "
        f"Last stage: {last_stage}. Last error: {last_error}"
    )


@mcp.tool()
def ruc_hello_world(name: str = "World") -> str:
    """Return a Hello World greeting.

    Args:
        name: The name to greet.
    """
    return f"Hello, {name}!"


@mcp.tool(
    description=(
        "Perform a task that mixes deterministic procedural work (primarily the "
        "domain of traditional programming languages) with semantic "
        "interpretation (primarily the domain of LLMs). Use this when a "
        "user describes a fuzzy task that must be executed rigorously "
        "across records, files, or other repeatable inputs."
    ),
)
async def ruc_execute_semantic_code_workflow(
    task_description: Annotated[
        str,
        Field(
            description=(
                "Plain-English description of the task the user wants performed."
            )
        ),
    ],
    ctx: fastmcp.Context,
    context_explanation: Annotated[
        str | None,
        Field(
            description=(
                "Optional background information needed to interpret the task correctly. "
                "Use this for domain context, terminology, business rules, assumptions, etc."
            )
        ),
    ] = None,
    data_source_uris: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional indicators of one or more sources of data upon which to perform "
                "the task. RUC will try to interpret each of these as a text-based collection "
                "of records, such as a CSV file, a JSON file with a list of objects, a "
                "SQL dump of a database table, etc. "
                "Currently only accepts file URIs with absolute paths, "
                "e.g. `file:///C:/Users/mvol/Documents/client_list.csv`"
            )
        ),
    ] = None,
    expected_result_schema: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Optional JSON Schema describing the expected shape of the final result. "
                "If not provided, RUC will decide for itself what an appropriate result schema "
                "should be, based on the task description."
            )
        ),
    ] = None,
    behavioral_requirements: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional non-negotiable requirements the workflow must obey, and which might "
                "not be immediately obvious from the task description. Use this to specify "
                'hard constraints or stipulations, such as "Don\'t process repeated records", '
                'or "Scrub personally identifiable information from the output", or '
                '"Interpret dates as eight-digit numerical sequences in DDMMYYYY format". '
                "Not all tasks will require this field, so feel free to leave it blank if "
                "there are no special requirements."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Perform a RUC task."""
    logger = logging.getLogger(__name__)
    logger.info("execute_semantic_code_workflow started for task: %s", task_description)

    # First step: load the data from the indicated sources, if any.
    data_source_records = {}
    data_source_uris = data_source_uris or []
    for data_source_uri in data_source_uris:
        data_source_records[data_source_uri] = await _load_data_from_uri(
            data_source_uri, ctx
        )

    logger.info("Data loading complete. Starting main workflow execution.")

    return {
        "status": "not_implemented",
        "message": "execute_semantic_code_workflow is not implemented yet.",
    }


def main() -> None:
    """Entrypoint for local development."""
    configure_logging()
    logging.getLogger(__name__).info("Starting RUC MCP server over stdio")
    mcp.run(transport="stdio")
