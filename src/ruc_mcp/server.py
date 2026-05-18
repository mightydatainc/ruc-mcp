"""Render Unto Caesar MCP server built with FastMCP."""

import json
import inspect
import logging
import os
from pathlib import Path
import re
import time
import traceback
from typing import Annotated, Any, cast
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
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

RUC_FUNCTION_WRITING_SYSTEM_PROMPT = """
In today's work session, you'll write a Python function called `execute_workflow`.
It will adhere to the following calling convention and structure:

```python
async def execute_workflow(data_source_records: dict[str, list[Any]], ctx: fastmcp.Context) -> dict:
    # TODO: Implement the task here
    # ...
    return retval # "retval" is some JSON-serializable dict or list.
```

The Python environment you'll be running in is a default Python 3.11 install.
You can import default packages like `json` or `csv`, but you don't have access to fancy
advanced tools like dataframes. Likewise, the __future__ library in your environment
is extremely flakey and unreliable, and must be avoided (not that you should need it anyway).
Also, it'll be running in a restricted VM, so you can't access the network or the filesystem
-- again, not that you should need such things for a data restructuring operation.

The one exception is that this environment *does* have a library called FastMCP installed,
so you can `import fastmcp` and declare `ctx: fastmcp.Context` in your type signature.
You shouldn't need to actually *use* the FastMCP library (we'll add calls to the LLM 
ourselves in a later pass), but the Context object will be passed into your function,
so I figure you'll want it for type signature purposes.

The "ctx" argument is a FastMCP Context object. It's what we'll be using to communicate
with the LLM. Don't worry about it for now. We only provide it here because we'll need to
pass it through to the LLM calling function stubs. More on that later.

You may, of course, write whatever support or helper functions you might need.

In an ideal world, you should be able to implement the entire task using only conventional
code, possibly with the help of heuristic tricks involving regexes or string operations
where necessary.

However, in practice, some portion (or multiple portions) of this task might require judgment
calls, inference, reconciliation of noisy or ambiguous information, and other tasks that are
more suitable to an LLM than to a Python function.

If you come across such operational requirements, then here's what I want you to do:
Invent async ad-hoc functions on the fly that would hypothetically send an LLM call. Each such
function should take a single JSON-serializable object as an argument (call it just "arg") and the
FastMCP Context object ("ctx"), and will return a JSON serializable dict as a reply.
Its function signature should look like this:
```python
async def some_made_up_function_name(arg: dict, ctx: fastmcp.Context) -> dict:
```

Whenever you invent such an ad-hoc function, write a placeholder stub for it, and describe in a
TODO statement what you'd imagine that the function will do. Make the body of the function throw
a NotImplementedError with a message saying what the function is and what it would do if it were
in fact implemented. Declare them with `async def`, and call them with `await` -- just like normal
async functions, even though they won't actually do anything yet.

To make it easier to find these stub functions later, use a special syntax for the TODO statement:
"TODO(llm_stub: stub_function_name): lorem ipsum dolor logit es..." In other words, mark the TODO
with the special label "llm_stub", followed by the name of the function, and then a description of
what the function would ask the LLM to do -- using the formatting exactly as shown.

Put all such stub functions at the end of your code. They, as well as the main workflow function
`execute_workflow`, must be at the top level of the module, i.e. not nested inside any other
function or class. I should be able to copy-paste your entire code into a Python environment and
run `execute_workflow(...)` without having to look for it inside a namespace or a class or 
something.

PRO TIP: Don't write stubs for functions that you don't plan to call. We *will* fill out their
implementations shortly, so don't bypass or avoid these stub functions simply in the interest of
writing code that works "for now". Treat the stub functions as though they actually do indeed
work in the here and now.

PRO TIP: As a courtesy to the end user, use the `ctx` object's logging and progress reporting
methods! RUC is often called to perform long-running operations, so it's an important courtesy
to the user to report things like the name of the current stage of the workflow or the current
progress through the current stage. The FastMCP Context object, `ctx`, has standard logger
methods to facilitate this: ctx.debug, ctx.info, ctx.warning, and ctx.error -- these work just
like the standard Python logger methods, except that they're asynchronous (so, call them 
with `await`, e.g. `await ctx.info("Hello world")`). In addition, the `ctx` object also has an
async method called `ctx.report_progress(progress: float, total: float, message: str)`, which
shows the user a very convenient progress bar along with a short status message. Use these
tools often to keep the user informed about what's going on!
"""

INJECT_RUC_LLM_CALL_FUNCTION = """
async def ruc_submit_sample_request_to_llm(
    messages: list[str],
    system_prompt: str,
    ctx: fastmcp.Context,
    result_type: type[pydantic.BaseModel],
) -> dict:
    convo = json.loads(json.dumps(messages))  # Deep copy for immutability.

    try:
        formal_structured_sample_result = await ctx.sample(
            messages=convo,
            system_prompt=system_prompt,
            max_tokens=5000,
            result_type=result_type,
        )
        return formal_structured_sample_result.result.model_dump()
    except Exception as e:
        # Check if the error is a ValueError whose message starts with:
        # "Client does not support sampling with tools."
        # If it's not, then this is an unexpected error and we should raise it.
        is_sampling_tool_error = isinstance(e, ValueError) and str(e).startswith(
            "Client does not support sampling with tools."
        )
        if not is_sampling_tool_error:
            raise e

    # If we get here, then the error is a sampling tool error, which means that the client
    # does not support sampling with tools. We can handle this case gracefully or provide
    # a fallback mechanism.
    convo.append(
        'Begin your reply with a fenced code block labeled "```json". '
        "Inside that block, provide a JSON object that matches the expected format."
    )
    formal_structured_sample_result = await ctx.sample(
        messages=convo,
        system_prompt=system_prompt,
        max_tokens=5000,
    )
    formal_structured_text = formal_structured_sample_result.text
    if not formal_structured_text:
        raise ValueError(
            "LLM returned no text in response to the structured output prompt."
        )

    if "```json" not in formal_structured_text:
        # That's okay-ish. The LLM should have followed the instructions to start its reply 
        # with a JSON code block, but maybe it forgot. Let's try to parse the whole thing as JSON
        # and see if we get anything.
        try:
            return json.loads(formal_structured_text)
        except json.JSONDecodeError:
            raise ValueError(
                "LLM did not include a JSON code block in its response to the structured output prompt, "
                "and the text of the response could not be parsed as JSON either. Here's what it said: "
                + formal_structured_text
            )

    split_on_json = formal_structured_text.split("```json", 1)
    if len(split_on_json) < 2:
        raise ValueError(
            "LLM did not include a JSON code block in its response to the structured output prompt."
            "Here's what it said instead: " + formal_structured_text
        )

    json_block_and_after = split_on_json[1]
    split_on_closing = json_block_and_after.split("```", 1)
    if len(split_on_closing) < 2:
        raise ValueError(
            "LLM did not include a properly closed JSON code block in its response to the structured output prompt."
        )
    json_block = split_on_closing[0]
    try:
        return json.loads(json_block)
    except json.JSONDecodeError as e:
        raise ValueError("LLM's JSON code block could not be parsed as JSON.") from e
"""

STUB_FUNCTION_IMPLEMENTATION_TEMPLATE = """
async def TODO_PROVIDE_FUNCTION_NAME(arg: dict, ctx: fastmcp.Context) -> dict:
    system_prompt = "TODO PASTE SYSTEM PROMPT CONTENTS HERE"

    convo = [json.dumps(arg, indent=2)]

    brainstorm_sample_result = await ctx.sample(
        messages=convo,
        system_prompt=system_prompt + (
            "\\n\\n"            
            "Brainstorm the question first, providing chain-of-thought reasoning to ensure "
            "you provide a well-thought-out answer. At the end of your consideration and "
            "reasoning, provide a final answer. Don't worry about formatting the final answer "
            "in a particular way, just provide it in a clear and concise manner. We'll have "
            "you formalize the output in the correct format in a later step."
        ),
        max_tokens=5000,
    )
    brainstorm_text = brainstorm_sample_result.text
    if not brainstorm_text:
        raise ValueError("LLM returned no text in response to the brainstorm prompt.")

    convo.append("LLM's brainstorm and reasoning:\\n" + brainstorm_text)
    convo.append(
        "Now, based on the above brainstorm and reasoning, provide a final answer to the "
        "question in a structured format as a JSON object."
    )

    result_type = pydantic.create_model(TODO_PROVIDE_PYDANTIC_MODEL_ARGS_HERE)

    retval = await ruc_submit_sample_request_to_llm(
        messages=convo,
        system_prompt=system_prompt,
        ctx=ctx,
        result_type=result_type,
    )
    return retval
"""


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


def _extract_labeled_code_block(response_text: str, label: str) -> str:
    """Extract the last fenced code block matching the given label."""
    if not label.strip():
        raise ValueError("Code block label must be a non-empty string.")

    pattern = rf"```{re.escape(label)}\s*(.*?)```"
    labeled_blocks = re.findall(pattern, response_text, flags=re.IGNORECASE | re.DOTALL)
    if labeled_blocks:
        return labeled_blocks[-1].strip()

    raise ValueError(
        f"Sampling response did not include a fenced code block labeled '{label}'."
    )


def _construct_data_source_previews(
    data_source_records: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Construct limited-length previews of JSON dumps of the data collections,
    suitable for inclusion in user messages to the LLM."""
    previews = []
    MAX_CHARACTERS_PER_SOURCE = 5000
    for uri, records in data_source_records.items():
        preview_str = f"Data source: {uri}\n"
        preview_str += (
            f"JSON dump preview (first {MAX_CHARACTERS_PER_SOURCE} characters):\n\n"
        )
        try:
            records_str = json.dumps(records, indent=2)
            if len(records_str) > MAX_CHARACTERS_PER_SOURCE:
                records_str = records_str[:MAX_CHARACTERS_PER_SOURCE] + "..."
            preview_str += f"\n\n{records_str}"
        except Exception as e:
            preview_str += (
                f"\n\n(Could not generate preview of records due to error: {e})"
            )
        previews.append(preview_str)
    return previews


def _send_result_to_uri(uri: str, file_contents: Any) -> str:
    """Write workflow results to a destination URI and return an execution note."""
    logger = logging.getLogger(__name__)
    logger.info("Sending workflow result to URI: %s", uri)

    # If file_contents is a string, then we probably want to write that string directly.
    # If it's anything else, we'll assume it's a JSON-serializable object and we'll write
    # the JSON dump of it.
    file_contents = (
        file_contents
        if isinstance(file_contents, str)
        else json.dumps(file_contents, ensure_ascii=False)
    )

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == "file":
        # Canonical Windows URI: file:///C:/path/to/output.json
        # Also preserves compatibility with file://C:/path style during transition.
        if parsed_uri.netloc and parsed_uri.netloc != "localhost":
            if len(parsed_uri.netloc) == 2 and parsed_uri.netloc[1] == ":":
                file_uri_path = f"/{parsed_uri.netloc}{parsed_uri.path}"
            else:
                file_uri_path = f"//{parsed_uri.netloc}{parsed_uri.path}"
        else:
            file_uri_path = parsed_uri.path

        file_path = Path(url2pathname(unquote(file_uri_path)))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_contents)

        logger.info("Workflow result written to %s", file_path)
        return f"Wrote workflow result to {uri}.\n\n"

    if parsed_uri.scheme in ("http", "https"):
        payload = file_contents.encode("utf-8")
        request = Request(
            uri,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            status_code = response.status

        if 200 <= status_code < 300:
            logger.info("Workflow result posted to %s with status %s", uri, status_code)
            return f"Posted workflow result to {uri} (HTTP {status_code}).\n\n"

        raise ValueError(
            f"Failed to post workflow result to {uri}: HTTP status {status_code}."
        )

    raise ValueError(
        f"Unsupported URI scheme in {uri}. Supported schemes are file, http, and https."
    )


async def _load_data_from_uri(ctx: fastmcp.Context, uri: str) -> list[dict[str, Any]]:
    """Load data from a given URI and return it as a list of records."""
    logger = logging.getLogger(__name__)
    await ctx.info(f"Loading data source: {uri}")

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme != "file":
        await ctx.error(
            f'Unsupported URI scheme "{parsed_uri.scheme}" in {uri}\n'
            "Currently only file URIs with absolute paths are supported "
            "(e.g. file:///C:/path/to/file.csv). If your data source is not a file, "
            "please download or export it to a file and try again, "
            "providing a file URI pointing to it."
        )

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
                raise ValueError(
                    f"Sampling returned no text to execute when trying to parse data from {uri}."
                )

            convo.append(
                SamplingMessage(
                    role="assistant",
                    content=TextContent(type="text", text=sample_result.text),
                )
            )

            error_stage = "extract"
            generated_code = _extract_labeled_code_block(sample_result.text, "python")

            # POC path: run generated code directly in-process.
            # Production hardening plan: execute this inside an ephemeral Docker container
            # with strict CPU/memory/time limits, then destroy the container immediately.
            error_stage = "exec"
            namespace: dict[str, Any] = {
                "__name__": "ruc_data_loader",
                "__package__": None,
            }
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

            if records:
                logger.info(
                    "First record keys for %s: %s",
                    uri,
                    sorted(records[0].keys()),
                )
            else:
                logger.info("Restructure returned an empty list for %s", uri)

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


async def _write_workflow(ctx: fastmcp.Context, convo: list[str]):
    """Write a Python function that performs a procedural workflow that includes
    "fuzzy" operations."""
    await ctx.info("Writing workflow code...")

    convo = json.loads(json.dumps(convo))  # Deep-copy to ensure mutability.

    convo.append("""
You now have everything you need (or at least, everything I can give you)
to write `execute_workflow`.

This will require some thought, so don't just start coding right away.
First, discuss a plan. Approach this with the mindset of a seasoned software engineer.
Feel free to talk your way through state management or support systems as needed.

Then, after your proverbial one-man design meeting, emit a block of Python code 
enclosed in triple-backticks, i.e. as "```python". Later, I'll look for this block,
and will copy-paste it into an execution environment.
""")

    MAX_ATTEMPTS = 5
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            sample_result = await ctx.sample(
                messages=convo,
                system_prompt=RUC_FUNCTION_WRITING_SYSTEM_PROMPT,
                max_tokens=60_000,
            )

            if not sample_result.text:
                raise ValueError(
                    "Sampling returned no text when trying to write the initial workflow code."
                )

            pycode = _extract_labeled_code_block(sample_result.text, "python")
            if not pycode:
                raise ValueError(
                    "Failed to extract Python code block when trying to write "
                    "initial workflow code."
                )

            # Let's clean up the code just a tiny bit.
            pycode = pycode.strip()

            # We need to make sure that it calls a few imports that will be necessary
            # for implementing the stubs.
            # Fortunately, Python allows multiple import statements for the same package, so we
            # don't need to check if they're already imported.
            # The only snag in this plan is that __future__ imports must be at the very top of
            # the file -- but we forbid the use of __future__ imports in the system prompt,
            # so we should be in the clear. Just in case, let's check for __future__ and
            # force a retry if it's there.
            if "__future__" in pycode:
                raise ValueError(
                    "The LLM included a __future__ import, which is forbidden in the "
                    "system prompt."
                )

            pycode = (
                "import fastmcp\nimport logging\nimport json\nimport pydantic\n\n"
                + pycode
            )

            # Inject some helper functions.
            pycode += "\n\n\n" + INJECT_RUC_LLM_CALL_FUNCTION

            return pycode
        except Exception as e:
            await ctx.warning(
                f"Attempt {attempt}/{MAX_ATTEMPTS} failed while writing workflow code: {e}"
            )

    raise ValueError(
        f"RUC failed to write any initial workflow code after {MAX_ATTEMPTS} attempts. "
        "There is probably something wrong with RUC's code generator prompt, or with the "
        "model it's calling."
    )


async def _is_ready_for_workflow(ctx: fastmcp.Context, convo: list[str]):
    """Sanity-checks to see if we have enough information to proceed with writing a workflow."""
    await ctx.info(
        "Confirming that we have adequate information to proceed with writing workflow..."
    )

    convo = json.loads(json.dumps(convo))  # Deep-copy to ensure mutability.

    convo.append("""
Before we get started, discuss the following matters:

- Is the task clear? Do you understand what you're being asked to do?

- Do you have all of the data sources that the task seems to need?

- For each data source, what's the *shape* or *structure* of the data? 
    Is the data something you can work with? (Note that I'm not asking you to make up what
    you think the shape or structure *should* be. I'm asking you to describe what you see.
    If there's a discrepancy between what you see and what you would expect, say so.)

- For each data source, look at the preview/snapshot. Do you see any lurking anomalies or
    inconsistencies that you'll have to code around? Basically, anything that might bite you
    in the ass once the code is running?

Discuss these matters. Take as much time or verbiage as you need; these are important to
get right.

Then, at the end of your discussion, answer one chief salient quesion: 
If I were to paste in the implementations of the stub functions, then would you be good to go?
Right now, unconditionally (assuming stub function flesh-out), without any further input... 
Would you be able to write and execute `execute_workflow`?
At the end of your response, write either "GOOD_TO_GO: YES", just like that, on its own line, 
in all caps -- or else "GOOD_TO_GO: NO. ", followed by an explanation of why not.
""")

    sample_result = await ctx.sample(
        messages=convo,
        system_prompt=RUC_FUNCTION_WRITING_SYSTEM_PROMPT,
        max_tokens=20_000,
    )

    if not sample_result.text:
        raise ValueError(
            "Sampling returned no text when sanity-checking whether or not we "
            "have enough information to proceed with writing a workflow."
        )

    # Grep sample_result for "GOOD_TO_GO: YES". If it's there, we can exit.
    if "GOOD_TO_GO: YES" in sample_result.text:
        # Returning no value is a signal to proceed!
        return

    # If we get here, the model is saying "GOOD_TO_GO: NO". We should raise an
    # error with the model's explanation of why not, which should be in the text
    # after "GOOD_TO_GO: NO".
    if "GOOD_TO_GO: NO" not in sample_result.text:
        raise ValueError(
            "When we asked the model to sanity-check whether or not we have enough "
            "information to proceed with writing a workflow, "
            "it gave us neither a clear YES nor a clear NO."
        )

    good_to_go_split = sample_result.text.split("GOOD_TO_GO: NO", 1)
    explanation = (
        good_to_go_split[1].strip()
        if len(good_to_go_split) > 1
        else (
            "When we asked the model to sanity-check whether or not we have enough information "
            "to proceed with writing a workflow, it said NO but did not provide an explanation."
        )
    )
    raise ValueError(
        "When we asked the model to sanity-check whether or not we have enough information to "
        "proceed with writing a workflow, it said NO. Explanation: "
        f"{explanation}"
    )


async def _write_implementation_for_stub(
    ctx: fastmcp.Context,
    stubname: str,
    pycode: str,
    convo: list[str],
) -> str:
    """Replace the given stub function with a real implementation."""
    logger = logging.getLogger(__name__)
    convo = json.loads(json.dumps(convo))  # Deep-copy to ensure mutability.

    logger.debug("Asking model to implement stub function %s", stubname)

    convo.append(f"""
Oh shoot, I'm so sorry -- I hit the Back button by accident and deleted your entire response.
I lost your whole reasoning process and self-discussion. From your POV, the conversation probably
has a major discontinuity. I apologize for the confusion.

But that's okay, because I had copy-pasted the code you wrote.

Check it out.

```python
{pycode}
```
""")

    stubfunction = f"{STUB_FUNCTION_IMPLEMENTATION_TEMPLATE}"
    stubfunction = stubfunction.replace("TODO_PROVIDE_FUNCTION_NAME", stubname)

    convo.append(f"""
I'm currently working on implementing the function {stubname}. Maybe you can help me out.
Here's what my implementation looks like so far...
""")

    # Save a snapshot of the convo in this state.
    convoSnapshotBeforePresentingPythonCode = json.loads(json.dumps(convo))

    convo.append(f"""
```python
{stubfunction}
```
""")

    convo.append("""
I need your help defining the result type.

Take a good look at how the function is called in the code, and how its results are used.
Based on your observations, the function's TODO notes, and so on, describe what its return
structure is like.

When you're done, write a Python block that I can use for declaring the result type.
Bear in mind that your Pydantic model will be passed to an LLM as a structured output
constraint, so be sure to include guidance (e.g. descriptions and commentary) and
guardrails (e.g. min and max values) where appropriate.

Your Python block should be enclosed in triple backticks labeled "```python".
Also, it should start with `result_type = pydantic.create_model(`,
so that I can just string-replace your code directly into my function.
""")

    logging.debug(f"Asking model to define result type for stub function {stubname}.")
    resulttype_sample_result = await ctx.sample(
        messages=convo,
        system_prompt=RUC_FUNCTION_WRITING_SYSTEM_PROMPT,
        max_tokens=20_000,
    )
    if not resulttype_sample_result.text:
        raise ValueError(
            "The model failed to provide any text when we asked it to define the result type for "
            f"stub function {stubname}."
        )

    resulttype_pysnippet = _extract_labeled_code_block(
        resulttype_sample_result.text, "python"
    )
    if not resulttype_pysnippet:
        raise ValueError(
            "Failed to extract a Python code block from the model's response when we asked it to "
            f"define the result type for stub function {stubname}."
        )

    # Now we have the code snippet for the result type declaration. Let's insert it into the
    # stub function template.
    # Make sure we get the initial indent level correct, because Python is sensitive about that.
    # Fortunately, we don't have to think too hard about this, because the replacement operation
    # preserves the whitespace before "result_type = ..." exactly as it is in the template.
    stubfunction = stubfunction.replace(
        "result_type = pydantic.create_model(TODO_PROVIDE_PYDANTIC_MODEL_ARGS_HERE)",
        resulttype_pysnippet.strip(),
    )

    # Next, let's do something about that system prompt.

    # Restore the conversation to the state it was in before we presented the Python code,
    # so that the model can see the code we just wrote. That way, it won't be distracted
    # by potential ambiguities in the return type.
    convo = json.loads(json.dumps(convoSnapshotBeforePresentingPythonCode))

    convo.append(f"""
```python
{stubfunction}
```
""")

    convo.append(f"""
I need your help writing a system prompt.
See where it says "TODO PASTE SYSTEM PROMPT CONTENTS HERE"?
Yeah, well, I need such contents. :)

Compose a focused, chain-of-thought optimized system prompt for this task.

Don't dive right into writing the prompt just yet.
First, talk about what would make a good prompt vs a bad prompt in this scenario.

Then, when you're ready, emit the system prompt in a block of
triple-backticks labeled "```systemprompt".
""")

    logging.debug(
        "Asking model to provide system prompt for stub function %s.", stubname
    )
    sysprompt_sample_result = await ctx.sample(
        messages=convo,
        system_prompt=RUC_FUNCTION_WRITING_SYSTEM_PROMPT,
        max_tokens=20_000,
    )
    if not sysprompt_sample_result.text:
        raise ValueError(
            "The model failed to provide any text when we asked it to provide a system prompt "
            f"for stub function {stubname}."
        )

    sysprompt = _extract_labeled_code_block(
        sysprompt_sample_result.text, "systemprompt"
    )
    if not sysprompt:
        raise ValueError(
            "Failed to extract a system prompt block from the model's response when we asked it "
            f"to provide a system prompt for stub function {stubname}."
        )
    # The replacement is nontrivial, because the system prompt is a string that might contain
    # delimiters that we ourselves are using in the stub function template. Our best bet
    # is to give it a JSONized string and parse it in the function.
    stubfunction = stubfunction.replace(
        '"TODO PASTE SYSTEM PROMPT CONTENTS HERE"',
        json.dumps(sysprompt),
    )

    return stubfunction


async def _replace_all_stubs_with_implementations(
    ctx: fastmcp.Context,
    pycode: str,
    convo: list[str],
):
    """Find any stub functions in the given code, and replace them with real implementations."""
    logger = logging.getLogger(__name__)
    convo = json.loads(json.dumps(convo))  # Deep-copy to ensure mutability.

    await ctx.info(
        "Wiring the workflow's non-procedural portions back into LLM calls..."
    )

    # We marked these stub functions with a special syntax in the TODO comment,
    # so we can grep for them.
    # The syntax is
    # "TODO(llm_stub: stub_function_name): description of what the function should do".
    stub_pattern = re.compile(r"TODO\(llm_stub:\s*(\w+)\):\s*(.*?)\n")
    stubs = stub_pattern.findall(pycode)
    logger.info("Found %d stub functions to implement: %s", len(stubs), stubs)

    stubfunctions_by_name: dict[str, str] = {}

    # NOTE: In theory this could be done in parallel, but to keep things simple and to
    # avoid any weirdness with the model getting confused by multiple simultaneous requests,
    # we'll do it sequentially for now.
    for stubname, stubdescription in stubs:
        stubfunction = await _write_implementation_for_stub(
            ctx, stubname, pycode, convo
        )
        stubfunctions_by_name[stubname] = stubfunction

    for stubname, stubfunction in stubfunctions_by_name.items():
        # Now that we've shown the current state of the code back to the model,
        # neuter the stub function by renaming it to {stubname}_obsolete_stub,
        # so that its definition won't interfere with the next round of code generation
        # for the real implementation.
        pycode = pycode.replace(f"def {stubname}(", f"def {stubname}_obsolete_stub(")

        # Add the now-implemented function to the code.
        pycode += "\n\n\n" + stubfunction

    return pycode


async def _execute_workflow_code(
    ctx: fastmcp.Context,
    pycode: str,
    data_source_records: dict[str, Any],
) -> Any:
    """Execute the given workflow code and return the result."""
    logger = logging.getLogger(__name__)
    await ctx.info("Executing workflow.")

    namespace: dict[str, Any] = {
        "__name__": "ruc_generated_workflow",
        "__package__": None,
    }
    exec(pycode, namespace, namespace)

    execute_workflow = namespace.get("execute_workflow")
    if not callable(execute_workflow):
        raise ValueError(
            "Generated code did not define callable execute_workflow(data_source_records, ctx)."
        )

    # The function execute_workflow takes an argument called "data_source_records".
    # Pass it in.
    workflow_result = execute_workflow(
        data_source_records=data_source_records,
        ctx=ctx,
    )
    result = (
        await cast(Any, workflow_result)
        if inspect.isawaitable(workflow_result)
        else workflow_result
    )
    logger.info("Workflow execution complete.")
    return result


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
    ctx: fastmcp.Context,
    task_description: Annotated[
        str,
        Field(
            description=(
                "Plain-English description of the task the user wants performed."
            )
        ),
    ],
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
    desired_result_schema: Annotated[
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
                "Optional list of requirements the workflow must obey, and which might "
                "not be immediately obvious from the task description. Use this to specify "
                'hard constraints or stipulations, such as "Don\'t process repeated records", '
                'or "Scrub personally identifiable information from the output", or '
                '"Interpret dates as eight-digit numerical sequences in DDMMYYYY format". '
                "Not all tasks will require this field, so feel free to leave it blank if "
                "there are no special requirements."
                "\n\n"
                "PRO TIP: Keep this *minimal*. The more behavioral requirements a task has, "
                "the more complex the implementation will be, and the higher the risk that "
                "something will go wrong. So provide only the most essential requirements "
                "here, and try to keep the task description itself as clear and comprehensive "
                "as possible, so that you don't have to rely on this field to convey "
                "important information about the task."
            )
        ),
    ] = None,
    result_uri: Annotated[
        str | None,
        Field(
            description=(
                "Optional destination to which to write bulk results. For most tasks, writing your "
                "output to a file will be more practical than returning a huge JSON blob. "
                "If this field isn't provided, RUC will just present the results directly to the "
                "agent LLM that called it. This is a bad idea for large results, because the "
                "LLM then has to load the entire result into its context window just to read it, "
                "which is inefficient, unreliable, and probably defeats the purpose of using RUC "
                "in the first place. So if your task is producing a large result, it's best to "
                "provide a URI here to write it to. "
                "If the URI is a file URI, RUC will write the result to the given file. "
                "If the URI is an HTTP endpoint, RUC will POST the result to that endpoint as JSON. "
                "Currently only accepts file URIs with absolute paths, "
                "e.g. `file:///C:/Users/mvol/Documents/client_list.csv`"
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Perform a RUC task."""
    logger = logging.getLogger(__name__)
    logger.info("execute_semantic_code_workflow started for task: %s", task_description)
    start_time = time.time()

    # First step: load the data from the indicated sources, if any.
    data_source_records = {}
    execution_notes = ""
    data_source_uris = data_source_uris or []
    for data_source_uri in data_source_uris:
        try:
            data_source_records[data_source_uri] = await _load_data_from_uri(
                ctx,
                data_source_uri,
            )
        except Exception as e:
            note = f"Failed to load data source {data_source_uri}: {e}"
            logger.warning(note)
            execution_notes += f"{note}\n\n"

    data_previews = _construct_data_source_previews(data_source_records)

    # Build the basic conversational context for the workflow execution. This will be fed into the LLM
    # as part of the prompt, and will serve as the definition of this task.

    convo = [f"TASK:\n\n{task_description}"]

    if context_explanation:
        convo.append(
            'Here is some context to help you understand the "big picture" of this task, '
            "including background information or an explanation about *why* it's being run:"
            "\n\n"
            f"{context_explanation}"
        )

    if behavioral_requirements:
        convo.append(
            "Here are some specific requirements and constraints that the execution of this task "
            "must adhere to:"
            "\n\n" + "\n\n".join(f"- {req}" for req in behavioral_requirements)
        )

    if desired_result_schema:
        convo.append(
            "The final result of this execution should adhere to this expected schema:"
            "\n\n" + json.dumps(desired_result_schema, indent=2)
        )

    if data_previews and len(data_previews) > 0:
        convo.append(
            f"I'll now, over the next few messages, show you {len(data_previews)} data sources, "
            "which the task presumably expects you to rely on. I'll show you a truncated "
            "preview of the contents of each data source, to help you understand what kind of "
            "data you're working with."
        )
        for preview in data_previews:
            convo.append(preview)
    else:
        convo.append(
            "No external data sources were provided, so you'll have to rely entirely on the task "
            "description and context explanation to understand what this task is asking you "
            "to do. If that's impossible, i.e. if the task is inherently asking you to operate "
            "on some data and that data is missing, then that's almost certainly an error on the "
            "part of either the end user or the AI agent that dispatched you."
        )

    if result_uri and len(result_uri) > 0:
        convo.append(
            'MAKE THE WORKFLOW RESULT USE {"file_contents": "lorem ipsum dolor..."}\n\n '
            "You've been asked to make the workflow write its results to a file or post them to "
            "an external URI. "
            "On the surface, this might seem impossible, because you don't have access to the "
            "filesystem nor the network. Fortunately, our runtime environment has special support "
            "for this exact scenario: the `file_contents` field! "
            "Make `execute_workflow` return a dict with a `file_contents` field, whose value is "
            "a string containing the full contents to write. Like this:\n\n"
            '{"file_contents": "the full contents string to write goes here"}\n\n'
            "Our runtime environment will look for this field in your workflow's output, "
            "and if it finds it, it will write the contents of that field to the given "
            "destination."
        )

    try:
        await _is_ready_for_workflow(ctx, convo)
    except Exception as e:
        note = f"Model indicated it was not ready to write workflow code: {e}"
        logger.warning(note)
        execution_notes += f"{note}\n\n"
        # If the model says it's not ready, we should stop here and return an error message to the user.
        return {
            "status": "error",
            "message": "Model indicated it was not ready to write workflow code.",
            "details": str(e),
            "execution_notes": execution_notes.strip()
            or "(no notes recorded during execution)",
        }

    pycode = await _write_workflow(ctx, convo)

    logger.info(
        "Workflow written. Now checking for any stub functions that need implementations, "
        "and replacing them with real code."
    )

    pycode = await _replace_all_stubs_with_implementations(ctx, pycode, convo)

    # For now, just log the generated code and return a placeholder response, since the main point
    # of this POC is to demonstrate the code generation aspect of RUC. The production version of
    # this function will need to execute the generated code in a sandboxed environment and return
    # the actual results of that execution.

    # DEBUG: Save pycode to a local file, so we can inspect it if anything goes wrong during execution.
    with open("./logs/temp_auto_generated_workflow.py", "w", encoding="utf-8") as f:
        f.write(pycode)

    try:
        runresult = await _execute_workflow_code(ctx, pycode, data_source_records)
        elapsed_seconds = int(time.time() - start_time)
        await ctx.info(f"Workflow execution complete after {elapsed_seconds} seconds.")
    except Exception as e:
        elapsed_seconds = int(time.time() - start_time)
        logger.error(
            f"Workflow execution failed after {elapsed_seconds} seconds: {e}",
            exc_info=True,
        )
        return {
            "status": "error",
            "message": f"Workflow execution failed after {elapsed_seconds} seconds.",
            "details": str(e),
            "execution_notes": execution_notes.strip()
            or "(no notes recorded during execution)",
        }

    # If we have a result_uri, write the result there instead of returning it directly.
    if result_uri and len(result_uri) > 0:
        file_contents = (
            (
                runresult.get("file_contents")
                if isinstance(runresult, dict)
                else json.dumps(runresult, indent=2)
            )
            if runresult is not None
            else ""
        )
        execution_notes += _send_result_to_uri(result_uri, file_contents)

    elapsed_seconds = int(time.time() - start_time)
    execution_notes += (
        f"\n\nWorkflow executed successfully after {elapsed_seconds} seconds. Please take a moment to "
        "inspect the results and confirm that everything looks correct. "
        "If not, please consider running RUC again with a clearer task description, "
        "more detailed context explanation, more specific behavioral requirements, "
        "or a more detailed expected result schema, as you see fit."
    )

    retval = {
        "status": "success",
        "execution_notes": execution_notes.strip()
        or "(no notes recorded during execution)",
    }
    if not result_uri:
        # If we didn't write the result to a URI, include it in the response.
        retval["result"] = json.dumps(runresult, indent=2)

    return retval


def main() -> None:
    """Entrypoint for local development."""
    configure_logging()
    logging.getLogger(__name__).info("Starting RUC MCP server over stdio")
    mcp.run(transport="stdio")
