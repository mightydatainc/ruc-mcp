"""
"Render Unto Caesar" FastMCP server for mixed semantic and procedural workflows.

This module defines an MCP tool that turns fuzzy task descriptions into
deterministic Python workflows with explicit semantic LLM call boundaries.
"""

import json
import inspect
import logging
import os
from pathlib import Path
import re
import time
import traceback
from typing import Annotated, Any, cast

import fastmcp
from mcp.types import SamplingMessage, TextContent
from pydantic import Field

mcp: fastmcp.FastMCP = fastmcp.FastMCP(
    name="ruc-mcp",
    instructions=(
        "Use this server when a user asks for a task that mixes deterministic "
        "procedural work with semantic interpretation or natural-language comprehension "
        "or output. RUC is appropriate when the task involves code-shaped work such as "
        "iteration, counting, sorting, validation, aggregation, state tracking, or "
        "repeatable file processing, but also contains LLM-shaped work such as "
        "classification, summarization, fuzzy matching, tone analysis, relevance judgment, "
        "ambiguity resolution, or creative writing."
        "\n\n"
        "RUC is ideal for tasks where the semantic question is fuzzy, but the execution "
        "must be exact."
        "\n\n"
        "For example, if the user asks to review a collection of support tickets "
        "and count how many are angry, frustrated, neutral, or positive, do not try "
        "to keep the whole process in conversational memory. Use RUC to run a "
        "procedural workflow that loops over the tickets and aggregates counts, "
        "while delegating only the tone classification step to an LLM-style semantic "
        "function."
        "\n\n"
        "RUC also applies when the user wants the LLM to generate content repeatedly "
        'at a specific scale. For example, "write 300 tweets about AI safety" is a RUC task: '
        "the LLM concentrates on writing a tweet, and the procedural code handles calling the "
        "LLM in a loop. This way, not only does the LLM not need to worry about keeping count, "
        "but it also allows each LLM invocation to focus solely on writing a single tweet, "
        "which is more likely to yield high-quality output."
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
async def execute_workflow(ctx: fastmcp.Context) -> dict:
    # TODO: Implement the task here
    # ...
    return retval # "retval" is some JSON-serializable dict or list.
```

---

Your Execution Environment: python-3.12:slim Docker image with /workspace mount and scientific/data libraries

The environment you'll be running in is a python-3.12:slim Docker image, with standard 
libraries such as `json`, `re`, `pydantic`, and so on. In addition, the environment comes installed
with the following scientific/data libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, and
`statsmodels`. The Docker container is sandboxed against network and filesystem access; but you
have read/write access to the folder /workspace, which you may use for reading source data and/or
writing output files as requested by the task. You can also write to /tmp for "scratchpad" work,
of course.

The environment also has `fastmcp` installed, which provides the `Context` class for communicating
with the LLM (more on that below).

(NOTE: the __future__ library in your environment is extremely flakey and unreliable, and
must be avoided (not that you should need it anyway)).

---

FastMCP library: fastmcp.Context imported for type safety, but you shouldn't need to use it.

This environment has a library called FastMCP installed, so you can `import fastmcp` and
declare `ctx: fastmcp.Context` in your type signatures. You shouldn't need to actually
*use* the FastMCP library (we'll add calls to the LLM  ourselves in a later pass), but the
Context object will be passed into your function, so I figure you'll want it for type signature
purposes. The "ctx" argument is a FastMCP Context object. It's what we'll be using to communicate
with the LLM. Don't worry about it for now. We only provide it here because we'll need to
pass it through to the LLM calling function stubs. More on that later.

---

How to use stub functions as placeholder LLM calls in `execute_workflow`

In an ideal world, you should be able to implement the entire task using only conventional
code, possibly with the help of heuristic tricks involving regexes or string operations
where necessary. Naturally, you may write whatever helper functions or support functions
you want, and to structure your code however you like. For some tasks, conventional procedural
code may entirely suffice to get the job done.

However, in practice, some portion (or multiple portions) of this task might require judgment
calls, inference, reconciliation of noisy or ambiguous information, creative writing, natural
language input or output, and other tasks that are more suitable to an LLM than to a Python
function.

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

Put all such stub functions at the end of your code. 

The stub functions, as well as the main workflow function `execute_workflow`, must be at the top
level of the module, i.e. not nested inside any other function or class. I should be able to
copy-paste your entire code into a Python environment and run `execute_workflow(...)` without 
having to look for it inside a namespace or a class or something.

---

Tips and Guidelines

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
# TODO: Instruct the LLM to write resumable code -- i.e. tell it to think about how it would
# handle interruptions and restarts, and to write code that can pick up where it left off if
# interrupted.

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
        "Here is the structure that your final answer must conform to, expressed as a JSON schema. "
        "Ensure that your final answer strictly adheres to this structure.\\n\\n"
        "```json-schema\\n"
        + json.dumps(result_type.model_json_schema(), indent=2)
        + "\\n```"
    )
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


def _delete_python_function_from_code(
    pycode: str,
    function_name: str,
) -> str:
    """Delete the function with the given name from the given Python code."""
    return _replace_python_function_in_code(pycode, function_name, "")


def _replace_python_function_in_code(
    pycode: str,
    function_name: str,
    new_function_code: str,
) -> str:
    """Replace the function with the given name in the given Python code with new code."""
    # We are *not* going to be clever and "Pythonic" about this.
    # We will use extremely simple iterative line examination using basic string matching.
    # We will not use regexes.
    pycode_lines = pycode.splitlines()

    # Find the index of the line that starts with either "def function_name(" or "async def function_name(".
    function_def_index = None
    for i, line in enumerate(pycode_lines):
        stripped_line = line.strip()
        if stripped_line.startswith(
            f"def {function_name}("
        ) or stripped_line.startswith(f"async def {function_name}("):
            function_def_index = i
            break

    if function_def_index is None:
        raise ValueError(f"Function {function_name} not found in code.")

    # Find the end of the function by looking for the next line that is not indented.
    function_end_index = function_def_index + 1
    while function_end_index < len(pycode_lines):
        if pycode_lines[function_end_index].strip() == "":
            # Blank lines are fine, even if they're not indented. Just skip over them.
            function_end_index += 1
            continue
        if not pycode_lines[function_end_index].startswith((" ", "\t")):
            # This line is not indented, which means it's outside the function.
            # We've found the end of the function.
            break
        function_end_index += 1

    # We also want to delete the comment above the function definition,
    # if there is one and if it's not separated by a blank line.
    # This is to ensure that we delete any TODO comments that might be attached to the function.
    comment_index = function_def_index - 1
    while comment_index >= 0:
        if pycode_lines[comment_index].strip() == "":
            # We've reached a blank line, so we should stop looking for comments.
            break
        if pycode_lines[comment_index].strip().startswith("#"):
            # This is a comment line, so we should include it in the deletion.
            comment_index -= 1
        else:
            # This is not a comment line, so we should stop looking for comments.
            break

    # Delete the function lines.
    code_before_function = "\n".join(pycode_lines[: comment_index + 1])
    code_after_function = "\n".join(pycode_lines[function_end_index:])

    new_pycode = (
        code_before_function + "\n" + new_function_code + "\n" + code_after_function
    )
    return new_pycode


def _delete_all_functions_with_designated_marker(
    pycode: str,
    marker: str,
) -> str:
    """Deletes all functions whose name is somethingsomething_<marker>."""
    # This regex works as follows:
    # - "def " matches the literal string "def ".
    # - "\w+" matches one or more word characters (i.e. the function name before the marker).
    # - "_{marker}" matches the literal string "_<marker>".
    # - "\s*\(" matches zero or more whitespace characters followed by an opening parenthesis.
    pattern = rf"def \w+_{re.escape(marker)}\s*\("
    function_names = re.findall(pattern, pycode)

    for function_name in function_names:
        # Extract the actual function name from the match.
        match = re.match(r"def (\w+)\s*\(", function_name)
        if match:
            actual_function_name = match.group(1)
            pycode = _delete_python_function_from_code(pycode, actual_function_name)
    return pycode


async def _write_workflow(ctx: fastmcp.Context, convo: list[str]) -> dict[str, str]:
    """Write a Python function that performs a procedural workflow that includes
    "fuzzy" operations."""
    await ctx.report_progress(
        progress=0,
        total=None,
        message="Writing workflow code",
    )

    convo = json.loads(json.dumps(convo))  # Deep-copy to ensure immutability.

    # Sidebar: sanity-check first.
    convo_sanity: list[str] = json.loads(
        json.dumps(convo)
    )  # Deep-copy for immutability.
    convo_sanity.append("""
Before we begin, let's perform a sanity-check to make sure that we are able to proceed.
Do you have enough information and guidance to start writing code for the workflow?
Is there anything else you need, that's stopping you from proceeding?
                 
This is your chance to early-out if you feel like you don't have enough to go on.
                 
Discuss the issue with yourself. Apply chain-of-thought reasoning to analyze whether
you should abort before we get started. Aborting is not desirable, but if you simply
can't perform the task with the information and resources at your disposal, then
now is the time to say so.
                 
When you're done with your deliberation, provide your final determination by emitting,
by itself, on its own line, either "READY TO PROCEED", or else 
"ABORT: reason for aborting goes here". This should be the last thing you say.
I will look for this exact line and use it to decide whether to proceed or not.
""")
    sanity_check_result = await ctx.sample(
        messages=convo_sanity,
        system_prompt=RUC_FUNCTION_WRITING_SYSTEM_PROMPT,
        max_tokens=10_000,
    )
    sanity_check_text = sanity_check_result.text
    if not sanity_check_text:
        raise ValueError(
            "Sampling returned no text when trying to perform the initial sanity check."
        )

    if "ABORT:" in sanity_check_text:
        abort_reason = sanity_check_text.split("ABORT:", 1)[1].strip()
        if abort_reason.endswith('"'):
            # Damn thing included the quotes.
            abort_reason = abort_reason[:-1].strip()
        # If there are newlines, split on newlines and keep only the first line.
        if "\n" in abort_reason:
            abort_reason = abort_reason.split("\n", 1)[0].strip()
        raise ValueError(f"Cannot proceed. Reason: {abort_reason}")

    if "READY TO PROCEED" not in sanity_check_text:
        await ctx.warning(
            "Model did not explicitly say 'READY TO PROCEED' in response to the sanity check. "
            "Here's what it said instead: " + sanity_check_text
        )

    # Sanity check complete. We can proceed with writing the workflow code.

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

            convo.append("ASSISTANT REPLY:\n\n" + sample_result.text)

            convo.append(
                "Please emit a very short summary, between one sentence and one paragraph, "
                "in which you explain your implementation approach and the structure of "
                "your code. I will attach this explanation to the results, so that when I "
                "present the results I can also explain the methodology by which these "
                "results were produced."
                "\n\n"
                "Your summary should be detailed but non-technical. Its target reader "
                "should be a PM or a business client. You should explain your approach "
                "in such a way that *they* understand what your code does."
                "\n\n"
                "Your explanation MUST make it very clear what's being handled procedurally "
                "in Python code, what's being handled semantically in LLM calls, and how the "
                "two parts interact with each other."
            )
            strategy_explanation_result = await ctx.sample(
                messages=convo,
                system_prompt=RUC_FUNCTION_WRITING_SYSTEM_PROMPT,
                max_tokens=60_000,
            )
            if not strategy_explanation_result.text:
                raise ValueError(
                    "Sampling returned no text when trying to write the "
                    "workflow strategy explanation."
                )
            strategy_explanation = strategy_explanation_result.text.strip()
            await ctx.info(f"Workflow strategy explanation: {strategy_explanation}")
            # TODO: Return the strategy explanation along with the workflow results, so that it can be presented to the user together with the results.

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

            return {
                "pycode": pycode,
                "implementation_strategy": strategy_explanation,
            }

        except Exception as e:
            await ctx.warning(
                f"Attempt {attempt}/{MAX_ATTEMPTS} failed while writing workflow code: {e}"
            )

    raise ValueError(
        f"RUC failed to write any initial workflow code after {MAX_ATTEMPTS} attempts. "
        "There is probably something wrong with RUC's code generator prompt, or with the "
        "model it's calling."
    )


async def _write_implementation_for_stub(
    ctx: fastmcp.Context,
    stubname: str,
    pycode: str,
    convo: list[str],
) -> str:
    """Replace the given stub function with a real implementation."""
    logger = logging.getLogger(__name__)
    convo = json.loads(json.dumps(convo))  # Deep-copy to ensure immutability.

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
) -> str:
    """Find any stub functions in the given code, and replace them with real implementations."""
    convo = json.loads(json.dumps(convo))  # Deep-copy to ensure immutability.

    # We marked these stub functions with a special syntax in the TODO comment,
    # so we can grep for them.
    # The syntax is
    # "TODO(llm_stub: stub_function_name): description of what the function should do".
    stub_pattern = re.compile(r"TODO\(llm_stub:\s*(\w+)\):\s*(.*?)\n")
    stubs = stub_pattern.findall(pycode)

    if not stubs or len(stubs) == 0:
        return pycode

    await ctx.report_progress(
        progress=0,
        total=None,
        message=f"Implementing {len(stubs)} mid-workflow LLM callbacks",
    )

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
) -> Any:
    """Execute the given workflow code and return the result."""
    namespace: dict[str, Any] = {
        "__name__": "ruc_generated_workflow",
        "__package__": None,
    }
    exec(pycode, namespace, namespace)

    execute_workflow = namespace.get("execute_workflow")
    if not callable(execute_workflow):
        raise ValueError(
            "Generated code did not define callable execute_workflow(ctx)."
        )

    workflow_result = execute_workflow(ctx=ctx)
    result = (
        await cast(Any, workflow_result)
        if inspect.isawaitable(workflow_result)
        else workflow_result
    )
    return result


async def _explore_data(
    ctx: fastmcp.Context,
    convo: list[str],
) -> str:
    """Explore the given data sources. Return a report containing observations
    about the data, as well as a sample of records from each data source."""

    convo = json.loads(json.dumps(convo))  # Deep-copy to ensure immutability.

    convo_sanity: list[str] = json.loads(
        json.dumps(convo)
    )  # Deep-copy for immutability.
    convo_sanity.append("""
Does the description of data sources literally just tell us that there are no
data sources? I.e. is it, like, a list saying, "This list is empty," or a string saying
something like, "No data sources are needed for this task," or something like that?

Talk about this with yourself. Perform a chain-of-thought reasoning process.

When you're done, emit a final conclusion by writing, on its own line by itself, either
"DATA SOURCE DESCRIPTION INDICATES NO DATA SOURCES", or else
"YES WE ARE BEING DIRECTED TO DATA SOURCES". Just like that, on its own line,
in all caps.
""")
    s_sanity_check_result = await ctx.sample(
        messages=convo_sanity,
        system_prompt=RUC_FUNCTION_WRITING_SYSTEM_PROMPT,
        max_tokens=10_000,
    )
    s_sanity_check_text = s_sanity_check_result.text
    if not s_sanity_check_text:
        await ctx.warning(
            "LLM returned no text when trying to perform the data source sanity check."
        )
    else:
        if "DATA SOURCE DESCRIPTION INDICATES NO DATA SOURCES" in s_sanity_check_text:
            await ctx.info("The data source description indicates no data sources.")
            return ""

        if not ("YES WE ARE BEING DIRECTED TO DATA SOURCES" in s_sanity_check_text):
            await ctx.warning(
                "LLM did not explicitly indicate that we are being directed to data sources."
            )

    report = ""

    system_prompt = """
You're an AI agent that's been assigned a task. The task involves working with some data sources.
You are just beginning to work on the task. The first step of any such endeavor is
data exploration. You need to get a good understanding of the data you'll be working with.

You'll eventually have to write Python code that performs a workflow involving this data,
and in order to do that, you need to understand the data's shape and structure, as well as any
quirks or gotchas or anomalies it might have. Your job at the moment is *not* to actually
perform the requested task. Your job is to explore the data and understand it well enough
that you could write code to process it later.

You're running in a Python 3.12 Docker container. The environment comes installed with `numpy`,
`pandas`, `scipy`, `scikit-learn`, and `statsmodels`, in addition to standard libraries. The
Docker container is sandboxed, but you can access files through a shared mount at `/workspace`
using standard Python file I/O. Using Python, you should be able to do everything you need,
including listing folder contents, reading files, parsing files, and so on. (You do, of course,
have to import the corresponding libraries first.)

The end result of your data exploration will be a report. This report will enumerate the
data source (or sources) that the workflow will access. For each data source, the report will
describe in exact detail how to access it, what format it's in, how to parse it or read it
or iterate it, and any potential irregularities in the data. The report will also include
samples of records from each data source, which can serve as a reference when you later
write code to process the data.

Our conversation will occur in stages. First, I'll show you all the information I have about
the task, the data sources, etc.
Then, we'll enter a loop. Each iteration of the loop will go like this:
STAGE 1: REPORT PRESENTATION. I'll show you the report in its current state.
STAGE 2: DELIBERATION. I'll ask you to perform a chain-of-thought reasoning process on your
    current understanding of the data. This stage is purely for your own benefit, so that
    you can square your thoughts and decide on what to do next.
STAGE 3: ACTION SELECTION. I'll ask you to choose one of the following actions:
    - AMEND_REPORT: Write an amendment to the report. This is your sole means of authoring
        the final report. You can write as many amendments as you want, and they will be
        combined together to form the final report. Each amendment should be a standalone
        chunk of text that can be added to the report to improve it in some way. For example,
        if you notice that a certain data source has some irregularities, you might write an
        amendment describing those irregularities and how to handle them. The report 
        grows monotonically. You may not delete or modify previous amendments; you can only
        add new ones. If you need to change something you wrote in a previous amendment,
        then write a new amendment describing the new knowledge or correction, and explicitly
        call out the fact that this is a correction to a previous amendment.
    - WRITE_CODE: Write a block of Python code that examines the data in some way. This is your
        means of performing "hands-on" exploration of the data. You can write code to read
        the data, to compute statistics about it, to visualize it, or to do anything else that
        might help you understand it better. You can write as many code blocks as you want,
        and they can be as long or as short as you want. Each code block should be a standalone
        chunk of Python code that can be executed independently. When you write a code block, I
        will execute it in a Python environment and show you the results. This can be a very
        powerful tool for understanding the data, so use it often and creatively!
    - FINISH: Declare that you're finished with data exploration, and that the report is complete.
        This action ends the loop.

I might abridge portions of our conversation for the sake of brevity, so you won't necessarily
see every single stage of every single iteration of the loop. But don't worry about that.
Just focus on doing a good job of writing a comprehensive and accurate report.
"""

    convo_base: list[str] = json.loads(json.dumps(convo))

    while True:
        convo = json.loads(json.dumps(convo_base))

        if report:
            convo.append(
                "STAGE 1: REPORT PRESENTATION. "
                "Here is the report as it currently stands:\n\n---\n\n" + report
            )
        else:
            convo.append(
                "STAGE 1: REPORT PRESENTATION. "
                "The report is currently blank, because we haven't done any exploration yet."
            )

        convo.append("""
STAGE 2: DELIBERATION. 
Take some time to think through your current understanding of the data.
Discuss the following topics with yourself.
- What you know about the data so far, based on the information you've been given and any
    exploration you've done up to this point.
- What you still need to learn about the data.
- What's captured so far in the report, and whether it's accurate and comprehensive.
- What still needs to be added to the report, and how you'll go about learning that information.
I don't need you to decide on an action or to write any code just yet.
Just talk through your thoughts.
""")
        s_deliberation = await ctx.sample(
            messages=convo,
            system_prompt=system_prompt,
            max_tokens=20_000,
        )
        if not s_deliberation.text:
            raise ValueError(
                "LLM returned no text in response to the data exploration deliberation prompt."
            )

        convo.append("===\nASSISTANT REPLIED\n===\n\n" + s_deliberation.text)
        convo_base.append("===\nASSISTANT REPLIED\n===\n\n" + s_deliberation.text)

        convo.append("""
STAGE 3: ACTION SELECTION.
What would you like to do next? Talk it over with yourself, and then choose one of the 
following actions:
- AMEND_REPORT: Write an amendment to the report.
- WRITE_CODE: Write a block of Python code that examines the data in some way, or performs
    some tentative or exploratory processing on the data.
- FINISH: Declare that you're finished with data exploration, and that the report is complete.
When you've decided on an option, write the words `ACTION_SELECTED: <chosen action>`,
where <chosen action> is one of the three options listed above. Make sure that
`ACTION_SELECTED: <chosen action>` on its own line, and is written in all caps, e.g.:
ACTION_SELECTED: AMEND_REPORT

If you chose WRITE_CODE, then after writing `ACTION_SELECTED: WRITE_CODE`, also write a block
of Python code enclosed in triple backticks labeled "```python". IMPORTANT: DO NOT PRINT TO
STDOUT IN THIS CODE BLOCK. Instead, append your output to a string variable called
`data_exploration_log`. This variable will be initialized as an empty string at the beginning
of your code block. I will execute your code block in a Python environment and then show you
the contents of `data_exploration_log`. This is how you should produce output from your
code block, since you won't have access to standard output when your code block is executed.
Try to write your code defensively -- be mindful of things like looping over contents of files
when you don't know their length, and so on. You don't want to accidentally write an infinite
loop that appends to `data_exploration_log`, or getting stuck on a locked resource or something.
Remember, you don't have to actually *do* the requested workflow task yet -- you're just
exploring for now.  Remember that you aren't being asked to write the final code that will 
ultimately perform the requested task; this is just for exploration and learning purposes.
Just be careful not to accidentally screw up the data while you're exploring it, e.g. by
stomping the data files or something. Keep your operations non-destructive.

If you chose AMEND_REPORT, then after writing `ACTION_SELECTED: AMEND_REPORT`, also write a
text block enclosed in triple backticks labeled "```reportamendment". This block should contain
the text of your amendment to the report. I will copy-paste this onto the end of the report,
so it should be written in a way that makes sense as an addition to the existing report.
""")

        s_action = await ctx.sample(
            messages=convo,
            system_prompt=system_prompt,
            max_tokens=20_000,
        )
        if not s_action.text:
            await ctx.warning(
                "LLM returned no text in response to the data exploration action selection prompt. "
                "I'll just prompt it again and hope for a better result this time."
            )
            continue

        if "ACTION_SELECTED: FINISH" in s_action.text:
            # This is our exit point from the data exploration loop!
            return report

        # Produce a status update.
        convo.append("===\nASSISTANT REPLIED\n===\n\n" + s_action.text)
        convo.append("""
In order to keep the user updated about what you're doing and to make sure you haven't frozen
or gotten stuck, please provide a status update to explain the action you're about to
perform. This status update should be a sentence fragment, only a handful of words long.

It should look something like this:
STATUS: Determining what version of Word quarterly_earnings.doc is in
or
STATUS: Reading first 100 records of customers.csv
or
STATUS: Found irregularities in orders.json
or
STATUS: Delimiters are pipes, not commas or tabs

You get the idea.

Try to make your status updates informative and specific, so that the user can understand
exactly what you're doing. Here are a few examples of BAD status updates:
- STATUS: Working on it (not informative, doesn't say what "it" is)
- STATUS: Exploring data (not informative, doesn't say what aspect of the data is being explored
      or how it's being explored)
- STATUS: Writing code (not informative, doesn't say what the code is doing or why you want it 
      to do that thing)
- STATUS: Amending report (not informative, doesn't say what the amendment is about or what it
      adds to the report)
                     
Because the space on the status line is so limited, try to avoid saying things like
"Writing code" or "Amending report". We *know* that you're doing one of those things,
because we are aware of how the data exploration loop works. Instead, focus on the specific
content of what you're doing. For example, don't say, "Writing code to determine number of
records in customers.csv". Instead, say, "Determining number of records in customers.csv".
Don't say, "Amending report to document 27 records with missing fields in orders.json".
Instead, say, "27 records with missing fields in orders.json".

Emit your reply as a line that starts with "STATUS:", followed by your status update.
""")
        status_update = await ctx.sample(
            messages=convo,
            system_prompt=system_prompt,
            max_tokens=1000,
        )
        if not status_update.text:
            await ctx.warning(
                "LLM returned no text in response to the data exploration status update prompt. "
            )
        else:
            status_text = status_update.text.strip()
            if "STATUS:" in status_text:
                status_text = status_text.split("STATUS:", 1)[1].strip()
            if "\n" in status_text:
                status_text = status_text.split("\n", 1)[0].strip()
            await ctx.report_progress(
                progress=0,
                total=None,
                message=status_text,
            )

        if "ACTION_SELECTED: AMEND_REPORT" in s_action.text:
            amendment = _extract_labeled_code_block(s_action.text, "reportamendment")
            if not amendment:
                await ctx.warning(
                    "LLM indicated that it wanted to amend the report, "
                    "but did not include a report amendment block."
                )
                continue
            report += "\n\n" + amendment.strip()

        elif "ACTION_SELECTED: WRITE_CODE" in s_action.text:
            code = _extract_labeled_code_block(s_action.text, "python")
            if not code:
                await ctx.warning(
                    "LLM indicated that it wanted to write code, "
                    "but did not include a Python code block."
                )
                continue

            code = 'data_exploration_log = ""\n\n' + code.strip()

            # Execute the code and capture its output.
            local_namespace: dict[str, Any] = {}

            try:
                exec(code, {}, local_namespace)
                if "data_exploration_log" not in local_namespace:
                    raise ValueError(
                        "The Python code block that the LLM provided did not define "
                        "a variable called `data_exploration_log`."
                    )
                s_data_exploration_log = local_namespace["data_exploration_log"]
                result_str = (
                    json.dumps(s_data_exploration_log, indent=2)
                    if not isinstance(s_data_exploration_log, str)
                    else s_data_exploration_log
                )
            except Exception as e:
                result_str = f"Error executing code block: {e}"
                # Also provide a stack trace for debugging purposes.
                result_str += "\n\nStack trace:\n" + traceback.format_exc()

            # Append the code itself and its results to the conversation,
            # so that the LLM can refer to them in future deliberation and action selection.
            convo_base.append(
                "===\nLLM-PROVIDED CODE BLOCK\n===\n\n```python\n"
                + code.strip()
                + "\n```\n\n===\nOUTPUT OF LLM-PROVIDED CODE\n===\n\n"
                + result_str
            )

        else:
            await ctx.warning(
                "LLM did not select a valid action. Here's what it said:\n"
                + s_action.text
            )


async def _repair_workflow_code(
    ctx: fastmcp.Context,
    pycode: str,
    errormessage: str,
) -> str:
    """If the workflow code produces an error, ask the model to debug it and fix any issues."""
    system_prompt = f"""
You're helping to fix an error in a Python script. The script was written to solve a one-off
request for a task that required interopration between LLM calls and traditional procedural code.

The code was invoked as an MCP server. It runs inside a Docker container built from a
python:3.12-slim image, with a few science and data libraries installed
(numpy, pandas, scipy, scikit-learn, and statsmodels).

The user will show you the code itself, and will then show you the error that caused the code
to break.

Your job will be to make as minimal a change as possible to the function that the error occurred
in, in order to resolve the error. We don't want you to change the intended behavior of the code.
We just want you to make it not crash.

EDITOR CONSTRAINTS
I don't want you to rewrite the entire file. That's too much work, and raises the risk of error
too high. Likewise, I don't want you to just apply a standalone diff or patch, because, again,
there's too much risk of error -- getting line numbers wrong, getting indentation wrong, etc.

Therefore, the granularity of your edit will be to determine which function in the root namespace
(i.e. whose definition has no indentation) the error occurred in. You will then rewrite that
function, and *only* that function. Your rewrite will be nearly identical to the function's
current form, with the only changes being the ones that are necessary in order to fix the error.
"""
    convo: list[str] = []
    convo.append(f"""
```python
{pycode}
```
""")
    convo.append(f"""
```error
{errormessage}
```
""")
    convo.append(
        "What exactly went wrong? "
        "Where did it go wrong? "
        "What function needs to be edited in order to fix it? "
        "\n\n"
        "Don't actually write the patch yet. "
        "Just discuss and deliberate the matter first. "
        "I want to hear your thoughts before we proceed."
    )
    s_deliberation = await ctx.sample(
        messages=convo,
        system_prompt=system_prompt,
        max_tokens=20_000,
    )
    if not s_deliberation.text:
        raise ValueError(
            "LLM returned no text in response to the workflow code repair deliberation prompt."
        )
    convo.append("===\nASSISTANT REPLIED\n===\n\n" + s_deliberation.text)

    convo_status: list[str] = json.loads(json.dumps(convo))
    convo_status.append("""
Before we proceed with the actual edit, write a status blurb that can be pasted into a progress bar
to explain what this problem is all about. This blurb should be a sentence fragment, extremely
concise but as specific as possible. Examples:
STATUS: Syntax error in function that imputes missing sales values
STATUS: KeyError in function that aggregates customer data by region
STATUS: Malformed regex in logfile parser
You get the idea.

Emit your reply as a line that starts with "STATUS:", followed by your status update.
""")
    s_status = await ctx.sample(
        messages=convo_status,
        system_prompt=system_prompt,
        max_tokens=1000,
    )
    if not s_status.text:
        await ctx.warning(
            "LLM returned no text in response to the workflow code repair status update prompt. "
        )
    else:
        status_text = s_status.text.strip()
        if "STATUS:" in status_text:
            status_text = status_text.split("STATUS:", 1)[1].strip()
        if "\n" in status_text:
            status_text = status_text.split("\n", 1)[0].strip()
        await ctx.report_progress(
            progress=0,
            total=None,
            message=status_text,
        )

    convo.append("""
Let's pause for a sanity check. Can this problem indeed be fixed?
Or is the code so mangled and distorted from previous edit attempts
as to be basically irreparable at this point?

Think about the matter for a minute. I'd like to hear your thoughts.

When you're ready, emit one of two conclusion patterns.

-EITHER-

Say, on its own line, just like this:
FUNCTION TO EDIT: function_name
followed by a triple-backticked block labeled "```python".
In that Python block, provide a complete rewrite of the fuction you've specified,
including its definition. Your rewrite should be as identical to the original as
possible, while still fixing the error.

-OR-

Say, on its own line, just like this:
IRREPARABLE: Lorem ipsum dolor logit...
Except, of course, replace the lorem ipsum text with an explanation of why you
feel the workflow code is beyond repair.
""")
    s_conclusion = await ctx.sample(
        messages=convo,
        system_prompt=system_prompt,
        max_tokens=20_000,
    )
    if not s_conclusion.text:
        raise ValueError(
            "LLM returned no text in response to the workflow code repair conclusion prompt."
        )
    convo.append("===\nASSISTANT REPLIED\n===\n\n" + s_conclusion.text)

    if "IRREPARABLE:" in s_conclusion.text:
        explanation = s_conclusion.text.split("IRREPARABLE:", 1)[1].strip()
        raise ValueError(f"The workflow code is broken beyond repair. {explanation}")

    if not "FUNCTION TO EDIT:" in s_conclusion.text:
        raise ValueError(
            "LLM did not provide a valid conclusion in response to the "
            "workflow code repair conclusion prompt. "
            "Here's what it said:\n" + s_conclusion.text
        )

    function_name = (
        s_conclusion.text.split("FUNCTION TO EDIT:", 1)[1].split("\n", 1)[0].strip()
    )
    new_function_code = _extract_labeled_code_block(s_conclusion.text, "python")
    if not new_function_code:
        raise ValueError(
            "Failed to extract a Python code block from the LLM's response "
            "when we asked it to provide a repaired function."
        )
    new_pycode = _replace_python_function_in_code(
        pycode,
        function_name,
        new_function_code,
    )
    return new_pycode


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
    data_sources: Annotated[
        str | None,
        Field(
            description=(
                "Optional. Provide instructions on where to find the data sources that the "
                "workflow should rely on. For each data source, explain how to access it "
                "(e.g. file URI, API endpoint, database connection string) and what format "
                "the data is in (e.g. CSV, JSON, SQL). Not all tasks rely on data sources, "
                "but those that do will require this information."
            )
        ),
    ] = None,
    how_to_present_output: Annotated[
        str | None,
        Field(
            description=(
                "Optional. Provide instructions for how the final output should be presented, "
                "what format it should be in, and where/how it should be delivered. For example, "
                "do you want just a string? Do you want a JSON object? Do you want the "
                "results written to a file? If so, what should the file be called, and where "
                "should it be saved? Should the results overwrite an existing file, or should "
                "they be appended to it, or should a new file be created with a unique name? "
                "Should a file be written in-place, or would you prefer to see a list of "
                "edits in unified diff format that you can review and apply using a diff tool? "
                "RUC doesn't have access to your entire filesystem, but it *does* run in a "
                "Docker container with /workspace as a mount point; so if you want it to "
                "write to the filesystem for output, specify a path under /workspace (e.g. "
                "/workspace/results.json)."
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
    write_execution_workflow_file: Annotated[
        str | None,
        Field(
            description=(
                "Optional, and purely for debugging purposes. If provided, RUC will save the "
                "generated workflow code to a file at this path. This enables the human user to "
                "inspect the generated code, which can be useful if something goes wrong during "
                "execution and you want to understand why."
                "\n\n"
                "The file should be specified with a full pathname under /workspace, "
                "since that's the only part of the filesystem that the RUC Docker container has "
                "access to. Ideally, you'd find a temp folder or gitignored folder somewhere in "
                "your project directory. For example, "
                '"/workspace/temp/ruc_generated_workflow.py" would be a good value for this '
                "field."
                "\n\n"
                "If the file already exists, it will be overwritten."
                "\n\n"
                "The LLM is never shown the value of this field. It's not passed to the LLM in "
                "its prompts, and the LLM has no ability to write its own code to this file "
                "anyway. Do not put instructions about saving the execution workflow file in the "
                "task description or any other fields, because the LLM won't be able to follow "
                "those instructions, and it might get confused by them."
                "\n\n"
                "Under normal circumstances, you should never need to use this field. It's "
                "provided merely as a debugging aid for particularly tech-savvy users. Do not "
                "use this field unless you have a specific reason to want to inspect the generated "
                "workflow code."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Execute a mixed semantic/procedural RUC workflow.

    This tool ingests optional data sources, asks the model to generate workflow
    code, replaces semantic stubs with structured LLM-call implementations, and
    runs the resulting workflow against loaded records.

    NOTE: Planned future expansions can include caches of code for frequently
    requested workflows.

    Returns a status payload containing execution notes and a `result` field.
    """
    logger = logging.getLogger(__name__)
    logger.info("execute_semantic_code_workflow started for task: %s", task_description)
    start_time = time.time()

    # Build the basic conversational context for the workflow execution. This will be fed
    # into the LLM as part of the prompt, and will serve as the definition of this task.

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

    if data_sources:
        convo.append(
            "Here are the data sources that you should rely on for this task:"
            "\n\n" + data_sources
        )

    if how_to_present_output:
        convo.append(
            "INSTRUCTIONS FOR HOW TO PRESENT YOUR OUTPUT: "
            "\n\n" + how_to_present_output
        )

    # First perform data exploration (assuming we have data to explore.)
    if data_sources:
        convo.append(
            "Here are the data sources that you should rely on for this task:"
            "\n\n" + data_sources
        )

        await ctx.report_progress(
            progress=0,
            total=None,
            message="Data exploration in progress",
        )

        data_exploration_report = await _explore_data(ctx, convo)
        await ctx.report_progress(
            progress=0,
            total=None,
            message="Data exploration complete",
        )
        convo.append(
            "We ran a preliminary exploratory analysis of the data sources. "
            "Here is the report we generated from that analysis:\n\n---\n\n"
            f"{data_exploration_report}"
        )

    workflow_generation_result = await _write_workflow(ctx, convo)
    pycode = workflow_generation_result["pycode"]
    implementation_strategy = workflow_generation_result["implementation_strategy"]

    logger.info(
        "Workflow written. Now checking for any stub functions that need implementations, "
        "and replacing them with real code."
    )

    await ctx.report_progress(
        progress=0,
        total=None,
        message="Implementing mid-workflow LLM calls (if any).",
    )
    pycode = await _replace_all_stubs_with_implementations(ctx, pycode, convo)
    pycode = _delete_all_functions_with_designated_marker(pycode, "obsolete_stub")

    # --------------------------------
    # DELIBERATELY INTRODUCE AN ERROR SO THAT WE CAN TEST OUR ERROR HANDLING CODE.
    # TODO DEBUG DELETE THIS LATER.
    # Find the string "async def execute_workflow".
    # Then, *after* that string, find the first occurrence of the word "return",
    # and replace it with "returnt" (a misspelling of "return" that will cause a syntax error when we try to execute the code).
    # This simulates a situation where the LLM made a mistake in its code generation.
    workflow_function_def_position = pycode.find("async def execute_workflow")
    if workflow_function_def_position != -1:
        return_position = pycode.find("return", workflow_function_def_position)
        if return_position != -1:
            pycode = (
                pycode[:return_position]
                + "returnt"
                + pycode[return_position + len("return") :]
            )
    # -----------------------------------

    execution_notes = "\n\n"

    while True:
        # DEBUG: Save pycode to a local file, so we can inspect it if anything goes wrong during
        # execution.
        if write_execution_workflow_file:
            try:
                with open(write_execution_workflow_file, "w", encoding="utf-8") as f:
                    f.write(pycode)
                execution_notes += f"Generated workflow code was written to file: {write_execution_workflow_file}\n\n"
            except Exception as e:
                await ctx.error(
                    f"Failed to write generated workflow code to file {write_execution_workflow_file}: {e}",
                )
                execution_notes += (
                    f"Warning: Failed to write generated workflow code to file: "
                    f"{write_execution_workflow_file}\n\n"
                )

        await ctx.report_progress(
            progress=0,
            total=None,
            message="Executing workflow.",
        )

        workflow_error = None
        try:
            runresult = await _execute_workflow_code(ctx, pycode)
            break  # If execution is successful, break out of the loop and return the result.

        except Exception as e:
            logger.error(
                f"Workflow execution failed: {e}",
                exc_info=True,
            )
            workflow_error = e

        await ctx.report_progress(
            progress=0,
            total=None,
            message=f"Workflow crashed. Repairing and retrying.",
        )
        pycode = await _repair_workflow_code(ctx, pycode, str(workflow_error))

    elapsed_seconds = int(time.time() - start_time)

    # Report time elapsed in the format "00h04m36s".
    # Hours can be any number of digits, but minutes and seconds should
    # always be two digits (e.g. "00h04m36s", not "00h4m36s").
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_elapsed_pretty_msg = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"

    await ctx.report_progress(
        progress=1,
        total=1,
        message=f"Done after {time_elapsed_pretty_msg}",
    )

    retval = {
        "status": "success",
        "execution_time_seconds": elapsed_seconds,
        "implementation_strategy": implementation_strategy,
        "execution_notes": execution_notes.strip(),
        "result": runresult,
    }

    return retval


def main() -> None:
    """Entrypoint for local development."""
    configure_logging()
    logging.getLogger(__name__).info("Starting RUC MCP server over stdio")
    mcp.run(transport="stdio")
