import fastmcp
import json
import pydantic

import math
from typing import Any

import fastmcp


async def execute_workflow(
    data_source_records: dict[str, list[Any]], ctx: fastmcp.Context
):
    rows: list[Any] = []
    for _source_name, source_rows in data_source_records.items():
        if isinstance(source_rows, list):
            rows = source_rows
            break

    total_rows = 0
    irish_sounding_count = 0
    irish_sounding_customers: list[dict[str, Any]] = []

    for row in rows:
        total_rows += 1

        if isinstance(row, dict):
            customer_id = str(row.get("Customer ID", ""))
            first_name = str(row.get("First Name", ""))
            last_name = str(row.get("Last Name", ""))
        else:
            customer_id = ""
            first_name = ""
            last_name = ""

        llm_arg = {
            "first_name": first_name,
            "last_name": last_name,
        }
        llm_result = await classify_irish_sounding_name(llm_arg, ctx)

        is_irish = bool(llm_result.get("is_irish_sounding", False))
        confidence_raw = llm_result.get("confidence", 0.0)
        reason_raw = llm_result.get("reason", "")

        try:
            confidence = float(confidence_raw)
            if math.isnan(confidence) or math.isinf(confidence):
                confidence = 0.0
        except (TypeError, ValueError):
            confidence = 0.0

        reason = str(reason_raw)

        if is_irish:
            irish_sounding_count += 1
            irish_sounding_customers.append(
                {
                    "customer_id": customer_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "confidence": confidence,
                    "reason": reason,
                }
            )

    non_irish_sounding_count = total_rows - irish_sounding_count

    retval = {
        "summary": {
            "total_rows": total_rows,
            "irish_sounding_count": irish_sounding_count,
            "non_irish_sounding_count": non_irish_sounding_count,
        },
        "irish_sounding_customers": irish_sounding_customers,
    }
    return retval


async def classify_irish_sounding_name_obsolete_stub(
    arg: dict[str, Any], ctx: fastmcp.Context
) -> dict[str, Any]:
    # TODO(llm_stub: classify_irish_sounding_name): Given only first_name and last_name, ask an LLM to classify whether the name is Irish-sounding using common Irish onomastics, and return {"is_irish_sounding": bool, "confidence": float, "reason": str}.
    raise NotImplementedError(
        "classify_irish_sounding_name is a placeholder LLM call; it would classify a first+last name as Irish-sounding or not and return structured confidence and reason."
    )


async def classify_irish_sounding_name(arg: dict, ctx: fastmcp.Context) -> dict:
    system_prompt = 'You are an expert linguistic classifier for personal names.\n\nTask:\nClassify whether a full name is Irish-sounding using only:\n- first_name\n- last_name\n\nInput:\nYou will receive a JSON object with keys:\n- "first_name"\n- "last_name"\n\nDecision rule:\n- Determine whether the combined full name sounds Irish based on common Irish onomastics (typical Irish given-name and surname patterns/usage).\n- This is a fuzzy linguistic judgment, not a claim about ethnicity, nationality, or identity.\n- Do not use any fields other than first_name and last_name.\n- Do not invent external facts about the person.\n\nOutput requirements:\nReturn a judgment with:\n1) is_irish_sounding: boolean\n2) confidence: float in [0.0, 1.0]\n3) reason: brief plain-English explanation grounded only in name-origin cues from first and last name (max ~1-3 sentences, no markdown)\n\nConfidence guidance:\n- High (0.80\u20131.00): strong Irish onomastic signal in first and/or last name.\n- Medium (0.50\u20130.79): mixed or plausible Irish signal with some ambiguity.\n- Low (0.00\u20130.49): weak or no Irish signal.\n\nQuality constraints:\n- Be consistent and deterministic in applying the same criteria.\n- If ambiguous, choose the best-supported label and lower confidence.\n- Keep explanation concise and specific to the provided names.'

    convo = [json.dumps(arg, indent=2)]

    brainstorm_sample_result = await ctx.sample(
        messages=convo,
        system_prompt=system_prompt
        + (
            "\n\n"
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

    convo.append("LLM's brainstorm and reasoning:\n" + brainstorm_text)
    convo.append(
        "Now, based on the above brainstorm and reasoning, provide a final answer to the "
        "question in a structured format as a JSON object."
    )

    result_type = pydantic.create_model(
        "IrishSoundingNameClassification",
        is_irish_sounding=(
            bool,
            pydantic.Field(
                ...,
                description=(
                    "True if the full name (using only first_name and last_name) sounds Irish "
                    "based on common Irish onomastics; otherwise False."
                ),
            ),
        ),
        confidence=(
            float,
            pydantic.Field(
                ...,
                ge=0.0,
                le=1.0,
                description=(
                    "Confidence score for the classification on a 0.0 to 1.0 scale, where 1.0 is highest confidence."
                ),
            ),
        ),
        reason=(
            str,
            pydantic.Field(
                ...,
                min_length=1,
                max_length=300,
                description=(
                    "Brief explanation of the decision grounded in name-origin cues from first and last name only."
                ),
            ),
        ),
    )

    formal_structured_sample_result = await ctx.sample(
        messages=convo,
        system_prompt=system_prompt,
        max_tokens=5000,
        result_type=result_type,
    )
    return formal_structured_sample_result.result.model_dump()


async def ruc_submit_sample_request_to_llm(
    messages: list[str],
    system_prompt: str,
    ctx: fastmcp.Context,
    result_type: type[pydantic.BaseModel],
) -> dict | list | str | int | float | bool:
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

    split_on_json = formal_structured_text.split("```json", 1)
    if len(split_on_json) < 2:
        raise ValueError(
            "LLM did not include a JSON code block in its response to the structured output prompt."
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


async def TODO_PROVIDE_FUNCTION_NAME(
    arg: dict, ctx: fastmcp.Context
) -> dict | list | str | int | float | bool:
    system_prompt = "TODO PASTE SYSTEM PROMPT CONTENTS HERE"

    convo = [json.dumps(arg, indent=2)]

    brainstorm_sample_result = await ctx.sample(
        messages=convo,
        system_prompt=system_prompt
        + (
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
