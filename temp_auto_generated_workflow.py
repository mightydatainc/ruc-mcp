import fastmcp
import json
import pydantic

import json
from typing import Any
import fastmcp


async def execute_workflow(data_source_records: dict[str, list[Any]], ctx: fastmcp.Context):
    all_rows: list[dict[str, Any]] = []

    for rows in data_source_records.values():
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    all_rows.append(row)

    unique_rows: list[dict[str, Any]] = []
    seen_row_fingerprints: set[str] = set()

    for row in all_rows:
        fingerprint = _record_fingerprint(row)
        if fingerprint in seen_row_fingerprints:
            continue
        seen_row_fingerprints.add(fingerprint)
        unique_rows.append(row)

    irish_sounding_customers: list[dict[str, str]] = []
    irish_sounding_count = 0
    non_irish_sounding_count = 0

    classification_cache: dict[str, dict[str, Any]] = {}

    for row in unique_rows:
        full_name = _extract_full_name(row)
        cache_key = full_name.strip().lower() if full_name else f"__row__:{_record_fingerprint(row)}"

        if cache_key in classification_cache:
            llm_result = classification_cache[cache_key]
        else:
            llm_result = await classify_name_irish_sounding(
                {"name": full_name, "record": row},
                ctx,
            )
            classification_cache[cache_key] = llm_result

        is_irish = bool(llm_result.get("is_irish_sounding", False))
        reason = str(llm_result.get("reason", ""))

        if is_irish:
            irish_sounding_count += 1
            irish_sounding_customers.append(
                {
                    "name": full_name,
                    "reason": reason,
                }
            )
        else:
            non_irish_sounding_count += 1

    return {
        "summary": {
            "total_rows": len(unique_rows),
            "irish_sounding_count": irish_sounding_count,
            "non_irish_sounding_count": non_irish_sounding_count,
        },
        "irish_sounding_customers": irish_sounding_customers,
    }


def _extract_full_name(row: dict[str, Any]) -> str:
    full_name = str(row.get("Full Name", "")).strip()
    if full_name:
        return full_name

    first = str(row.get("First Name", "")).strip()
    last = str(row.get("Last Name", "")).strip()
    combined = f"{first} {last}".strip()
    if combined:
        return combined

    fallback = str(row.get("Name", "")).strip()
    return fallback


def _record_fingerprint(row: dict[str, Any]) -> str:
    try:
        return json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        # Fallback if row contains non-JSON-serializable values.
        items = sorted((str(k), str(v)) for k, v in row.items())
        return json.dumps(items, ensure_ascii=False)


async def classify_name_irish_sounding_obsolete_stub(arg: dict[str, Any], ctx: fastmcp.Context) -> dict[str, Any]:
    # TODO(llm_stub: classify_name_irish_sounding): Determine whether the provided customer full name is Irish-sounding, and return structured output like {"is_irish_sounding": bool, "reason": str}.
    raise NotImplementedError(
        "classify_name_irish_sounding is a placeholder for an LLM call that classifies a customer name as Irish-sounding or not and provides a reason."
    )



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




async def classify_name_irish_sounding(arg: dict, ctx: fastmcp.Context) -> dict:
    system_prompt = "You are a strict name-origin classifier for data processing.\n\nTask:\nGiven an input JSON object containing a customer name (typically in `name`, possibly with fields in `record`), decide whether the full name is Irish-sounding.\n\nDefinition of Irish-sounding:\nA name is Irish-sounding if the given name and/or surname strongly matches common Irish/Gaelic naming patterns, such as:\n- Gaelic given names (e.g., Saoirse, Cian, Niamh, Aoife, Ronan, Siobhan, Declan, Maeve, Eoin, Aisling, Padraig, Caoimhe, Finbarr, Brigid)\n- Common Irish surnames and variants (e.g., O'Connell, Murphy, Doherty, Byrne, McCarthy, Gallagher, Reilly, O'Neill, Fitzpatrick, Keane, Quinn, Walsh, Kelly, Roche)\n- Recognizable Irish orthographic patterns (e.g., O', Mc/Mac, certain Gaelic spellings)\n\nImportant constraints:\n- Classify only from name patterns; do not infer nationality, ethnicity, or citizenship.\n- Be conservative on ambiguous/global names: mark true only when Irish signal is reasonably strong.\n- If name is missing/blank/unusable, return false with a brief reason.\n\nReasoning behavior:\n- Think carefully before answering.\n- If asked to brainstorm, provide analytical reasoning.\n- When asked for structured output, return only the final structured answer.\n\nStructured output requirements:\nReturn a JSON object with exactly:\n- `is_irish_sounding`: boolean\n- `reason`: one concise factual sentence (8\u2013240 chars), referencing name pattern evidence.\nNo extra keys. No markdown."

    convo = [json.dumps(arg, indent=2)]

    brainstorm_sample_result = await ctx.sample(
        messages=convo,
        system_prompt=system_prompt + (
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
    "IrishNameClassificationResult",
    is_irish_sounding=(
        bool,
        pydantic.Field(
            ...,
            description=(
                "Set to true only if the person's full name sounds Irish based on naming conventions. "
                "Set to false otherwise."
            ),
        ),
    ),
    reason=(
        str,
        pydantic.Field(
            ...,
            min_length=8,
            max_length=240,
            description=(
                "A brief, factual justification for the classification, referencing name patterns "
                "(given name and/or surname). One concise sentence."
            ),
        ),
    ),
    __config__={"extra": "forbid"},
)

    retval = await ruc_submit_sample_request_to_llm(
        messages=convo,
        system_prompt=system_prompt,
        ctx=ctx,
        result_type=result_type,
    )
    return retval
