import fastmcp
import json
import pydantic
import typing

import asyncio
from typing import Any
import fastmcp


async def execute_workflow(
    data_source_records: dict[str, list[Any]], ctx: fastmcp.Context
):
    records = next(iter(data_source_records.values()))

    tasks = [
        classify_irish_name(
            {
                "customer_id": row["customer_id"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
            },
            ctx,
        )
        for row in records
    ]

    classifications = await asyncio.gather(*tasks)

    per_row = []
    irish_rows = []

    for row, classification in zip(records, classifications):
        entry = {
            "customer_id": row["customer_id"],
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "classification": classification["classification"],
            "confidence": classification["confidence"],
            "reason": classification["reason"],
        }
        per_row.append(entry)
        if classification["classification"] == "irish_sounding":
            irish_rows.append(entry)

    return {
        "per_row_classifications": per_row,
        "irish_sounding_rows": irish_rows,
    }


# TODO(llm_stub: classify_irish_name): Given a customer's first and last name, classify
# whether the full name sounds Irish based on linguistic cues (e.g., Saoirse, Niamh, Aoife,
# O'-, Mc-/Mac- prefixes, typical Irish phonetic patterns). Return a dict with keys:
# "classification" (one of: "irish_sounding", "not_irish_sounding", "uncertain"),
# "confidence" (float 0-1), and "reason" (short string explanation).
async def classify_irish_name_obsolete_stub(arg: dict, ctx: fastmcp.Context) -> dict:
    raise NotImplementedError(
        "classify_irish_name: LLM stub that classifies whether a given first+last name "
        "sounds Irish, returning classification, confidence, and reason."
    )


async def classify_irish_name(arg: dict, ctx: fastmcp.Context) -> dict:
    system_prompt = "TODO PASTE SYSTEM PROMPT CONTENTS HERE"

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

    convo.append(f"LLM's brainstorm and reasoning:\n{brainstorm_text}")
    convo.append(
        "Now, based on the above brainstorm and reasoning, provide a final answer to the "
        "question in a structured format as a JSON object."
    )

    result_type = pydantic.create_model(TODO_PROVIDE_PYDANTIC_MODEL_ARGS_HERE)

    formal_structured_sample_result = await ctx.sample(
        messages=convo,
        system_prompt=system_prompt,
        max_tokens=5000,
        result_type=result_type,
    )
    return formal_structured_sample_result.result.model_dump()
