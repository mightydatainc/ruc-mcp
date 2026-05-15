import fastmcp
import json
import pydantic

from typing import Any
import math
import fastmcp


async def execute_workflow(data_source_records: dict[str, list[Any]], ctx: fastmcp.Context):
    rows = _extract_single_source_rows(data_source_records)

    irish_sounding_customers: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            row = {}

        first_name = _coerce_string(row.get("first_name", ""))
        last_name = _coerce_string(row.get("last_name", ""))
        customer_id = _coerce_string(row.get("customer_id", ""))

        classification = await classify_irish_sounding_name(
            {
                "first_name": first_name,
                "last_name": last_name,
            },
            ctx,
        )

        is_irish_sounding = bool(classification.get("is_irish_sounding", False))
        confidence = _coerce_confidence(classification.get("confidence", 0.0))
        reason = _coerce_string(classification.get("reason", ""))

        if is_irish_sounding:
            irish_sounding_customers.append(
                {
                    "customer_id": customer_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "confidence": confidence,
                    "reason": reason,
                }
            )

    total_rows = len(rows)
    irish_sounding_count = len(irish_sounding_customers)
    non_irish_sounding_count = total_rows - irish_sounding_count

    return {
        "summary": {
            "total_rows": total_rows,
            "irish_sounding_count": irish_sounding_count,
            "non_irish_sounding_count": non_irish_sounding_count,
        },
        "irish_sounding_customers": irish_sounding_customers,
    }


def _extract_single_source_rows(data_source_records: dict[str, list[Any]]) -> list[Any]:
    if not isinstance(data_source_records, dict):
        raise ValueError("data_source_records must be a dict[str, list[Any]].")

    if len(data_source_records) == 0:
        return []

    if len(data_source_records) != 1:
        raise ValueError("Exactly one data source is expected for this workflow.")

    rows = next(iter(data_source_records.values()))

    if not isinstance(rows, list):
        raise ValueError("The data source value must be a list of rows.")

    return rows


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(result):
        return 0.0

    return result


async def classify_irish_sounding_name_obsolete_stub(arg: dict[str, Any], ctx: fastmcp.Context):
    # TODO(llm_stub: classify_irish_sounding_name): Determine whether the provided first name and last name, using only those two fields and common Irish onomastics, should be classified as Irish-sounding; return a structured dict with keys "is_irish_sounding" (bool), "confidence" (number), and "reason" (string).
    raise NotImplementedError(
        "classify_irish_sounding_name would ask the LLM to classify whether the provided first and last name sound Irish and return a structured result."
    )



async def classify_irish_sounding_name(arg: dict, ctx: fastmcp.Context) -> dict:
    system_prompt = "You are classifying whether a customer\u2019s full name is Irish-sounding.\n\nTask:\nGiven only:\n- first_name\n- last_name\n\ndecide whether the full name should be classified as Irish-sounding based on common Irish onomastics. This is a fuzzy linguistic judgment about how the name sounds or is commonly recognized, not a claim about the person\u2019s actual nationality, ethnicity, ancestry, or identity.\n\nRules:\n1. Use only the provided first_name and last_name.\n2. Ignore all other data, metadata, and outside context.\n3. Base the judgment on common Irish naming patterns, including well-known Irish given names, Irish surnames, anglicized Irish forms, and the overall combined impression of the full name.\n4. Do not use demographic stereotypes or non-name-based assumptions.\n5. If the evidence is mixed or ambiguous, make the best single classification you can and lower confidence accordingly.\n6. Be consistent and deterministic: apply the same standard across similar names.\n7. Reason carefully internally, but keep the final explanation brief and do not expose hidden reasoning.\n\nOutput requirements:\nReturn a result with:\n- is_irish_sounding: boolean\n- confidence: number from 0.0 to 1.0\n- reason: brief, specific explanation based only on the first and last name\n\nConfidence guidance:\n- 0.90 to 1.00: very strong Irish onomastic signals\n- 0.70 to 0.89: likely Irish-sounding\n- 0.40 to 0.69: ambiguous or mixed signals\n- 0.00 to 0.39: weak basis for Irish-sounding classification\n\nInterpretation guidance:\n- \u201cIrish-sounding\u201d means the name would commonly be perceived as Irish or plausibly Irish in ordinary onomastic judgment.\n- A clearly Irish surname can outweigh a non-distinctive first name.\n- A clearly Irish given name alone may support classification, but confidence should reflect whether the surname strengthens or weakens the impression.\n- Common English-language names without distinct Irish signals should not be treated as Irish-sounding unless the combined name provides a convincing Irish impression.\n\nBe concise, accurate, and disciplined."

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
                "Whether the full name should be classified as Irish-sounding, using only "
                "the provided first_name and last_name and common Irish onomastics. "
                "Return true or false only; do not use any other fields or outside context."
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
                "Confidence in the classification on a 0.0 to 1.0 scale, where 0.0 means "
                "very uncertain and 1.0 means very confident."
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
                "A brief explanation of why the name does or does not sound Irish, based "
                "only on the first_name and last_name. Keep it concise and specific."
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
