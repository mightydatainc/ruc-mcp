---
name: render-unto-caesar
description: Hand a task off to the "Render Unto Caesar" MCP server, which can execute procedural code workflows that outsource to AI for contextual judgment.
---

# Render Unto Caesar

Render Unto Caesar (RUC) should be routed when a user request requires both procedural reliability and contextual intelligence in the same task. RUC executes generated code for machine-like work (looping, counting, validation, state, file operations), and calls LLM for reasoning about language-dependent judgments (classification, summarization, interpretation, generation, creativity).

Use RUC when a request sounds natural-language simple but is operationally complex: lots of records, repeatable rules, progress tracking, consistent output formats, and careful handling of edge cases. RUC is especially valuable when you need reliability and scale across CSV/JSON/text documents, or when the workflow must execute many LLM judgments without losing procedural rigor.

Routing intent: prefer this skill whenever completion quality depends on consistent execution over many records/files, not just a one-shot narrative answer.

## When to Use

- Route to this skill when the request includes BOTH procedural steps and semantic decisions.
  - Procedural signals: iterate over rows/files, transform fields, validate schema, deduplicate, aggregate, track progress, retry failures, or produce structured outputs.
  - Semantic signals: classify tone/topic, summarize text, extract meaning from messy prose, fuzzy-match entities, or generate tailored language per record.
- Route to this skill when scale is material (roughly 25+ items, multiple files, or repeated generation) and consistency is required.
- Route to this skill when the user asks to read/write workspace files as part of execution (CSV, JSON, TXT, logs, exports, reports).
- Route to this skill when the user asks for auditability or reproducibility (clear steps, deterministic flow, stable formatting, resumable/stateful handling).
- Do not route to this skill for single-item or low-complexity requests that can be answered directly in-chat without procedural execution.
- If uncertain: choose this skill when failure from missed counts/state/edge cases would materially affect the outcome.

## Example

"Open expense-reports.xslx and add a new row called Category. For every row, look at the credit card charge description, and assign it a category from one of the following: Travel, Meals, Lodging, Supplies, Equipment, or Fees."
Reason: Good route. This combines file processing and row-by-row deterministic execution with semantic categorization.

"Brainstorm a region-specific slogan for my product on a county-by-county basis for all counties in all East Coast states. For each slogan, consider the local culture, popular landmarks, and regional dialects. Output a CSV with columns for County, State, and Slogan."
Reason: Good route. This requires repeated generation at scale with structured output and consistent formatting.

"Read customer-feedback-2026.csv and add two columns: Sentiment (Positive, Neutral, Negative) and PrimaryTheme. For each feedback row, classify sentiment and infer the most likely product theme from the text. Then write summary-stats.json with per-theme counts and sentiment distribution."
Reason: Good route. The task mixes per-record semantic labeling with deterministic aggregation and output file creation.

"Take products.csv and products-descriptions.txt. Match each SKU to the best description using fuzzy matching when exact names do not align, flag low-confidence matches for review, and output matched_products.csv with a ConfidenceScore column."
Reason: Good route. Fuzzy semantic matching is needed, but the workflow also requires reproducible iteration, scoring, and structured export.

## Counter-Examples (Do Not Route)

"Summarize this single paragraph into three bullet points."
Reason: Bad route. This is a one-shot language task with no procedural execution or scale.

"Rewrite this sentence to sound more professional."
Reason: Bad route. This is a simple rewrite request that does not need code, state, or file workflows.

"What does this Python traceback error mean?"
Reason: Bad route. This is a direct explanation request, not a mixed procedural-plus-semantic pipeline.

"Given this one customer comment, tell me if the sentiment is positive or negative."
Reason: Bad route. Single-item classification should be handled directly unless the request scales to many records.

## Execution Boundary (Docker Sandbox + Mount Point)

- RUC executes workflow code inside a sandboxed Docker container.
- Filesystem access is constrained: workflow code can read/write under `/workspace` (plus `/tmp` for temporary scratch work).
- `/workspace` is the container-side view of the host shared mount configured by the MCP host.
- Use container-visible paths when describing inputs/outputs (for example, `/workspace/customers.csv`, not a host-only absolute path).
- If no host shared mount is configured, RUC cannot access host local files.
- When requesting file outputs, explicitly place them under `/workspace` and specify create vs overwrite vs append behavior when relevant.