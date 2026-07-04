# Architecture

This document describes the restaurant outdoor seating extraction pipeline: how data flows, where responsibilities live, and how errors and evaluation are handled. For module-level detail, see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Overview

The system is a batch CLI pipeline. Each restaurant record flows through three agents that read and update a single shared `AgentState` object. Predictions are evaluated against optional ground-truth labels, written to CSV, and low-confidence cases are routed to a human review queue.

No orchestration framework is used — a plain Python function chain over a typed Pydantic state object is sufficient at this scale.

## Pipeline diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         restaurant_agent.cli                            │
│  argparse · logging setup · ClaudeClient or DryRunClaudeClient          │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         restaurant_agent.pipeline                       │
│  load → per-record agent chain → evaluate → write CSV outputs           │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
     data/restaurants.csv           │
              │                     │
              ▼                     │
       ┌──────────────┐             │
       │ data_loader  │             │
       └──────┬───────┘             │
              │ RestaurantRecord    │
              ▼                     │
       ┌──────────────┐             │
       │ AgentState   │◄────────────┘  (one state per record)
       └──────┬───────┘
              │
              ▼
       ┌──────────────────────┐
       │ preprocessing_agent  │  deterministic · preprocessing.py
       └──────────┬───────────┘
                  │ cleaned_text
                  ▼
       ┌──────────────────────┐
       │ extraction_agent     │  LLM · prompts/outdoor_seating.py
       │                      │  → claude_client.py (or DryRunClaudeClient)
       └──────────┬───────────┘
                  │ prediction, confidence, evidence, reasoning
                  ▼
       ┌──────────────────────┐
       │ validation_agent     │  deterministic · validation.py
       └──────────┬───────────┘
                  │ validation_status, needs_review
                  ▼
       ┌──────────────────────┐
       │ evaluation.py        │  aggregate metrics
       └──────────┬───────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 outputs/results.csv   outputs/review_queue.csv
                       (needs_review == True)
```

## Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `cli.py` | Argument parsing, logging setup, client selection (`ClaudeClient` vs `DryRunClaudeClient`), summary output |
| `pipeline.py` | Orchestration: load records, run agent chain per row, write outputs, return `EvaluationSummary` |
| `config.py` | Environment-backed `Settings`; API key required only at real client construction |
| `schemas.py` | Pydantic data contracts: `RestaurantRecord`, `ExtractionResult`, `EvaluationSummary` |
| `state.py` | Shared `AgentState` passed between all agents |
| `data_loader.py` | CSV I/O, row validation, `to_initial_state()` |
| `preprocessing.py` | Pure text cleaning (whitespace, HTML, boilerplate) |
| `validation.py` | Pure business rules (confidence bounds, evidence checks, review threshold) |
| `evaluation.py` | Batch accuracy, precision/recall, failure and low-confidence counts |
| `claude_client.py` | **Only module that imports `anthropic`**; also provides `DryRunClaudeClient` |
| `prompts/outdoor_seating.py` | Versioned system/user prompt templates |
| `agents/preprocessing_agent.py` | Thin wrapper: `clean_evidence` → `cleaned_text` |
| `agents/extraction_agent.py` | Prompt build, LLM call, JSON parse + one repair retry, schema validation |
| `agents/validation_agent.py` | Thin wrapper: business rules → `validation_status`, `needs_review` |
| `logging_config.py` | Centralised `configure_logging()` |

## AgentState flow

Every agent shares the same contract: `AgentState → AgentState`.

```
Initial state (from data_loader)
  restaurant_id, raw_text

After preprocessing_agent
  + cleaned_text

After extraction_agent
  + prediction, confidence, evidence, reasoning
  (or error if LLM/parse failed)

After validation_agent
  + validation_status ("ok" | "flagged" | "failed")
  + needs_review (bool)
```

Fields are additive across stages. Errors set on extraction propagate; validation marks failed records and routes them to review. See `state.py` for the full model.

## Validation layers

Two layers are intentional — schema validity and business trustworthiness are separate concerns.

### Layer 1: Schema validation (`schemas.py`)

Enforced in `extraction_agent.py` immediately after JSON parsing via Pydantic `ExtractionResult`:

- `label` must be `yes`, `no`, or `unknown`
- `confidence` must be 0.0–1.0
- `evidence` is a list of strings; `reasoning` is required

Malformed JSON triggers one repair retry with an explicit “JSON only” follow-up prompt before the record is marked failed.

### Layer 2: Business-rule validation (`validation.py` → `validation_agent.py`)

Applied after a schema-valid extraction:

- Confidence within range (redundant guard)
- `yes`/`no` predictions must include non-empty supporting evidence quotes
- Confidence below threshold (default 0.6) → `needs_review = True`

`validation_status` is `ok` when rules pass, `flagged` when review is needed but extraction succeeded, `failed` when extraction failed.

## Error handling

Errors are isolated **per record** — one bad LLM response does not abort the batch.

| Failure | Handling |
|---------|----------|
| Invalid CSV row | Logged and skipped at load time |
| Missing `cleaned_text` | `error` set on state in extraction agent |
| Anthropic API error | Caught as `LLMRequestError`; `error` set on state |
| JSON parse / schema failure | One repair retry; then `error` set |
| Unexpected exception in agent chain | Caught in `pipeline._process_record`; `validation_status = "failed"`, `needs_review = True` |

Failed records appear in `outputs/results.csv` with an `error` column. They are excluded from accuracy/precision/recall denominators but counted in `EvaluationSummary.failure_count`.

## Evaluation approach

`evaluation.py` compares predictions to optional `expected_label` values from the input CSV:

- **Accuracy** — exact match over all non-failed records with expected labels (including `unknown` vs `unknown`)
- **Precision / recall (yes)** — standard binary yes vs not-yes; `unknown` predictions count as not-yes
- **Low-confidence count** — non-failed records with confidence below threshold
- **Failure count** — records with `error` or `validation_status == "failed"`

The CLI prints these metrics after each run. The sample dataset is for demonstration, not production benchmarking.

## Why the LLM boundary is isolated

All Anthropic SDK usage lives in `claude_client.py`. Extraction logic in `extraction_agent.py` depends on a duck-typed `complete(system, user)` interface — the same shape as `DryRunClaudeClient` and `FakeClaudeClient` in tests.

This design:

- Keeps unit and integration tests **network-free** without mocking frameworks
- Defers API key validation until a real client is constructed
- Allows `--dry-run` with deterministic local responses
- Leaves room for a future `openai_client.py` or similar without touching agents or the pipeline

The provider-specific name (`claude_client.py`, not `llm_client.py`) reflects what the PoC actually uses today rather than a premature multi-provider abstraction.

## Package layout

```
src/restaurant_agent/
├── cli.py, pipeline.py, config.py
├── schemas.py, state.py
├── data_loader.py, preprocessing.py, validation.py, evaluation.py
├── claude_client.py, logging_config.py
├── prompts/outdoor_seating.py
└── agents/
    ├── preprocessing_agent.py
    ├── extraction_agent.py
    └── validation_agent.py
```

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Shared `AgentState` | Uniform agent contract; easy to trace, log, and test stage-by-stage |
| Two validation layers | Schema correctness vs operational trustworthiness |
| Per-record error isolation | Batch resilience; failures tracked separately from model disagreement |
| No orchestration framework | Three agents and one attribute do not justify LangGraph/LangChain overhead |
| Provider-specific LLM client | Mockable boundary; explicit about current provider |
| Human review queue as output | Makes low-confidence routing concrete, not just a metric |
