# Restaurant Agent PoC

A proof-of-concept AI agent pipeline that extracts a single structured attribute — **outdoor seating (yes / no / unknown)** — from unstructured restaurant evidence text using Anthropic Claude.

## Problem statement

Restaurant listings, websites, and reviews often describe seating in free text rather than structured fields. Product teams need reliable, auditable answers to questions like “Does this venue offer outdoor seating?” — with confidence scores, supporting quotes, and a path to human review when the model is uncertain.

This PoC demonstrates how to build a small, production-style extraction pipeline: typed data contracts, specialised agents over shared state, strict validation, batch evaluation, and a review queue — without unnecessary framework overhead.

## What this PoC does

For each row in an input CSV, the pipeline:

1. Loads and validates the restaurant record.
2. Cleans the evidence text (deterministic preprocessing).
3. Calls Claude to extract a structured JSON prediction (label, confidence, evidence, reasoning).
4. Applies business-rule validation and routes low-trust cases to a review queue.
5. Writes per-record results and prints batch evaluation metrics (accuracy, precision, recall, failure counts).

The sample dataset (`data/restaurants.csv`) includes optional `expected_label` values for offline evaluation. Metrics are illustrative on ~12 rows, not statistically meaningful.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline diagram, module responsibilities, validation layers, and error-handling approach.

High-level flow:

```
CSV → Data Loader → AgentState → Preprocessing Agent → Extraction Agent (Claude)
    → Validation Agent → Evaluation → results.csv + review_queue.csv
```

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # add ANTHROPIC_API_KEY for live Claude runs
```

## Environment variables

Copy `.env.example` to `.env` and adjust as needed.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Live runs only | — | Anthropic API key for Claude calls |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-20250514` | Model name |
| `ANTHROPIC_MAX_TOKENS` | No | `1024` | Max tokens per request |
| `ANTHROPIC_TEMPERATURE` | No | — (omitted) | Sampling temperature; set e.g. `0.0` for older models only (`claude-sonnet-5` rejects it) |
| `ANTHROPIC_REQUEST_TIMEOUT` | No | `60` | Request timeout (seconds) |
| `CONFIDENCE_THRESHOLD` | No | `0.6` | Below this, records are flagged for human review |
| `DEFAULT_INPUT_PATH` | No | `data/restaurants.csv` | Default `--input` path |
| `DEFAULT_OUTPUT_PATH` | No | `outputs/results.csv` | Default `--output` path |
| `DEFAULT_REVIEW_QUEUE_PATH` | No | `outputs/review_queue.csv` | Default review queue path |
| `LOG_LEVEL` | No | `INFO` | Logging level |

API key validation happens only when constructing the real `ClaudeClient`. Tests and `--dry-run` never require a key.

## CLI usage

```bash
python -m restaurant_agent.cli --help
```

| Flag | Description |
|------|-------------|
| `--input PATH` | Input restaurants CSV |
| `--output PATH` | Results CSV path |
| `--review-queue-output PATH` | Review queue CSV path |
| `--model NAME` | Override Anthropic model |
| `--limit N` | Process at most N records (useful for smoke runs) |
| `--confidence-threshold FLOAT` | Review routing threshold |
| `--backend {pipeline,graph}` | Orchestration backend (default: `pipeline`) |
| `--dry-run` | Deterministic local fake LLM responses; no API calls |

### Example: live run

Requires `ANTHROPIC_API_KEY` in `.env` or the environment.

```bash
python -m restaurant_agent.cli \
  --input data/restaurants.csv \
  --output outputs/results.csv
```

### Example: dry run (no API key)

Runs the full pipeline shape with heuristic local responses:

```bash
python -m restaurant_agent.cli --dry-run --limit 3
```

### Example: LangGraph backend (dry run)

Same behaviour as the default pipeline, using LangGraph orchestration:

```bash
python -m restaurant_agent.cli --backend graph --dry-run --limit 3
```

## Output files

| File | Contents |
|------|----------|
| `outputs/results.csv` | One row per processed record with prediction, confidence, evidence, validation status, correctness (when expected labels exist), and errors |
| `outputs/review_queue.csv` | Subset of results where `needs_review == True` |

Generated output files are gitignored; only `outputs/.gitkeep` is tracked.

## Example output schema

Each results row contains:

| Column | Type | Description |
|--------|------|-------------|
| `restaurant_id` | string | Input record ID |
| `name` | string | Restaurant name |
| `prediction` | `yes` \| `no` \| `unknown` | Extracted label |
| `confidence` | float | Model confidence (0.0–1.0) |
| `evidence` | JSON string | List of verbatim supporting quotes |
| `reasoning` | string | Short explanation |
| `validation_status` | `ok` \| `flagged` \| `failed` | Business-rule outcome |
| `needs_review` | bool | Whether the record was routed to the review queue |
| `is_correct` | bool \| empty | Exact match vs `expected_label` when available |
| `error` | string \| empty | Error message if extraction failed |

Example row (values abbreviated):

```csv
restaurant_id,name,prediction,confidence,evidence,reasoning,validation_status,needs_review,is_correct,error
r001,Harbor View Cafe,yes,0.92,"[""Enjoy breakfast on our sunny patio...""]",Evidence mentions patio.,ok,False,True,
```

After a run, the CLI prints an evaluation summary: total records, failures, accuracy, precision/recall for `yes`, false positives/negatives, low-confidence count, and review-queue count.

## Testing

All automated tests are network-free and do not require an API key.

```bash
pytest
```

Run only the end-to-end smoke tests:

```bash
pytest tests/test_smoke.py
```

Unit tests cover preprocessing, validation, evaluation, agents, pipeline orchestration, and the dry-run client. Agent and pipeline tests use `FakeClaudeClient` from `tests/conftest.py` or `DryRunClaudeClient` for deterministic LLM responses.

## Limitations

- Batch CLI only — no REST API, database, or live scraping.
- Single attribute (outdoor seating) in this PoC.
- Sample evaluation set is too small for meaningful production metrics.
- `--dry-run` uses simple keyword heuristics, not Claude — useful for local demos, not for measuring extraction quality.
- No retry/backoff for rate limits beyond one JSON repair attempt per record.
- English evidence text only; no translation stage.

## Future production improvements

- **More attributes** — extend the agent chain with additional extractors (cuisine, hours, accessibility).
- **API and async batching** — expose a service endpoint with job queues and idempotent runs.
- **Observability** — structured tracing, prompt/response logging with PII controls, cost tracking.
- **Human review workflow** — UI for the review queue with feedback loops into prompt tuning.
- **Stronger evaluation** — larger labelled datasets, confusion matrices, regression suites on prompt changes.
- **Resilience** — rate-limit handling, circuit breakers, configurable retries.
- **Data scale** — Polars or streaming CSV for larger catalogues.
- **Orchestration** — LangGraph or similar if the pipeline grows beyond a handful of agents.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system design
- [docs/LANGGRAPH_ORCHESTRATION.md](docs/LANGGRAPH_ORCHESTRATION.md) — LangGraph backend design
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — detailed build specification
