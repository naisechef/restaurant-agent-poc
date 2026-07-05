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

Alternatively, the **`gather`** command accepts a restaurant name and city, collects evidence from multiple static source adapters in parallel, then runs the same extraction pipeline. See [docs/EVIDENCE_GATHERING.md](docs/EVIDENCE_GATHERING.md).

A **web demo** (FastAPI + server-rendered HTML) exposes the same gather flow via a browser form and optional JSON API. See [docs/WEB_DEMO.md](docs/WEB_DEMO.md) for local run, Docker, and Cloud Run deployment. API keys stay server-side; default dry-run mode needs no keys.

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

For the web demo: `pip install -e ".[web,dev]"` then `uvicorn restaurant_agent.web.app:app --reload`.

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
| `DEFAULT_GATHER_OUTPUT_PATH` | No | `outputs/gather_results.csv` | Default gather results path |
| `DEFAULT_GATHER_REVIEW_QUEUE_PATH` | No | `outputs/gather_review_queue.csv` | Default gather review queue path |
| `DEFAULT_FIXTURES_PATH` | No | `data/evidence_fixtures` | Static evidence fixtures for gather |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `GOOGLE_PLACES_API_KEY` | `gather --live --source google` only | — | Google Places API key (see [Google Places setup](#google-places-setup-optional-live-adapter)) |
| `GOOGLE_PLACES_TIMEOUT` | No | `10` | Per-request timeout (seconds) for Text Search / Place Details |
| `GOOGLE_PLACES_MAX_REVIEWS` | No | `3` | Max review snippets converted to evidence per gather call |
| `GOOGLE_PLACES_REVIEW_SNIPPET_CHARS` | No | `300` | Max characters per review snippet |

API key validation happens only when constructing the real `ClaudeClient`, or when `gather --live --source google` is used. Tests, `--dry-run`, and the default gather flow never require either key.

## CLI usage

After `pip install -e ".[dev]"`, use the `restaurant-agent` console script (recommended):

```bash
restaurant-agent --help
restaurant-agent run --help
restaurant-agent gather --help
```

Alternatively, run the module directly:

```bash
python -m restaurant_agent.cli --help
```

### `run` — CSV batch extraction

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

Backward-compatible usage without the `run` subcommand still works:

```bash
restaurant-agent --dry-run --limit 3
```

### `gather` — parallel evidence gathering

| Flag | Description |
|------|-------------|
| `--name TEXT` | Restaurant name (required) |
| `--city TEXT` | City (required) |
| `--fixtures-path PATH` | Static evidence fixtures directory |
| `--output PATH` | Gather results CSV path |
| `--review-queue-output PATH` | Gather review queue path |
| `--backend {pipeline,graph}` | Orchestration backend (default: `pipeline`) |
| `--dry-run` | Deterministic local LLM; static fixtures for sources |
| `--live` | Use a live source instead of static fixtures for the `maps` role (see `--source`) |
| `--source {google}` | Live source provider to use with `--live` (default: `google`); has no effect without `--live` |

### Example: live run

Requires `ANTHROPIC_API_KEY` in `.env` or the environment.

```bash
restaurant-agent run \
  --input data/restaurants.csv \
  --output outputs/results.csv
```

### Example: gather (dry run)

```bash
restaurant-agent gather \
  --name "The River Cafe" \
  --city "London" \
  --dry-run
```

### Example: gather with LangGraph backend

```bash
restaurant-agent gather \
  --name "The River Cafe" \
  --city "London" \
  --backend graph \
  --dry-run
```

### Example: gather with the live Google Places adapter

Requires `GOOGLE_PLACES_API_KEY` — see [Google Places setup](#google-places-setup-optional-live-adapter). All other source roles (`search`, `reviews`, `website`) remain static/fake; only the `maps` role becomes live.

```bash
restaurant-agent gather \
  --name "The River Cafe" \
  --city "London" \
  --backend graph \
  --live \
  --source google
```

### Example: dry run (no API key)

Runs the full pipeline shape with heuristic local responses:

```bash
restaurant-agent --dry-run --limit 3
```

### Example: LangGraph backend (dry run)

Same behaviour as the default pipeline, using LangGraph orchestration:

```bash
restaurant-agent --backend graph --dry-run --limit 12
```

## Google Places setup (optional live adapter)

`gather --live --source google` replaces the static `maps` source with a live lookup against the **official Google Places API (New)** — Text Search followed by Place Details with an explicit field mask. No other source (Google Maps scraping, Google Search, Google Reviews pages, Tripadvisor, or website crawling) is used.

1. Enable the "Places API (New)" for a Google Cloud project and create an API key.
2. Add it to `.env`:

   ```bash
   GOOGLE_PLACES_API_KEY=your-key-here
   ```

3. Run gather with `--live --source google` (see example above).

Notes:

- The key is only required when `--live --source google` is selected — the default gather flow, `run`, and the full test suite never need it.
- Exactly one Text Search request and one Place Details request are made per gather call; the field mask never uses a wildcard, keeping cost and payload size bounded.
- `websiteUri` returned by Google Places is stored only as a citation URL — it is never fetched or crawled. Website crawling is out of scope for this adapter and may become a separate adapter later.
- See [docs/EVIDENCE_GATHERING.md](docs/EVIDENCE_GATHERING.md) for the evidence-mapping details and safeguards.

## Output files

| File | Contents |
|------|----------|
| `outputs/results.csv` | One row per processed record with prediction, confidence, evidence, validation status, correctness (when expected labels exist), and errors |
| `outputs/review_queue.csv` | Subset of results where `needs_review == True` |
| `outputs/gather_results.csv` | Single-row gather run with source provenance columns |
| `outputs/gather_review_queue.csv` | Gather subset routed to review |

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

- Batch CLI only — no REST API or database.
- Gather mode uses static JSON fixtures by default; an optional live Google Places adapter is available via `--live --source google` for the `maps` role only (no scraping of Google Maps/Search/Reviews pages, no other live sources, no website crawling).
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
- [docs/EVIDENCE_GATHERING.md](docs/EVIDENCE_GATHERING.md) — parallel evidence gathering
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — detailed build specification
