# Evidence Gathering

This document describes the parallel evidence gathering feature: how sources are queried, merged, and fed into the existing extraction pipeline.

## Overview

The `gather` CLI command accepts a restaurant **name** and **city**, collects evidence from multiple source adapters in parallel, merges and deduplicates snippets, then runs the same preprocessing → extraction → validation agent chain used by the CSV `run` command.

Two orchestration backends are available:

- **`pipeline`** (default) — thread-pool parallel gathering in `gather_pipeline.py`
- **`graph`** — LangGraph fan-out/fan-in in `gather_graph.py`

## Architecture

```mermaid
flowchart TD
    START([START]) --> resolve[resolve_restaurant]
    resolve --> gw[gather_website]
    resolve --> gs[gather_search]
    resolve --> gr[gather_reviews]
    resolve --> gm[gather_maps]
    gw --> merge[merge_evidence]
    gs --> merge
    gr --> merge
    gm --> merge
    merge --> preprocess
    preprocess --> extract
    extract --> validate
    validate --> router{route_after_validate}
    router -->|failed| failed
    router -->|needs_review| needsReview
    router -->|success| success
    failed --> END([END])
    needsReview --> END
    success --> END
```

The graph backend uses true LangGraph parallel branches. Each gather node writes to `source_results` via an `operator.add` reducer. Downstream agent nodes return partial state updates to avoid re-accumulating source results.

## Source adapters (v1)

| Adapter | Role | Behaviour |
|---------|------|-----------|
| `FakeSourceAdapter` | Tests, empty website slot | Returns canned evidence or raises when configured |
| `StaticSearchAdapter` | search, reviews, maps | Loads snippets from JSON fixtures keyed by `name\|city` |

Fixtures live in `data/evidence_fixtures/` (`search.json`, `reviews.json`, `maps.json`). No live web scraping in v1.

## Data contracts

See `schemas.py`:

- `RestaurantQuery` — name, city, optional website URL hint
- `Evidence` — snippet with source type, URL, reliability metadata
- `SourceResult` — per-adapter outcome including optional error
- `GatheredEvidence` — deduplicated items plus aggregated errors

`GatherState` extends `AgentState` with gather-specific fields for LangGraph orchestration.

## Failure isolation

- Each adapter runs inside `run_adapter_safe()` — exceptions become `SourceResult.error`
- One failed source never aborts the batch or graph run
- If no evidence is gathered from any source, merge short-circuits with `error = "No evidence gathered from any source"` and skips the LLM call

## CLI usage

```bash
restaurant-agent gather --name "The River Cafe" --city "London" --dry-run
restaurant-agent gather --name "The River Cafe" --city "London" --backend graph --dry-run
```

The existing CSV path is unchanged:

```bash
restaurant-agent --dry-run --limit 3          # backward compatible
restaurant-agent run --dry-run --limit 3      # explicit subcommand
```

## Output

Gather results are written to `outputs/gather_results.csv` with columns including source provenance:

- `source_evidence_snippets` — JSON list of gathered snippets
- `source_urls` — JSON list of source URLs (nullable entries)
- `source_types` — JSON list of source categories
- `gather_errors` — JSON list of per-source errors

## Testing

All gather tests use `FakeSourceAdapter` or static fixtures — no network calls.

```bash
pytest tests/test_source_adapters.py tests/test_gather.py tests/test_evidence_merge.py
pytest tests/test_gather_pipeline.py tests/test_gather_graph.py tests/test_cli_gather.py
```

## Future extensions

- Optional live adapters (website fetch, search API) behind explicit flags
- Semantic deduplication and conflict resolution
- Batch gather mode for multiple restaurants
