# LangGraph Orchestration

This document describes the LangGraph-backed orchestration layer introduced alongside the existing imperative `pipeline.py`. Business logic in agents, preprocessing, validation, evaluation, prompts, and the Claude client is unchanged — only orchestration differs.

## Goals

- Replace the orchestration layer only; reuse all existing agents and helpers.
- Use `AgentState` (Pydantic) directly as the LangGraph state — no duplicate `GraphState`.
- Expose orchestration via a `GraphPipeline` class with explicit node methods.
- Keep `pipeline.py` as the default CLI backend; add `--backend graph` as an opt-in.
- Structure the graph for future conditional routing without changing current behaviour.

## Architecture

```mermaid
flowchart TD
    START([START]) --> preprocess
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

### GraphPipeline class

```python
class GraphPipeline:
    def __init__(self, client: LLMClient, threshold: float) -> None: ...

    def preprocess_node(self, state: AgentState) -> AgentState: ...
    def extraction_node(self, state: AgentState) -> AgentState: ...
    def validation_node(self, state: AgentState) -> AgentState: ...
    def success_node(self, state: AgentState) -> AgentState: ...
    def needs_review_node(self, state: AgentState) -> AgentState: ...
    def failed_node(self, state: AgentState) -> AgentState: ...

    def route_after_validate(self, state: AgentState) -> str: ...
    def build(self) -> CompiledStateGraph: ...
```

Each agent node delegates to the existing `agents/*.run()` functions. Terminal nodes (`success`, `needs_review`, `failed`) are pass-through in Phase 1 — they exist to demonstrate LangGraph routing and to host future logic (logging, metrics, human-in-the-loop hooks).

### State

`AgentState` from `state.py` is passed directly to `StateGraph(AgentState)`. No conversion layer.

### Batch orchestration

`run_graph_pipeline()` mirrors `run_pipeline()`:

1. Load records via `data_loader`.
2. For each record: seed `AgentState`, invoke the compiled graph, collect final state.
3. Reuse `_state_to_row`, `_write_csv`, `RESULT_COLUMNS`, and `evaluate_predictions` from `pipeline.py`.

Per-record error isolation matches `pipeline._process_record`: unexpected exceptions are caught, recorded on `state.error`, and the batch continues.

## Files

| File | Action |
|------|--------|
| `src/restaurant_agent/graph_pipeline.py` | Create — `GraphPipeline`, `run_graph_pipeline()` |
| `src/restaurant_agent/cli.py` | Change — add `--backend {pipeline,graph}` |
| `tests/test_graph_pipeline.py` | Create — graph tests + pipeline comparison |
| `tests/test_smoke.py` | Change — add `--backend graph` smoke test |
| `docs/ARCHITECTURE.md` | Change — document graph backend |
| `pyproject.toml` | Change — add `langgraph` dependency |

Unchanged: all agents, `pipeline.py`, `state.py`, `schemas.py`, preprocessing, validation, evaluation, prompts, claude_client.

## Routing logic (Phase 1)

| Route | Condition |
|-------|-----------|
| `failed` | `validation_status == "failed"` |
| `needs_review` | `needs_review == True` (and not failed) |
| `success` | otherwise |

Terminal nodes return state unchanged. Outputs are identical to the imperative pipeline.

## Test plan

- All existing tests pass unchanged.
- `tests/test_graph_pipeline.py`:
  - Happy path (results + review queue)
  - Error isolation (`ErrorClaudeClient`)
  - `--limit` respected
  - Graph node unit test via `GraphPipeline.build().invoke()`
  - **Comparison test**: `run_pipeline()` vs `run_graph_pipeline()` produce identical CSVs with `FakeClaudeClient`

## Phased checklist

1. Add `langgraph` dependency; verify existing tests pass.
2. Create `graph_pipeline.py` with `GraphPipeline` and `run_graph_pipeline()`.
3. Add graph tests including pipeline comparison.
4. Wire `--backend` flag in CLI; add smoke test.
5. Update `ARCHITECTURE.md`.

## Risks and scope limits

- **Over-engineering**: Phase 1 is linear processing with routing scaffolding only; no async, fan-out, or retry nodes yet.
- **State drift**: Shared helpers imported from `pipeline.py` — never duplicate CSV/evaluation logic.
- **Pydantic + LangGraph**: Slightly less performant than TypedDict; acceptable for a batch CLI PoC.
