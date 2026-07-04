"""LangGraph orchestration for parallel evidence gathering plus extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from restaurant_agent.agents import (
    extraction_agent,
    preprocessing_agent,
    validation_agent,
)
from restaurant_agent.config import Settings
from restaurant_agent.evidence_merge import combine_evidence_text, merge_evidence
from restaurant_agent.gather import run_adapter_safe
from restaurant_agent.gather_pipeline import (
    GATHER_RESULT_COLUMNS,
    _gather_state_to_row,
    _initial_gather_state,
    _resolve_restaurant_id,
)
from restaurant_agent.gather_result import build_gather_run_result
from restaurant_agent.pipeline import LLMClient, _write_csv
from restaurant_agent.schemas import GatherRunResult, GraphNodeExecution, RestaurantQuery
from restaurant_agent.sources.base import SourceAdapter
from restaurant_agent.state import GatherState

logger = logging.getLogger(__name__)

RouteLabel = Literal["success", "needs_review", "failed"]
MergeRoute = Literal["preprocess", "validate"]


class EvidenceGatherPipeline:
    """LangGraph orchestrator with fan-out/fan-in evidence gathering."""

    def __init__(
        self,
        client: LLMClient,
        threshold: float,
        adapters: dict[str, SourceAdapter],
    ) -> None:
        self._client = client
        self._threshold = threshold
        self._adapters = adapters

    def resolve_restaurant_node(self, state: GatherState) -> dict[str, object]:
        if state.query is None:
            raise ValueError("GatherState.query is required")
        return {"restaurant_id": _resolve_restaurant_id(state.query)}

    def gather_website_node(self, state: GatherState) -> dict[str, object]:
        assert state.query is not None
        result = run_adapter_safe(self._adapters["website"], state.query)
        return {"source_results": [result]}

    def gather_search_node(self, state: GatherState) -> dict[str, object]:
        assert state.query is not None
        result = run_adapter_safe(self._adapters["search"], state.query)
        return {"source_results": [result]}

    def gather_reviews_node(self, state: GatherState) -> dict[str, object]:
        assert state.query is not None
        result = run_adapter_safe(self._adapters["reviews"], state.query)
        return {"source_results": [result]}

    def gather_maps_node(self, state: GatherState) -> dict[str, object]:
        assert state.query is not None
        result = run_adapter_safe(self._adapters["maps"], state.query)
        return {"source_results": [result]}

    def merge_evidence_node(self, state: GatherState) -> dict[str, object]:
        gathered = merge_evidence(state.source_results)
        combined = combine_evidence_text(gathered.items)

        if not gathered.items:
            return {
                "gathered_evidence": [],
                "gather_errors": gathered.errors,
                "raw_text": "",
                "error": "No evidence gathered from any source",
                "validation_status": "failed",
                "needs_review": True,
            }

        return {
            "gathered_evidence": gathered.items,
            "gather_errors": gathered.errors,
            "raw_text": combined,
            "error": None,
        }

    def preprocess_node(self, state: GatherState) -> dict[str, object]:
        updated = preprocessing_agent.run(state)
        return {"cleaned_text": updated.cleaned_text}

    def extraction_node(self, state: GatherState) -> dict[str, object]:
        updated = extraction_agent.run(state, self._client)
        return {
            "prediction": updated.prediction,
            "confidence": updated.confidence,
            "evidence": updated.evidence,
            "reasoning": updated.reasoning,
            "error": updated.error,
        }

    def validation_node(self, state: GatherState) -> dict[str, object]:
        updated = validation_agent.run(state, self._threshold)
        return {
            "validation_status": updated.validation_status,
            "needs_review": updated.needs_review,
        }

    def success_node(self, state: GatherState) -> dict[str, object]:
        return {}

    def needs_review_node(self, state: GatherState) -> dict[str, object]:
        return {}

    def failed_node(self, state: GatherState) -> dict[str, object]:
        return {}

    def route_after_merge(self, state: GatherState) -> MergeRoute:
        if state.error:
            return "validate"
        return "preprocess"

    def route_after_validate(self, state: GatherState) -> RouteLabel:
        if state.validation_status == "failed":
            return "failed"
        if state.needs_review:
            return "needs_review"
        return "success"

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(GatherState)

        graph.add_node("resolve_restaurant", self.resolve_restaurant_node)
        graph.add_node("gather_website", self.gather_website_node)
        graph.add_node("gather_search", self.gather_search_node)
        graph.add_node("gather_reviews", self.gather_reviews_node)
        graph.add_node("gather_maps", self.gather_maps_node)
        graph.add_node("merge_evidence", self.merge_evidence_node)
        graph.add_node("preprocess", self.preprocess_node)
        graph.add_node("extract", self.extraction_node)
        graph.add_node("validate", self.validation_node)
        graph.add_node("success", self.success_node)
        graph.add_node("needs_review", self.needs_review_node)
        graph.add_node("failed", self.failed_node)

        graph.add_edge(START, "resolve_restaurant")
        graph.add_edge("resolve_restaurant", "gather_website")
        graph.add_edge("resolve_restaurant", "gather_search")
        graph.add_edge("resolve_restaurant", "gather_reviews")
        graph.add_edge("resolve_restaurant", "gather_maps")
        graph.add_edge("gather_website", "merge_evidence")
        graph.add_edge("gather_search", "merge_evidence")
        graph.add_edge("gather_reviews", "merge_evidence")
        graph.add_edge("gather_maps", "merge_evidence")
        graph.add_conditional_edges(
            "merge_evidence",
            self.route_after_merge,
            {
                "preprocess": "preprocess",
                "validate": "validate",
            },
        )
        graph.add_edge("preprocess", "extract")
        graph.add_edge("extract", "validate")
        graph.add_conditional_edges(
            "validate",
            self.route_after_validate,
            {
                "success": "success",
                "needs_review": "needs_review",
                "failed": "failed",
            },
        )
        graph.add_edge("success", END)
        graph.add_edge("needs_review", END)
        graph.add_edge("failed", END)

        return graph.compile()


def _final_state(result: GatherState | dict[str, object]) -> GatherState:
    if isinstance(result, GatherState):
        return result
    return GatherState.model_validate(result)


def _normalize_graph_update(update: dict[str, object] | None) -> dict[str, object]:
    """LangGraph may emit ``None`` for terminal nodes (e.g. success) — treat as empty."""
    return update if update is not None else {}


def _apply_graph_update(
    state: GatherState, update: dict[str, object] | None
) -> GatherState:
    """Apply a LangGraph node update, mirroring the source_results reducer."""
    update = _normalize_graph_update(update)
    if "source_results" in update:
        incoming = update["source_results"]
        if isinstance(incoming, list):
            update = {
                **update,
                "source_results": [*state.source_results, *incoming],
            }
    return state.model_copy(update=update)


def _serialize_graph_update(update: dict[str, object] | None) -> dict[str, object]:
    """Convert node updates to JSON-serializable dicts for graph_trace."""
    update = _normalize_graph_update(update)
    serialized: dict[str, object] = {}
    for key, value in update.items():
        if key in {"raw_text", "cleaned_text"}:
            continue
        if key == "source_results" and isinstance(value, list):
            serialized[key] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in value
            ]
        elif key == "gathered_evidence" and isinstance(value, list):
            serialized[key] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in value
            ]
        elif hasattr(value, "model_dump"):
            serialized[key] = value.model_dump()
        else:
            serialized[key] = value
    return serialized


def run_gather_graph(
    query: RestaurantQuery,
    adapters: dict[str, SourceAdapter],
    client: LLMClient,
    threshold: float,
    *,
    dry_run: bool = False,
    live_google: bool = False,
) -> GatherRunResult:
    """Gather via LangGraph without writing CSV files; includes execution trace."""
    pipeline = EvidenceGatherPipeline(client, threshold, adapters)
    graph = pipeline.build()
    state = _initial_gather_state(query)
    trace: list[GraphNodeExecution] = []

    try:
        for step in graph.stream(state, stream_mode="updates"):
            for node, raw_update in step.items():
                normalized = _normalize_graph_update(raw_update)
                trace.append(
                    GraphNodeExecution(
                        node=node,
                        update=_serialize_graph_update(normalized),
                    )
                )
                state = _apply_graph_update(state, normalized)
    except Exception as exc:
        logger.error(
            "Unexpected error in gather graph for %s: %s",
            query.name,
            exc,
            exc_info=True,
        )
        state = state.model_copy(
            update={
                "error": str(exc),
                "validation_status": "failed",
                "needs_review": True,
            }
        )

    return build_gather_run_result(
        state,
        backend="graph",
        graph_trace=trace,
        dry_run=dry_run,
        live_google=live_google,
    )


def run_gather_graph_pipeline(
    query: RestaurantQuery,
    adapters: dict[str, SourceAdapter],
    output_path: Path,
    review_queue_path: Path,
    settings: Settings,
    client: LLMClient,
) -> GatherState:
    """Gather evidence via LangGraph, extract, validate, and write CSV outputs."""
    logger.info(
        "Gathering evidence (graph) for %s, %s",
        query.name,
        query.city,
    )

    pipeline = EvidenceGatherPipeline(
        client,
        settings.confidence_threshold,
        adapters,
    )
    graph = pipeline.build()
    initial = _initial_gather_state(query)

    try:
        state = _final_state(graph.invoke(initial))
    except Exception as exc:
        logger.error(
            "Unexpected error in gather graph for %s: %s",
            query.name,
            exc,
            exc_info=True,
        )
        state = initial.model_copy(
            update={
                "error": str(exc),
                "validation_status": "failed",
                "needs_review": True,
            }
        )

    row = _gather_state_to_row(state)
    results_rows = [row]
    review_rows = [row] if state.needs_review else []

    _write_csv(output_path, results_rows, GATHER_RESULT_COLUMNS)
    _write_csv(review_queue_path, review_rows, GATHER_RESULT_COLUMNS)

    logger.info("Wrote gather result to %s", output_path)
    if review_rows:
        logger.info("Wrote gather review-queue row to %s", review_queue_path)

    return state
