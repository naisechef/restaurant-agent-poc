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
from restaurant_agent.pipeline import LLMClient, _write_csv
from restaurant_agent.schemas import RestaurantQuery
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
