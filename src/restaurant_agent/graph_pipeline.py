"""LangGraph-backed orchestration via GraphPipeline over shared AgentState."""

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
from restaurant_agent.data_loader import load_restaurants, to_initial_state
from restaurant_agent.evaluation import evaluate_predictions
from restaurant_agent.pipeline import (
    LLMClient,
    RESULT_COLUMNS,
    _state_to_row,
    _write_csv,
)
from restaurant_agent.schemas import EvaluationSummary, RestaurantRecord
from restaurant_agent.state import AgentState

logger = logging.getLogger(__name__)

RouteLabel = Literal["success", "needs_review", "failed"]


class GraphPipeline:
    """LangGraph orchestrator that delegates to existing agents on AgentState."""

    def __init__(self, client: LLMClient, threshold: float) -> None:
        self._client = client
        self._threshold = threshold

    def preprocess_node(self, state: AgentState) -> AgentState:
        return preprocessing_agent.run(state)

    def extraction_node(self, state: AgentState) -> AgentState:
        return extraction_agent.run(state, self._client)

    def validation_node(self, state: AgentState) -> AgentState:
        return validation_agent.run(state, self._threshold)

    def success_node(self, state: AgentState) -> AgentState:
        return state

    def needs_review_node(self, state: AgentState) -> AgentState:
        return state

    def failed_node(self, state: AgentState) -> AgentState:
        return state

    def route_after_validate(self, state: AgentState) -> RouteLabel:
        if state.validation_status == "failed":
            return "failed"
        if state.needs_review:
            return "needs_review"
        return "success"

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("preprocess", self.preprocess_node)
        graph.add_node("extract", self.extraction_node)
        graph.add_node("validate", self.validation_node)
        graph.add_node("success", self.success_node)
        graph.add_node("needs_review", self.needs_review_node)
        graph.add_node("failed", self.failed_node)

        graph.add_edge(START, "preprocess")
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


def _final_state(result: AgentState | dict[str, object]) -> AgentState:
    if isinstance(result, AgentState):
        return result
    return AgentState.model_validate(result)


def _invoke_record(
    graph: CompiledStateGraph,
    record: RestaurantRecord,
) -> AgentState:
    """Run the graph for one record; isolate unexpected failures on state.error."""
    state = to_initial_state(record)
    try:
        return _final_state(graph.invoke(state))
    except Exception as exc:
        logger.error(
            "Unexpected error processing record %s: %s",
            record.id,
            exc,
            exc_info=True,
        )
        return state.model_copy(
            update={
                "error": str(exc),
                "validation_status": "failed",
                "needs_review": True,
            }
        )


def run_graph_pipeline(
    input_path: Path,
    output_path: Path,
    review_queue_path: Path,
    settings: Settings,
    client: LLMClient,
    limit: int | None = None,
) -> EvaluationSummary:
    """Load records, run LangGraph per row, evaluate, write CSV outputs, return summary."""
    records = load_restaurants(input_path)
    if limit is not None:
        records = records[:limit]

    total = len(records)
    logger.info("Loaded %s record(s) from %s", total, input_path)

    graph_pipeline = GraphPipeline(client, settings.confidence_threshold)
    graph = graph_pipeline.build()

    results_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    states: list[AgentState] = []
    expected: dict[str, str] = {}

    for index, record in enumerate(records, start=1):
        logger.info("Processing record %s/%s (%s)", index, total, record.id)

        if record.expected_label is not None:
            expected[record.id] = record.expected_label

        state = _invoke_record(graph, record)
        states.append(state)

        row = _state_to_row(record, state)
        results_rows.append(row)
        if state.needs_review:
            review_rows.append(row)

    summary = evaluate_predictions(
        states,
        expected,
        confidence_threshold=settings.confidence_threshold,
    )

    _write_csv(output_path, results_rows, RESULT_COLUMNS)
    _write_csv(review_queue_path, review_rows, RESULT_COLUMNS)

    logger.info("Wrote %s result row(s) to %s", len(results_rows), output_path)
    logger.info(
        "Wrote %s review-queue row(s) to %s",
        len(review_rows),
        review_queue_path,
    )

    return summary
