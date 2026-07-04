"""End-to-end orchestration: load data, run agents, evaluate, and write outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

from restaurant_agent.agents import (
    extraction_agent,
    preprocessing_agent,
    validation_agent,
)
from restaurant_agent.config import Settings
from restaurant_agent.data_loader import load_restaurants, to_initial_state
from restaurant_agent.evaluation import evaluate_predictions
from restaurant_agent.schemas import EvaluationSummary, RestaurantRecord
from restaurant_agent.state import AgentState

logger = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "restaurant_id",
    "name",
    "prediction",
    "confidence",
    "evidence",
    "reasoning",
    "validation_status",
    "needs_review",
    "is_correct",
    "error",
]


class LLMClient(Protocol):
    """Duck-typed client interface shared by ClaudeClient and test doubles."""

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


def _compute_is_correct(
    record: RestaurantRecord,
    state: AgentState,
) -> bool | None:
    if state.error is not None or state.validation_status == "failed":
        return None
    if state.prediction is None or record.expected_label is None:
        return None
    return state.prediction == record.expected_label


def _state_to_row(record: RestaurantRecord, state: AgentState) -> dict[str, object]:
    return {
        "restaurant_id": record.id,
        "name": record.name,
        "prediction": state.prediction,
        "confidence": state.confidence,
        "evidence": json.dumps(state.evidence, ensure_ascii=False),
        "reasoning": state.reasoning,
        "validation_status": state.validation_status,
        "needs_review": state.needs_review,
        "is_correct": _compute_is_correct(record, state),
        "error": state.error,
    }


def _process_record(
    record: RestaurantRecord,
    client: LLMClient,
    threshold: float,
) -> AgentState:
    """Run the agent chain for one record; isolate unexpected failures on state.error."""
    state = to_initial_state(record)
    try:
        state = preprocessing_agent.run(state)
        state = extraction_agent.run(state, client)
        state = validation_agent.run(state, threshold)
        return state
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


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)


def run_pipeline(
    input_path: Path,
    output_path: Path,
    review_queue_path: Path,
    settings: Settings,
    client: LLMClient,
    limit: int | None = None,
) -> EvaluationSummary:
    """Load records, run agents per row, evaluate, write CSV outputs, return summary."""
    records = load_restaurants(input_path)
    if limit is not None:
        records = records[:limit]

    total = len(records)
    logger.info("Loaded %s record(s) from %s", total, input_path)

    results_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    states: list[AgentState] = []
    expected: dict[str, str] = {}

    for index, record in enumerate(records, start=1):
        logger.info("Processing record %s/%s (%s)", index, total, record.id)

        if record.expected_label is not None:
            expected[record.id] = record.expected_label

        state = _process_record(record, client, settings.confidence_threshold)
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
