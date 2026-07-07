"""Shared result-row shaping and CSV output helpers for the `run` orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import pandas as pd

from restaurant_agent.schemas import RestaurantRecord
from restaurant_agent.state import AgentState

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


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)
