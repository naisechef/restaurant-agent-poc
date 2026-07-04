"""Load restaurant input rows and seed initial pipeline state."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from restaurant_agent.schemas import RestaurantRecord
from restaurant_agent.state import AgentState

logger = logging.getLogger(__name__)


def _row_to_record(row: pd.Series, row_index: int) -> RestaurantRecord | None:
    """Validate a single CSV row; return None if the row is invalid."""
    payload: dict[str, object] = {
        "id": row.get("id"),
        "name": row.get("name"),
        "evidence_text": row.get("evidence_text"),
    }

    expected_label = row.get("expected_label")
    if expected_label is None or (isinstance(expected_label, float) and pd.isna(expected_label)):
        payload["expected_label"] = None
    elif isinstance(expected_label, str) and not expected_label.strip():
        payload["expected_label"] = None
    else:
        payload["expected_label"] = expected_label

    try:
        return RestaurantRecord.model_validate(payload)
    except ValidationError as exc:
        logger.warning("Skipping invalid row at index %s: %s", row_index, exc)
        return None


def load_restaurants(path: Path) -> list[RestaurantRecord]:
    """Read a restaurants CSV and return validated records; invalid rows are logged and skipped."""
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    records: list[RestaurantRecord] = []

    for row_index, row in df.iterrows():
        record = _row_to_record(row, int(row_index))
        if record is not None:
            records.append(record)

    return records


def to_initial_state(record: RestaurantRecord) -> AgentState:
    """Map a validated input record to the initial AgentState for the pipeline."""
    return AgentState(
        restaurant_id=record.id,
        raw_text=record.evidence_text,
    )
