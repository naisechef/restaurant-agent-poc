"""Unit tests for restaurant CSV loading and initial state seeding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from restaurant_agent.data_loader import load_restaurants, to_initial_state
from restaurant_agent.schemas import RestaurantRecord
from restaurant_agent.state import AgentState

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "restaurants.csv"


def test_load_restaurants_reads_sample_csv() -> None:
    records = load_restaurants(DATA_PATH)

    assert len(records) == 12
    assert all(isinstance(record, RestaurantRecord) for record in records)


def test_load_restaurants_parses_optional_expected_label() -> None:
    records = load_restaurants(DATA_PATH)
    by_id = {record.id: record for record in records}

    assert by_id["r001"].expected_label == "yes"
    assert by_id["r012"].expected_label is None


def test_load_restaurants_skips_invalid_rows(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "restaurants.csv"
    pd.DataFrame(
        [
            {
                "id": "good-1",
                "name": "Valid Place",
                "evidence_text": "Patio seating available.",
                "expected_label": "yes",
            },
            {
                "id": "",
                "name": "Missing ID",
                "evidence_text": "Some text.",
                "expected_label": "yes",
            },
            {
                "id": "bad-label",
                "name": "Bad Label",
                "evidence_text": "Some text.",
                "expected_label": "maybe",
            },
        ]
    ).to_csv(csv_path, index=False)

    with caplog.at_level("WARNING"):
        records = load_restaurants(csv_path)

    assert len(records) == 1
    assert records[0].id == "good-1"
    assert any("Skipping invalid row" in message for message in caplog.messages)


def test_to_initial_state_maps_record_fields(sample_restaurant_record: RestaurantRecord) -> None:
    state = to_initial_state(sample_restaurant_record)

    assert isinstance(state, AgentState)
    assert state.restaurant_id == sample_restaurant_record.id
    assert state.raw_text == sample_restaurant_record.evidence_text
    assert state.cleaned_text is None
    assert state.prediction is None
