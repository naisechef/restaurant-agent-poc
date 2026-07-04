"""Tests for imperative gather pipeline (network-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from restaurant_agent.config import Settings
from restaurant_agent.gather_pipeline import GATHER_RESULT_COLUMNS, run_gather_pipeline
from restaurant_agent.schemas import Evidence, EvidenceSourceType, RestaurantQuery
from restaurant_agent.sources.fake import FakeSourceAdapter
from tests.conftest import FakeClaudeClient, valid_extraction_json

QUERY = RestaurantQuery(name="The River Cafe", city="London")


@pytest.fixture
def gather_settings(tmp_path: Path) -> Settings:
    return Settings(
        confidence_threshold=0.6,
        output_path=tmp_path / "gather_results.csv",
        review_queue_path=tmp_path / "gather_review_queue.csv",
    )


def test_run_gather_pipeline_happy_path(
    gather_settings: Settings,
    tmp_path: Path,
) -> None:
    patio_evidence = Evidence(
        source_type=EvidenceSourceType.SEARCH,
        source_name="fake_search",
        snippet="Enjoy our sunny patio with riverside views.",
        reliability="medium",
    )
    adapters = {
        "website": FakeSourceAdapter("fake_website", evidence=[]),
        "search": FakeSourceAdapter("fake_search", evidence=[patio_evidence]),
        "reviews": FakeSourceAdapter("fake_reviews", evidence=[]),
        "maps": FakeSourceAdapter("fake_maps", evidence=[]),
    }
    client = FakeClaudeClient([valid_extraction_json(label="yes", confidence=0.9)])

    output_path = tmp_path / "gather_results.csv"
    review_path = tmp_path / "gather_review_queue.csv"

    state = run_gather_pipeline(
        query=QUERY,
        adapters=adapters,
        output_path=output_path,
        review_queue_path=review_path,
        settings=gather_settings,
        client=client,
    )

    assert output_path.exists()
    assert state.prediction == "yes"
    assert state.validation_status == "ok"
    assert len(state.gathered_evidence) == 1

    results = pd.read_csv(output_path)
    assert list(results.columns) == GATHER_RESULT_COLUMNS
    assert results.iloc[0]["name"] == "The River Cafe"
    assert results.iloc[0]["city"] == "London"

    snippets = json.loads(results.iloc[0]["source_evidence_snippets"])
    assert "patio" in snippets[0].lower()


def test_run_gather_pipeline_no_evidence_routes_to_failed(
    gather_settings: Settings,
    tmp_path: Path,
) -> None:
    adapters = {
        role: FakeSourceAdapter(f"fake_{role}", evidence=[])
        for role in ("website", "search", "reviews", "maps")
    }
    client = FakeClaudeClient([])

    output_path = tmp_path / "gather_results.csv"
    review_path = tmp_path / "gather_review_queue.csv"

    state = run_gather_pipeline(
        query=QUERY,
        adapters=adapters,
        output_path=output_path,
        review_queue_path=review_path,
        settings=gather_settings,
        client=client,
    )

    assert state.validation_status == "failed"
    assert state.needs_review is True
    assert "No evidence gathered" in (state.error or "")

    results = pd.read_csv(output_path)
    assert results.iloc[0]["validation_status"] == "failed"
    assert len(client.calls) == 0


def test_run_gather_pipeline_isolates_adapter_failures(
    gather_settings: Settings,
    tmp_path: Path,
) -> None:
    good_evidence = Evidence(
        source_type=EvidenceSourceType.MAPS,
        source_name="fake_maps",
        snippet="Outdoor seating available on terrace.",
        reliability="medium",
    )
    adapters = {
        "website": FakeSourceAdapter("fake_website", raises=RuntimeError("boom")),
        "search": FakeSourceAdapter("fake_search", evidence=[good_evidence]),
        "reviews": FakeSourceAdapter("fake_reviews", evidence=[]),
        "maps": FakeSourceAdapter("fake_maps", evidence=[]),
    }
    client = FakeClaudeClient([valid_extraction_json(label="yes", confidence=0.85)])

    output_path = tmp_path / "gather_results.csv"
    review_path = tmp_path / "gather_review_queue.csv"

    state = run_gather_pipeline(
        query=QUERY,
        adapters=adapters,
        output_path=output_path,
        review_queue_path=review_path,
        settings=gather_settings,
        client=client,
    )

    assert state.prediction == "yes"
    gather_errors = json.loads(
        pd.read_csv(output_path).iloc[0]["gather_errors"]
    )
    assert any("boom" in err for err in gather_errors)
