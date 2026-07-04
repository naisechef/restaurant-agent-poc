"""Tests for LangGraph evidence gathering (network-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from restaurant_agent.config import Settings
from restaurant_agent.gather_graph import EvidenceGatherPipeline, run_gather_graph, run_gather_graph_pipeline
from restaurant_agent.gather_pipeline import GATHER_RESULT_COLUMNS
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


def test_gather_graph_accumulates_four_source_results() -> None:
    search_evidence = Evidence(
        source_type=EvidenceSourceType.SEARCH,
        source_name="fake_search",
        snippet="Riverside terrace seating.",
        reliability="medium",
    )
    review_evidence = Evidence(
        source_type=EvidenceSourceType.REVIEW,
        source_name="fake_reviews",
        snippet="Lovely patio area.",
        reliability="low",
    )
    adapters = {
        "website": FakeSourceAdapter("fake_website", evidence=[]),
        "search": FakeSourceAdapter("fake_search", evidence=[search_evidence]),
        "reviews": FakeSourceAdapter("fake_reviews", evidence=[review_evidence]),
        "maps": FakeSourceAdapter("fake_maps", evidence=[]),
    }
    client = FakeClaudeClient([valid_extraction_json(label="yes", confidence=0.88)])
    pipeline = EvidenceGatherPipeline(client, threshold=0.6, adapters=adapters)
    graph = pipeline.build()

    from restaurant_agent.gather_pipeline import _initial_gather_state

    state = graph.invoke(_initial_gather_state(QUERY))
    if isinstance(state, dict):
        source_count = len(state.get("source_results", []))
        gathered_count = len(state.get("gathered_evidence", []))
    else:
        source_count = len(state.source_results)
        gathered_count = len(state.gathered_evidence)

    assert source_count == 4
    assert gathered_count == 2


def test_gather_graph_isolates_adapter_failure() -> None:
    good_evidence = Evidence(
        source_type=EvidenceSourceType.MAPS,
        source_name="fake_maps",
        snippet="Outdoor seating on terrace.",
        reliability="medium",
    )
    adapters = {
        "website": FakeSourceAdapter("fake_website", raises=RuntimeError("boom")),
        "search": FakeSourceAdapter("fake_search", evidence=[good_evidence]),
        "reviews": FakeSourceAdapter("fake_reviews", evidence=[]),
        "maps": FakeSourceAdapter("fake_maps", evidence=[]),
    }
    client = FakeClaudeClient([valid_extraction_json(label="yes", confidence=0.9)])
    pipeline = EvidenceGatherPipeline(client, threshold=0.6, adapters=adapters)
    graph = pipeline.build()

    from restaurant_agent.gather_pipeline import _initial_gather_state

    state = graph.invoke(_initial_gather_state(QUERY))
    if isinstance(state, dict):
        gather_errors = state.get("gather_errors", [])
        prediction = state.get("prediction")
    else:
        gather_errors = state.gather_errors
        prediction = state.prediction

    assert any("boom" in err for err in gather_errors)
    assert prediction == "yes"


def test_gather_graph_no_evidence_routes_to_failed(
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

    state = run_gather_graph_pipeline(
        query=QUERY,
        adapters=adapters,
        output_path=output_path,
        review_queue_path=review_path,
        settings=gather_settings,
        client=client,
    )

    assert state.validation_status == "failed"
    assert "No evidence gathered" in (state.error or "")
    assert len(client.calls) == 0

    results = pd.read_csv(output_path)
    assert list(results.columns) == GATHER_RESULT_COLUMNS
    assert results.iloc[0]["validation_status"] == "failed"


def test_run_gather_graph_pipeline_writes_output(
    gather_settings: Settings,
    tmp_path: Path,
) -> None:
    patio_evidence = Evidence(
        source_type=EvidenceSourceType.SEARCH,
        source_name="fake_search",
        snippet="Sunny patio with garden views.",
        reliability="medium",
    )
    adapters = {
        "website": FakeSourceAdapter("fake_website", evidence=[]),
        "search": FakeSourceAdapter("fake_search", evidence=[patio_evidence]),
        "reviews": FakeSourceAdapter("fake_reviews", evidence=[]),
        "maps": FakeSourceAdapter("fake_maps", evidence=[]),
    }
    client = FakeClaudeClient([valid_extraction_json(label="yes", confidence=0.92)])

    output_path = tmp_path / "gather_results.csv"
    review_path = tmp_path / "gather_review_queue.csv"

    state = run_gather_graph_pipeline(
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

    snippets = json.loads(pd.read_csv(output_path).iloc[0]["source_evidence_snippets"])
    assert "patio" in snippets[0].lower()


def test_run_gather_graph_stream_preserves_ok_validation() -> None:
    patio_evidence = Evidence(
        source_type=EvidenceSourceType.SEARCH,
        source_name="fake_search",
        snippet="Sunny patio with garden views.",
        reliability="medium",
    )
    adapters = {
        "website": FakeSourceAdapter("fake_website", evidence=[]),
        "search": FakeSourceAdapter("fake_search", evidence=[patio_evidence]),
        "reviews": FakeSourceAdapter("fake_reviews", evidence=[]),
        "maps": FakeSourceAdapter("fake_maps", evidence=[]),
    }
    client = FakeClaudeClient([valid_extraction_json(label="yes", confidence=0.98)])

    result = run_gather_graph(QUERY, adapters, client, threshold=0.6)

    assert result.prediction == "yes"
    assert result.confidence == 0.98
    assert result.validation_status == "ok"
    assert result.route == "success"
    assert result.needs_review is False
    assert result.error is None
    assert result.graph_trace is not None
    assert result.graph_trace[-1].node == "success"
    assert all(step.duration_ms is not None for step in result.graph_trace)
