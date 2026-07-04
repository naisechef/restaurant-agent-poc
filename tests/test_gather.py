"""Tests for parallel evidence gathering."""

from __future__ import annotations

from restaurant_agent.gather import gather_all, run_adapter_safe
from restaurant_agent.schemas import Evidence, EvidenceSourceType, RestaurantQuery
from restaurant_agent.sources.fake import FakeSourceAdapter

QUERY = RestaurantQuery(name="Test Cafe", city="London")


def test_run_adapter_safe_captures_exception() -> None:
    adapter = FakeSourceAdapter("broken", raises=RuntimeError("timeout"))
    result = run_adapter_safe(adapter, QUERY)
    assert result.error == "timeout"
    assert result.evidence == []


def test_run_adapter_safe_returns_evidence() -> None:
    evidence = Evidence(
        source_type=EvidenceSourceType.SEARCH,
        source_name="fake",
        snippet="Patio seating.",
    )
    adapter = FakeSourceAdapter("fake", evidence=[evidence])
    result = run_adapter_safe(adapter, QUERY)
    assert result.error is None
    assert len(result.evidence) == 1


def test_gather_all_isolates_failures() -> None:
    good_evidence = Evidence(
        source_type=EvidenceSourceType.SEARCH,
        source_name="good",
        snippet="Outdoor deck.",
    )
    adapters = [
        FakeSourceAdapter("good", evidence=[good_evidence]),
        FakeSourceAdapter("bad", raises=RuntimeError("network error")),
        FakeSourceAdapter("also_good", evidence=[]),
    ]
    results = gather_all(QUERY, adapters)
    assert len(results) == 3
    assert results[0].error is None
    assert results[1].error == "network error"
    assert results[2].error is None


def test_gather_all_empty_adapters() -> None:
    assert gather_all(QUERY, []) == []
