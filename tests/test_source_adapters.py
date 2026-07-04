"""Tests for evidence source adapters (network-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from restaurant_agent.config import MissingGooglePlacesAPIKeyError, Settings
from restaurant_agent.schemas import Evidence, EvidenceSourceType, RestaurantQuery
from restaurant_agent.sources.factory import build_adapters
from restaurant_agent.sources.fake import FakeSourceAdapter
from restaurant_agent.sources.google_places import GooglePlacesAdapter
from restaurant_agent.sources.static_search import StaticSearchAdapter

QUERY = RestaurantQuery(name="The River Cafe", city="London")


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "search.json").write_text(
        json.dumps(
            {
                "the river cafe|london": [
                    {
                        "url": "https://example.com/s",
                        "snippet": "Outdoor terrace seating.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (fixtures / "reviews.json").write_text("{}", encoding="utf-8")
    (fixtures / "maps.json").write_text("{}", encoding="utf-8")
    return fixtures


def test_fake_source_adapter_returns_evidence() -> None:
    evidence = Evidence(
        source_type=EvidenceSourceType.SEARCH,
        source_name="fake",
        snippet="Patio seating available.",
    )
    adapter = FakeSourceAdapter("fake", evidence=[evidence])
    results = adapter.gather(QUERY)
    assert len(results) == 1
    assert results[0].snippet == "Patio seating available."


def test_fake_source_adapter_raises_when_configured() -> None:
    adapter = FakeSourceAdapter("broken", raises=RuntimeError("timeout"))
    with pytest.raises(RuntimeError, match="timeout"):
        adapter.gather(QUERY)


def test_static_search_adapter_fixture_hit(fixtures_dir: Path) -> None:
    adapter = StaticSearchAdapter(
        "static_search",
        fixtures_dir / "search.json",
        EvidenceSourceType.SEARCH,
    )
    results = adapter.gather(QUERY)
    assert len(results) == 1
    assert "Outdoor terrace" in results[0].snippet
    assert results[0].reliability == "medium"


def test_static_search_adapter_case_insensitive(fixtures_dir: Path) -> None:
    adapter = StaticSearchAdapter(
        "static_search",
        fixtures_dir / "search.json",
        EvidenceSourceType.SEARCH,
    )
    query = RestaurantQuery(name="THE RIVER CAFE", city="London")
    results = adapter.gather(query)
    assert len(results) == 1


def test_static_search_adapter_miss_returns_empty(fixtures_dir: Path) -> None:
    adapter = StaticSearchAdapter(
        "static_search",
        fixtures_dir / "search.json",
        EvidenceSourceType.SEARCH,
    )
    query = RestaurantQuery(name="Unknown Place", city="Paris")
    assert adapter.gather(query) == []


def test_static_search_adapter_missing_file_returns_empty(tmp_path: Path) -> None:
    adapter = StaticSearchAdapter(
        "static_search",
        tmp_path / "missing.json",
        EvidenceSourceType.SEARCH,
    )
    assert adapter.gather(QUERY) == []


def test_build_adapters_dry_run(fixtures_dir: Path) -> None:
    adapters = build_adapters(fixtures_dir, dry_run=True)
    assert set(adapters) == {"website", "search", "reviews", "maps"}
    assert len(adapters["search"].gather(QUERY)) == 1
    assert adapters["website"].gather(QUERY) == []


def test_build_adapters_static(fixtures_dir: Path) -> None:
    adapters = build_adapters(fixtures_dir, dry_run=False)
    search_results = adapters["search"].gather(QUERY)
    assert len(search_results) == 1
    assert adapters["website"].gather(QUERY) == []


def test_build_adapters_live_google_missing_key_raises(fixtures_dir: Path) -> None:
    settings = Settings(google_places_api_key=None)
    with pytest.raises(MissingGooglePlacesAPIKeyError):
        build_adapters(fixtures_dir, live_source="google", settings=settings)


def test_build_adapters_live_google_swaps_maps_slot_only(fixtures_dir: Path) -> None:
    settings = Settings(google_places_api_key="test-key")
    adapters = build_adapters(fixtures_dir, live_source="google", settings=settings)

    assert isinstance(adapters["maps"], GooglePlacesAdapter)
    assert isinstance(adapters["search"], StaticSearchAdapter)
    assert isinstance(adapters["reviews"], StaticSearchAdapter)
    assert isinstance(adapters["website"], FakeSourceAdapter)


def test_build_adapters_without_live_source_never_requires_google_key(
    fixtures_dir: Path,
) -> None:
    settings = Settings(google_places_api_key=None)
    # Must not raise even though the Google key is unset, since Google is
    # never selected here.
    adapters = build_adapters(fixtures_dir, dry_run=True, settings=settings)
    assert isinstance(adapters["maps"], StaticSearchAdapter)
