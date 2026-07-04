"""Tests for evidence-gathering data contracts."""

from __future__ import annotations

import json

from restaurant_agent.schemas import (
    Evidence,
    EvidenceSourceType,
    GatheredEvidence,
    RestaurantQuery,
    SourceResult,
)


def test_restaurant_query_serialisation() -> None:
    query = RestaurantQuery(name="The River Cafe", city="London")
    payload = json.loads(query.model_dump_json())
    assert payload == {"name": "The River Cafe", "city": "London", "website_url": None}
    restored = RestaurantQuery.model_validate_json(query.model_dump_json())
    assert restored == query


def test_evidence_source_type_values() -> None:
    assert EvidenceSourceType.WEBSITE.value == "website"
    assert EvidenceSourceType.SEARCH.value == "search"
    assert EvidenceSourceType.MAPS.value == "maps"
    assert EvidenceSourceType.REVIEW.value == "review"


def test_evidence_round_trip() -> None:
    evidence = Evidence(
        source_type=EvidenceSourceType.SEARCH,
        source_name="static_search",
        url="https://example.com",
        snippet="Outdoor terrace seating available.",
        reliability="medium",
    )
    restored = Evidence.model_validate_json(evidence.model_dump_json())
    assert restored == evidence


def test_source_result_defaults() -> None:
    result = SourceResult(source_name="fake")
    assert result.evidence == []
    assert result.error is None


def test_gathered_evidence_round_trip() -> None:
    item = Evidence(
        source_type=EvidenceSourceType.REVIEW,
        source_name="static_reviews",
        snippet="Lovely patio area.",
        reliability="low",
    )
    gathered = GatheredEvidence(
        items=[item],
        errors=["broken_source: timeout"],
    )
    restored = GatheredEvidence.model_validate_json(gathered.model_dump_json())
    assert restored == gathered
