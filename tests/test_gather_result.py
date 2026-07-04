"""Tests for GatherRunResult construction helpers."""

from __future__ import annotations

from restaurant_agent.gather_result import (
    build_gather_run_result,
    summarize_graph_update,
)
from restaurant_agent.schemas import (
    Evidence,
    EvidenceSourceType,
    GraphNodeExecution,
    PlaceLocation,
    RestaurantQuery,
    SourceResult,
)
from restaurant_agent.state import GatherState


def test_summarize_graph_update_redacts_large_fields() -> None:
    summary = summarize_graph_update(
        "merge_evidence",
        {
            "gathered_evidence": [{"snippet": "Terrace seating."}],
            "gather_errors": [],
            "error": None,
        },
    )
    assert "Terrace" not in summary
    assert "Merged 1 evidence snippet(s)." in summary


def test_build_gather_run_result_includes_google_places_summary() -> None:
    query = RestaurantQuery(name="Example", city="London")
    state = GatherState(
        restaurant_id="example-london",
        raw_text="",
        query=query,
        source_results=[
            SourceResult(
                source_name="google_places",
                evidence=[],
                place_location=PlaceLocation(
                    place_name="Example Restaurant",
                    formatted_address="1 Example Street, London",
                    google_maps_url="https://maps.google.com/?cid=123",
                    latitude=51.5,
                    longitude=-0.12,
                ),
            )
        ],
        gathered_evidence=[
            Evidence(
                source_type=EvidenceSourceType.MAPS,
                source_name="google_places",
                url="https://maps.google.com/?cid=123",
                snippet="Outdoor seating available.",
                reliability="high",
            ),
            Evidence(
                source_type=EvidenceSourceType.REVIEW,
                source_name="google_places",
                url="https://maps.google.com/?cid=123",
                snippet="Great patio.",
                reliability="low",
            ),
        ],
        validation_status="ok",
        needs_review=False,
    )

    result = build_gather_run_result(
        state,
        backend="pipeline",
        graph_trace=[
            GraphNodeExecution(node="success", update={}, summary="Completed."),
        ],
        dry_run=True,
        live_google=True,
    )

    assert result.google_places is not None
    assert result.google_places.maps_evidence_count == 1
    assert result.google_places.review_evidence_count == 1
    assert result.google_places.reliability_high == 1
    assert result.google_places.reliability_low == 1
    assert result.google_places.maps_url == "https://maps.google.com/?cid=123"
    assert result.google_places.place_name == "Example Restaurant"
    assert result.google_places.formatted_address == "1 Example Street, London"
    assert result.google_places.google_maps_url == "https://maps.google.com/?cid=123"
    assert result.google_places.latitude == 51.5
    assert result.google_places.longitude == -0.12
    assert result.dry_run is True
    assert result.live_google is True
