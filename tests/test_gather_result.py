"""Tests for GatherRunResult construction helpers."""

from __future__ import annotations

from restaurant_agent.gather_result import (
    build_gather_run_result,
    enrich_graph_trace,
    summarize_graph_update,
)
from restaurant_agent.schemas import (
    Evidence,
    EvidenceSourceType,
    GraphNodeExecution,
    PlaceLocation,
    RestaurantQuery,
    SourceResult,
    StructuredSourceAttributes,
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
        backend="graph",
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
    assert len(result.structured_validations) == 1
    assert result.structured_validations[0].status == "unavailable"


def _base_state(**overrides: object) -> GatherState:
    query = RestaurantQuery(name="Example", city="London")
    defaults: dict[str, object] = {
        "restaurant_id": "example-london",
        "raw_text": "evidence",
        "query": query,
        "prediction": "yes",
        "confidence": 0.9,
        "validation_status": "ok",
        "needs_review": False,
    }
    defaults.update(overrides)
    return GatherState(**defaults)  # type: ignore[arg-type]


def test_verified_validation_keeps_success_route() -> None:
    state = _base_state(
        source_results=[
            SourceResult(
                source_name="google_places",
                structured_attributes=StructuredSourceAttributes(
                    source="google_places",
                    outdoor_seating=True,
                ),
            )
        ],
    )
    result = build_gather_run_result(state, backend="graph")
    assert result.structured_validations[0].status == "verified"
    assert result.route == "success"
    assert result.needs_review is False


def test_conflict_escalates_success_to_needs_review() -> None:
    state = _base_state(
        prediction="no",
        source_results=[
            SourceResult(
                source_name="google_places",
                structured_attributes=StructuredSourceAttributes(
                    source="google_places",
                    outdoor_seating=True,
                ),
            )
        ],
    )
    result = build_gather_run_result(state, backend="graph")
    assert result.structured_validations[0].status == "conflict"
    assert result.route == "needs_review"
    assert result.needs_review is True
    assert result.prediction == "no"


def test_google_places_summary_includes_rating_and_place_id() -> None:
    state = _base_state(
        source_results=[
            SourceResult(
                source_name="google_places",
                structured_attributes=StructuredSourceAttributes(
                    source="google_places",
                    outdoor_seating=True,
                    rating=4.6,
                    user_rating_count=812,
                    place_id="ChIJ123",
                    place_name="Example Restaurant",
                ),
                place_location=PlaceLocation(
                    google_maps_url="https://maps.google.com/?cid=123",
                ),
            )
        ],
        gathered_evidence=[
            Evidence(
                source_type=EvidenceSourceType.MAPS,
                source_name="google_places",
                snippet="Outdoor seating available.",
                reliability="high",
            ),
        ],
    )
    result = build_gather_run_result(state, backend="graph", live_google=True)
    assert result.google_places is not None
    assert result.google_places.rating == 4.6
    assert result.google_places.user_rating_count == 812
    assert result.google_places.place_id == "ChIJ123"


def test_enrich_graph_trace_adds_observability_metadata() -> None:
    trace = [
        GraphNodeExecution(
            node="gather_search",
            update={
                "source_results": [
                    {
                        "source_name": "fake_search",
                        "evidence": [{"snippet": "patio"}],
                    }
                ]
            },
            duration_ms=12.5,
        ),
        GraphNodeExecution(node="success", update={}, duration_ms=0.1),
    ]
    enriched = enrich_graph_trace(
        trace,
        structured_status="verified",
        route="success",
        route_escalated=False,
    )
    assert enriched[0].status == "completed"
    assert enriched[0].evidence_count == 1
    assert enriched[0].update_type == "source_gather"
    assert enriched[0].duration_ms == 12.5
    assert "Structured validation: verified" in enriched[1].summary


def test_build_gather_run_result_includes_validation_summary_and_decision_path() -> None:
    state = _base_state(
        source_results=[
            SourceResult(
                source_name="google_places",
                evidence=[
                    Evidence(
                        source_type=EvidenceSourceType.MAPS,
                        source_name="google_places",
                        snippet="Outdoor seating available.",
                        reliability="high",
                    )
                ],
                structured_attributes=StructuredSourceAttributes(
                    source="google_places",
                    outdoor_seating=True,
                ),
            )
        ],
    )
    result = build_gather_run_result(
        state,
        backend="graph",
        graph_trace=[
            GraphNodeExecution(node="success", update={}, duration_ms=1.0),
        ],
    )
    assert result.validation_summary is not None
    assert result.validation_summary.structured_status == "verified"
    assert result.validation_summary.route == "success"
    assert "Claude predicted YES" in result.decision_path
    assert "Google Places structured outdoor seating=TRUE" in result.decision_path
    assert "Decision verified" in result.decision_path
    assert "Route success" in result.decision_path
    assert result.total_duration_ms == 1.0
