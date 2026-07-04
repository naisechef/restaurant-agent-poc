"""Build structured GatherRunResult objects from pipeline state."""

from __future__ import annotations

from restaurant_agent.schemas import (
    EvidenceSourceType,
    GatherBackend,
    GatherRoute,
    GatherRunResult,
    GooglePlacesSummary,
    GraphNodeExecution,
    SourceAdapterExecution,
)
from restaurant_agent.state import GatherState


def _resolve_route(state: GatherState) -> GatherRoute:
    if state.validation_status == "failed":
        return "failed"
    if state.needs_review:
        return "needs_review"
    return "success"


def _reliability_mix(evidence: list) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for item in evidence:
        reliability = getattr(item, "reliability", None) or "medium"
        if reliability in counts:
            counts[reliability] += 1
    return counts


def _build_google_places_summary(state: GatherState) -> GooglePlacesSummary | None:
    google_items = [
        item for item in state.gathered_evidence if item.source_name == "google_places"
    ]
    google_source = next(
        (result for result in state.source_results if result.source_name == "google_places"),
        None,
    )
    place_location = google_source.place_location if google_source else None

    if not google_items and place_location is None:
        return None

    maps_count = sum(
        1 for item in google_items if item.source_type == EvidenceSourceType.MAPS
    )
    review_count = sum(
        1 for item in google_items if item.source_type == EvidenceSourceType.REVIEW
    )
    mix = _reliability_mix(google_items)
    maps_url = (
        place_location.google_maps_url
        if place_location and place_location.google_maps_url
        else next((item.url for item in google_items if item.url), None)
    )

    return GooglePlacesSummary(
        maps_evidence_count=maps_count,
        review_evidence_count=review_count,
        reliability_high=mix["high"],
        reliability_medium=mix["medium"],
        reliability_low=mix["low"],
        maps_url=maps_url,
        place_name=place_location.place_name if place_location else None,
        formatted_address=place_location.formatted_address if place_location else None,
        google_maps_url=maps_url,
        latitude=place_location.latitude if place_location else None,
        longitude=place_location.longitude if place_location else None,
    )


def summarize_graph_update(node: str, update: dict[str, object]) -> str:
    """Produce a safe, human-readable summary of a graph node update."""
    if not update:
        return "Completed with no state changes."

    parts: list[str] = []

    if "restaurant_id" in update:
        parts.append("Resolved restaurant identifier.")

    if "source_results" in update:
        results = update["source_results"]
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue
                name = str(result.get("source_name", "source"))
                evidence = result.get("evidence")
                count = len(evidence) if isinstance(evidence, list) else 0
                error = result.get("error")
                if error:
                    parts.append(f"{name}: adapter error recorded.")
                else:
                    parts.append(f"{name}: gathered {count} snippet(s).")

    if "gathered_evidence" in update:
        items = update["gathered_evidence"]
        if isinstance(items, list):
            parts.append(f"Merged {len(items)} evidence snippet(s).")
        errors = update.get("gather_errors")
        if isinstance(errors, list) and errors:
            parts.append(f"{len(errors)} gather error(s) recorded.")

    if "prediction" in update:
        prediction = update.get("prediction", "unknown")
        confidence = update.get("confidence")
        if confidence is not None:
            parts.append(f"Extracted prediction {prediction} ({confidence:.0%} confidence).")
        else:
            parts.append(f"Extracted prediction {prediction}.")

    if "validation_status" in update:
        status = update["validation_status"]
        needs_review = update.get("needs_review")
        if needs_review:
            parts.append(f"Validation {status}; flagged for review.")
        else:
            parts.append(f"Validation {status}.")

    if update.get("error"):
        parts.append("Recorded pipeline error.")

    if parts:
        return " ".join(parts)

    return f"Updated {', '.join(sorted(str(k) for k in update))}."


def enrich_graph_trace(trace: list[GraphNodeExecution]) -> list[GraphNodeExecution]:
    """Attach redacted summaries to graph trace entries."""
    return [
        step.model_copy(
            update={"summary": summarize_graph_update(step.node, step.update)}
        )
        for step in trace
    ]


def build_gather_run_result(
    state: GatherState,
    *,
    backend: GatherBackend,
    graph_trace: list[GraphNodeExecution] | None = None,
    dry_run: bool = False,
    live_google: bool = False,
) -> GatherRunResult:
    """Convert final GatherState into a structured API result."""
    name = state.query.name if state.query else ""
    city = state.query.city if state.query else ""

    source_results = [
        SourceAdapterExecution(
            source_name=result.source_name,
            evidence_count=len(result.evidence),
            error=result.error,
        )
        for result in state.source_results
    ]

    trace = enrich_graph_trace(graph_trace) if graph_trace else None

    return GatherRunResult(
        restaurant_id=state.restaurant_id,
        name=name,
        city=city,
        backend=backend,
        dry_run=dry_run,
        live_google=live_google,
        prediction=state.prediction,
        confidence=state.confidence,
        reasoning=state.reasoning,
        evidence=list(state.evidence),
        gathered_evidence=list(state.gathered_evidence),
        source_results=source_results,
        gather_errors=list(state.gather_errors),
        validation_status=state.validation_status,
        needs_review=state.needs_review,
        route=_resolve_route(state),
        error=state.error,
        reliability_mix=_reliability_mix(state.gathered_evidence),
        google_places=_build_google_places_summary(state),
        graph_trace=trace,
    )
