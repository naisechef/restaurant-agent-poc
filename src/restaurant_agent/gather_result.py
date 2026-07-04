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
    StructuredSourceAttributes,
    StructuredValidation,
    StructuredValidationStatus,
    ValidationSummary,
)
from restaurant_agent.state import GatherState
from restaurant_agent.structured_validation import (
    build_structured_validations,
    escalation_needs_review,
)


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


def _find_structured_attribute_sources(
    state: GatherState,
) -> list[StructuredSourceAttributes]:
    sources: list[StructuredSourceAttributes] = []
    for result in state.source_results:
        if result.structured_attributes is not None:
            sources.append(result.structured_attributes)
    return sources


def _build_google_places_summary(state: GatherState) -> GooglePlacesSummary | None:
    google_items = [
        item for item in state.gathered_evidence if item.source_name == "google_places"
    ]
    google_source = next(
        (result for result in state.source_results if result.source_name == "google_places"),
        None,
    )
    place_location = google_source.place_location if google_source else None
    attributes = google_source.structured_attributes if google_source else None

    if not google_items and place_location is None and attributes is None:
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
        place_name=(
            place_location.place_name
            if place_location and place_location.place_name
            else (attributes.place_name if attributes else None)
        ),
        formatted_address=place_location.formatted_address if place_location else None,
        google_maps_url=maps_url,
        latitude=place_location.latitude if place_location else None,
        longitude=place_location.longitude if place_location else None,
        rating=attributes.rating if attributes else None,
        user_rating_count=attributes.user_rating_count if attributes else None,
        place_id=attributes.place_id if attributes else None,
    )


def _node_update_type(node: str) -> str:
    if node == "resolve_restaurant":
        return "resolve"
    if node.startswith("gather_"):
        return "source_gather"
    if node == "merge_evidence":
        return "merge"
    if node == "preprocess":
        return "preprocess"
    if node == "extract":
        return "extract"
    if node == "validate":
        return "validate"
    if node in {"success", "needs_review", "failed"}:
        return "terminal"
    return "update"


def _node_status(node: str, update: dict[str, object]) -> str:
    if node in {"success", "needs_review", "failed"}:
        return node
    if update.get("error"):
        return "error"
    if "validation_status" in update:
        return str(update["validation_status"])
    if "source_results" in update:
        results = update["source_results"]
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict) and result.get("error"):
                    return "error"
        return "completed"
    return "completed"


def _node_evidence_count(node: str, update: dict[str, object]) -> int | None:
    if "source_results" in update:
        results = update["source_results"]
        if isinstance(results, list):
            total = 0
            for result in results:
                if isinstance(result, dict):
                    evidence = result.get("evidence")
                    if isinstance(evidence, list):
                        total += len(evidence)
            return total
    if "gathered_evidence" in update:
        items = update["gathered_evidence"]
        if isinstance(items, list):
            return len(items)
    return None


def _primary_structured_status(
    validations: list[StructuredValidation],
) -> StructuredValidationStatus | None:
    priority = ("conflict", "structured_only", "verified", "unavailable")
    statuses = {v.status for v in validations}
    for status in priority:
        if status in statuses:
            return status
    return validations[0].status if validations else None


def _build_validation_summary(
    *,
    pipeline_status: str | None,
    route: GatherRoute,
    structured_validations: list[StructuredValidation],
    route_escalated: bool,
) -> ValidationSummary:
    return ValidationSummary(
        pipeline_status=pipeline_status,  # type: ignore[arg-type]
        route=route,
        structured_status=_primary_structured_status(structured_validations),
        route_escalated=route_escalated,
    )


def _format_source_label(source_name: str) -> str:
    if source_name == "google_places":
        return "Google Places"
    return source_name.replace("_", " ").title()


def _build_decision_path(
    state: GatherState,
    *,
    structured_validations: list[StructuredValidation],
    route: GatherRoute,
    route_escalated: bool,
) -> list[str]:
    steps: list[str] = []

    for result in state.source_results:
        if result.evidence:
            label = _format_source_label(result.source_name)
            steps.append(f"Gathered {label} evidence")

    if state.prediction:
        steps.append(f"Claude predicted {state.prediction.upper()}")

    for validation in structured_validations:
        if validation.source_value is not None:
            label = _format_source_label(validation.source)
            value = "TRUE" if validation.source_value else "FALSE"
            steps.append(f"{label} structured outdoor seating={value}")

        if validation.status == "verified":
            steps.append("Decision verified")
        elif validation.status == "conflict":
            steps.append("Decision conflict detected")
        elif validation.status == "structured_only":
            steps.append("Structured data only (LLM unknown)")
        elif validation.status == "unavailable":
            steps.append("Structured validation unavailable")

    route_label = route.replace("_", " ")
    steps.append(f"Route {route_label}")
    if route_escalated:
        steps.append("Route escalated due to structured validation")

    return steps


def _append_validation_to_summary(
    summary: str,
    *,
    structured_status: StructuredValidationStatus | None,
    route: GatherRoute | None,
    route_escalated: bool,
) -> str:
    parts = [summary] if summary else []
    if structured_status:
        parts.append(f"Structured validation: {structured_status}.")
    if route:
        parts.append(f"Route: {route.replace('_', ' ')}.")
    if route_escalated:
        parts.append("Route escalated due to structured validation.")
    return " ".join(part for part in parts if part)


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


def enrich_graph_trace(
    trace: list[GraphNodeExecution],
    *,
    structured_status: StructuredValidationStatus | None = None,
    route: GatherRoute | None = None,
    route_escalated: bool = False,
) -> list[GraphNodeExecution]:
    """Attach redacted summaries and observability metadata to graph trace entries."""
    enriched: list[GraphNodeExecution] = []
    terminal_nodes = {"success", "needs_review", "failed"}

    for step in trace:
        summary = summarize_graph_update(step.node, step.update)
        if step.node in terminal_nodes:
            summary = _append_validation_to_summary(
                summary,
                structured_status=structured_status,
                route=route,
                route_escalated=route_escalated,
            )

        enriched.append(
            step.model_copy(
                update={
                    "summary": summary,
                    "status": _node_status(step.node, step.update),
                    "evidence_count": _node_evidence_count(step.node, step.update),
                    "update_type": _node_update_type(step.node),
                }
            )
        )
    return enriched


def _total_duration_ms(trace: list[GraphNodeExecution]) -> float | None:
    durations = [step.duration_ms for step in trace if step.duration_ms is not None]
    if not durations:
        return None
    return round(sum(durations), 2)


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

    attribute_sources = _find_structured_attribute_sources(state)
    structured_validations = build_structured_validations(
        state.prediction, attribute_sources
    )

    base_route = _resolve_route(state)
    needs_review = state.needs_review
    route_escalated = base_route == "success" and escalation_needs_review(
        structured_validations
    )
    route = base_route
    if route_escalated:
        route = "needs_review"
        needs_review = True

    validation_summary = _build_validation_summary(
        pipeline_status=state.validation_status,
        route=route,
        structured_validations=structured_validations,
        route_escalated=route_escalated,
    )
    decision_path = _build_decision_path(
        state,
        structured_validations=structured_validations,
        route=route,
        route_escalated=route_escalated,
    )
    total_duration_ms = _total_duration_ms(graph_trace) if graph_trace else None

    trace = (
        enrich_graph_trace(
            graph_trace,
            structured_status=validation_summary.structured_status,
            route=route,
            route_escalated=route_escalated,
        )
        if graph_trace
        else None
    )

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
        needs_review=needs_review,
        route=route,
        error=state.error,
        reliability_mix=_reliability_mix(state.gathered_evidence),
        google_places=_build_google_places_summary(state),
        structured_validations=structured_validations,
        validation_summary=validation_summary,
        decision_path=decision_path,
        total_duration_ms=total_duration_ms,
        graph_trace=trace,
    )
