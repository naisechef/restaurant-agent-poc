"""Shared gather-state seeding and CSV row-shaping helpers for the `gather` orchestration."""

from __future__ import annotations

import json
import re

from restaurant_agent.schemas import RestaurantQuery
from restaurant_agent.state import GatherState

GATHER_RESULT_COLUMNS = [
    "restaurant_id",
    "name",
    "city",
    "prediction",
    "confidence",
    "evidence",
    "source_evidence_snippets",
    "source_urls",
    "source_types",
    "reasoning",
    "validation_status",
    "needs_review",
    "gather_errors",
    "error",
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _resolve_restaurant_id(query: RestaurantQuery) -> str:
    slug = _SLUG_RE.sub("-", f"{query.name}-{query.city}".lower()).strip("-")
    return slug or "unknown-restaurant"


def _initial_gather_state(query: RestaurantQuery) -> GatherState:
    return GatherState(
        restaurant_id=_resolve_restaurant_id(query),
        raw_text="",
        query=query,
    )


def _gather_state_to_row(state: GatherState) -> dict[str, object]:
    snippets = [item.snippet for item in state.gathered_evidence]
    urls = [item.url for item in state.gathered_evidence]
    types = [item.source_type.value for item in state.gathered_evidence]
    city = state.query.city if state.query else ""

    return {
        "restaurant_id": state.restaurant_id,
        "name": state.query.name if state.query else "",
        "city": city,
        "prediction": state.prediction,
        "confidence": state.confidence,
        "evidence": json.dumps(state.evidence, ensure_ascii=False),
        "source_evidence_snippets": json.dumps(snippets, ensure_ascii=False),
        "source_urls": json.dumps(urls, ensure_ascii=False),
        "source_types": json.dumps(types, ensure_ascii=False),
        "reasoning": state.reasoning,
        "validation_status": state.validation_status,
        "needs_review": state.needs_review,
        "gather_errors": json.dumps(state.gather_errors, ensure_ascii=False),
        "error": state.error,
    }
