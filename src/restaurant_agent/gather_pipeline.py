"""Imperative orchestration for evidence gathering plus extraction."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from restaurant_agent.agents import (
    extraction_agent,
    preprocessing_agent,
    validation_agent,
)
from restaurant_agent.config import Settings
from restaurant_agent.evidence_merge import combine_evidence_text, merge_evidence
from restaurant_agent.gather import gather_all
from restaurant_agent.pipeline import LLMClient, _write_csv
from restaurant_agent.schemas import RestaurantQuery
from restaurant_agent.sources.base import SourceAdapter
from restaurant_agent.state import GatherState

logger = logging.getLogger(__name__)

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


def _apply_merge(state: GatherState, source_results: list) -> GatherState:
    gathered = merge_evidence(source_results)
    combined = combine_evidence_text(gathered.items)

    if not gathered.items:
        return state.model_copy(
            update={
                "source_results": source_results,
                "gathered_evidence": [],
                "gather_errors": gathered.errors,
                "raw_text": "",
                "error": "No evidence gathered from any source",
                "validation_status": "failed",
                "needs_review": True,
            }
        )

    return state.model_copy(
        update={
            "source_results": source_results,
            "gathered_evidence": gathered.items,
            "gather_errors": gathered.errors,
            "raw_text": combined,
            "error": None,
        }
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


def _process_gather(
    query: RestaurantQuery,
    adapters: dict[str, SourceAdapter],
    client: LLMClient,
    threshold: float,
) -> GatherState:
    """Gather evidence, merge, then run the existing agent chain."""
    state = _initial_gather_state(query)
    try:
        source_results = gather_all(query, list(adapters.values()))
        state = _apply_merge(state, source_results)

        if state.error:
            return state

        state = preprocessing_agent.run(state)
        state = extraction_agent.run(state, client)
        state = validation_agent.run(state, threshold)
        return state
    except Exception as exc:
        logger.error(
            "Unexpected error gathering for %s: %s",
            query.name,
            exc,
            exc_info=True,
        )
        return state.model_copy(
            update={
                "error": str(exc),
                "validation_status": "failed",
                "needs_review": True,
            }
        )


def run_gather_pipeline(
    query: RestaurantQuery,
    adapters: dict[str, SourceAdapter],
    output_path: Path,
    review_queue_path: Path,
    settings: Settings,
    client: LLMClient,
) -> GatherState:
    """Gather evidence for one restaurant, extract, validate, and write CSV outputs."""
    logger.info("Gathering evidence for %s, %s", query.name, query.city)

    state = _process_gather(
        query,
        adapters,
        client,
        settings.confidence_threshold,
    )

    row = _gather_state_to_row(state)
    results_rows = [row]
    review_rows = [row] if state.needs_review else []

    _write_csv(output_path, results_rows, GATHER_RESULT_COLUMNS)
    _write_csv(review_queue_path, review_rows, GATHER_RESULT_COLUMNS)

    logger.info("Wrote gather result to %s", output_path)
    if review_rows:
        logger.info("Wrote gather review-queue row to %s", review_queue_path)

    return state
