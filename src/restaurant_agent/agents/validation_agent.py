"""Validation agent — applies business rules and sets review routing fields."""

from __future__ import annotations

from restaurant_agent.schemas import ExtractionResult
from restaurant_agent.state import AgentState, ValidationStatus
from restaurant_agent.validation import (
    has_supporting_evidence,
    is_confidence_valid,
    needs_human_review,
)


def run(state: AgentState, threshold: float) -> AgentState:
    """Validate extraction output and populate validation_status and needs_review."""
    if state.error or state.prediction is None:
        return state.model_copy(
            update={
                "validation_status": "failed",
                "needs_review": True,
            }
        )

    result = ExtractionResult(
        label=state.prediction,
        confidence=state.confidence if state.confidence is not None else 0.0,
        evidence=state.evidence,
        reasoning=state.reasoning or "",
    )

    review = needs_human_review(result, threshold)
    status: ValidationStatus

    if not is_confidence_valid(result) or not has_supporting_evidence(result):
        status = "flagged"
    elif review:
        status = "flagged"
    else:
        status = "ok"

    return state.model_copy(
        update={
            "validation_status": status,
            "needs_review": review,
        }
    )
