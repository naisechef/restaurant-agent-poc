"""Unit tests for validation agent."""

from __future__ import annotations

import pytest

from restaurant_agent.agents import validation_agent
from restaurant_agent.state import AgentState


def test_validation_agent_ok_for_high_confidence_with_evidence() -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="patio seating",
        cleaned_text="patio seating",
        prediction="yes",
        confidence=0.92,
        evidence=["patio seating"],
        reasoning="Clear outdoor seating mention.",
    )

    result = validation_agent.run(state, threshold=0.6)

    assert result.validation_status == "ok"
    assert result.needs_review is False


def test_validation_agent_flags_low_confidence() -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="maybe a terrace",
        cleaned_text="maybe a terrace",
        prediction="yes",
        confidence=0.45,
        evidence=["maybe a terrace"],
        reasoning="Weak inference.",
    )

    result = validation_agent.run(state, threshold=0.6)

    assert result.validation_status == "flagged"
    assert result.needs_review is True


def test_validation_agent_flags_missing_evidence_for_yes_no() -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="indoor dining only",
        cleaned_text="indoor dining only",
        prediction="no",
        confidence=0.95,
        evidence=[],
        reasoning="No outdoor seating mentioned.",
    )

    result = validation_agent.run(state, threshold=0.6)

    assert result.validation_status == "flagged"
    assert result.needs_review is True


def test_validation_agent_allows_unknown_without_evidence() -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="great food downtown",
        cleaned_text="great food downtown",
        prediction="unknown",
        confidence=0.75,
        evidence=[],
        reasoning="No outdoor seating information.",
    )

    result = validation_agent.run(state, threshold=0.6)

    assert result.validation_status == "ok"
    assert result.needs_review is False


def test_validation_agent_marks_extraction_error_as_failed() -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="patio seating",
        cleaned_text="patio seating",
        error="Failed to parse extraction JSON after repair retry.",
    )

    result = validation_agent.run(state, threshold=0.6)

    assert result.validation_status == "failed"
    assert result.needs_review is True


def test_validation_agent_marks_missing_prediction_as_failed() -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="patio seating",
        cleaned_text="patio seating",
    )

    result = validation_agent.run(state, threshold=0.6)

    assert result.validation_status == "failed"
    assert result.needs_review is True


@pytest.mark.parametrize("threshold", [0.6, 0.85])
def test_validation_agent_respects_threshold(threshold: float) -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="terrace seating",
        cleaned_text="terrace seating",
        prediction="yes",
        confidence=0.7,
        evidence=["terrace seating"],
        reasoning="Terrace implies outdoor seating.",
    )

    result = validation_agent.run(state, threshold=threshold)

    if threshold <= 0.7:
        assert result.validation_status == "ok"
        assert result.needs_review is False
    else:
        assert result.validation_status == "flagged"
        assert result.needs_review is True
