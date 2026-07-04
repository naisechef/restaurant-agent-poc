"""Unit tests for validation pure functions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from restaurant_agent.schemas import ExtractionResult
from restaurant_agent.validation import (
    has_supporting_evidence,
    is_confidence_valid,
    needs_human_review,
)


def test_is_confidence_valid_accepts_in_range_values() -> None:
    result = ExtractionResult(
        label="yes",
        confidence=0.75,
        evidence=["patio seating"],
        reasoning="Clear outdoor seating mention.",
    )
    assert is_confidence_valid(result) is True


def test_extraction_result_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        ExtractionResult(
            label="yes",
            confidence=1.5,
            evidence=["patio seating"],
            reasoning="Invalid confidence should fail schema validation.",
        )


def test_has_supporting_evidence_requires_quotes_for_yes_no() -> None:
    with_evidence = ExtractionResult(
        label="yes",
        confidence=0.9,
        evidence=["patio seating"],
        reasoning="Supported.",
    )
    without_evidence = ExtractionResult(
        label="no",
        confidence=0.9,
        evidence=[],
        reasoning="Unsupported.",
    )

    assert has_supporting_evidence(with_evidence) is True
    assert has_supporting_evidence(without_evidence) is False


def test_has_supporting_evidence_allows_unknown_without_quotes() -> None:
    result = ExtractionResult(
        label="unknown",
        confidence=0.4,
        evidence=[],
        reasoning="Ambiguous evidence.",
    )
    assert has_supporting_evidence(result) is True


def test_has_supporting_evidence_ignores_blank_quote_entries() -> None:
    result = ExtractionResult(
        label="yes",
        confidence=0.8,
        evidence=["", "   "],
        reasoning="Blank evidence entries should not count.",
    )
    assert has_supporting_evidence(result) is False


@pytest.mark.parametrize(
    ("confidence", "label", "evidence", "expected"),
    [
        (0.95, "yes", ["patio"], False),
        (0.45, "yes", ["patio"], True),
        (0.95, "yes", [], True),
        (0.95, "unknown", [], False),
    ],
)
def test_needs_human_review(
    confidence: float,
    label: str,
    evidence: list[str],
    expected: bool,
) -> None:
    result = ExtractionResult(
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=evidence,
        reasoning="Review rule test.",
    )
    assert needs_human_review(result, threshold=0.6) is expected
