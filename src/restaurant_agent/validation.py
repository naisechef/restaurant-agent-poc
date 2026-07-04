"""Pure business-rule validation for extraction results."""

from __future__ import annotations

from restaurant_agent.schemas import ExtractionResult


def is_confidence_valid(result: ExtractionResult) -> bool:
    """Return True when confidence is within the expected 0.0–1.0 range."""
    return 0.0 <= result.confidence <= 1.0


def has_supporting_evidence(result: ExtractionResult) -> bool:
    """Return True when yes/no predictions include non-empty supporting quotes."""
    if result.label == "unknown":
        return True

    return any(quote.strip() for quote in result.evidence)


def needs_human_review(result: ExtractionResult, threshold: float) -> bool:
    """Return True when confidence is low or yes/no predictions lack evidence."""
    if not is_confidence_valid(result):
        return True

    if result.confidence < threshold:
        return True

    if not has_supporting_evidence(result):
        return True

    return False
