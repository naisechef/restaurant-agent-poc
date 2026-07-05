"""Tests for structured validation helpers."""

from __future__ import annotations

from restaurant_agent.schemas import StructuredSourceAttributes
from restaurant_agent.structured_validation import (
    build_structured_validations,
    escalation_needs_review,
    validate_outdoor_seating,
)

_GOOGLE = StructuredSourceAttributes(
    source="google_places",
    outdoor_seating=True,
    rating=4.6,
    user_rating_count=812,
    place_id="ChIJ123",
    place_name="The River Cafe",
)


def test_yes_and_true_is_verified_without_review() -> None:
    result = validate_outdoor_seating("yes", _GOOGLE)
    assert result.status == "verified"
    assert result.needs_review is False
    assert result.source_value is True
    assert result.prediction == "yes"
    assert result.source == "google_places"


def test_no_and_false_is_verified_without_review() -> None:
    attrs = _GOOGLE.model_copy(update={"outdoor_seating": False})
    result = validate_outdoor_seating("no", attrs)
    assert result.status == "verified"
    assert result.needs_review is False


def test_yes_and_false_is_conflict_with_review() -> None:
    attrs = _GOOGLE.model_copy(update={"outdoor_seating": False})
    result = validate_outdoor_seating("yes", attrs)
    assert result.status == "conflict"
    assert result.needs_review is True


def test_no_and_true_is_conflict_with_review() -> None:
    result = validate_outdoor_seating("no", _GOOGLE)
    assert result.status == "conflict"
    assert result.needs_review is True


def test_unknown_and_true_is_structured_only_with_review() -> None:
    result = validate_outdoor_seating("unknown", _GOOGLE)
    assert result.status == "structured_only"
    assert result.needs_review is True


def test_none_prediction_and_true_is_structured_only_with_review() -> None:
    result = validate_outdoor_seating(None, _GOOGLE)
    assert result.status == "structured_only"
    assert result.needs_review is True


def test_absent_field_is_unavailable_without_review() -> None:
    result = validate_outdoor_seating("yes", None)
    assert result.status == "unavailable"
    assert result.needs_review is False
    assert result.source_value is None


def test_provider_without_outdoor_seating_is_unavailable() -> None:
    attrs = StructuredSourceAttributes(source="google_places", outdoor_seating=None)
    result = validate_outdoor_seating("yes", attrs)
    assert result.status == "unavailable"
    assert result.needs_review is False
    assert result.source == "google_places"


def test_build_validations_empty_sources_returns_unavailable() -> None:
    results = build_structured_validations("yes", [])
    assert len(results) == 1
    assert results[0].status == "unavailable"


def test_build_validations_multiple_sources() -> None:
    apple = StructuredSourceAttributes(source="apple_maps", outdoor_seating=False)
    results = build_structured_validations("yes", [_GOOGLE, apple])
    assert len(results) == 2
    assert results[0].source == "google_places"
    assert results[1].source == "apple_maps"


def test_escalation_needs_review_on_conflict() -> None:
    conflict = validate_outdoor_seating("no", _GOOGLE)
    assert escalation_needs_review([conflict]) is True


def test_escalation_needs_review_on_structured_only() -> None:
    only = validate_outdoor_seating("unknown", _GOOGLE)
    assert escalation_needs_review([only]) is True


def test_escalation_not_needed_for_verified() -> None:
    verified = validate_outdoor_seating("yes", _GOOGLE)
    assert escalation_needs_review([verified]) is False


def test_escalation_not_needed_for_unavailable() -> None:
    unavailable = validate_outdoor_seating("yes", None)
    assert escalation_needs_review([unavailable]) is False
