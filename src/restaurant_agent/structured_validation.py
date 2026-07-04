"""Structured validation: compare LLM predictions against provider attribute data."""

from __future__ import annotations

from restaurant_agent.schemas import (
    OutdoorSeatingLabel,
    StructuredSourceAttributes,
    StructuredValidation,
)

_OUTDOOR_SEATING = "outdoor_seating"


def validate_outdoor_seating(
    prediction: OutdoorSeatingLabel | None,
    attributes: StructuredSourceAttributes | None,
) -> StructuredValidation:
    """Compare an LLM outdoor-seating prediction with a provider's structured value."""
    source = attributes.source if attributes else "unknown"
    source_value = attributes.outdoor_seating if attributes else None

    if source_value is None:
        return StructuredValidation(
            source=source,
            attribute=_OUTDOOR_SEATING,
            source_value=None,
            prediction=prediction,
            status="unavailable",
            needs_review=False,
            explanation=(
                "Structured outdoor seating data is unavailable from this provider."
            ),
        )

    if prediction is None or prediction == "unknown":
        return StructuredValidation(
            source=source,
            attribute=_OUTDOOR_SEATING,
            source_value=source_value,
            prediction=prediction,
            status="structured_only",
            needs_review=True,
            explanation=(
                "The LLM could not determine outdoor seating from evidence text; "
                f"the provider reports outdoor seating as {source_value}."
            ),
        )

    agrees = (prediction == "yes") == source_value
    return StructuredValidation(
        source=source,
        attribute=_OUTDOOR_SEATING,
        source_value=source_value,
        prediction=prediction,
        status="verified" if agrees else "conflict",
        needs_review=not agrees,
        explanation=(
            f"LLM prediction ({prediction}) "
            f"{'matches' if agrees else 'conflicts with'} the provider's "
            f"structured outdoor seating value ({source_value})."
        ),
    )


def build_structured_validations(
    prediction: OutdoorSeatingLabel | None,
    attribute_sources: list[StructuredSourceAttributes],
) -> list[StructuredValidation]:
    """Run all applicable structured validations for the current prediction."""
    if not attribute_sources:
        return [validate_outdoor_seating(prediction, None)]
    return [validate_outdoor_seating(prediction, attrs) for attrs in attribute_sources]


def escalation_needs_review(validations: list[StructuredValidation]) -> bool:
    """Return True when any validation requires human review."""
    return any(
        v.needs_review and v.status in ("conflict", "structured_only")
        for v in validations
    )
