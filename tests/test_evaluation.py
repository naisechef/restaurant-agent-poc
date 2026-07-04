"""Unit tests for evaluation metrics."""

from __future__ import annotations

from restaurant_agent.evaluation import evaluate_predictions
from restaurant_agent.state import AgentState


def _state(
    restaurant_id: str,
    *,
    prediction: str | None,
    confidence: float | None = None,
    error: str | None = None,
    validation_status: str | None = None,
) -> AgentState:
    return AgentState(
        restaurant_id=restaurant_id,
        raw_text="evidence",
        prediction=prediction,  # type: ignore[arg-type]
        confidence=confidence,
        error=error,
        validation_status=validation_status,  # type: ignore[arg-type]
    )


def test_evaluate_predictions_computes_core_metrics() -> None:
    states = [
        _state("r1", prediction="yes", confidence=0.9),
        _state("r2", prediction="no", confidence=0.85),
        _state("r3", prediction="unknown", confidence=0.5),
        _state("r4", prediction="yes", confidence=0.55),
    ]
    expected = {
        "r1": "yes",
        "r2": "no",
        "r3": "unknown",
        "r4": "no",
    }

    summary = evaluate_predictions(states, expected)

    assert summary.total_records == 4
    assert summary.failure_count == 0
    assert summary.accuracy == 0.75
    assert summary.false_positives == 1
    assert summary.false_negatives == 0
    assert summary.precision_yes == 0.5
    assert summary.recall_yes == 1.0
    assert summary.low_confidence_count == 2


def test_evaluate_predictions_excludes_failed_records_from_metrics() -> None:
    states = [
        _state("r1", prediction="yes", confidence=0.9),
        _state("r2", prediction=None, confidence=None, error="LLM timeout"),
        _state(
            "r3",
            prediction="no",
            confidence=0.8,
            validation_status="failed",
        ),
    ]
    expected = {"r1": "yes", "r2": "no", "r3": "no"}

    summary = evaluate_predictions(states, expected)

    assert summary.total_records == 3
    assert summary.failure_count == 2
    assert summary.accuracy == 1.0
    assert summary.false_positives == 0
    assert summary.false_negatives == 0


def test_evaluate_predictions_handles_empty_input() -> None:
    summary = evaluate_predictions([], {})

    assert summary.total_records == 0
    assert summary.failure_count == 0
    assert summary.accuracy == 0.0
    assert summary.precision_yes == 0.0
    assert summary.recall_yes == 0.0
    assert summary.false_positives == 0
    assert summary.false_negatives == 0
    assert summary.low_confidence_count == 0


def test_evaluate_predictions_handles_all_unknown_labels() -> None:
    states = [
        _state("r1", prediction="unknown", confidence=0.4),
        _state("r2", prediction="unknown", confidence=0.3),
    ]
    expected = {"r1": "unknown", "r2": "unknown"}

    summary = evaluate_predictions(states, expected)

    assert summary.accuracy == 1.0
    assert summary.precision_yes == 0.0
    assert summary.recall_yes == 0.0
    assert summary.false_positives == 0
    assert summary.false_negatives == 0
    assert summary.low_confidence_count == 2


def test_evaluate_predictions_respects_custom_threshold() -> None:
    states = [_state("r1", prediction="yes", confidence=0.55)]
    expected = {"r1": "yes"}

    default_summary = evaluate_predictions(states, expected)
    custom_summary = evaluate_predictions(
        states,
        expected,
        confidence_threshold=0.5,
    )

    assert default_summary.low_confidence_count == 1
    assert custom_summary.low_confidence_count == 0


def test_evaluate_predictions_handles_all_wrong_predictions() -> None:
    states = [
        _state("r1", prediction="no", confidence=0.9),
        _state("r2", prediction="unknown", confidence=0.8),
    ]
    expected = {"r1": "yes", "r2": "yes"}

    summary = evaluate_predictions(states, expected)

    assert summary.accuracy == 0.0
    assert summary.false_positives == 0
    assert summary.false_negatives == 2
    assert summary.recall_yes == 0.0
