"""Batch evaluation metrics for pipeline predictions."""

from __future__ import annotations

from restaurant_agent.config import _DEFAULT_CONFIDENCE_THRESHOLD
from restaurant_agent.schemas import EvaluationSummary
from restaurant_agent.state import AgentState


def _is_failed(state: AgentState) -> bool:
    return state.error is not None or state.validation_status == "failed"


def evaluate_predictions(
    states: list[AgentState],
    expected: dict[str, str],
    confidence_threshold: float | None = None,
) -> EvaluationSummary:
    """Compare predictions to expected labels and compute aggregate metrics."""
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else _DEFAULT_CONFIDENCE_THRESHOLD
    )

    total_records = len(states)
    failure_count = sum(1 for state in states if _is_failed(state))

    evaluated: list[tuple[str, str]] = []
    low_confidence_count = 0

    for state in states:
        if _is_failed(state):
            continue

        if state.prediction is None:
            failure_count += 1
            continue

        expected_label = expected.get(state.restaurant_id)
        if expected_label is None:
            continue

        evaluated.append((state.prediction, expected_label))

        if state.confidence is not None and state.confidence < threshold:
            low_confidence_count += 1

    if not evaluated:
        return EvaluationSummary(
            accuracy=0.0,
            precision_yes=0.0,
            recall_yes=0.0,
            false_positives=0,
            false_negatives=0,
            low_confidence_count=low_confidence_count,
            total_records=total_records,
            failure_count=failure_count,
        )

    correct = sum(1 for predicted, label in evaluated if predicted == label)
    accuracy = correct / len(evaluated)

    false_positives = sum(
        1 for predicted, label in evaluated if predicted == "yes" and label != "yes"
    )
    false_negatives = sum(
        1 for predicted, label in evaluated if predicted != "yes" and label == "yes"
    )

    true_positives = sum(
        1 for predicted, label in evaluated if predicted == "yes" and label == "yes"
    )
    predicted_yes = sum(1 for predicted, _ in evaluated if predicted == "yes")
    actual_yes = sum(1 for _, label in evaluated if label == "yes")

    precision_yes = true_positives / predicted_yes if predicted_yes else 0.0
    recall_yes = true_positives / actual_yes if actual_yes else 0.0

    return EvaluationSummary(
        accuracy=accuracy,
        precision_yes=precision_yes,
        recall_yes=recall_yes,
        false_positives=false_positives,
        false_negatives=false_negatives,
        low_confidence_count=low_confidence_count,
        total_records=total_records,
        failure_count=failure_count,
    )
