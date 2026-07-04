"""Pydantic data contracts for pipeline I/O and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

OutdoorSeatingLabel = Literal["yes", "no", "unknown"]


class RestaurantRecord(BaseModel):
    """Raw input row from the restaurants CSV."""

    id: str
    name: str
    evidence_text: str
    expected_label: OutdoorSeatingLabel | None = None


class ExtractionResult(BaseModel):
    """Structured output contract for the extraction agent (Claude JSON response)."""

    label: OutdoorSeatingLabel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]
    reasoning: str


class EvaluationSummary(BaseModel):
    """Aggregate metrics from a batch evaluation run."""

    accuracy: float
    precision_yes: float
    recall_yes: float
    false_positives: int
    false_negatives: int
    low_confidence_count: int
    total_records: int
    failure_count: int
