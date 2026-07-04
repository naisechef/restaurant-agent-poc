"""Pydantic data contracts for pipeline I/O and evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

OutdoorSeatingLabel = Literal["yes", "no", "unknown"]
EvidenceReliability = Literal["high", "medium", "low"]


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


class RestaurantQuery(BaseModel):
    """Input for evidence gathering by restaurant name and city."""

    name: str
    city: str
    website_url: str | None = None


class EvidenceSourceType(str, Enum):
    """Category of evidence source."""

    WEBSITE = "website"
    SEARCH = "search"
    MAPS = "maps"
    REVIEW = "review"


class Evidence(BaseModel):
    """A single evidence snippet from a source adapter."""

    source_type: EvidenceSourceType
    source_name: str
    url: str | None = None
    snippet: str
    reliability: EvidenceReliability = "medium"


class SourceResult(BaseModel):
    """Result from one source adapter invocation."""

    source_name: str
    evidence: list[Evidence] = Field(default_factory=list)
    error: str | None = None


class GatheredEvidence(BaseModel):
    """Merged evidence from all sources after deduplication."""

    items: list[Evidence] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
