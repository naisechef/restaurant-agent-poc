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


class PlaceLocation(BaseModel):
    """Sanitized place metadata for web display (no raw API payloads)."""

    place_name: str | None = None
    formatted_address: str | None = None
    google_maps_url: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class SourceResult(BaseModel):
    """Result from one source adapter invocation."""

    source_name: str
    evidence: list[Evidence] = Field(default_factory=list)
    error: str | None = None
    place_location: PlaceLocation | None = None


class GatheredEvidence(BaseModel):
    """Merged evidence from all sources after deduplication."""

    items: list[Evidence] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


ValidationStatus = Literal["ok", "flagged", "failed"]
GatherRoute = Literal["success", "needs_review", "failed"]
GatherBackend = Literal["pipeline", "graph"]


class SourceAdapterExecution(BaseModel):
    """One adapter's execution outcome for web/API responses."""

    source_name: str
    evidence_count: int
    error: str | None = None


class GraphNodeExecution(BaseModel):
    """One LangGraph node's partial state update, in execution order."""

    node: str
    update: dict[str, object] = Field(default_factory=dict)
    summary: str = ""


class GooglePlacesSummary(BaseModel):
    """Aggregated Google Places evidence for observability (no API secrets)."""

    source_name: str = "google_places"
    maps_evidence_count: int = 0
    review_evidence_count: int = 0
    reliability_high: int = 0
    reliability_medium: int = 0
    reliability_low: int = 0
    maps_url: str | None = None
    place_name: str | None = None
    formatted_address: str | None = None
    google_maps_url: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class GatherRunResult(BaseModel):
    """Structured gather outcome returned by the web layer (in-memory, no CSV)."""

    restaurant_id: str
    name: str
    city: str
    backend: GatherBackend
    dry_run: bool = False
    live_google: bool = False
    prediction: OutdoorSeatingLabel | None
    confidence: float | None
    reasoning: str | None
    evidence: list[str]
    gathered_evidence: list[Evidence]
    source_results: list[SourceAdapterExecution]
    gather_errors: list[str]
    validation_status: ValidationStatus | None
    needs_review: bool
    route: GatherRoute
    error: str | None
    reliability_mix: dict[str, int] = Field(default_factory=dict)
    google_places: GooglePlacesSummary | None = None
    graph_trace: list[GraphNodeExecution] | None = None
