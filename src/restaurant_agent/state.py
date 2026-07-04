"""Shared pipeline state passed between agents."""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from restaurant_agent.schemas import Evidence, OutdoorSeatingLabel, RestaurantQuery, SourceResult

ValidationStatus = Literal["ok", "flagged", "failed"]


class AgentState(BaseModel):
    """Single state object flowing through preprocessing, extraction, and validation."""

    restaurant_id: str
    raw_text: str
    cleaned_text: str | None = None
    prediction: OutdoorSeatingLabel | None = None
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)
    reasoning: str | None = None
    validation_status: ValidationStatus | None = None
    needs_review: bool = False
    error: str | None = None


class GatherState(AgentState):
    """Extended state for evidence gathering plus extraction pipeline."""

    query: RestaurantQuery | None = None
    source_results: Annotated[list[SourceResult], operator.add] = Field(
        default_factory=list
    )
    gathered_evidence: list[Evidence] = Field(default_factory=list)
    gather_errors: list[str] = Field(default_factory=list)
