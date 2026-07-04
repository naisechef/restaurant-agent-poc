"""Shared pipeline state passed between agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from restaurant_agent.schemas import OutdoorSeatingLabel

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
