"""Request/response schemas for the web demo API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_NAME_LENGTH = 200
MAX_CITY_LENGTH = 100


class GatherRequest(BaseModel):
    """Validated input for a single gather operation."""

    name: str = Field(..., min_length=1, max_length=MAX_NAME_LENGTH)
    city: str = Field(..., min_length=1, max_length=MAX_CITY_LENGTH)
    backend: Literal["pipeline", "graph"] = "pipeline"
    dry_run: bool = True
    live_google: bool = False

    @field_validator("name", "city", mode="before")
    @classmethod
    def strip_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("name", "city")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value


class GatherErrorResponse(BaseModel):
    """User-safe error payload for HTML/JSON responses."""

    message: str
