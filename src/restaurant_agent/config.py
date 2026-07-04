"""Centralised configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_REQUEST_TIMEOUT = 60
_DEFAULT_CONFIDENCE_THRESHOLD = 0.6
_DEFAULT_INPUT_PATH = Path("data/restaurants.csv")
_DEFAULT_OUTPUT_PATH = Path("outputs/results.csv")
_DEFAULT_REVIEW_QUEUE_PATH = Path("outputs/review_queue.csv")
_DEFAULT_GATHER_OUTPUT_PATH = Path("outputs/gather_results.csv")
_DEFAULT_GATHER_REVIEW_QUEUE_PATH = Path("outputs/gather_review_queue.csv")
_DEFAULT_FIXTURES_PATH = Path("data/evidence_fixtures")
_DEFAULT_LOG_LEVEL = "INFO"


class MissingAPIKeyError(RuntimeError):
    """Raised when ClaudeClient is constructed without a configured API key."""

    def __init__(
        self,
        message: str = (
            "ANTHROPIC_API_KEY is not set. "
            "Add it to .env or export it in your environment."
        ),
    ) -> None:
        super().__init__(message)


class Settings(BaseModel):
    """Application settings. API key presence is not validated at load time."""

    anthropic_api_key: str | None = None
    model: str = _DEFAULT_MODEL
    max_tokens: int = Field(default=_DEFAULT_MAX_TOKENS, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    request_timeout: float = Field(default=_DEFAULT_REQUEST_TIMEOUT, gt=0)
    confidence_threshold: float = Field(
        default=_DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    input_path: Path = Field(default=_DEFAULT_INPUT_PATH)
    output_path: Path = Field(default=_DEFAULT_OUTPUT_PATH)
    review_queue_path: Path = Field(default=_DEFAULT_REVIEW_QUEUE_PATH)
    gather_output_path: Path = Field(default=_DEFAULT_GATHER_OUTPUT_PATH)
    gather_review_queue_path: Path = Field(
        default=_DEFAULT_GATHER_REVIEW_QUEUE_PATH
    )
    fixtures_path: Path = Field(default=_DEFAULT_FIXTURES_PATH)
    log_level: str = _DEFAULT_LOG_LEVEL

    def require_api_key(self) -> str:
        """Return the API key or raise MissingAPIKeyError.

        Intended for ClaudeClient construction — not called at import time.
        """
        if not self.anthropic_api_key:
            raise MissingAPIKeyError()
        return self.anthropic_api_key


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _parse_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _parse_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _parse_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw)


def load_settings(*, env_file: str | Path | None = None) -> Settings:
    """Load settings from environment variables and an optional .env file."""
    load_dotenv(env_file)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key is not None and api_key.strip() == "":
        api_key = None

    return Settings(
        anthropic_api_key=api_key,
        model=os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL),
        max_tokens=_parse_int("ANTHROPIC_MAX_TOKENS", _DEFAULT_MAX_TOKENS),
        temperature=_parse_optional_float("ANTHROPIC_TEMPERATURE"),
        request_timeout=_parse_float(
            "ANTHROPIC_REQUEST_TIMEOUT",
            _DEFAULT_REQUEST_TIMEOUT,
        ),
        confidence_threshold=_parse_float(
            "CONFIDENCE_THRESHOLD",
            _DEFAULT_CONFIDENCE_THRESHOLD,
        ),
        input_path=_parse_path("DEFAULT_INPUT_PATH", _DEFAULT_INPUT_PATH),
        output_path=_parse_path("DEFAULT_OUTPUT_PATH", _DEFAULT_OUTPUT_PATH),
        review_queue_path=_parse_path(
            "DEFAULT_REVIEW_QUEUE_PATH",
            _DEFAULT_REVIEW_QUEUE_PATH,
        ),
        gather_output_path=_parse_path(
            "DEFAULT_GATHER_OUTPUT_PATH",
            _DEFAULT_GATHER_OUTPUT_PATH,
        ),
        gather_review_queue_path=_parse_path(
            "DEFAULT_GATHER_REVIEW_QUEUE_PATH",
            _DEFAULT_GATHER_REVIEW_QUEUE_PATH,
        ),
        fixtures_path=_parse_path(
            "DEFAULT_FIXTURES_PATH",
            _DEFAULT_FIXTURES_PATH,
        ),
        log_level=os.getenv("LOG_LEVEL", _DEFAULT_LOG_LEVEL),
    )
