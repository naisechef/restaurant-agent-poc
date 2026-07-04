"""Shared pytest fixtures for restaurant agent tests."""

from __future__ import annotations

import json

import pytest

from restaurant_agent.claude_client import LLMRequestError
from restaurant_agent.schemas import ExtractionResult, RestaurantRecord


class FakeClaudeClient:
    """Hand-rolled test double for ClaudeClient.complete()."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self._responses:
            raise RuntimeError("FakeClaudeClient has no remaining responses.")
        return self._responses.pop(0)


class ErrorClaudeClient:
    """Test double that always raises LLMRequestError."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMRequestError("API unavailable")


@pytest.fixture
def sample_restaurant_record() -> RestaurantRecord:
    return RestaurantRecord(
        id="r1",
        name="Garden Bistro",
        evidence_text="Enjoy our patio seating with views of the park.",
        expected_label="yes",
    )


@pytest.fixture
def sample_extraction_result() -> ExtractionResult:
    return ExtractionResult(
        label="yes",
        confidence=0.92,
        evidence=["Enjoy our patio seating"],
        reasoning="Evidence explicitly mentions patio seating.",
    )


def valid_extraction_json(
    *,
    label: str = "yes",
    confidence: float = 0.92,
    evidence: list[str] | None = None,
    reasoning: str = "Evidence explicitly mentions patio seating.",
) -> str:
    """Return a JSON string matching the ExtractionResult schema."""
    payload = {
        "label": label,
        "confidence": confidence,
        "evidence": evidence if evidence is not None else ["Enjoy our patio seating"],
        "reasoning": reasoning,
    }
    return json.dumps(payload)
