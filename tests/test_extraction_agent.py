"""Unit tests for extraction agent (network-free via FakeClaudeClient)."""

from __future__ import annotations

import pytest

from restaurant_agent.agents import extraction_agent
from restaurant_agent.state import AgentState
from tests.conftest import ErrorClaudeClient, FakeClaudeClient, valid_extraction_json


@pytest.fixture
def preprocessed_state() -> AgentState:
    return AgentState(
        restaurant_id="r1",
        raw_text="Enjoy our patio seating with views of the park.",
        cleaned_text="Enjoy our patio seating with views of the park.",
    )


def test_extraction_agent_valid_json(preprocessed_state: AgentState) -> None:
    client = FakeClaudeClient([valid_extraction_json()])

    result = extraction_agent.run(preprocessed_state, client)

    assert result.error is None
    assert result.prediction == "yes"
    assert result.confidence == pytest.approx(0.92)
    assert result.evidence == ["Enjoy our patio seating"]
    assert "patio seating" in (result.reasoning or "")
    assert len(client.calls) == 1


def test_extraction_agent_strips_fenced_json(preprocessed_state: AgentState) -> None:
    fenced = f"```json\n{valid_extraction_json()}\n```"
    client = FakeClaudeClient([fenced])

    result = extraction_agent.run(preprocessed_state, client)

    assert result.error is None
    assert result.prediction == "yes"
    assert len(client.calls) == 1


def test_extraction_agent_repairs_malformed_json(preprocessed_state: AgentState) -> None:
    client = FakeClaudeClient(
        [
            "not valid json at all",
            valid_extraction_json(label="no", confidence=0.88, evidence=["indoor only"]),
        ]
    )

    result = extraction_agent.run(preprocessed_state, client)

    assert result.error is None
    assert result.prediction == "no"
    assert result.confidence == pytest.approx(0.88)
    assert len(client.calls) == 2
    assert "Previous response:" in client.calls[1][1]
    assert "Original evidence:" in client.calls[1][1]


def test_extraction_agent_handles_evidence_with_braces() -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="Menu item {special} includes patio seating.",
        cleaned_text="Menu item {special} includes patio seating.",
    )
    client = FakeClaudeClient([valid_extraction_json()])

    result = extraction_agent.run(state, client)

    assert result.error is None
    assert result.prediction == "yes"
    assert "{special}" in client.calls[0][1]


def test_extraction_agent_repair_handles_braces_in_malformed_response() -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="Enjoy our patio seating.",
        cleaned_text="Enjoy our patio seating.",
    )
    client = FakeClaudeClient(
        [
            'Here is JSON: {"label": "yes", "bad": {unclosed}',
            valid_extraction_json(label="yes"),
        ]
    )

    result = extraction_agent.run(state, client)

    assert result.error is None
    assert result.prediction == "yes"
    assert len(client.calls) == 2
    assert "{unclosed}" in client.calls[1][1]


def test_extraction_agent_sets_error_after_failed_repair(
    preprocessed_state: AgentState,
) -> None:
    client = FakeClaudeClient(["still not json", "also not json"])

    result = extraction_agent.run(preprocessed_state, client)

    assert result.prediction is None
    assert result.confidence is None
    assert result.error is not None
    assert "repair retry" in result.error
    assert len(client.calls) == 2


def test_extraction_agent_sets_error_on_llm_failure(
    preprocessed_state: AgentState,
) -> None:
    result = extraction_agent.run(preprocessed_state, ErrorClaudeClient())

    assert result.prediction is None
    assert result.error == "API unavailable"


def test_extraction_agent_requires_cleaned_text() -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="Some evidence.",
        cleaned_text=None,
    )
    client = FakeClaudeClient([valid_extraction_json()])

    result = extraction_agent.run(state, client)

    assert result.error == "Missing cleaned_text: run preprocessing first."
    assert client.calls == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"label": "yes"}', '{"label": "yes"}'),
        ("```json\n{\"label\": \"yes\"}\n```", '{"label": "yes"}'),
        ("```\n{\"label\": \"no\"}\n```", '{"label": "no"}'),
        (
            'Here is the result:\n```json\n{"label": "yes"}\n```',
            '{"label": "yes"}',
        ),
    ],
)
def test_strip_markdown_fences(raw: str, expected: str) -> None:
    assert extraction_agent.strip_markdown_fences(raw) == expected
