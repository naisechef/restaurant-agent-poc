"""Unit tests for preprocessing agent."""

from __future__ import annotations

from restaurant_agent.agents import preprocessing_agent
from restaurant_agent.state import AgentState


def test_preprocessing_agent_populates_cleaned_text() -> None:
    state = AgentState(
        restaurant_id="r1",
        raw_text="<p>Enjoy our   patio seating.</p>\n\nMENU",
    )

    result = preprocessing_agent.run(state)

    assert result.cleaned_text == "Enjoy our patio seating."
    assert result.restaurant_id == "r1"
    assert result.prediction is None
