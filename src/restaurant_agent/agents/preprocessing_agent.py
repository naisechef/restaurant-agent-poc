"""Preprocessing agent — cleans raw evidence text into state.cleaned_text."""

from __future__ import annotations

from restaurant_agent.preprocessing import clean_evidence
from restaurant_agent.state import AgentState


def run(state: AgentState) -> AgentState:
    """Clean raw evidence and populate cleaned_text."""
    cleaned = clean_evidence(state.raw_text)
    return state.model_copy(update={"cleaned_text": cleaned})
