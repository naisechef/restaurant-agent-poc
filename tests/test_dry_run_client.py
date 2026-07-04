"""Unit tests for DryRunClaudeClient (network-free)."""

from __future__ import annotations

import json

from restaurant_agent.claude_client import DryRunClaudeClient
from restaurant_agent.prompts.outdoor_seating import build_prompt


def test_dry_run_client_returns_valid_json_for_outdoor_evidence() -> None:
    client = DryRunClaudeClient()
    _, user_prompt = build_prompt(
        "Enjoy breakfast on our sunny patio overlooking the marina."
    )

    raw = client.complete("system", user_prompt)
    payload = json.loads(raw)

    assert payload["label"] == "yes"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["evidence"]


def test_dry_run_client_returns_no_for_indoor_only_evidence() -> None:
    client = DryRunClaudeClient()
    _, user_prompt = build_prompt(
        "All seating is inside the dining room; we do not offer patio tables."
    )

    payload = json.loads(client.complete("system", user_prompt))

    assert payload["label"] == "no"


def test_dry_run_client_returns_unknown_for_contradictory_evidence() -> None:
    client = DryRunClaudeClient()
    _, user_prompt = build_prompt(
        "We do not have outdoor dining. Our sidewalk patio opens on warm evenings."
    )

    payload = json.loads(client.complete("system", user_prompt))

    assert payload["label"] == "unknown"
