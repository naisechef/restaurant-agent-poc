"""Unit tests for ClaudeClient configuration (no network calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from restaurant_agent.claude_client import ClaudeClient, _model_supports_temperature
from restaurant_agent.config import MissingAPIKeyError, Settings


def test_claude_client_requires_api_key_at_construction() -> None:
    settings = Settings(anthropic_api_key=None)

    with pytest.raises(MissingAPIKeyError):
        ClaudeClient(settings=settings)


def test_claude_client_accepts_configured_settings() -> None:
    settings = Settings(
        anthropic_api_key="test-key",
        model="claude-test-model",
        max_tokens=512,
        temperature=0.2,
        request_timeout=30.0,
    )

    client = ClaudeClient(settings=settings)

    assert client._settings.model == "claude-test-model"
    assert client._settings.max_tokens == 512
    assert client._settings.temperature == pytest.approx(0.2)
    assert client._settings.request_timeout == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-sonnet-5", False),
        ("claude-sonnet-4-20250514", True),
        ("claude-test-model", True),
    ],
)
def test_model_supports_temperature(model: str, expected: bool) -> None:
    assert _model_supports_temperature(model) is expected


def _mock_message(text: str = "ok") -> MagicMock:
    block = MagicMock()
    block.text = text
    message = MagicMock()
    message.content = [block]
    return message


@patch("restaurant_agent.claude_client.anthropic.Anthropic")
def test_complete_omits_temperature_for_sonnet_5(
    mock_anthropic_cls: MagicMock,
) -> None:
    mock_create = MagicMock(return_value=_mock_message())
    mock_anthropic_cls.return_value.messages.create = mock_create

    settings = Settings(
        anthropic_api_key="test-key",
        model="claude-sonnet-5",
        temperature=None,
    )
    client = ClaudeClient(settings=settings)
    client.complete("system", "user")

    assert "temperature" not in mock_create.call_args.kwargs


@patch("restaurant_agent.claude_client.anthropic.Anthropic")
def test_complete_omits_temperature_for_sonnet_5_even_when_set(
    mock_anthropic_cls: MagicMock,
) -> None:
    mock_create = MagicMock(return_value=_mock_message())
    mock_anthropic_cls.return_value.messages.create = mock_create

    settings = Settings(
        anthropic_api_key="test-key",
        model="claude-sonnet-5",
        temperature=0.0,
    )
    client = ClaudeClient(settings=settings)
    client.complete("system", "user")

    assert "temperature" not in mock_create.call_args.kwargs


@patch("restaurant_agent.claude_client.anthropic.Anthropic")
def test_complete_omits_temperature_when_unset_on_older_model(
    mock_anthropic_cls: MagicMock,
) -> None:
    mock_create = MagicMock(return_value=_mock_message())
    mock_anthropic_cls.return_value.messages.create = mock_create

    settings = Settings(
        anthropic_api_key="test-key",
        model="claude-sonnet-4-20250514",
        temperature=None,
    )
    client = ClaudeClient(settings=settings)
    client.complete("system", "user")

    assert "temperature" not in mock_create.call_args.kwargs


@patch("restaurant_agent.claude_client.anthropic.Anthropic")
def test_complete_includes_temperature_for_older_model_when_set(
    mock_anthropic_cls: MagicMock,
) -> None:
    mock_create = MagicMock(return_value=_mock_message())
    mock_anthropic_cls.return_value.messages.create = mock_create

    settings = Settings(
        anthropic_api_key="test-key",
        model="claude-sonnet-4-20250514",
        temperature=0.2,
    )
    client = ClaudeClient(settings=settings)
    client.complete("system", "user")

    assert mock_create.call_args.kwargs["temperature"] == pytest.approx(0.2)
