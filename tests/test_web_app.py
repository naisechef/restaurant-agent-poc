"""FastAPI web demo tests (network-free via mocked Claude client)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from restaurant_agent.claude_client import DryRunClaudeClient
from restaurant_agent.schemas import GatherRunResult
from restaurant_agent.sources.factory import build_adapters as real_build_adapters
from restaurant_agent.sources.google_places import GooglePlacesAdapter
from restaurant_agent.web.app import create_app
from tests.test_google_places_adapter import FakeTransport, _FULL_DETAILS

FIXTURES_PATH = Path(__file__).resolve().parents[1] / "data" / "evidence_fixtures"
QUERY_NAME = "The River Cafe"
QUERY_CITY = "London"


class _YesDryRunClient:
    """Deterministic client that always extracts yes for verification tests."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "label": "yes",
                "confidence": 0.95,
                "evidence": ["Google Places lists outdoor seating as available."],
                "reasoning": "Test fixture: outdoor seating confirmed from evidence.",
            }
        )


def _google_places_build_adapters(
    fixtures_path: Path,
    *,
    dry_run: bool = False,
    live_source: str | None = None,
    settings: Any = None,
) -> dict[str, Any]:
    adapters = real_build_adapters(
        fixtures_path,
        dry_run=dry_run,
        live_source=None,
        settings=settings,
    )
    if live_source == "google":
        transport = FakeTransport(
            search_response={"places": [{"id": "ChIJ123"}]},
            details_response=_FULL_DETAILS,
        )
        adapters["maps"] = GooglePlacesAdapter(
            "google_places",
            "test-key",
            transport=transport,
        )
    return adapters


@pytest.fixture(autouse=True)
def _mock_claude_client(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("no_claude_mock"):
        return
    monkeypatch.setattr(
        "restaurant_agent.web.service.ClaudeClient",
        lambda settings: DryRunClaudeClient(),
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DEFAULT_FIXTURES_PATH", str(FIXTURES_PATH))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    return TestClient(create_app())


def test_get_landing_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Restaurant Outdoor Seating Demo" in response.text
    assert "gather evidence, run extraction" in response.text
    assert 'name="name"' in response.text
    assert 'name="city"' in response.text
    assert 'name="backend"' not in response.text
    assert 'name="dry_run"' not in response.text
    assert ">Search</button>" in response.text
    assert 'data-loading-text="Searching..."' in response.text
    assert "Precision, recall, and F1 are dataset-level" not in response.text
    assert "Location will appear after search." in response.text
    assert 'class="location-panel' in response.text
    assert "Results for" not in response.text


def test_get_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_demo_redirects_to_landing(client: TestClient) -> None:
    response = client.get("/demo", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/"

    response = client.get("/demo", follow_redirects=True)
    assert response.status_code == 200
    assert "Restaurant Outdoor Seating Demo" in response.text


def test_post_demo_search(client: TestClient) -> None:
    response = client.post(
        "/",
        data={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 200
    assert "Results for" in response.text
    assert "Execution details" in response.text
    assert "Graph execution trace" in response.text
    assert "merge_evidence" in response.text
    assert 'class="evidence-table"' in response.text
    assert "cell-url" in response.text
    assert "No live location data available." in response.text
    assert "Precision (yes)" not in response.text
    assert "Recall (yes)" not in response.text
    assert "<dt>Backend</dt>" not in response.text
    assert "<dt>Dry-run</dt>" not in response.text


def test_post_demo_missing_name(client: TestClient) -> None:
    response = client.post(
        "/",
        data={"name": "", "city": QUERY_CITY},
    )
    assert response.status_code == 422


def test_api_gather(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "live_google": False,
        },
    )
    assert response.status_code == 200
    result = GatherRunResult.model_validate(response.json())
    assert result.name == QUERY_NAME
    assert result.city == QUERY_CITY
    assert result.backend == "graph"
    assert result.dry_run is False
    assert result.graph_trace is not None
    assert result.validation_status == "ok"
    assert result.route == "success"
    assert result.needs_review is False
    assert result.error is None
    node_names = [step.node for step in result.graph_trace]
    assert "gather_search" in node_names
    assert "merge_evidence" in node_names
    assert "success" in node_names
    assert all(step.summary for step in result.graph_trace)


def test_graph_trace_includes_duration_ms(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 200
    result = GatherRunResult.model_validate(response.json())
    assert result.graph_trace is not None
    assert result.total_duration_ms is not None
    assert result.total_duration_ms >= 0
    assert all(step.duration_ms is not None for step in result.graph_trace)
    assert all(step.duration_ms >= 0 for step in result.graph_trace)


def test_graph_trace_summaries_include_structured_validation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")
    monkeypatch.setattr(
        "restaurant_agent.web.service.build_adapters",
        _google_places_build_adapters,
    )
    monkeypatch.setattr(
        "restaurant_agent.web.service.ClaudeClient",
        lambda settings: _YesDryRunClient(),
    )

    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "live_google": True,
        },
    )
    assert response.status_code == 200
    result = GatherRunResult.model_validate(response.json())
    assert result.validation_summary is not None
    assert result.validation_summary.structured_status == "verified"
    assert result.validation_summary.route == "success"
    assert result.validation_summary.route_escalated is False

    terminal = result.graph_trace[-1]
    assert terminal.node == "success"
    assert "Structured validation: verified" in terminal.summary
    assert "Route: success" in terminal.summary


def test_demo_html_renders_duration_and_decision_path(client: TestClient) -> None:
    response = client.post(
        "/",
        data={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 200
    text = response.text
    assert "Graph execution trace" in text
    assert "graph-trace-table" in text
    assert " ms" in text
    assert "Decision path" in text
    assert "decision-path" in text
    assert "Claude predicted" in text


def test_api_includes_sanitized_timing_and_trace_fields(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("total_duration_ms") is not None
    assert payload.get("validation_summary") is not None
    assert payload.get("decision_path")

    trace = payload["graph_trace"]
    for step in trace:
        assert "duration_ms" in step
        assert "status" in step
        assert "update_type" in step
        update = step.get("update") or {}
        assert "raw_text" not in update
        assert "cleaned_text" not in update


def test_timing_fields_never_include_sensitive_data(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 200
    text = response.text.lower()
    for forbidden in (
        "raw_text",
        "cleaned_text",
        "api_key",
        "traceback",
        "anthropic_api_key",
        "google_places_api_key",
        "evidence_fixtures",
        "system_prompt",
        "user_prompt",
    ):
        assert forbidden not in text


@pytest.mark.no_claude_mock
def test_api_gather_missing_api_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert "message" in body
    assert "ANTHROPIC_API_KEY" not in body["message"]
    assert "api_key" not in body["message"].lower()


def test_api_gather_missing_google_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "live_google": True,
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert "message" in body
    assert "GOOGLE_PLACES_API_KEY" not in body["message"]


def test_api_gather_name_too_long(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": "x" * 201,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 422


def test_no_static_data_exposure(client: TestClient) -> None:
    for path in ("/data/evidence_fixtures/search.json", "/outputs/gather_results.csv", "/docs/WEB_DEMO.md"):
        response = client.get(path)
        assert response.status_code == 404


def test_openapi_disabled(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404


def test_graph_trace_omits_raw_text(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    trace = payload.get("graph_trace") or []
    for step in trace:
        update = step.get("update") or {}
        assert "raw_text" not in update
        assert "cleaned_text" not in update


@pytest.mark.no_claude_mock
def test_error_responses_never_include_secrets(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 400
    text = response.text.lower()
    assert "anthropic_api_key" not in text
    assert "traceback" not in text


def test_demo_html_never_includes_sensitive_fields(client: TestClient) -> None:
    response = client.post(
        "/",
        data={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 200
    text = response.text.lower()
    for forbidden in (
        "raw_text",
        "cleaned_text",
        "anthropic_api_key",
        "google_places_api_key",
        "traceback",
        "evidence_fixtures",
        "restaurants.csv",
    ):
        assert forbidden not in text


def test_api_gather_json_never_includes_sensitive_fields(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
        },
    )
    assert response.status_code == 200
    text = response.text.lower()
    for forbidden in ("raw_text", "cleaned_text", "api_key", "traceback"):
        assert forbidden not in text


def test_execution_details_in_api_response(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "live_google": False,
        },
    )
    assert response.status_code == 200
    result = GatherRunResult.model_validate(response.json())
    assert result.dry_run is False
    assert result.live_google is False
    assert result.reliability_mix["medium"] >= 1
    assert len(result.source_results) == 4
    assert result.graph_trace is not None
    assert result.graph_trace[-1].node == "success"


def test_demo_location_panel_with_mocked_live_google(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")
    monkeypatch.setattr(
        "restaurant_agent.web.service.build_adapters",
        _google_places_build_adapters,
    )
    monkeypatch.setattr(
        "restaurant_agent.web.service.ClaudeClient",
        lambda settings: _YesDryRunClient(),
    )

    response = client.post(
        "/",
        data={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "live_google": "true",
        },
    )

    assert response.status_code == 200
    text = response.text
    assert "The River Cafe" in text
    assert "Thames Wharf, Rainville Rd, London" in text
    assert "Open in Google Maps" in text
    assert "https://maps.google.com/?cid=123" in text
    assert 'class="map-preview"' in text
    assert "openstreetmap.org/export/embed.html" in text
    assert "No live location data available." not in text
    assert "Decision Verification" in text
    assert "validation-verified" in text
    assert "Google Places outdoor seating" in text
    assert "Verification status" in text
    assert "displayName" not in text
    assert "outdoorSeating" not in text
    assert "test-key" not in text


def test_api_gather_with_mocked_live_google_includes_location(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")
    monkeypatch.setattr(
        "restaurant_agent.web.service.build_adapters",
        _google_places_build_adapters,
    )
    monkeypatch.setattr(
        "restaurant_agent.web.service.ClaudeClient",
        lambda settings: _YesDryRunClient(),
    )

    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "live_google": True,
        },
    )

    assert response.status_code == 200
    result = GatherRunResult.model_validate(response.json())
    assert result.google_places is not None
    assert result.google_places.place_name == "The River Cafe"
    assert result.google_places.formatted_address == "Thames Wharf, Rainville Rd, London"
    assert result.google_places.latitude == pytest.approx(51.4839)
    assert result.google_places.longitude == pytest.approx(-0.2234)
    assert result.google_places.rating == pytest.approx(4.6)
    assert result.google_places.user_rating_count == 812
    assert len(result.structured_validations) == 1
    assert result.structured_validations[0].source == "google_places"
    assert result.structured_validations[0].status == "verified"
    assert result.structured_validations[0].source_value is True
    assert result.structured_validations[0].prediction == "yes"
    assert result.route == "success"

    text = response.text.lower()
    assert "displayname" not in text
    assert "outdoorseating" not in text
    assert "test-key" not in text
    assert "x-goog-api-key" not in text
    assert result.google_places.place_id == "ChIJ123"
