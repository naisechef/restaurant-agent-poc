"""FastAPI web demo tests (network-free, dry-run only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from restaurant_agent.schemas import GatherRunResult
from restaurant_agent.sources.factory import build_adapters as real_build_adapters
from restaurant_agent.sources.google_places import GooglePlacesAdapter
from restaurant_agent.web.app import create_app
from tests.test_google_places_adapter import FakeTransport, _FULL_DETAILS

FIXTURES_PATH = Path(__file__).resolve().parents[1] / "data" / "evidence_fixtures"
QUERY_NAME = "The River Cafe"
QUERY_CITY = "London"


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


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DEFAULT_FIXTURES_PATH", str(FIXTURES_PATH))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    return TestClient(create_app())


def test_get_landing_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Outdoor Seating Demo" in response.text


def test_get_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_demo_form(client: TestClient) -> None:
    response = client.get("/demo")
    assert response.status_code == 200
    assert 'name="name"' in response.text
    assert 'name="city"' in response.text
    assert ">Search</button>" in response.text
    assert 'data-loading-text="Searching..."' in response.text
    assert "Precision, recall, and F1 are dataset-level" in response.text
    assert "Location will appear after search." in response.text
    assert 'class="location-panel' in response.text


def test_post_demo_pipeline_dry_run(client: TestClient) -> None:
    response = client.post(
        "/demo",
        data={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "backend": "pipeline",
            "dry_run": "true",
        },
    )
    assert response.status_code == 200
    assert "Results for" in response.text
    assert "Execution details" in response.text
    assert 'class="evidence-table"' in response.text
    assert "cell-url" in response.text
    assert "No live location data available." in response.text


def test_post_demo_graph_dry_run(client: TestClient) -> None:
    response = client.post(
        "/demo",
        data={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "backend": "graph",
            "dry_run": "true",
        },
    )
    assert response.status_code == 200
    assert "Results for" in response.text
    assert "Execution details" in response.text
    assert "Graph nodes executed" in response.text
    assert "merge_evidence" in response.text
    assert "validation-ok" in response.text
    assert "route-success" in response.text
    assert "Precision (yes)" not in response.text
    assert "Recall (yes)" not in response.text


def test_post_demo_missing_name(client: TestClient) -> None:
    response = client.post(
        "/demo",
        data={"name": "", "city": QUERY_CITY, "backend": "pipeline", "dry_run": "true"},
    )
    assert response.status_code == 422


def test_post_demo_invalid_backend(client: TestClient) -> None:
    response = client.post(
        "/demo",
        data={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "backend": "invalid",
            "dry_run": "true",
        },
    )
    assert response.status_code == 400
    assert "Invalid request" in response.text


def test_api_gather_pipeline(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "backend": "pipeline",
            "dry_run": True,
            "live_google": False,
        },
    )
    assert response.status_code == 200
    result = GatherRunResult.model_validate(response.json())
    assert result.name == QUERY_NAME
    assert result.city == QUERY_CITY
    assert result.backend == "pipeline"
    assert result.graph_trace is None


def test_api_gather_graph(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "backend": "graph",
            "dry_run": True,
        },
    )
    assert response.status_code == 200
    result = GatherRunResult.model_validate(response.json())
    assert result.backend == "graph"
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


def test_api_gather_missing_api_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "backend": "pipeline",
            "dry_run": False,
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
            "backend": "pipeline",
            "dry_run": True,
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
            "backend": "pipeline",
            "dry_run": True,
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
            "backend": "graph",
            "dry_run": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    trace = payload.get("graph_trace") or []
    for step in trace:
        update = step.get("update") or {}
        assert "raw_text" not in update
        assert "cleaned_text" not in update


def test_error_responses_never_include_secrets(client: TestClient) -> None:
    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "backend": "pipeline",
            "dry_run": False,
        },
    )
    assert response.status_code == 400
    text = response.text.lower()
    assert "anthropic_api_key" not in text
    assert "traceback" not in text


def test_demo_html_never_includes_sensitive_fields(client: TestClient) -> None:
    response = client.post(
        "/demo",
        data={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "backend": "graph",
            "dry_run": "true",
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
            "backend": "graph",
            "dry_run": True,
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
            "backend": "graph",
            "dry_run": True,
            "live_google": False,
        },
    )
    assert response.status_code == 200
    result = GatherRunResult.model_validate(response.json())
    assert result.dry_run is True
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

    response = client.post(
        "/demo",
        data={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "backend": "pipeline",
            "dry_run": "true",
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
    assert "displayName" not in text
    assert "ChIJ123" not in text
    assert "test-key" not in text


def test_api_gather_with_mocked_live_google_includes_location(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")
    monkeypatch.setattr(
        "restaurant_agent.web.service.build_adapters",
        _google_places_build_adapters,
    )

    response = client.post(
        "/api/gather",
        json={
            "name": QUERY_NAME,
            "city": QUERY_CITY,
            "backend": "pipeline",
            "dry_run": True,
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

    text = response.text.lower()
    assert "displayname" not in text
    assert "chij123" not in text
    assert "test-key" not in text
    assert "x-goog-api-key" not in text
