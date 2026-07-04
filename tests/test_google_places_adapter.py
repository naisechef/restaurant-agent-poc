"""Tests for the Google Places evidence adapter (fully network-free).

Two layers of fakery are used, both offline:
- Most tests exercise GooglePlacesAdapter against an in-file FakeTransport.
- A couple of tests exercise HttpxPlacesTransport wired to httpx.MockTransport
  to prove the real transport never needs a live socket.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from restaurant_agent.config import MissingGooglePlacesAPIKeyError, Settings
from restaurant_agent.schemas import EvidenceSourceType, PlaceLocation, RestaurantQuery
from restaurant_agent.sources.google_places import (
    GooglePlacesAdapter,
    GooglePlacesError,
    HttpxPlacesTransport,
)

QUERY = RestaurantQuery(name="The River Cafe", city="London")


class FakeTransport:
    """Records calls and returns canned responses; never touches the network."""

    def __init__(
        self,
        search_response: dict[str, object] | None = None,
        details_response: dict[str, object] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self.search_response = search_response if search_response is not None else {"places": []}
        self.details_response = details_response or {}
        self._raises = raises
        self.search_calls: list[tuple[str, float]] = []
        self.details_calls: list[tuple[str, float]] = []

    def search_text(self, text_query: str, *, timeout: float) -> dict[str, object]:
        self.search_calls.append((text_query, timeout))
        if self._raises is not None:
            raise self._raises
        return self.search_response

    def get_place_details(self, place_id: str, *, timeout: float) -> dict[str, object]:
        self.details_calls.append((place_id, timeout))
        return self.details_response


_FULL_DETAILS = {
    "id": "ChIJ123",
    "displayName": {"text": "The River Cafe", "languageCode": "en"},
    "formattedAddress": "Thames Wharf, Rainville Rd, London",
    "location": {"latitude": 51.4839, "longitude": -0.2234},
    "googleMapsUri": "https://maps.google.com/?cid=123",
    "websiteUri": "https://rivercafe.co.uk",
    "outdoorSeating": True,
    "rating": 4.6,
    "userRatingCount": 812,
    "reviewSummary": {"text": {"text": "Diners love the riverside terrace."}},
    "editorialSummary": {"text": "An Italian restaurant with a famous terrace."},
    "reviews": [
        {"text": {"text": "Wonderful terrace seating by the river."}, "rating": 5},
        {"text": {"text": "Great food, lovely outdoor area."}, "rating": 4},
        {"originalText": {"text": "Cozy indoor space too."}, "rating": 4},
        {"text": {"text": "A fifth review that should be truncated by max_reviews."}},
    ],
}


def _make_adapter(transport: FakeTransport, **kwargs: object) -> GooglePlacesAdapter:
    return GooglePlacesAdapter("google_places", "test-key", transport=transport, **kwargs)


def test_successful_place_lookup_returns_evidence() -> None:
    transport = FakeTransport(
        search_response={"places": [{"id": "ChIJ123"}]},
        details_response=_FULL_DETAILS,
    )
    adapter = _make_adapter(transport)

    evidence = adapter.gather(QUERY)

    assert len(transport.search_calls) == 1
    assert len(transport.details_calls) == 1
    assert transport.details_calls[0][0] == "ChIJ123"

    maps_items = [item for item in evidence if item.source_type == EvidenceSourceType.MAPS]
    review_items = [item for item in evidence if item.source_type == EvidenceSourceType.REVIEW]
    assert len(maps_items) == 3  # outdoor seating + review summary + editorial summary
    assert len(review_items) == 3  # max_reviews default is 3

    outdoor_item = next(
        item for item in maps_items if "available" in item.snippet.lower()
    )
    assert outdoor_item.reliability == "high"
    assert outdoor_item.url == "https://maps.google.com/?cid=123"
    assert outdoor_item.source_name == "google_places"
    assert adapter._place_location == PlaceLocation(
        place_name="The River Cafe",
        formatted_address="Thames Wharf, Rainville Rd, London",
        google_maps_url="https://maps.google.com/?cid=123",
        latitude=51.4839,
        longitude=-0.2234,
    )


def test_no_candidates_found_returns_empty() -> None:
    transport = FakeTransport(search_response={"places": []})
    adapter = _make_adapter(transport)

    evidence = adapter.gather(QUERY)

    assert evidence == []
    assert len(transport.search_calls) == 1
    assert len(transport.details_calls) == 0


def test_outdoor_seating_true_produces_high_reliability_evidence() -> None:
    transport = FakeTransport(
        search_response={"places": [{"id": "p1"}]},
        details_response={"outdoorSeating": True, "googleMapsUri": "https://maps/x"},
    )
    adapter = _make_adapter(transport)

    evidence = adapter.gather(QUERY)

    assert len(evidence) == 1
    assert evidence[0].source_type == EvidenceSourceType.MAPS
    assert evidence[0].reliability == "high"
    assert "available" in evidence[0].snippet.lower()
    assert "not" not in evidence[0].snippet.lower()


def test_outdoor_seating_false_produces_high_reliability_evidence() -> None:
    transport = FakeTransport(
        search_response={"places": [{"id": "p1"}]},
        details_response={"outdoorSeating": False, "googleMapsUri": "https://maps/x"},
    )
    adapter = _make_adapter(transport)

    evidence = adapter.gather(QUERY)

    assert len(evidence) == 1
    assert evidence[0].reliability == "high"
    assert "does not list" in evidence[0].snippet.lower()


def test_outdoor_seating_absent_produces_no_evidence() -> None:
    transport = FakeTransport(
        search_response={"places": [{"id": "p1"}]},
        details_response={"googleMapsUri": "https://maps/x"},
    )
    adapter = _make_adapter(transport)

    evidence = adapter.gather(QUERY)

    assert evidence == []


def test_reviews_mapped_into_low_reliability_evidence_with_limit() -> None:
    transport = FakeTransport(
        search_response={"places": [{"id": "p1"}]},
        details_response=_FULL_DETAILS,
    )
    adapter = _make_adapter(transport, max_reviews=2, max_review_chars=10)

    evidence = adapter.gather(QUERY)
    review_items = [item for item in evidence if item.source_type == EvidenceSourceType.REVIEW]

    assert len(review_items) == 2
    for item in review_items:
        assert item.reliability == "low"
        assert len(item.snippet) <= 10


def test_review_summary_and_editorial_summary_produce_medium_reliability_evidence() -> None:
    transport = FakeTransport(
        search_response={"places": [{"id": "p1"}]},
        details_response={
            "googleMapsUri": "https://maps/x",
            "websiteUri": "https://example.com",
            "reviewSummary": {"text": {"text": "Great riverside views."}},
            "editorialSummary": {"text": "Famous Italian restaurant."},
        },
    )
    adapter = _make_adapter(transport)

    evidence = adapter.gather(QUERY)

    assert len(evidence) == 2
    assert all(item.reliability == "medium" for item in evidence)
    assert all(item.source_type == EvidenceSourceType.MAPS for item in evidence)
    editorial_item = next(item for item in evidence if "Italian" in item.snippet)
    assert editorial_item.url == "https://example.com"


def test_place_location_captured_even_without_evidence() -> None:
    transport = FakeTransport(
        search_response={"places": [{"id": "p1"}]},
        details_response={
            "displayName": {"text": "Quiet Place"},
            "formattedAddress": "1 Side St, London",
            "googleMapsUri": "https://maps.google.com/?cid=999",
            "location": {"latitude": 51.48, "longitude": -0.22},
        },
    )
    adapter = _make_adapter(transport)

    evidence = adapter.gather(QUERY)

    assert evidence == []
    assert adapter._place_location == PlaceLocation(
        place_name="Quiet Place",
        formatted_address="1 Side St, London",
        google_maps_url="https://maps.google.com/?cid=999",
        latitude=51.48,
        longitude=-0.22,
    )


def test_api_error_raises_google_places_error() -> None:
    transport = FakeTransport(raises=GooglePlacesError("boom"))
    adapter = _make_adapter(transport)

    with pytest.raises(GooglePlacesError, match="boom"):
        adapter.gather(QUERY)


def test_missing_api_key_only_errors_at_construction() -> None:
    with pytest.raises(MissingGooglePlacesAPIKeyError):
        Settings(google_places_api_key=None).require_google_places_api_key()

    # Settings with no Google key never raises unless explicitly required.
    Settings(google_places_api_key=None)


def test_observability_log_emitted_on_success(caplog: pytest.LogCaptureFixture) -> None:
    transport = FakeTransport(
        search_response={"places": [{"id": "ChIJ123"}]},
        details_response=_FULL_DETAILS,
    )
    adapter = _make_adapter(transport)

    with caplog.at_level(logging.INFO, logger="restaurant_agent.sources.google_places"):
        adapter.gather(QUERY)

    combined = "\n".join(record.message for record in caplog.records)
    assert "Google Places" in combined
    assert "Search latency" in combined
    assert "Evidence extracted" in combined


def test_httpx_transport_uses_mock_transport_no_network() -> None:
    """HttpxPlacesTransport must work entirely against httpx.MockTransport."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/places:searchText":
            assert request.headers["X-Goog-Api-Key"] == "test-key"
            assert request.headers["X-Goog-FieldMask"] != "*"
            return httpx.Response(200, json={"places": [{"id": "ChIJ123"}]})
        if request.url.path == "/v1/places/ChIJ123":
            assert request.headers["X-Goog-FieldMask"] != "*"
            return httpx.Response(200, json=_FULL_DETAILS)
        raise AssertionError(f"Unexpected request: {request.url}")

    mock_transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=mock_transport)
    transport = HttpxPlacesTransport("test-key", client=client)
    adapter = GooglePlacesAdapter("google_places", "test-key", transport=transport)

    evidence = adapter.gather(QUERY)

    assert len(evidence) > 0
    assert all(item.source_name == "google_places" for item in evidence)


def test_httpx_transport_raises_google_places_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpxPlacesTransport("test-key", client=client)

    with pytest.raises(GooglePlacesError):
        transport.search_text("The River Cafe, London", timeout=5.0)


def test_httpx_transport_raises_google_places_error_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpxPlacesTransport("test-key", client=client)

    with pytest.raises(GooglePlacesError):
        transport.search_text("The River Cafe, London", timeout=5.0)
