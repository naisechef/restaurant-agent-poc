"""CLI tests for gather command (network-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from restaurant_agent.cli import main
from restaurant_agent.gather_pipeline import GATHER_RESULT_COLUMNS
from restaurant_agent.schemas import Evidence, EvidenceSourceType, RestaurantQuery

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_PATH = PROJECT_ROOT / "data" / "evidence_fixtures"


class _StubGooglePlacesAdapter:
    """Stand-in for GooglePlacesAdapter used to prove CLI wiring with zero network."""

    def __init__(self, name: str, api_key: str, **_: object) -> None:
        assert api_key == "test-google-key"
        self.name = name

    def gather(self, query: RestaurantQuery) -> list[Evidence]:
        return [
            Evidence(
                source_type=EvidenceSourceType.MAPS,
                source_name=self.name,
                url="https://maps.google.com/?cid=999",
                snippet="Google Places lists outdoor seating as available.",
                reliability="high",
            )
        ]


def test_cli_gather_dry_run_pipeline(tmp_path: Path) -> None:
    output_path = tmp_path / "gather_results.csv"
    review_path = tmp_path / "gather_review_queue.csv"

    exit_code = main(
        [
            "gather",
            "--name",
            "The River Cafe",
            "--city",
            "London",
            "--dry-run",
            "--fixtures-path",
            str(FIXTURES_PATH),
            "--output",
            str(output_path),
            "--review-queue-output",
            str(review_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()

    results = pd.read_csv(output_path)
    assert list(results.columns) == GATHER_RESULT_COLUMNS
    assert results.iloc[0]["name"] == "The River Cafe"
    assert results.iloc[0]["city"] == "London"
    assert results.iloc[0]["prediction"] in ("yes", "no", "unknown")


def test_cli_gather_dry_run_graph(tmp_path: Path) -> None:
    output_path = tmp_path / "gather_results.csv"
    review_path = tmp_path / "gather_review_queue.csv"

    exit_code = main(
        [
            "gather",
            "--name",
            "The River Cafe",
            "--city",
            "London",
            "--backend",
            "graph",
            "--dry-run",
            "--fixtures-path",
            str(FIXTURES_PATH),
            "--output",
            str(output_path),
            "--review-queue-output",
            str(review_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert len(pd.read_csv(output_path)) == 1


def test_cli_gather_live_google_uses_google_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--live --source google should construct GooglePlacesAdapter, with zero network."""
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-google-key")
    monkeypatch.setattr(
        "restaurant_agent.sources.factory.GooglePlacesAdapter",
        _StubGooglePlacesAdapter,
    )

    output_path = tmp_path / "gather_results.csv"
    review_path = tmp_path / "gather_review_queue.csv"

    exit_code = main(
        [
            "gather",
            "--name",
            "The River Cafe",
            "--city",
            "London",
            "--dry-run",
            "--live",
            "--source",
            "google",
            "--fixtures-path",
            str(FIXTURES_PATH),
            "--output",
            str(output_path),
            "--review-queue-output",
            str(review_path),
        ]
    )

    assert exit_code == 0
    results = pd.read_csv(output_path)
    row = results.iloc[0]
    # source_urls carries the stub adapter's URL, proving the live Google
    # path (not the static maps.json fixture) supplied this evidence.
    urls = json.loads(row["source_urls"])
    assert "https://maps.google.com/?cid=999" in urls
    assert "maps" in json.loads(row["source_types"])


def test_cli_gather_source_without_live_is_usage_error(tmp_path: Path) -> None:
    output_path = tmp_path / "gather_results.csv"
    review_path = tmp_path / "gather_review_queue.csv"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "gather",
                "--name",
                "The River Cafe",
                "--city",
                "London",
                "--dry-run",
                "--source",
                "google",
                "--fixtures-path",
                str(FIXTURES_PATH),
                "--output",
                str(output_path),
                "--review-queue-output",
                str(review_path),
            ]
        )

    assert exc_info.value.code == 2
    assert not output_path.exists()


def test_cli_gather_live_google_missing_key_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)

    output_path = tmp_path / "gather_results.csv"
    review_path = tmp_path / "gather_review_queue.csv"

    from restaurant_agent.config import MissingGooglePlacesAPIKeyError

    with pytest.raises(MissingGooglePlacesAPIKeyError):
        main(
            [
                "gather",
                "--name",
                "The River Cafe",
                "--city",
                "London",
                "--dry-run",
                "--live",
                "--source",
                "google",
                "--fixtures-path",
                str(FIXTURES_PATH),
                "--output",
                str(output_path),
                "--review-queue-output",
                str(review_path),
            ]
        )

    assert not output_path.exists()
