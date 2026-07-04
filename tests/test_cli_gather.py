"""CLI tests for gather command (network-free)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from restaurant_agent.cli import main
from restaurant_agent.gather_pipeline import GATHER_RESULT_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_PATH = PROJECT_ROOT / "data" / "evidence_fixtures"


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
