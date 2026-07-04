"""Integration-style tests for pipeline orchestration (network-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from restaurant_agent.config import Settings
from restaurant_agent.pipeline import RESULT_COLUMNS, run_pipeline
from tests.conftest import ErrorClaudeClient, FakeClaudeClient, valid_extraction_json


@pytest.fixture
def pipeline_settings(tmp_path: Path) -> Settings:
    return Settings(
        confidence_threshold=0.6,
        output_path=tmp_path / "results.csv",
        review_queue_path=tmp_path / "review_queue.csv",
    )


@pytest.fixture
def sample_input_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "restaurants.csv"
    csv_path.write_text(
        "id,name,evidence_text,expected_label\n"
        "r1,Sunny Patio,Cozy patio seating with garden views.,yes\n"
        "r2,Indoor Only,All seating is indoors with no patio.,no\n"
        "r3,Low Confidence,Outdoor deck when weather permits.,yes\n",
        encoding="utf-8",
    )
    return csv_path


def test_run_pipeline_writes_results_and_review_queue(
    sample_input_csv: Path,
    pipeline_settings: Settings,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results.csv"
    review_path = tmp_path / "review_queue.csv"
    client = FakeClaudeClient(
        [
            valid_extraction_json(label="yes", confidence=0.92),
            valid_extraction_json(label="no", confidence=0.88),
            valid_extraction_json(label="yes", confidence=0.45),
        ]
    )

    summary = run_pipeline(
        input_path=sample_input_csv,
        output_path=output_path,
        review_queue_path=review_path,
        settings=pipeline_settings,
        client=client,
    )

    assert output_path.exists()
    assert review_path.exists()
    assert summary.total_records == 3
    assert summary.failure_count == 0
    assert summary.accuracy == pytest.approx(1.0)
    assert summary.low_confidence_count == 1

    results = pd.read_csv(output_path)
    assert list(results.columns) == RESULT_COLUMNS
    assert len(results) == 3

    assert bool(results.loc[results["restaurant_id"] == "r1", "is_correct"].iloc[0])
    assert bool(results.loc[results["restaurant_id"] == "r2", "is_correct"].iloc[0])
    assert bool(results.loc[results["restaurant_id"] == "r3", "is_correct"].iloc[0])

    evidence = json.loads(results.loc[0, "evidence"])
    assert isinstance(evidence, list)

    review = pd.read_csv(review_path)
    assert len(review) == 1
    assert review.iloc[0]["restaurant_id"] == "r3"
    assert bool(review.iloc[0]["needs_review"])


def test_run_pipeline_isolates_record_failures(
    sample_input_csv: Path,
    pipeline_settings: Settings,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results.csv"
    review_path = tmp_path / "review_queue.csv"
    client = ErrorClaudeClient()

    summary = run_pipeline(
        input_path=sample_input_csv,
        output_path=output_path,
        review_queue_path=review_path,
        settings=pipeline_settings,
        client=client,
    )

    results = pd.read_csv(output_path)
    assert len(results) == 3
    assert summary.total_records == 3
    assert summary.failure_count == 3
    assert results["error"].notna().all()
    assert results["validation_status"].eq("failed").all()
    assert results["needs_review"].all()


def test_run_pipeline_respects_limit(
    sample_input_csv: Path,
    pipeline_settings: Settings,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results.csv"
    review_path = tmp_path / "review_queue.csv"
    client = FakeClaudeClient([valid_extraction_json()])

    summary = run_pipeline(
        input_path=sample_input_csv,
        output_path=output_path,
        review_queue_path=review_path,
        settings=pipeline_settings,
        client=client,
        limit=1,
    )

    results = pd.read_csv(output_path)
    assert len(results) == 1
    assert summary.total_records == 1
    assert len(client.calls) == 1
