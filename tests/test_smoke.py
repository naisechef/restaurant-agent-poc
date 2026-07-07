"""End-to-end CLI smoke tests (network-free by default)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from restaurant_agent.cli import main
from restaurant_agent.result_output import RESULT_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "restaurants.csv"
_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        [str(PROJECT_ROOT / "src"), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep),
}


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "restaurant_agent.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=_SUBPROCESS_ENV,
        check=False,
    )

    assert result.returncode == 0
    assert "run" in result.stdout
    assert "gather" in result.stdout


def test_cli_run_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "restaurant_agent.cli", "run", "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=_SUBPROCESS_ENV,
        check=False,
    )

    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    assert "--input" in result.stdout
    assert "--backend" not in result.stdout


def test_cli_main_dry_run_smoke(tmp_path: Path) -> None:
    """Invoke cli.main() directly (no installed console script required)."""
    output_path = tmp_path / "results.csv"
    review_path = tmp_path / "review_queue.csv"

    exit_code = main(
        [
            "--dry-run",
            "--limit",
            "3",
            "--input",
            str(SAMPLE_CSV),
            "--output",
            str(output_path),
            "--review-queue-output",
            str(review_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert review_path.exists()

    results = pd.read_csv(output_path)
    assert list(results.columns) == RESULT_COLUMNS
    assert len(results) == 3
    assert results["prediction"].notna().all()
    assert results["validation_status"].notna().all()


def test_cli_dry_run_subprocess_smoke(tmp_path: Path) -> None:
    """Exercise the module entry point via subprocess (closer to manual usage)."""
    output_path = tmp_path / "results.csv"
    review_path = tmp_path / "review_queue.csv"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "restaurant_agent.cli",
            "--dry-run",
            "--limit",
            "2",
            "--input",
            str(SAMPLE_CSV),
            "--output",
            str(output_path),
            "--review-queue-output",
            str(review_path),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=_SUBPROCESS_ENV,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Dry-run mode" in result.stdout
    assert output_path.exists()
    assert len(pd.read_csv(output_path)) == 2
