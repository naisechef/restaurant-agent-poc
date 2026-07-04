"""Command-line entry point for the restaurant agent pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from restaurant_agent.claude_client import ClaudeClient, DryRunClaudeClient
from restaurant_agent.config import load_settings
from restaurant_agent.logging_config import configure_logging
from restaurant_agent.pipeline import run_pipeline
from restaurant_agent.schemas import EvaluationSummary


def _build_parser() -> argparse.ArgumentParser:
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description=(
            "Run the outdoor seating extraction pipeline over a restaurants CSV."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.input_path,
        help=f"Input CSV path (default: {settings.input_path})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.output_path,
        help=f"Results CSV path (default: {settings.output_path})",
    )
    parser.add_argument(
        "--review-queue-output",
        type=Path,
        default=settings.review_queue_path,
        help=f"Review queue CSV path (default: {settings.review_queue_path})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Anthropic model name override",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many records (for quick smoke runs)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "Confidence threshold for human review routing "
            f"(default: {settings.confidence_threshold})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use deterministic local fake LLM responses instead of Claude "
            "(no ANTHROPIC_API_KEY required)"
        ),
    )
    return parser


def _print_summary(
    summary: EvaluationSummary,
    *,
    review_queue_path: Path,
    review_queue_count: int,
) -> None:
    print("\nEvaluation summary")
    print("------------------")
    print(f"Total records:        {summary.total_records}")
    print(f"Failures:             {summary.failure_count}")
    print(f"Accuracy:             {summary.accuracy:.2%}")
    print(f"Precision (yes):      {summary.precision_yes:.2%}")
    print(f"Recall (yes):         {summary.recall_yes:.2%}")
    print(f"False positives:      {summary.false_positives}")
    print(f"False negatives:      {summary.false_negatives}")
    print(f"Low-confidence cases: {summary.low_confidence_count}")
    print(
        f"Review queue:         {review_queue_count} record(s) -> {review_queue_path}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.model is not None:
        settings = settings.model_copy(update={"model": args.model})
    if args.confidence_threshold is not None:
        settings = settings.model_copy(
            update={"confidence_threshold": args.confidence_threshold}
        )

    configure_logging(settings.log_level)

    if args.dry_run:
        print("Dry-run mode: using deterministic local responses (no API calls).")
        client = DryRunClaudeClient()
    else:
        client = ClaudeClient(settings)

    summary = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        review_queue_path=args.review_queue_output,
        settings=settings,
        client=client,
        limit=args.limit,
    )

    review_queue_count = 0
    if args.review_queue_output.exists():
        review_queue_count = len(pd.read_csv(args.review_queue_output))

    _print_summary(
        summary,
        review_queue_path=args.review_queue_output,
        review_queue_count=review_queue_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
