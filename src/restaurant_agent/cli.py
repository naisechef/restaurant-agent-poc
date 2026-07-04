"""Command-line entry point for the restaurant agent pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from restaurant_agent.claude_client import ClaudeClient, DryRunClaudeClient
from restaurant_agent.config import Settings, load_settings
from restaurant_agent.gather_graph import run_gather_graph_pipeline
from restaurant_agent.gather_pipeline import run_gather_pipeline
from restaurant_agent.graph_pipeline import run_graph_pipeline
from restaurant_agent.logging_config import configure_logging
from restaurant_agent.pipeline import run_pipeline
from restaurant_agent.schemas import EvaluationSummary, RestaurantQuery
from restaurant_agent.sources.factory import build_adapters

_KNOWN_COMMANDS = frozenset({"run", "gather"})


def _normalise_argv(argv: list[str]) -> list[str]:
    """Preserve backward compatibility: bare flags imply the run subcommand."""
    if not argv:
        return ["run"]
    if argv[0] in _KNOWN_COMMANDS or argv[0] in ("-h", "--help"):
        return argv
    return ["run", *argv]


def _add_shared_run_flags(parser: argparse.ArgumentParser, settings: Settings) -> None:
    parser.add_argument(
        "--model",
        default=None,
        help="Anthropic model name override",
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
        "--backend",
        choices=("pipeline", "graph"),
        default="pipeline",
        help="Orchestration backend: imperative pipeline (default) or LangGraph",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use deterministic local fake LLM responses instead of Claude "
            "(no ANTHROPIC_API_KEY required)"
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description=(
            "Extract outdoor seating predictions from restaurant evidence."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser(
        "run",
        help="Run extraction over a restaurants CSV (default command)",
    )
    run_parser.add_argument(
        "--input",
        type=Path,
        default=settings.input_path,
        help=f"Input CSV path (default: {settings.input_path})",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=settings.output_path,
        help=f"Results CSV path (default: {settings.output_path})",
    )
    run_parser.add_argument(
        "--review-queue-output",
        type=Path,
        default=settings.review_queue_path,
        help=f"Review queue CSV path (default: {settings.review_queue_path})",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many records (for quick smoke runs)",
    )
    _add_shared_run_flags(run_parser, settings)

    gather_parser = subparsers.add_parser(
        "gather",
        help="Gather evidence by restaurant name and city, then extract",
    )
    gather_parser.add_argument(
        "--name",
        required=True,
        help="Restaurant name",
    )
    gather_parser.add_argument(
        "--city",
        required=True,
        help="City where the restaurant is located",
    )
    gather_parser.add_argument(
        "--output",
        type=Path,
        default=settings.gather_output_path,
        help=f"Results CSV path (default: {settings.gather_output_path})",
    )
    gather_parser.add_argument(
        "--review-queue-output",
        type=Path,
        default=settings.gather_review_queue_path,
        help=(
            "Review queue CSV path "
            f"(default: {settings.gather_review_queue_path})"
        ),
    )
    gather_parser.add_argument(
        "--fixtures-path",
        type=Path,
        default=settings.fixtures_path,
        help=f"Static evidence fixtures directory (default: {settings.fixtures_path})",
    )
    gather_parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Enable a live evidence source for the 'maps' role instead of "
            "static fixtures (see --source; requires GOOGLE_PLACES_API_KEY)"
        ),
    )
    gather_parser.add_argument(
        "--source",
        choices=("google",),
        default=None,
        help=(
            "Live source provider to use with --live (default: google when "
            "--live is set). Passing --source without --live is an error."
        ),
    )
    _add_shared_run_flags(gather_parser, settings)

    return parser


def _apply_settings_overrides(args: argparse.Namespace, settings: Settings) -> Settings:
    if args.model is not None:
        settings = settings.model_copy(update={"model": args.model})
    if args.confidence_threshold is not None:
        settings = settings.model_copy(
            update={"confidence_threshold": args.confidence_threshold}
        )
    return settings


def _build_client(args: argparse.Namespace, settings: Settings):
    if args.dry_run:
        print("Dry-run mode: using deterministic local responses (no API calls).")
        return DryRunClaudeClient()
    return ClaudeClient(settings)


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


def _print_gather_summary(
    *,
    output_path: Path,
    review_queue_path: Path,
) -> None:
    results = pd.read_csv(output_path)
    row = results.iloc[0]
    review_count = len(pd.read_csv(review_queue_path)) if review_queue_path.exists() else 0

    print("\nGather summary")
    print("--------------")
    print(f"Restaurant:           {row['name']}, {row['city']}")
    print(f"Prediction:           {row['prediction']}")
    print(f"Confidence:           {row['confidence']}")
    print(f"Validation status:    {row['validation_status']}")
    print(f"Needs review:         {row['needs_review']}")
    print(f"Results:              {output_path}")
    print(f"Review queue:         {review_count} record(s) -> {review_queue_path}")


def _run_command(args: argparse.Namespace, settings: Settings) -> int:
    client = _build_client(args, settings)
    run_fn = run_graph_pipeline if args.backend == "graph" else run_pipeline

    summary = run_fn(
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


def _resolve_live_source(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str | None:
    """Return the live source name, or None for the default static/fake flow.

    --source only has meaning alongside --live; passing it alone is a usage
    error rather than a silently ignored flag.
    """
    if not args.live:
        if args.source is not None:
            parser.error("--source requires --live")
        return None
    return args.source or "google"


def _gather_command(
    args: argparse.Namespace, settings: Settings, parser: argparse.ArgumentParser
) -> int:
    client = _build_client(args, settings)
    query = RestaurantQuery(name=args.name, city=args.city)
    live_source = _resolve_live_source(args, parser)
    adapters = build_adapters(
        args.fixtures_path,
        dry_run=args.dry_run,
        live_source=live_source,
        settings=settings,
    )

    if args.backend == "graph":
        state = run_gather_graph_pipeline(
            query=query,
            adapters=adapters,
            output_path=args.output,
            review_queue_path=args.review_queue_output,
            settings=settings,
            client=client,
        )
    else:
        state = run_gather_pipeline(
            query=query,
            adapters=adapters,
            output_path=args.output,
            review_queue_path=args.review_queue_output,
            settings=settings,
            client=client,
        )

    _ = state
    _print_gather_summary(
        output_path=args.output,
        review_queue_path=args.review_queue_output,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    normalised = _normalise_argv(list(sys.argv[1:] if argv is None else argv))
    parser = _build_parser()
    args = parser.parse_args(normalised)

    command = args.command or "run"
    settings = _apply_settings_overrides(args, load_settings())
    configure_logging(settings.log_level)

    if command == "gather":
        return _gather_command(args, settings, parser)
    return _run_command(args, settings)


if __name__ == "__main__":
    sys.exit(main())
