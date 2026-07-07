"""CLI entry point: build the curated restaurant-features Parquet dataset.

Usage:
    python -m restaurant_agent.pipelines.build_restaurant_features \
        [--raw-dir data/raw] [--output-dir data/curated/restaurant_features]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from restaurant_agent.pipelines.features import build_restaurant_feature_table
from restaurant_agent.pipelines.ingest import read_restaurants, read_reviews
from restaurant_agent.pipelines.spark_session import build_spark_session, stop_spark_session
from restaurant_agent.pipelines.write import write_restaurant_features

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/curated/restaurant_features")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build curated restaurant outdoor-seating features from raw CSVs.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing restaurants.csv and reviews.csv (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the curated Parquet dataset (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def run_pipeline(
    *,
    raw_dir: Path,
    output_dir: Path,
    spark: SparkSession | None = None,
) -> DataFrame:
    """Run ingest -> clean -> features -> write and return the curated DataFrame.

    If `spark` is None, this builds and owns its own session, stopping it in
    a finally block. If a session is passed in (e.g. by a test or caller
    that manages its own lifecycle), it is reused and left running.
    """
    owns_session = spark is None
    session = spark or build_spark_session()
    try:
        restaurants = read_restaurants(session, raw_dir / "restaurants.csv")
        reviews = read_reviews(session, raw_dir / "reviews.csv")
        features = build_restaurant_feature_table(restaurants, reviews)
        write_restaurant_features(features, output_dir)
        return features
    finally:
        if owns_session:
            stop_spark_session(session)


def _print_summary(features: DataFrame, *, output_dir: Path) -> None:
    totals = features.agg(
        F.count("*").alias("restaurant_count"),
        F.sum("total_reviews").alias("review_count"),
        F.sum(F.col("has_outdoor_seating_inferred").cast("int")).alias("inferred_outdoor_count"),
        F.sum((F.col("total_reviews") == 0).cast("int")).alias("no_review_count"),
        F.sum("outdoor_review_mentions").alias("outdoor_mention_count"),
    ).collect()[0]

    print("\nSpark pipeline summary")
    print("-----------------------")
    print(f"Restaurants processed:            {totals['restaurant_count']}")
    print(f"Reviews processed:                {totals['review_count'] or 0}")
    print(f"Restaurants inferred outdoor:      {totals['inferred_outdoor_count'] or 0}")
    print(f"Restaurants with no reviews:       {totals['no_review_count'] or 0}")
    print(f"Reviews mentioning outdoor phrases: {totals['outdoor_mention_count'] or 0}")
    print(f"Output:                            {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    # Owned here (rather than left to run_pipeline) so the session stays
    # alive long enough to compute the post-run summary below.
    spark = build_spark_session()
    try:
        features = run_pipeline(raw_dir=args.raw_dir, output_dir=args.output_dir, spark=spark)
        _print_summary(features, output_dir=args.output_dir)
    finally:
        stop_spark_session(spark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
