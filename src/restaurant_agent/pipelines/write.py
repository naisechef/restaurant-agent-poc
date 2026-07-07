"""Parquet I/O for the curated restaurant-features dataset."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession


def write_restaurant_features(
    df: DataFrame,
    output_dir: str | Path,
    *,
    mode: str = "overwrite",
    num_output_files: int = 1,
) -> None:
    """Write the curated feature table to Parquet.

    Coalesced to `num_output_files` part-file(s) for PoC readability (easy to
    inspect a single file locally). This is a demo-scale convenience, not a
    production partitioning strategy.
    """
    df.coalesce(num_output_files).write.mode(mode).parquet(str(output_dir))


def read_restaurant_features(spark: SparkSession, output_dir: str | Path) -> DataFrame:
    """Read back a previously written curated Parquet dataset."""
    return spark.read.parquet(str(output_dir))
