"""Schema-on-read CSV ingestion for the raw restaurant/review datasets."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from restaurant_agent.pipelines.schemas import RESTAURANTS_SCHEMA, REVIEWS_SCHEMA


def read_restaurants(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read restaurants.csv against RESTAURANTS_SCHEMA (header row expected)."""
    return (
        spark.read.option("header", True)
        .schema(RESTAURANTS_SCHEMA)
        .csv(str(path))
    )


def read_reviews(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read reviews.csv against REVIEWS_SCHEMA (header row expected)."""
    return spark.read.option("header", True).schema(REVIEWS_SCHEMA).csv(str(path))
