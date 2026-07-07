"""Field-level normalization and range validation for raw ingested data.

Column names are preserved; only values change. Out-of-range or malformed
values are nulled rather than guessed, in keeping with "unknown is
preferable to incorrect."
"""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def _collapse_whitespace(column: str) -> F.Column:
    return F.trim(F.regexp_replace(F.col(column), r"\s+", " "))


def _title_case(column: str) -> F.Column:
    return F.initcap(F.lower(_collapse_whitespace(column)))


def _null_if_blank(col: F.Column) -> F.Column:
    return F.when(col == "", None).otherwise(col)


def clean_restaurants(df: DataFrame) -> DataFrame:
    """Trim/normalise restaurant fields and null out-of-range numeric values.

    Rows with a blank/null ``restaurant_id`` are dropped since they cannot be
    joined against review features.
    """
    cleaned = (
        df.withColumn("restaurant_id", _null_if_blank(F.trim(F.col("restaurant_id"))))
        .withColumn("name", _null_if_blank(_collapse_whitespace("name")))
        .withColumn("cuisine", _null_if_blank(_title_case("cuisine")))
        .withColumn("city", _null_if_blank(_title_case("city")))
        .withColumn("area", _null_if_blank(_title_case("area")))
        .withColumn(
            "rating",
            F.when(
                F.col("rating").between(0.0, 5.0), F.round(F.col("rating"), 1)
            ).otherwise(None),
        )
        .withColumn(
            "review_count",
            F.when(F.col("review_count") >= 0, F.col("review_count")).otherwise(None),
        )
        .withColumn(
            "price_level",
            F.when(F.col("price_level").isin(1, 2, 3, 4), F.col("price_level")).otherwise(
                None
            ),
        )
        .withColumn(
            "latitude",
            F.when(F.col("latitude").between(-90.0, 90.0), F.col("latitude")).otherwise(
                None
            ),
        )
        .withColumn(
            "longitude",
            F.when(
                F.col("longitude").between(-180.0, 180.0), F.col("longitude")
            ).otherwise(None),
        )
    )
    return cleaned.filter(F.col("restaurant_id").isNotNull())


def clean_reviews(df: DataFrame) -> DataFrame:
    """Trim/normalise review fields; drop rows with no review_id or restaurant_id.

    ``review_text`` case and punctuation are preserved exactly so downstream
    matching stays accurate and any sampled evidence is a verbatim quote.
    Null/empty text is allowed through: it means "no signal", not an error.
    """
    cleaned = (
        df.withColumn("review_id", _null_if_blank(F.trim(F.col("review_id"))))
        .withColumn("restaurant_id", _null_if_blank(F.trim(F.col("restaurant_id"))))
        .withColumn(
            "review_text",
            F.when(
                F.col("review_text").isNotNull(), _collapse_whitespace("review_text")
            ).otherwise(None),
        )
    )
    return cleaned.filter(
        F.col("review_id").isNotNull() & F.col("restaurant_id").isNotNull()
    )
