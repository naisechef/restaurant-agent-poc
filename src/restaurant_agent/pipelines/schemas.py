"""Explicit Spark schemas for raw CSV ingestion.

Schemas are defined up front rather than relying on ``inferSchema`` so that
column types are stable and documented. Note that Spark's CSV reader treats
``nullable`` as documentary metadata only: it does not reject rows, and a
value that fails to cast to its declared type becomes ``null`` for that cell
rather than raising an error. Real validation (range checks, required-field
filtering) happens in ``clean.py``, not here.
"""

from __future__ import annotations

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

RESTAURANTS_SCHEMA = StructType(
    [
        StructField("restaurant_id", StringType(), nullable=True),
        StructField("name", StringType(), nullable=True),
        StructField("cuisine", StringType(), nullable=True),
        StructField("city", StringType(), nullable=True),
        StructField("area", StringType(), nullable=True),
        StructField("rating", DoubleType(), nullable=True),
        StructField("review_count", IntegerType(), nullable=True),
        StructField("price_level", IntegerType(), nullable=True),
        StructField("latitude", DoubleType(), nullable=True),
        StructField("longitude", DoubleType(), nullable=True),
    ]
)

REVIEWS_SCHEMA = StructType(
    [
        StructField("review_id", StringType(), nullable=True),
        StructField("restaurant_id", StringType(), nullable=True),
        StructField("review_text", StringType(), nullable=True),
    ]
)
