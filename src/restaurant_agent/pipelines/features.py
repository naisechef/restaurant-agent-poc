"""Outdoor-seating keyword detection and restaurant-level feature aggregation.

Keyword matching uses Spark SQL's ``regexp_extract_all`` rather than a Python
UDF. This keeps matching Catalyst-optimized (whole-stage codegen, no
JVM<->Python row serialization) and keeps the entire vocabulary inspectable
as a single generated regex string rather than opaque Python bytecode.

Known limitation: this is literal phrase matching, not NLP. It has no
negation handling, so a review saying "no patio" or "the outdoor seating was
closed" still counts as a mention. This is an intentional, acknowledged
simplification for this PoC's keyword/phrase-matching scope, not a bug.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from restaurant_agent.pipelines.clean import clean_restaurants, clean_reviews

OUTDOOR_PHRASES: tuple[str, ...] = (
    "outdoor seating",
    "terrace",
    "patio",
    "garden",
    "courtyard",
    "outside tables",
    "sea view terrace",
    "rooftop",
    "al fresco",
)

# Below this many reviews, confidence is discounted proportionally: a single
# anecdote should never read as certain. At/above this count, only the
# mention ratio matters. See the worked examples in docs/SPARK_PIPELINE.md.
MIN_REVIEWS_FOR_FULL_CONFIDENCE: int = 5

# Minimum outdoor_confidence_score for a restaurant to be inferred as having
# outdoor seating.
OUTDOOR_CONFIDENCE_THRESHOLD: float = 0.3

FEATURE_COLUMNS: list[str] = [
    "restaurant_id",
    "name",
    "cuisine",
    "city",
    "area",
    "rating",
    "review_count",
    "price_level",
    "latitude",
    "longitude",
    "total_reviews",
    "outdoor_review_mentions",
    "outdoor_confidence_score",
    "has_outdoor_seating_inferred",
    "sample_outdoor_evidence",
]


def _build_outdoor_pattern(phrases: Sequence[str] = OUTDOOR_PHRASES) -> str:
    """Build a case-insensitive regex alternation of escaped phrases.

    The whole alternation is wrapped in a single capturing group so
    ``regexp_extract_all(..., idx=1)`` returns the matched phrase text.
    """
    alternation = "|".join(re.escape(phrase) for phrase in phrases)
    return f"(?i)({alternation})"


def detect_outdoor_mentions(
    reviews: DataFrame, *, phrases: Sequence[str] = OUTDOOR_PHRASES
) -> DataFrame:
    """Add matched_phrases: array<string> and has_outdoor_mention: bool per review."""
    pattern = _build_outdoor_pattern(phrases)
    safe_text = F.coalesce(F.col("review_text"), F.lit(""))
    return reviews.withColumn(
        "matched_phrases", F.regexp_extract_all(safe_text, F.lit(pattern), 1)
    ).withColumn("has_outdoor_mention", F.size(F.col("matched_phrases")) > 0)


def aggregate_outdoor_features(annotated_reviews: DataFrame) -> DataFrame:
    """Group annotated reviews by restaurant into per-restaurant outdoor features.

    outdoor_confidence_score = mention_ratio * volume_factor, where:
      mention_ratio = outdoor_review_mentions / total_reviews
      volume_factor = min(1.0, total_reviews / MIN_REVIEWS_FOR_FULL_CONFIDENCE)

    volume_factor discounts confidence when there isn't enough review volume
    to corroborate the signal yet (e.g. a single review with one mention
    scores 0.20, not a false-certain 1.0), consistent with "unknown is
    preferable to incorrect."

    sample_outdoor_evidence is picked deterministically via min_by on
    review_id among matching reviews only, so it is a stable, traceable
    verbatim quote rather than an arbitrary row.
    """
    aggregated = annotated_reviews.groupBy("restaurant_id").agg(
        F.count("*").cast("int").alias("total_reviews"),
        F.sum(F.col("has_outdoor_mention").cast("int"))
        .cast("int")
        .alias("outdoor_review_mentions"),
        F.min_by(
            F.when(F.col("has_outdoor_mention"), F.col("review_text")),
            F.when(F.col("has_outdoor_mention"), F.col("review_id")),
        ).alias("sample_outdoor_evidence"),
    )
    mention_ratio = F.col("outdoor_review_mentions") / F.col("total_reviews")
    volume_factor = F.least(
        F.lit(1.0), F.col("total_reviews") / F.lit(MIN_REVIEWS_FOR_FULL_CONFIDENCE)
    )
    return aggregated.withColumn(
        "outdoor_confidence_score", F.round(mention_ratio * volume_factor, 2)
    ).withColumn(
        "has_outdoor_seating_inferred",
        F.col("outdoor_confidence_score") >= F.lit(OUTDOOR_CONFIDENCE_THRESHOLD),
    )


def build_restaurant_feature_table(
    restaurants: DataFrame,
    reviews: DataFrame,
    *,
    phrases: Sequence[str] = OUTDOOR_PHRASES,
) -> DataFrame:
    """End-to-end: clean inputs, detect + aggregate outdoor mentions, join, order columns.

    Uses a left join so restaurants with zero matching reviews still appear
    exactly once, with mentions/confidence zeroed and no evidence.
    """
    cleaned_restaurants = clean_restaurants(restaurants)
    annotated_reviews = detect_outdoor_mentions(clean_reviews(reviews), phrases=phrases)
    features = aggregate_outdoor_features(annotated_reviews)

    joined = cleaned_restaurants.join(features, on="restaurant_id", how="left")
    joined = (
        joined.withColumn("total_reviews", F.coalesce(F.col("total_reviews"), F.lit(0)))
        .withColumn(
            "outdoor_review_mentions",
            F.coalesce(F.col("outdoor_review_mentions"), F.lit(0)),
        )
        .withColumn(
            "outdoor_confidence_score",
            F.coalesce(F.col("outdoor_confidence_score"), F.lit(0.0)),
        )
        .withColumn(
            "has_outdoor_seating_inferred",
            F.coalesce(F.col("has_outdoor_seating_inferred"), F.lit(False)),
        )
    )
    return joined.select(FEATURE_COLUMNS)
