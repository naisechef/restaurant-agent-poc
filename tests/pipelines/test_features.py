from __future__ import annotations

import pytest
from pyspark.sql import DataFrame, SparkSession

from restaurant_agent.pipelines.clean import clean_restaurants, clean_reviews
from restaurant_agent.pipelines.features import (
    FEATURE_COLUMNS,
    OUTDOOR_PHRASES,
    aggregate_outdoor_features,
    build_restaurant_feature_table,
    detect_outdoor_mentions,
)
from restaurant_agent.pipelines.schemas import RESTAURANTS_SCHEMA, REVIEWS_SCHEMA


def _reviews_df(spark: SparkSession, rows: list[tuple[str, str, str | None]]) -> DataFrame:
    return spark.createDataFrame(rows, schema=REVIEWS_SCHEMA)


def _restaurants_df(
    spark: SparkSession,
    rows: list[
        tuple[
            str,
            str | None,
            str | None,
            str | None,
            str | None,
            float | None,
            int | None,
            int | None,
            float | None,
            float | None,
        ]
    ],
) -> DataFrame:
    return spark.createDataFrame(rows, schema=RESTAURANTS_SCHEMA)


# --- detect_outdoor_mentions -------------------------------------------------


@pytest.mark.parametrize("phrase", OUTDOOR_PHRASES)
def test_detect_outdoor_mentions_matches_all_required_phrases(
    spark: SparkSession, phrase: str
) -> None:
    rows = [("rev-1", "rest-1", f"We loved the {phrase.upper()} here.")]
    result = detect_outdoor_mentions(_reviews_df(spark, rows)).collect()[0]
    assert result["has_outdoor_mention"] is True


def test_detect_outdoor_mentions_no_match_for_unrelated_text(spark: SparkSession) -> None:
    rows = [("rev-1", "rest-1", "The service was slow and the food was cold.")]
    result = detect_outdoor_mentions(_reviews_df(spark, rows)).collect()[0]
    assert result["has_outdoor_mention"] is False
    assert result["matched_phrases"] == []


def test_detect_outdoor_mentions_extracts_multiple_phrases_in_one_review(
    spark: SparkSession,
) -> None:
    rows = [("rev-1", "rest-1", "Lovely patio and a rooftop bar upstairs.")]
    result = detect_outdoor_mentions(_reviews_df(spark, rows)).collect()[0]
    assert set(p.lower() for p in result["matched_phrases"]) == {"patio", "rooftop"}


def test_detect_outdoor_mentions_is_null_safe(spark: SparkSession) -> None:
    rows: list[tuple[str, str, str | None]] = [("rev-1", "rest-1", None)]
    result = detect_outdoor_mentions(_reviews_df(spark, rows)).collect()[0]
    assert result["has_outdoor_mention"] is False


# --- aggregate_outdoor_features ----------------------------------------------


def test_aggregate_outdoor_features_counts_mentions_per_restaurant(
    spark: SparkSession,
) -> None:
    rows = [
        ("rev-1", "rest-1", "Great patio."),
        ("rev-2", "rest-1", "Nothing special."),
        ("rev-3", "rest-2", "No outdoor mention here."),
    ]
    annotated = detect_outdoor_mentions(_reviews_df(spark, rows))
    features = {
        row["restaurant_id"]: row
        for row in aggregate_outdoor_features(annotated).collect()
    }
    assert features["rest-1"]["total_reviews"] == 2
    assert features["rest-1"]["outdoor_review_mentions"] == 1
    assert features["rest-2"]["total_reviews"] == 1
    assert features["rest-2"]["outdoor_review_mentions"] == 0


def test_aggregate_outdoor_features_computes_confidence_formula(
    spark: SparkSession,
) -> None:
    rows = [
        ("rev-1", "rest-1", "Great patio."),
        ("rev-2", "rest-1", "Lovely terrace too."),
        ("rev-3", "rest-1", "Just average food."),
    ]
    annotated = detect_outdoor_mentions(_reviews_df(spark, rows))
    result = aggregate_outdoor_features(annotated).collect()[0]
    assert result["outdoor_confidence_score"] == pytest.approx(0.40)


def test_aggregate_outdoor_features_threshold_boundary(spark: SparkSession) -> None:
    single_anecdote = detect_outdoor_mentions(
        _reviews_df(spark, [("rev-1", "rest-1", "Great patio.")])
    )
    corroborated = detect_outdoor_mentions(
        _reviews_df(
            spark,
            [
                ("rev-1", "rest-2", "Great patio."),
                ("rev-2", "rest-2", "Lovely terrace too."),
                ("rev-3", "rest-2", "Just average food."),
            ],
        )
    )
    single_result = aggregate_outdoor_features(single_anecdote).collect()[0]
    corroborated_result = aggregate_outdoor_features(corroborated).collect()[0]

    assert single_result["outdoor_confidence_score"] == pytest.approx(0.20)
    assert single_result["has_outdoor_seating_inferred"] is False

    assert corroborated_result["outdoor_confidence_score"] == pytest.approx(0.40)
    assert corroborated_result["has_outdoor_seating_inferred"] is True


def test_aggregate_outdoor_features_sample_evidence_is_deterministic_and_traceable(
    spark: SparkSession,
) -> None:
    rows = [
        ("rev-2", "rest-1", "Second matching review about the rooftop."),
        ("rev-1", "rest-1", "First matching review about the patio."),
        ("rev-3", "rest-1", "No outdoor mention here."),
    ]
    annotated = detect_outdoor_mentions(_reviews_df(spark, rows))
    result = aggregate_outdoor_features(annotated).collect()[0]
    assert result["sample_outdoor_evidence"] == "First matching review about the patio."


def test_aggregate_outdoor_features_zero_matches_has_null_evidence(
    spark: SparkSession,
) -> None:
    rows = [("rev-1", "rest-1", "Nothing outdoor-related here.")]
    annotated = detect_outdoor_mentions(_reviews_df(spark, rows))
    result = aggregate_outdoor_features(annotated).collect()[0]
    assert result["sample_outdoor_evidence"] is None
    assert result["outdoor_confidence_score"] == pytest.approx(0.0)


# --- build_restaurant_feature_table ------------------------------------------


def test_build_restaurant_feature_table_left_join_keeps_zero_review_restaurants(
    spark: SparkSession,
) -> None:
    restaurants = _restaurants_df(
        spark,
        [
            ("rest-1", "Harbor View", "Seafood", "Brighton", None, 4.2, 10, 2, None, None),
            ("rest-2", "No Reviews Yet", "Steakhouse", "Denver", None, 3.9, 5, 3, None, None),
        ],
    )
    reviews = _reviews_df(spark, [("rev-1", "rest-1", "Lovely patio out back.")])

    result = {
        row["restaurant_id"]: row
        for row in build_restaurant_feature_table(restaurants, reviews).collect()
    }

    zero_review_row = result["rest-2"]
    assert zero_review_row["total_reviews"] == 0
    assert zero_review_row["outdoor_review_mentions"] == 0
    assert zero_review_row["outdoor_confidence_score"] == pytest.approx(0.0)
    assert zero_review_row["has_outdoor_seating_inferred"] is False
    assert zero_review_row["sample_outdoor_evidence"] is None


def test_build_restaurant_feature_table_output_columns_and_order(
    spark: SparkSession,
) -> None:
    restaurants = _restaurants_df(
        spark,
        [("rest-1", "Harbor View", "Seafood", "Brighton", None, 4.2, 10, 2, None, None)],
    )
    reviews = _reviews_df(spark, [("rev-1", "rest-1", "Lovely patio out back.")])
    result = build_restaurant_feature_table(restaurants, reviews)
    assert result.columns == FEATURE_COLUMNS


# --- clean_restaurants / clean_reviews ----------------------------------------


def test_clean_restaurants_trims_and_title_cases_text_fields(spark: SparkSession) -> None:
    restaurants = _restaurants_df(
        spark,
        [("rest-1", "  Rooftop Bistro  ", " italian ", "  LONDON ", None, 4.0, 10, 2, None, None)],
    )
    result = clean_restaurants(restaurants).collect()[0]
    assert result["name"] == "Rooftop Bistro"
    assert result["cuisine"] == "Italian"
    assert result["city"] == "London"


def test_clean_restaurants_nulls_out_of_range_numeric_fields(spark: SparkSession) -> None:
    restaurants = _restaurants_df(
        spark,
        [("rest-1", "Messy Fields Grill", "BBQ", "Austin", None, 7.5, 10, 9, 200.0, None)],
    )
    result = clean_restaurants(restaurants).collect()[0]
    assert result["rating"] is None
    assert result["price_level"] is None
    assert result["latitude"] is None


def test_clean_reviews_collapses_whitespace_but_preserves_case(spark: SparkSession) -> None:
    reviews = _reviews_df(spark, [("rev-1", "rest-1", "  Great   PATIO  view ")])
    result = clean_reviews(reviews).collect()[0]
    assert result["review_text"] == "Great PATIO view"
