"""Local SparkSession construction shared by the CLI entry point and tests."""

from __future__ import annotations

from pyspark.sql import SparkSession


def build_spark_session(
    app_name: str = "restaurant_features",
    *,
    master: str = "local[*]",
    shuffle_partitions: int = 8,
    extra_conf: dict[str, str] | None = None,
) -> SparkSession:
    """Create or fetch a local SparkSession tuned for this batch pipeline.

    Pins the driver to loopback to avoid a known local-Spark hang on
    networks where the driver cannot resolve its own hostname.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )
    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def stop_spark_session(spark: SparkSession) -> None:
    """Stop a SparkSession; thin wrapper kept for symmetry and testability."""
    spark.stop()
