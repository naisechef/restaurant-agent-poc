from __future__ import annotations

from collections.abc import Iterator

import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402

from restaurant_agent.pipelines.spark_session import build_spark_session  # noqa: E402


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Session-scoped local SparkSession for pipeline tests."""
    session = build_spark_session(
        app_name="restaurant-agent-tests", master="local[1]", shuffle_partitions=1
    )
    yield session
    session.stop()
