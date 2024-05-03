"""Module containing common fixtures for pytest."""

from collections.abc import Generator

import polars as pl
import pytest


@pytest.fixture(scope="session")
def answer_df() -> Generator[pl.DataFrame, None, None]:
    """Fixture for a dataframe with a single column 'answer' containing numbers."""
    df = pl.DataFrame((pl.int_range(end=42, eager=True).alias("answer"),))
    yield df
