"""Test asset storing."""

import os
import tempfile

import polars as pl

from flood_prediction import assets


def test_store_df(answer_df):
    """Test that storing assets works when no location provided."""
    with tempfile.TemporaryDirectory() as tmp_dir:

        @assets.store(auto=True, directory=tmp_dir)
        def test() -> pl.LazyFrame:
            return answer_df

        assert "test.jsonl" in os.listdir(
            tmp_dir
        ), "Expected the auto flag to create a file in the directory."


def test_store_df_location_specified(answer_df):
    """Test that storing assets works when location is provided."""
    with tempfile.TemporaryDirectory() as tmp_dir:

        @assets.store(location="test.jsonl", auto=True, directory=tmp_dir)
        def test():
            return answer_df

        assert "test.jsonl" in os.listdir(
            tmp_dir
        ), "Expected the auto flag to create a file in the directory."


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
