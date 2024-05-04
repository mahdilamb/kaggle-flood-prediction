import polars as pl

from flood_prediction import constants, dataset


def one_liner():
    """One liner taken from https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499263."""
    return dataset.load("test").select(
        "id",
        ((pl.sum_horizontal(pl.exclude("id")) * 0.0056) - 0.05).alias(
            constants.OUTPUT_FEATURE
        ),
    )
