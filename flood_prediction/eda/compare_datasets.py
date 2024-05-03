"""Module containing functions to compare the test and train datasets."""

import typing
from collections.abc import Callable

import numpy as np
import polars as pl
import statsmodels.stats.weightstats  # type: ignore

from flood_prediction import _type_aliases, assets, dataset, utils


@assets.store()
def compare_features(
    test: Callable[[np.ndarray, np.ndarray], float] = utils.return_arg(  # noqa: B008
        statsmodels.stats.weightstats.ztest, -1
    ),
) -> pl.LazyFrame:
    """Compare the distributions of each feature."""
    features = typing.get_args(_type_aliases.DatasetFeature)
    train_df = dataset.load("train")
    test_df = dataset.load("test")
    return pl.LazyFrame(
        {
            feature: test(
                utils.column_to_series(train_df, feature).to_numpy(),
                utils.column_to_series(test_df, feature).to_numpy(),
            )
            for feature in features
        }
    )
