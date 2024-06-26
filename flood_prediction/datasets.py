"""Methods for loading datasets."""

import os
from typing import Literal, overload

import pandas as pd
import polars as pl

from flood_prediction import _type_aliases, constants


@overload
def load(
    dataset: _type_aliases.Dataset, as_pandas: Literal[False] = ...
) -> pl.LazyFrame: ...
@overload
def load(dataset: _type_aliases.Dataset, as_pandas: Literal[True]) -> pd.DataFrame: ...


def load(
    dataset: _type_aliases.Dataset, as_pandas: bool = False
) -> pl.LazyFrame | pd.DataFrame:
    """Load either train or test dataset.

    See `constants.DATASET_SCHEMA` for the expected schema.

    Args:
        dataset (Literal['train', 'test']): Which dataset to load.
        as_pandas (bool, optional): Whether to retrieve a pandas dataframe or not.

    Returns:
        pl.LazyFrame: The data frame.
    """
    if as_pandas:
        return pd.read_csv(
            os.path.join(constants.DATA_DIR, f"{dataset}.csv"), index_col=0
        )
    schema = dict(constants.DATASET_SCHEMA.items())
    if dataset == "train":
        schema.update(**{constants.TARGET_FEATURE: pl.Float64})
    return pl.scan_csv(
        os.path.join(constants.DATA_DIR, f"{dataset}.csv"),
        schema=schema,
    )
