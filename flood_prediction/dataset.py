"""Methods for loading datasets."""

import os
import typing

import pandas as pd
import polars as pl

from flood_prediction import _type_aliases, constants


@typing.overload
def load(dataset: _type_aliases.Dataset, as_pandas: bool = False) -> pl.LazyFrame: ...
@typing.overload
def load(dataset: _type_aliases.Dataset, as_pandas: bool = True) -> pd.DataFrame: ...


def load(dataset: _type_aliases.Dataset, as_pandas: bool = False):
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
    return pl.scan_csv(
        os.path.join(constants.DATA_DIR, f"{dataset}.csv"),
        schema=dict(constants.DATASET_SCHEMA.items()),
    )
