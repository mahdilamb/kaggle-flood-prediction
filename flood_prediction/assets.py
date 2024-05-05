"""Methods for working with evaluation."""

import functools
import os
import typing
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from flood_prediction import constants

T = TypeVar("T")
P = ParamSpec("P")

ALWAYS_OVERWRITE: bool = False


def store(
    location: str | None = None,
    directory: str = constants.ASSETS_DIR,
    overwrite: bool = ALWAYS_OVERWRITE,
    auto: bool = False,
):
    """Store an asset.

    Args:
        location (str | None, optional): The location to store the file to.
            If None, use the function name. Defaults to None.
        directory (str, optional): The directory to store to.
            Defaults to constants.ASSETS_DIRECTORY.
        overwrite (bool, optional): Whether to always overwrite the file.
            Defaults to False.
        auto (bool, optional): Whether to auto-call the function.
            Assumes the function either has no args or all args are defaulted.
            Defaults to False.
    """
    os.makedirs(directory, exist_ok=True)
    if location:
        location = os.path.join(directory, location)
        _, ext = os.path.splitext(location)

    @typing.overload
    def wrapper(
        fn: Callable[P, T],
    ) -> Callable[P, None]: ...
    @typing.overload
    def wrapper(
        fn: Callable[[], T],
    ) -> Callable[[], None]: ...

    def wrapper(
        fn: Callable[P, T] | Callable[[], T],
    ):
        def call(*args: P.args, **kwargs: P.kwargs):
            nonlocal location, ext
            if not location:
                import pandas as pd
                import polars as pl

                location = (
                    fn.__name__ + ".jsonl"
                    if fn.__annotations__["return"]
                    in (pl.LazyFrame, pl.DataFrame, pd.DataFrame)
                    else ".png"
                )
                location = os.path.join(directory, location)
                _, ext = os.path.splitext(location)
            if overwrite or not os.path.exists(location):
                result = fn(*args, **kwargs)
                if result is not None:
                    if ext in (".png", ".jpg", ".jpeg"):
                        import matplotlib.pyplot as plt

                        plt.savefig(location)
                    elif ext in (".json", ".jsonl", ".csv"):
                        import pandas as pd
                        import polars as pl

                        if isinstance(result, pl.LazyFrame | pl.DataFrame):
                            {
                                ".json": pl.DataFrame.write_json,
                                ".jsonl": pl.DataFrame.write_ndjson,
                                ".csv": pl.DataFrame.write_csv,
                            }[ext](
                                result.collect().rechunk()
                                if isinstance(result, pl.LazyFrame)
                                else result,
                                location,
                            )
                        elif isinstance(result, pd.DataFrame):
                            {
                                ".csv": pd.DataFrame.to_csv,
                                ".json": pd.DataFrame.to_json,
                                ".jsonl": functools.partial(
                                    pd.DataFrame.to_json, lines=True
                                ),
                            }[ext](result, location)

        return call if not auto else call()

    return wrapper
