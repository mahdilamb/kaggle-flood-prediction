"""Package containing the regressor models."""

import functools
from collections.abc import Mapping
from types import MappingProxyType

from flood_prediction import api


@functools.cache
def registry() -> Mapping[str, type[api.RegressorModel]]:
    """List all the RegressorModels."""
    import importlib
    import inspect
    import os

    from flood_prediction import models

    result = {}
    for file in os.listdir(os.path.dirname(__file__)):
        if file.endswith(".py") and not file.startswith("_"):
            module_name = file[:-3]
            module = importlib.import_module(f"{models.__name__}.{module_name}")
            try:
                result[module_name] = next(
                    obj
                    for _, obj in inspect.getmembers(module)
                    if isinstance(obj, type) and issubclass(obj, api.RegressorModel)
                )
            except StopIteration:
                ...
    return MappingProxyType(result)
