"""Model containing the linear regressor model."""

import dataclasses
from typing import final

import numpy as np
from sklearn import linear_model

from flood_prediction import api


@dataclasses.dataclass(kw_only=True, frozen=True)
class LinearRegressorParams:
    """Parameters for the linear regressor model."""

    n_jobs: int = -1


@final
class LinearRegressorModel(
    api.JoblibMixin,
    api.RegressorModel,
    api.TrainerMixin,
    api.ModelConstructorMixin[linear_model.LinearRegression, LinearRegressorParams],
):
    """Linear regressor model."""

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the linear regressor model."""
        return self._model.fit(X_train, y_train)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Create predictions from this model."""
        return self._model.predict(x)
