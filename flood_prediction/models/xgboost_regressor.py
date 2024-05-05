"""Model containing the XGBoost regressor model."""

import dataclasses
from typing import final

import numpy as np
import xgboost

from flood_prediction import api


@dataclasses.dataclass(kw_only=True, frozen=True)
class XGBoostRegressorParams:
    """Parameters for the XGBoost regressor model."""

    n_jobs: int = -1


@final
class XGBoostRegressorModel(
    api.JoblibMixin,
    api.RegressorModel,
    api.TrainerMixin,
    api.ModelConstructorMixin[xgboost.XGBRegressor, XGBoostRegressorParams],
):
    """XGBoost regressor model."""

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the XGBoost regressor model."""
        return self._model.fit(X_train, y_train)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Create predictions from this model."""
        return self._model.predict(x)
