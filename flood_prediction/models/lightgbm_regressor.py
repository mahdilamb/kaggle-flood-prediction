"""Model containing the LightGBM regressor model."""

import dataclasses
from typing import final

import lightgbm
import numpy as np

from flood_prediction import api, constants


@dataclasses.dataclass(kw_only=True, frozen=True)
class LightGBMParams:
    """Params for LightGBM regressor."""

    n_jobs: int | None = -1
    random_state: int | None = constants.SEED


@final
class LinearRegressorModel(
    api.JoblibMixin,
    api.RegressorModel,
    api.TrainerWithValidationMixin,
    api.ModelConstructorMixin[lightgbm.LGBMRegressor, LightGBMParams],
):
    """LightGBM regressor model."""

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
    ):
        """Train the LightGBM regressor model."""
        return self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Create predictions from this model."""
        return self._model.predict(x)
