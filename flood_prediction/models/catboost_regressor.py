"""Model containing the catboost regressor model."""

import dataclasses
from collections.abc import Sequence
from typing import final

import catboost
import numpy as np

from flood_prediction import api, constants


@dataclasses.dataclass(kw_only=True, frozen=True)
class CatBoostParams:
    """Params for Catboost regressor."""

    random_seed: int | None = constants.SEED
    custom_metric: str | Sequence[str] = "Accuracy"
    logging_level: str | None = None


@final
class CatBoostRegressorModel(
    api.JoblibMixin,
    api.RegressorModel,
    api.TrainerWithValidationMixin,
    api.ModelConstructorMixin[catboost.CatBoostClassifier, CatBoostParams],
):
    """Catboost regressor model."""

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
    ):
        """Train the Catboost regressor model."""
        return self._model.fit(
            X_train,
            y_train,
            eval_set=(X_validation, y_validation),
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Create predictions from this model."""
        return self._model.predict(x)
