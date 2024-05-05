"""Abstract classes for declaring model."""

import abc
import dataclasses
from typing import Generic, TypeVar, get_args

import joblib
import numpy as np

from flood_prediction import _type_aliases

T = TypeVar("T")


class TrainerMixin(abc.ABC):
    """Mixin for a regressor model that trains without a validation set."""

    @abc.abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        """Train a model without a validation set."""


class TrainerWithValidationMixin(abc.ABC):
    """Mixin for a regressor model that trains with a validation set."""

    @abc.abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
    ):
        """Train this model with a validation set."""


class RegressorModel(abc.ABC):
    """Abstract class for a regressor model."""

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Save the state of this model."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load the model."""

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Use the model to generate prediction."""

    def __new__(cls, *args, **kwargs):
        """Ensure that the model subclasses a train function.

        Raises:
            NotImplementedError: If the model doesn't subclass
              `TrainerMixin` or `TrainerWithValidationMixin`.


        """
        if not issubclass(cls, TrainerMixin | TrainerWithValidationMixin):
            raise NotImplementedError(
                "Regressor model must either inherit a `train` function from either "
                + "`TrainerMixin` or `TrainerWithValidationMixin`."
            )
        return super().__new__(cls, *args, **kwargs)


class JoblibMixin:
    """Mixin for a class that uses JobLib for saving/loading."""

    __model_attribute__: str

    def __init_subclass__(cls, model_attribute: str = "_model") -> None:
        """Initialize the subclass.

        Args:
            model_attribute (str, optional): The attribute that stores the model.
                Defaults to "_model".
        """
        cls.__model_attribute__ = model_attribute

    def save(self, path: str) -> None:
        """Save the state of this model."""
        joblib.dump(getattr(self, self.__model_attribute__), path)

    def load(self, path: str) -> None:
        """Load the model."""
        setattr(self, self.__model_attribute__, joblib.load(path))


class ModelConstructorMixin(
    abc.ABC,
    Generic[T, _type_aliases.DataClass],
):
    """Mixin for a model whose constructor either takes kwargs or a dataclass."""

    _model: T

    def __new__(cls, *args, **kwargs):
        """Override the init function.

        The init function will either accept kwargs or a dataclass and set the `_model`
          attribute.
        """
        model_class, param_class = get_args(
            next(
                base
                for base in cls.__orig_bases__
                if getattr(base, "__origin__", None) == ModelConstructorMixin
            )
        )

        def init(self, params: _type_aliases.DataClass | None = None, **kwargs):
            super().__init__()
            if params is None:
                params = param_class(**kwargs)
            self._model = model_class(**dataclasses.asdict(params))  # type: ignore

        cls.__init__ = init
        return super().__new__(cls, *args, **kwargs)
