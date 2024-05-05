"""CLI entry for predicting using a specific model."""

import argparse
import os
import typing
from collections.abc import Sequence

from sklearn import metrics
from sklearn.model_selection import train_test_split

from flood_prediction import _type_aliases, api, constants, datasets, models


def create_parser():
    """Create the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=tuple(models.registry().keys()))
    return parser


def main(args: Sequence[str] | None = None):
    """Process the CLI args and run the module."""
    values, config = create_parser().parse_known_args(args)
    train_df = datasets.load("train", as_pandas=True)
    X, y = (
        train_df[list(typing.get_args(_type_aliases.DatasetFeature))],
        train_df[constants.TARGET_FEATURE],
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, train_size=0.75, random_state=constants.SEED
    )
    print(X.shape, y.shape)
    model_name = values.model_name
    model = models.registry()[model_name]()
    checkpoint_file = os.path.join(constants.CHECKPOINTS_DIR, f"{model_name}-.75.ckpt")
    if os.path.exists(checkpoint_file):
        model.load(checkpoint_file)
    else:
        if isinstance(model, api.TrainerWithValidationMixin):
            # TODO validation set
            model.train(
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
            )
        elif isinstance(model, api.TrainerMixin):
            model.train(X_train=X_train, y_train=y_train)
        model.save(checkpoint_file)
    print(metrics.r2_score(y_validation, model(X_validation)))


if __name__ == "__main__":
    main()
