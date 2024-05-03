"""CLI entry for predicting using a specific model."""

from collections.abc import Sequence


def main(args: Sequence[str] | None = None):
    """Process the CLI args and run the module."""
    print(args)


if __name__ == "__main__":
    main()
