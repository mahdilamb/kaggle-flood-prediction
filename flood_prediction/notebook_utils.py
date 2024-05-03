"""Utility functions for using jupyter notebooks."""

import flood_prediction


def find_project_root() -> str:
    """Find the package root."""
    import os

    while not os.path.exists(os.path.join(os.getcwd(), "pyproject.toml")):
        os.chdir("..")
    return os.getcwd()


def init():
    """Function to reset the notebook so it works consistently."""
    import importlib
    import subprocess

    print(f"Package root is '{find_project_root()}'.")
    if subprocess.check_call("make dataset", shell=True) != 0:
        raise RuntimeError(
            "Failed to download the dataset and/or install the requirements."
        )
    importlib.reload(flood_prediction)
