"""Constants used throughout the package."""

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

import polars as pl

from flood_prediction import _type_aliases

PACKAGE_DIR = os.path.dirname(__file__)  # The root of the package
ROOT_DIR = os.path.normpath(os.path.join(__file__, ".."))  # The parent to the package
DATA_DIR = os.path.join(PACKAGE_DIR, "data")  # The location of the data
ASSETS_DIR = os.getenv(
    "FLOOD_PREDICTION_ASSETS", os.path.join(ROOT_DIR, "assets")
)
CHECKPOINTS_DIR = os.getenv(
    "FLOOD_PREDICTION_CHECKPOINTS", os.path.join(ROOT_DIR, "checkpoints")
)
TRAINING: bool = os.getenv("FLOOD_PREDICTION_TRAINING", "true").strip().lower() in (
    "1",
    "true",
)

SEED = 42 if TRAINING else None
DATASET_SCHEMA: Mapping[
    Literal["id"] | _type_aliases.DatasetFeature, pl.PolarsDataType
] = MappingProxyType(
    {
        "id": pl.UInt64,
        "MonsoonIntensity": pl.Int64,
        "TopographyDrainage": pl.Int64,
        "RiverManagement": pl.Int64,
        "Deforestation": pl.Int64,
        "Urbanization": pl.Int64,
        "ClimateChange": pl.Int64,
        "DamsQuality": pl.Int64,
        "Siltation": pl.Int64,
        "AgriculturalPractices": pl.Int64,
        "Encroachments": pl.Int64,
        "IneffectiveDisasterPreparedness": pl.Int64,
        "DrainageSystems": pl.Int64,
        "CoastalVulnerability": pl.Int64,
        "Landslides": pl.Int64,
        "Watersheds": pl.Int64,
        "DeterioratingInfrastructure": pl.Int64,
        "PopulationScore": pl.Int64,
        "WetlandLoss": pl.Int64,
        "InadequatePlanning": pl.Int64,
        "PoliticalFactors": pl.Int64,
    }
)
TARGET_FEATURE = "FloodProbability"
