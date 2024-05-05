"""Type aliases used throughout the package."""

from typing import Literal, Protocol, TypeAlias, TypeVar


class _DataClass(Protocol):
    __dataclass_fields__ = ()


DataClass = TypeVar("DataClass", bound=_DataClass)

Dataset: TypeAlias = Literal["train", "test"]
DatasetFeature: TypeAlias = Literal[
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]
