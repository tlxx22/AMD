"""Canonical UrbanEV feature schemas and deterministic fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


TARGET_NAME = "volume"
WEATHER_SOURCE = "weather_central.csv"
WEATHER_FIELD_MAP: Mapping[str, str] = MappingProxyType(
    {"T": "Ta", "P": "P", "U": "h"}
)

CANONICAL_FEATURE_NAMES = (
    "volume",
    "e_price",
    "s_price",
    "Ta",
    "P",
    "h",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
)

FEATURE_PRESETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "F0": ("volume",),
        "F1": (
            "volume",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "is_weekend",
        ),
        "F2": (
            "volume",
            "e_price",
            "s_price",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "is_weekend",
        ),
        "F3": (
            "volume",
            "Ta",
            "P",
            "h",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "is_weekend",
        ),
        "F4": CANONICAL_FEATURE_NAMES,
    }
)

EXCLUDED_V1_FEATURES = (
    "volume-11kW",
    "occupancy",
    "duration",
    "weather_airport",
    "P0",
    "nRAIN",
    "Td",
    "POI",
    "area",
    "road_length",
    "pile_count",
    "station_count",
    "future_observed_price",
    "future_observed_weather",
)


def deterministic_fingerprint(payload: Any) -> str:
    """Return a stable SHA-256 for a JSON-serializable payload."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FeatureSchema:
    """A validated, immutable feature preset."""

    preset: str
    feature_names: tuple[str, ...]
    target_name: str = TARGET_NAME

    def __post_init__(self) -> None:
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if self.target_name not in self.feature_names:
            raise ValueError(f"target {self.target_name!r} is absent from feature_names")
        unknown = set(self.feature_names).difference(CANONICAL_FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown UrbanEV features: {sorted(unknown)}")
        canonical_subset = tuple(
            name for name in CANONICAL_FEATURE_NAMES if name in self.feature_names
        )
        if self.feature_names != canonical_subset:
            raise ValueError("features must preserve canonical feature order")

    @property
    def target_idx(self) -> int:
        """Index of ``volume`` in this preset."""

        return self.feature_names.index(self.target_name)

    @property
    def fingerprint(self) -> str:
        """Deterministic schema fingerprint."""

        return deterministic_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the stable serialization contract."""

        return {
            "preset": self.preset,
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
            "target_idx": self.target_idx,
            "weather_source": WEATHER_SOURCE,
            "weather_field_map": dict(WEATHER_FIELD_MAP),
            "excluded_v1_features": list(EXCLUDED_V1_FEATURES),
        }


def get_feature_schema(preset: str) -> FeatureSchema:
    """Resolve one of the frozen F0--F4 UrbanEV presets."""

    key = preset.upper()
    try:
        names = FEATURE_PRESETS[key]
    except KeyError as exc:
        allowed = ", ".join(FEATURE_PRESETS)
        raise ValueError(f"unknown feature preset {preset!r}; expected one of {allowed}") from exc
    return FeatureSchema(preset=key, feature_names=names)
