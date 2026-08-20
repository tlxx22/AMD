"""Strict UrbanEV loading and fold-local, leakage-safe preprocessing.

This module reads the official one-hour, 275-region data without repairing,
sorting, interpolating, or persisting transformed data.  Both public Dataset
views consume the same :class:`UrbanEVFoldBundle` produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from utils.feature_schema import (
    CANONICAL_FEATURE_NAMES,
    FeatureSchema,
    WEATHER_FIELD_MAP,
    WEATHER_SOURCE,
    deterministic_fingerprint,
    get_feature_schema,
)


EXPECTED_TIMESTAMPS = 4344
EXPECTED_NODES = 275
EXPECTED_TIMESTAMP_ORDER_SHA256 = (
    "35b37018ba38a902e856e5edc6a9640dc144b276c3e535ea01260635a27a8677"
)
EXPECTED_NODE_ORDER_SHA256 = (
    "fd1557ca6b1a61c26e1ca16a6229a3aeb9c4bda5b731bd8db56188bda7509299"
)
ALLOWED_HORIZONS = (3, 6, 9, 12)
SPLIT_NAMES = ("train", "validation", "test")


class UrbanEVDataContractError(ValueError):
    """Raised when source data violates a frozen UrbanEV contract."""


def sequence_sha256(values: Sequence[str]) -> str:
    """Hash an ordered string sequence using the M1-audit serialization."""

    payload = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _read_csv_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            return tuple(next(csv.reader(handle)))
        except StopIteration as exc:
            raise UrbanEVDataContractError(f"empty CSV file: {path}") from exc


def _require_files(data_root: Path, names: Iterable[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in names:
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(f"required UrbanEV file is missing: {path}")
        paths[name] = path
    return paths


def _parse_timestamps(
    raw_values: Sequence[Any], *, source: str
) -> tuple[pd.DatetimeIndex, tuple[str, ...]]:
    raw_series = pd.Series(raw_values, dtype="string")
    if raw_series.isna().any():
        raise UrbanEVDataContractError(f"{source}: timestamp column contains missing values")
    try:
        parsed = pd.DatetimeIndex(pd.to_datetime(raw_series, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise UrbanEVDataContractError(
            f"{source}: timestamp parsing failed: {exc}"
        ) from exc
    if parsed.tz is not None:
        raise UrbanEVDataContractError(
            f"{source}: timestamps must remain timezone-naive wall-clock values"
        )
    if parsed.hasnans:
        raise UrbanEVDataContractError(f"{source}: parsed timestamps contain NaT")
    if not parsed.is_monotonic_increasing:
        raise UrbanEVDataContractError(
            f"{source}: timestamps are not strictly ordered; automatic sorting is forbidden"
        )
    if not parsed.is_unique:
        raise UrbanEVDataContractError(f"{source}: duplicate timestamps are forbidden")
    if len(parsed) > 1:
        deltas = parsed[1:].asi8 - parsed[:-1].asi8
        expected_delta = pd.Timedelta(hours=1).value
        if not np.all(deltas == expected_delta):
            bad = int(np.flatnonzero(deltas != expected_delta)[0])
            raise UrbanEVDataContractError(
                f"{source}: non-hourly interval between positions {bad} and {bad + 1}; "
                "automatic filling is forbidden"
            )
    normalized = tuple(timestamp.isoformat() for timestamp in parsed)
    return parsed, normalized


def _as_finite_matrix(
    values: Any, *, source: str, expected_shape: tuple[int, int]
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise UrbanEVDataContractError(f"{source}: non-numeric value encountered") from exc
    if array.shape != expected_shape:
        raise UrbanEVDataContractError(
            f"{source}: expected shape {expected_shape}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        bad_count = int(array.size - np.isfinite(array).sum())
        raise UrbanEVDataContractError(
            f"{source}: found {bad_count} NaN/Inf values; repair is forbidden"
        )
    return np.array(array, dtype=np.float64, copy=True)


def _calendar_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    hours = timestamps.hour.to_numpy(dtype=np.float64)
    weekdays = timestamps.weekday.to_numpy(dtype=np.float64)
    radians_hour = 2.0 * np.pi * hours / 24.0
    radians_weekday = 2.0 * np.pi * weekdays / 7.0
    return np.column_stack(
        (
            np.sin(radians_hour),
            np.cos(radians_hour),
            np.sin(radians_weekday),
            np.cos(radians_weekday),
            np.isin(weekdays, (5.0, 6.0)).astype(np.float64),
        )
    )


@dataclass(frozen=True)
class UrbanEVRawData:
    """Validated in-memory representation of the M1 UrbanEV source data."""

    timestamps: pd.DatetimeIndex
    timestamp_strings: tuple[str, ...]
    node_ids: tuple[str, ...]
    volume: np.ndarray
    e_price: np.ndarray
    s_price: np.ndarray
    weather_central: np.ndarray
    calendar: np.ndarray
    timestamp_order_sha256: str
    node_order_sha256: str
    data_fingerprint: str
    file_sha256_items: tuple[tuple[str, str], ...]
    loaded_files: tuple[str, ...]
    weather_available_fields: tuple[str, ...]
    graph_headers_verified: bool
    inf_node_coverage_verified: bool
    data_root: Path | None = None
    weather_source: str = WEATHER_SOURCE
    weather_raw_fields: tuple[str, ...] = ("T", "P", "U")
    weather_feature_names: tuple[str, ...] = ("Ta", "P", "h")
    timezone: str = "unknown"
    timestamp_semantics: str = "naive_wall_clock"

    @property
    def num_timestamps(self) -> int:
        return len(self.timestamps)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def file_sha256(self) -> dict[str, str]:
        return dict(self.file_sha256_items)

    @classmethod
    def load(
        cls,
        data_root: str | Path,
        *,
        expected_timestamp_hash: str = EXPECTED_TIMESTAMP_ORDER_SHA256,
        expected_node_hash: str = EXPECTED_NODE_ORDER_SHA256,
    ) -> "UrbanEVRawData":
        """Load and strictly validate the audited official source files."""

        root = Path(data_root).expanduser().resolve()
        names = (
            "volume.csv",
            "e_price.csv",
            "s_price.csv",
            WEATHER_SOURCE,
            "adj.csv",
            "distance.csv",
            "inf.csv",
        )
        paths = _require_files(root, names)

        volume_header = _read_csv_header(paths["volume.csv"])
        if not volume_header or volume_header[0] != "time":
            raise UrbanEVDataContractError("volume.csv: first column must be 'time'")
        node_ids = tuple(volume_header[1:])
        if len(node_ids) != EXPECTED_NODES:
            raise UrbanEVDataContractError(
                f"volume.csv: expected {EXPECTED_NODES} nodes, got {len(node_ids)}"
            )
        if any(node == "" for node in node_ids) or len(set(node_ids)) != len(node_ids):
            raise UrbanEVDataContractError(
                "volume.csv: node headers must be non-empty and unique"
            )

        matrices: dict[str, np.ndarray] = {}
        reference_timestamps: pd.DatetimeIndex | None = None
        timestamp_strings: tuple[str, ...] | None = None
        for filename, key in (
            ("volume.csv", "volume"),
            ("e_price.csv", "e_price"),
            ("s_price.csv", "s_price"),
        ):
            header = _read_csv_header(paths[filename])
            if header != ("time", *node_ids):
                raise UrbanEVDataContractError(
                    f"{filename}: node columns differ in order from volume.csv"
                )
            frame = pd.read_csv(paths[filename], dtype={"time": "string"})
            if tuple(str(column) for column in frame.columns) != header:
                raise UrbanEVDataContractError(f"{filename}: parsed header changed unexpectedly")
            parsed, normalized = _parse_timestamps(frame["time"], source=filename)
            if reference_timestamps is None:
                reference_timestamps = parsed
                timestamp_strings = normalized
            elif not np.array_equal(parsed.asi8, reference_timestamps.asi8):
                raise UrbanEVDataContractError(
                    f"{filename}: timestamp order differs from volume.csv"
                )
            matrices[key] = _as_finite_matrix(
                frame.loc[:, list(node_ids)].to_numpy(),
                source=filename,
                expected_shape=(len(parsed), len(node_ids)),
            )

        assert reference_timestamps is not None and timestamp_strings is not None
        if len(reference_timestamps) != EXPECTED_TIMESTAMPS:
            raise UrbanEVDataContractError(
                f"expected {EXPECTED_TIMESTAMPS} hourly timestamps, "
                f"got {len(reference_timestamps)}"
            )

        weather_header = _read_csv_header(paths[WEATHER_SOURCE])
        required_weather = ("time", "T", "P", "U")
        missing_weather = set(required_weather).difference(weather_header)
        if missing_weather:
            raise UrbanEVDataContractError(
                f"{WEATHER_SOURCE}: missing required fields {sorted(missing_weather)}"
            )
        weather_frame = pd.read_csv(
            paths[WEATHER_SOURCE],
            usecols=list(required_weather),
            dtype={"time": "string"},
        )
        weather_timestamps, _ = _parse_timestamps(
            weather_frame["time"], source=WEATHER_SOURCE
        )
        if not np.array_equal(weather_timestamps.asi8, reference_timestamps.asi8):
            raise UrbanEVDataContractError(
                f"{WEATHER_SOURCE}: timestamp order differs from volume.csv"
            )
        weather = _as_finite_matrix(
            weather_frame.loc[:, ["T", "P", "U"]].to_numpy(),
            source=WEATHER_SOURCE,
            expected_shape=(len(reference_timestamps), 3),
        )

        for graph_name in ("adj.csv", "distance.csv"):
            graph_header = list(_read_csv_header(paths[graph_name]))
            if graph_header and graph_header[-1] == "":
                graph_header.pop()
            if tuple(graph_header) != node_ids:
                raise UrbanEVDataContractError(
                    f"{graph_name}: header does not match canonical node order"
                )
            if any(value == "" for value in graph_header):
                raise UrbanEVDataContractError(
                    f"{graph_name}: unexpected empty node label in header"
                )

        inf_header = _read_csv_header(paths["inf.csv"])
        if "TAZID" not in inf_header:
            raise UrbanEVDataContractError("inf.csv: missing TAZID column")
        inf_nodes = pd.read_csv(
            paths["inf.csv"], usecols=["TAZID"], dtype={"TAZID": "string"}
        )["TAZID"]
        if inf_nodes.isna().any():
            raise UrbanEVDataContractError("inf.csv: TAZID contains missing values")
        inf_set = set(inf_nodes.astype(str))
        canonical_set = set(node_ids)
        if inf_set != canonical_set:
            missing = sorted(canonical_set.difference(inf_set))
            extra = sorted(inf_set.difference(canonical_set))
            raise UrbanEVDataContractError(
                f"inf.csv: TAZID coverage mismatch; missing={missing}, extra={extra}"
            )

        timestamp_hash = sequence_sha256(timestamp_strings)
        node_hash = sequence_sha256(node_ids)
        if timestamp_hash != expected_timestamp_hash:
            raise UrbanEVDataContractError(
                "timestamp_order_sha256 mismatch: "
                f"expected {expected_timestamp_hash}, got {timestamp_hash}"
            )
        if node_hash != expected_node_hash:
            raise UrbanEVDataContractError(
                f"node_order_sha256 mismatch: expected {expected_node_hash}, got {node_hash}"
            )

        file_hashes = tuple((name, _file_sha256(paths[name])) for name in names)
        data_fingerprint = deterministic_fingerprint(
            {
                "source_files": dict(file_hashes),
                "timestamp_order_sha256": timestamp_hash,
                "node_order_sha256": node_hash,
                "timestamp_semantics": "naive_wall_clock",
                "weather_source": WEATHER_SOURCE,
                "weather_field_map": dict(WEATHER_FIELD_MAP),
            }
        )
        return cls(
            timestamps=reference_timestamps,
            timestamp_strings=timestamp_strings,
            node_ids=node_ids,
            volume=matrices["volume"],
            e_price=matrices["e_price"],
            s_price=matrices["s_price"],
            weather_central=weather,
            calendar=_calendar_features(reference_timestamps),
            timestamp_order_sha256=timestamp_hash,
            node_order_sha256=node_hash,
            data_fingerprint=data_fingerprint,
            file_sha256_items=file_hashes,
            loaded_files=names,
            weather_available_fields=tuple(weather_header[1:]),
            graph_headers_verified=True,
            inf_node_coverage_verified=True,
            data_root=root,
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        timestamps: Sequence[Any] | pd.DatetimeIndex,
        node_ids: Sequence[str],
        volume: Any,
        e_price: Any,
        s_price: Any,
        weather_central: Any,
    ) -> "UrbanEVRawData":
        """Construct validated in-memory data for deterministic contract tests."""

        if any(not isinstance(node, str) for node in node_ids):
            raise UrbanEVDataContractError(
                "node IDs must be supplied as strings to preserve their exact spelling"
            )
        canonical_nodes = tuple(node_ids)
        if not canonical_nodes or len(set(canonical_nodes)) != len(canonical_nodes):
            raise UrbanEVDataContractError("node IDs must be non-empty and unique")
        parsed, normalized = _parse_timestamps(timestamps, source="in-memory timestamps")
        shape = (len(parsed), len(canonical_nodes))
        arrays = {
            "volume": _as_finite_matrix(volume, source="volume", expected_shape=shape),
            "e_price": _as_finite_matrix(
                e_price, source="e_price", expected_shape=shape
            ),
            "s_price": _as_finite_matrix(
                s_price, source="s_price", expected_shape=shape
            ),
            "weather_central": _as_finite_matrix(
                weather_central,
                source="weather_central",
                expected_shape=(len(parsed), 3),
            ),
        }
        timestamp_hash = sequence_sha256(normalized)
        node_hash = sequence_sha256(canonical_nodes)
        array_hashes = {
            name: _array_sha256(array) for name, array in arrays.items()
        }
        data_fingerprint = deterministic_fingerprint(
            {
                "arrays": array_hashes,
                "timestamp_order_sha256": timestamp_hash,
                "node_order_sha256": node_hash,
                "timestamp_semantics": "naive_wall_clock",
            }
        )
        return cls(
            timestamps=parsed,
            timestamp_strings=normalized,
            node_ids=canonical_nodes,
            volume=arrays["volume"],
            e_price=arrays["e_price"],
            s_price=arrays["s_price"],
            weather_central=arrays["weather_central"],
            calendar=_calendar_features(parsed),
            timestamp_order_sha256=timestamp_hash,
            node_order_sha256=node_hash,
            data_fingerprint=data_fingerprint,
            file_sha256_items=tuple(sorted(array_hashes.items())),
            loaded_files=tuple(),
            weather_available_fields=("T", "P", "U"),
            graph_headers_verified=False,
            inf_node_coverage_verified=False,
        )


@dataclass(frozen=True)
class UrbanEVFoldDefinition:
    """Cumulative-month fold and its integer split boundaries."""

    fold: int
    months: tuple[str, ...]
    fold_length: int
    n_train: int
    n_validation: int
    n_test: int

    def split_slice(self, split: str) -> slice:
        if split == "train":
            return slice(0, self.n_train)
        if split == "validation":
            return slice(self.n_train, self.n_train + self.n_validation)
        if split == "test":
            return slice(self.n_train + self.n_validation, self.fold_length)
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")

    def split_length(self, split: str) -> int:
        selected = self.split_slice(split)
        return int(selected.stop - selected.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "months": list(self.months),
            "fold_length": self.fold_length,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "split_semantics": "split_first_no_cross_split_context",
        }


def build_fold_definition(
    timestamps: pd.DatetimeIndex, fold: int
) -> UrbanEVFoldDefinition:
    """Build the official cumulative-month fold with floor 80/10/rest splits."""

    ordered_months: list[tuple[int, int]] = []
    for timestamp in timestamps:
        value = (int(timestamp.year), int(timestamp.month))
        if not ordered_months or value != ordered_months[-1]:
            if value in ordered_months:
                raise UrbanEVDataContractError(
                    "calendar month reappears non-contiguously in the timestamp sequence"
                )
            ordered_months.append(value)
    if fold < 1 or fold > len(ordered_months):
        raise ValueError(f"fold must be in [1, {len(ordered_months)}], got {fold}")
    selected = set(ordered_months[:fold])
    mask = np.asarray(
        [(int(ts.year), int(ts.month)) in selected for ts in timestamps], dtype=bool
    )
    indices = np.flatnonzero(mask)
    if indices.size == 0 or not np.array_equal(indices, np.arange(indices[-1] + 1)):
        raise UrbanEVDataContractError("cumulative fold is not a contiguous prefix")
    fold_length = int(indices[-1] + 1)
    n_train = int(np.floor(0.8 * fold_length))
    n_validation = int(np.floor(0.1 * fold_length))
    n_test = fold_length - n_train - n_validation
    return UrbanEVFoldDefinition(
        fold=fold,
        months=tuple(f"{year:04d}-{month:02d}" for year, month in ordered_months[:fold]),
        fold_length=fold_length,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
    )


def validate_label_horizon(label_horizon: int) -> None:
    if label_horizon not in ALLOWED_HORIZONS:
        raise ValueError(
            f"label_horizon must be one of {ALLOWED_HORIZONS}, got {label_horizon}"
        )


def window_count(split_length: int, label_horizon: int, history_len: int = 12) -> int:
    """Return the frozen inclusive-last-label window count."""

    if split_length < 0:
        raise ValueError("split_length must be non-negative")
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    validate_label_horizon(label_horizon)
    return max(0, split_length - history_len - label_horizon + 1)


def _standard_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    variance = np.mean(np.square(values - mean), axis=0, dtype=np.float64)
    scale = np.sqrt(variance)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def _minmax_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = values.min(axis=0)
    raw_range = values.max(axis=0) - minimum
    safe_range = np.where(raw_range == 0.0, 1.0, raw_range)
    return minimum, raw_range, safe_range


@dataclass(frozen=True)
class UrbanEVPreprocessingState:
    """Explicit, JSON-serializable fold preprocessing contract."""

    fold: int
    train_start_idx: int
    train_end_idx: int
    train_start_timestamp: str
    train_end_timestamp: str
    node_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    target_idx: int
    volume_mean: np.ndarray
    volume_scale: np.ndarray
    e_price_min: np.ndarray
    e_price_range: np.ndarray
    e_price_safe_range: np.ndarray
    s_price_min: np.ndarray
    s_price_range: np.ndarray
    s_price_safe_range: np.ndarray
    weather_mean: np.ndarray
    weather_scale: np.ndarray
    timezone: str
    timestamp_semantics: str
    timestamp_order_sha256: str
    node_order_sha256: str
    data_fingerprint: str
    feature_schema_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_start_idx": self.train_start_idx,
            "train_end_idx": self.train_end_idx,
            "train_end_idx_semantics": "exclusive",
            "train_start_timestamp": self.train_start_timestamp,
            "train_end_timestamp": self.train_end_timestamp,
            "node_ids": list(self.node_ids),
            "feature_names": list(self.feature_names),
            "target_idx": self.target_idx,
            "volume_mean": self.volume_mean.tolist(),
            "volume_scale": self.volume_scale.tolist(),
            "e_price_min": self.e_price_min.tolist(),
            "e_price_range": self.e_price_range.tolist(),
            "e_price_safe_range": self.e_price_safe_range.tolist(),
            "s_price_min": self.s_price_min.tolist(),
            "s_price_range": self.s_price_range.tolist(),
            "s_price_safe_range": self.s_price_safe_range.tolist(),
            "weather_mean": self.weather_mean.tolist(),
            "weather_scale": self.weather_scale.tolist(),
            "weather_stats_scope": "city_fields_before_node_broadcast",
            "timezone": self.timezone,
            "timestamp_semantics": self.timestamp_semantics,
            "timestamp_order_sha256": self.timestamp_order_sha256,
            "node_order_sha256": self.node_order_sha256,
            "data_fingerprint": self.data_fingerprint,
            "feature_schema_fingerprint": self.feature_schema_fingerprint,
            "fit_scope": "current_fold_raw_train_time_slice_only",
        }

    @property
    def fingerprint(self) -> str:
        return deterministic_fingerprint(self.to_dict())


@dataclass(frozen=True)
class UrbanEVFoldBundle:
    """One transformed fold shared by temporal and graph Dataset views."""

    raw: UrbanEVRawData
    fold_definition: UrbanEVFoldDefinition
    schema: FeatureSchema
    features: np.ndarray
    target: np.ndarray
    preprocessing_state: UrbanEVPreprocessingState

    @property
    def node_ids(self) -> tuple[str, ...]:
        return self.raw.node_ids

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self.raw.timestamps[: self.fold_definition.fold_length]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.schema.feature_names

    @property
    def target_idx(self) -> int:
        return self.schema.target_idx

    @property
    def feature_schema_fingerprint(self) -> str:
        return self.schema.fingerprint

    @property
    def preprocessing_state_fingerprint(self) -> str:
        return self.preprocessing_state.fingerprint

    def split_slice(self, split: str) -> slice:
        return self.fold_definition.split_slice(split)

    def split_features(self, split: str) -> np.ndarray:
        return self.features[self.split_slice(split)]

    def split_target(self, split: str) -> np.ndarray:
        return self.target[self.split_slice(split)]

    def split_timestamps(self, split: str) -> pd.DatetimeIndex:
        return self.timestamps[self.split_slice(split)]

    def transform_target(
        self, values: np.ndarray | torch.Tensor, node_position: Any = None
    ) -> np.ndarray | torch.Tensor:
        """Apply the fold's per-node volume StandardScaler."""

        return self._target_operation(values, node_position=node_position, inverse=False)

    def inverse_transform_target(
        self, values: np.ndarray | torch.Tensor, node_position: Any = None
    ) -> np.ndarray | torch.Tensor:
        """Invert temporal ``[...,1]`` or graph ``[...,N,1]`` targets."""

        return self._target_operation(values, node_position=node_position, inverse=True)

    def _target_operation(
        self,
        values: np.ndarray | torch.Tensor,
        *,
        node_position: Any,
        inverse: bool,
    ) -> np.ndarray | torch.Tensor:
        n_nodes = len(self.node_ids)
        shape = tuple(values.shape)
        if not shape:
            raise ValueError("target values must have at least one dimension")

        if node_position is not None:
            positions = np.asarray(node_position)
            if not np.issubdtype(positions.dtype, np.integer):
                raise TypeError("node_position must contain integer positions")
            if np.any(positions < 0) or np.any(positions >= n_nodes):
                raise IndexError("node_position is outside canonical node order")
            mean_np = self.preprocessing_state.volume_mean[positions]
            scale_np = self.preprocessing_state.volume_scale[positions]
            while np.ndim(mean_np) < len(shape):
                mean_np = np.expand_dims(mean_np, axis=-1)
                scale_np = np.expand_dims(scale_np, axis=-1)
        elif shape[-1] == n_nodes:
            stat_shape = (1,) * (len(shape) - 1) + (n_nodes,)
            mean_np = self.preprocessing_state.volume_mean.reshape(stat_shape)
            scale_np = self.preprocessing_state.volume_scale.reshape(stat_shape)
        elif len(shape) >= 2 and shape[-2:] == (n_nodes, 1):
            stat_shape = (1,) * (len(shape) - 2) + (n_nodes, 1)
            mean_np = self.preprocessing_state.volume_mean.reshape(stat_shape)
            scale_np = self.preprocessing_state.volume_scale.reshape(stat_shape)
        else:
            raise ValueError(
                "cannot infer node axis; provide node_position for temporal targets or "
                f"use a graph target ending in [{n_nodes}] or [{n_nodes},1]"
            )

        if isinstance(values, torch.Tensor):
            if not values.is_floating_point():
                raise TypeError("target tensor must use a floating dtype")
            mean = torch.as_tensor(mean_np, dtype=values.dtype, device=values.device)
            scale = torch.as_tensor(scale_np, dtype=values.dtype, device=values.device)
        else:
            values = np.asarray(values)
            mean = mean_np
            scale = scale_np
        if inverse:
            return values * scale + mean
        return (values - mean) / scale


class UrbanEVFoldPreprocessor:
    """Fit train-only fold scalers and construct an immutable shared bundle."""

    def __init__(self, raw: UrbanEVRawData) -> None:
        self.raw = raw

    def fit_transform(self, fold: int, preset: str = "F4") -> UrbanEVFoldBundle:
        definition = build_fold_definition(self.raw.timestamps, fold)
        schema = get_feature_schema(preset)
        train = slice(0, definition.n_train)
        if definition.n_train <= 0:
            raise UrbanEVDataContractError("fold train split is empty")

        volume_mean, volume_scale = _standard_stats(self.raw.volume[train])
        e_min, e_range, e_safe = _minmax_stats(self.raw.e_price[train])
        s_min, s_range, s_safe = _minmax_stats(self.raw.s_price[train])
        weather_mean, weather_scale = _standard_stats(
            self.raw.weather_central[train]
        )

        fold_end = definition.fold_length
        volume = (self.raw.volume[:fold_end] - volume_mean) / volume_scale
        e_price = (self.raw.e_price[:fold_end] - e_min) / e_safe
        s_price = (self.raw.s_price[:fold_end] - s_min) / s_safe
        weather_city = (
            self.raw.weather_central[:fold_end] - weather_mean
        ) / weather_scale

        n_nodes = self.raw.num_nodes
        weather = np.broadcast_to(
            weather_city[:, np.newaxis, :], (fold_end, n_nodes, 3)
        )
        calendar = np.broadcast_to(
            self.raw.calendar[:fold_end, np.newaxis, :], (fold_end, n_nodes, 5)
        )
        canonical: Mapping[str, np.ndarray] = {
            "volume": volume,
            "e_price": e_price,
            "s_price": s_price,
            "Ta": weather[..., 0],
            "P": weather[..., 1],
            "h": weather[..., 2],
            "hour_sin": calendar[..., 0],
            "hour_cos": calendar[..., 1],
            "weekday_sin": calendar[..., 2],
            "weekday_cos": calendar[..., 3],
            "is_weekend": calendar[..., 4],
        }
        if tuple(canonical) != CANONICAL_FEATURE_NAMES:
            raise RuntimeError("internal canonical feature order changed unexpectedly")
        features = np.stack(
            [canonical[name] for name in schema.feature_names], axis=-1
        ).astype(np.float32, copy=False)
        target = volume.astype(np.float32, copy=False)
        if features.shape != (fold_end, n_nodes, len(schema.feature_names)):
            raise RuntimeError(f"unexpected transformed feature shape: {features.shape}")
        if schema.target_idx != schema.feature_names.index("volume"):
            raise RuntimeError("target index does not select volume")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise UrbanEVDataContractError("preprocessing produced NaN/Inf values")

        state = UrbanEVPreprocessingState(
            fold=fold,
            train_start_idx=0,
            train_end_idx=definition.n_train,
            train_start_timestamp=self.raw.timestamp_strings[0],
            train_end_timestamp=self.raw.timestamp_strings[definition.n_train - 1],
            node_ids=self.raw.node_ids,
            feature_names=schema.feature_names,
            target_idx=schema.target_idx,
            volume_mean=volume_mean,
            volume_scale=volume_scale,
            e_price_min=e_min,
            e_price_range=e_range,
            e_price_safe_range=e_safe,
            s_price_min=s_min,
            s_price_range=s_range,
            s_price_safe_range=s_safe,
            weather_mean=weather_mean,
            weather_scale=weather_scale,
            timezone=self.raw.timezone,
            timestamp_semantics=self.raw.timestamp_semantics,
            timestamp_order_sha256=self.raw.timestamp_order_sha256,
            node_order_sha256=self.raw.node_order_sha256,
            data_fingerprint=self.raw.data_fingerprint,
            feature_schema_fingerprint=schema.fingerprint,
        )
        return UrbanEVFoldBundle(
            raw=self.raw,
            fold_definition=definition,
            schema=schema,
            features=features,
            target=target,
            preprocessing_state=state,
        )


def load_urbanev_raw(data_root: str | Path) -> UrbanEVRawData:
    """Public convenience wrapper for strict source loading."""

    return UrbanEVRawData.load(data_root)


def build_urbanev_fold_bundle(
    data_root: str | Path, *, fold: int, preset: str = "F4"
) -> UrbanEVFoldBundle:
    """Load source data and construct one train-only-preprocessed fold."""

    raw = load_urbanev_raw(data_root)
    return UrbanEVFoldPreprocessor(raw).fit_transform(fold=fold, preset=preset)
