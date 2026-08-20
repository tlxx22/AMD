"""Strict checks against the server's audited official UrbanEV data."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from utils.dataloader_urbanev import (
    EXPECTED_NODE_ORDER_SHA256,
    EXPECTED_TIMESTAMP_ORDER_SHA256,
    UrbanEVDataContractError,
    UrbanEVFoldPreprocessor,
    UrbanEVRawData,
)
from utils.feature_schema import (
    CANONICAL_FEATURE_NAMES,
    EXCLUDED_V1_FEATURES,
    FEATURE_PRESETS,
    WEATHER_FIELD_MAP,
)


EXPECTED_SOURCE_HASHES = {
    "volume.csv": "a55a095ce75af33c59aece2643d5d71b5cd5a0dc73bb97bc553f0a48f40ace32",
    "e_price.csv": "0076d03b8e400c3e911789e2c7ffb7dd0d44a4414247ead676b508def95bcef4",
    "s_price.csv": "d125783e042024157f38d1749232696ea2aa893c61fc31672a3c54374498d3dc",
    "weather_central.csv": "da8c16dcc6a25eadc97ca062998b5dbb01efbb4569efdd693ac98fb5bbc6d065",
    "adj.csv": "93100d3b042086159387ec069efbaf411b90298cdf8a7ada64de214c6bdb5c00",
    "distance.csv": "3630642ddce0e4aac440804c134f3424614ce2bd34fc7bcadd1bc1a3de0d303e",
    "inf.csv": "03c9830965e9e99b29adfb8cceed0eba98d37631f514273cb3fe61f80d63de7c",
}


class UrbanEVDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "data" / "UrbanEV" / "data"
        cls.raw = UrbanEVRawData.load(root)
        cls.bundle = UrbanEVFoldPreprocessor(cls.raw).fit_transform(1, "F4")

    def test_actual_source_dimensions_time_axis_and_fingerprints(self):
        raw = self.raw
        self.assertEqual(raw.num_timestamps, 4344)
        self.assertEqual(raw.num_nodes, 275)
        self.assertEqual(raw.timestamps[0], pd.Timestamp("2022-09-01 00:00:00"))
        self.assertEqual(raw.timestamps[-1], pd.Timestamp("2023-02-28 23:00:00"))
        self.assertIsNone(raw.timestamps.tz)
        self.assertTrue(raw.timestamps.is_monotonic_increasing)
        self.assertTrue(raw.timestamps.is_unique)
        deltas = raw.timestamps[1:].asi8 - raw.timestamps[:-1].asi8
        self.assertTrue(np.all(deltas == pd.Timedelta(hours=1).value))
        self.assertEqual(raw.timestamp_order_sha256, EXPECTED_TIMESTAMP_ORDER_SHA256)
        self.assertEqual(raw.node_order_sha256, EXPECTED_NODE_ORDER_SHA256)
        self.assertEqual(len(raw.data_fingerprint), 64)
        self.assertEqual(raw.timezone, "unknown")
        self.assertEqual(raw.timestamp_semantics, "naive_wall_clock")

    def test_actual_source_hashes_and_first_version_file_scope(self):
        raw = self.raw
        self.assertEqual(raw.file_sha256, EXPECTED_SOURCE_HASHES)
        self.assertEqual(set(raw.loaded_files), set(EXPECTED_SOURCE_HASHES))
        for excluded_file in (
            "weather_airport.csv",
            "poi.csv",
            "occupancy.csv",
            "duration.csv",
            "volume-11kW.csv",
        ):
            with self.subTest(excluded_file=excluded_file):
                self.assertNotIn(excluded_file, raw.loaded_files)
        self.assertTrue(raw.graph_headers_verified)
        self.assertTrue(raw.inf_node_coverage_verified)

    def test_weather_central_mapping_calendar_and_exclusions(self):
        raw = self.raw
        self.assertEqual(raw.weather_source, "weather_central.csv")
        self.assertEqual(raw.weather_raw_fields, ("T", "P", "U"))
        self.assertEqual(raw.weather_feature_names, ("Ta", "P", "h"))
        self.assertEqual(dict(WEATHER_FIELD_MAP), {"T": "Ta", "P": "P", "U": "h"})
        self.assertEqual(
            raw.weather_available_fields, ("T", "P0", "P", "U", "nRAIN", "Td")
        )
        self.assertIn("nRAIN", EXCLUDED_V1_FEATURES)
        self.assertIn("weather_airport", EXCLUDED_V1_FEATURES)
        self.assertNotIn("nRAIN", CANONICAL_FEATURE_NAMES)
        self.assertNotIn("P0", CANONICAL_FEATURE_NAMES)
        self.assertNotIn("Td", CANONICAL_FEATURE_NAMES)

        hour_sin, hour_cos, weekday_sin, weekday_cos, is_weekend = raw.calendar[0]
        self.assertAlmostEqual(hour_sin, 0.0, places=12)
        self.assertAlmostEqual(hour_cos, 1.0, places=12)
        self.assertAlmostEqual(weekday_sin, np.sin(2.0 * np.pi * 3.0 / 7.0))
        self.assertAlmostEqual(weekday_cos, np.cos(2.0 * np.pi * 3.0 / 7.0))
        self.assertEqual(is_weekend, 0.0)

    def test_actual_canonical_schema_is_eleven_channels_and_ordered(self):
        bundle = self.bundle
        self.assertEqual(FEATURE_PRESETS["F4"], CANONICAL_FEATURE_NAMES)
        self.assertEqual(bundle.feature_names, CANONICAL_FEATURE_NAMES)
        self.assertEqual(bundle.target_idx, 0)
        self.assertEqual(bundle.features.shape, (720, 275, 11))
        self.assertEqual(bundle.features.dtype, np.float32)
        self.assertEqual(bundle.target.dtype, np.float32)
        self.assertTrue(np.isfinite(bundle.features).all())
        self.assertTrue(np.isfinite(bundle.target).all())
        self.assertEqual(len(bundle.feature_schema_fingerprint), 64)
        self.assertEqual(len(bundle.preprocessing_state_fingerprint), 64)

    def test_strict_validation_rejects_gaps_and_nonfinite_values_without_repair(self):
        node_ids = ("001", "002")
        base = np.ones((3, 2), dtype=np.float64)
        weather = np.ones((3, 3), dtype=np.float64)
        with self.assertRaisesRegex(UrbanEVDataContractError, "non-hourly"):
            UrbanEVRawData.from_arrays(
                timestamps=pd.to_datetime(
                    [
                        "2022-01-01 00:00",
                        "2022-01-01 01:00",
                        "2022-01-01 03:00",
                    ]
                ),
                node_ids=node_ids,
                volume=base,
                e_price=base,
                s_price=base,
                weather_central=weather,
            )
        bad = base.copy()
        bad[1, 0] = np.nan
        with self.assertRaisesRegex(UrbanEVDataContractError, "NaN/Inf"):
            UrbanEVRawData.from_arrays(
                timestamps=pd.date_range("2022-01-01", periods=3, freq="h"),
                node_ids=node_ids,
                volume=bad,
                e_price=base,
                s_price=base,
                weather_central=weather,
            )


if __name__ == "__main__":
    unittest.main()
