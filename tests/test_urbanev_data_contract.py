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
    def setUp(self):
        # This method needs no real fixture; do not make its synthetic assertions
        # depend on unrelated full-data setup.
        if self._testMethodName == "test_strict_validation_rejects_gaps_and_nonfinite_values_without_repair":
            return
        root = Path(__file__).resolve().parents[1] / "data" / "UrbanEV" / "data"
        self.raw = UrbanEVRawData.load(root)
        self.bundle = UrbanEVFoldPreprocessor(self.raw).fit_transform(1, "F4")

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




class UrbanEVRestrictedPrefixTests(unittest.TestCase):
    """The parser and scaler never receive synthetic test observation rows."""

    def test_forbidden_suffix_never_read_or_parsed_and_bundle_rejects_test(self):
        import tempfile
        from io import StringIO
        from unittest import mock
        import utils.dataloader_urbanev as urban
        from utils.temporal_region_dataset import TemporalRegionDataset
        stop = 3909
        node_ids = ("001", "002")
        clock = pd.date_range("2022-09-01", periods=4344, freq="h")
        with tempfile.TemporaryDirectory(prefix="urban-prefix-sentinel-") as directory:
            root = Path(directory)
            time_values = clock[:stop].strftime("%Y-%m-%d %H:%M:%S")
            t = np.arange(stop, dtype=float)
            for name in ("volume.csv", "e_price.csv", "s_price.csv"):
                pd.DataFrame({"time": time_values, "001": t + 1, "002": 2*t + 5}).to_csv(root/name, index=False)
                with (root/name).open("a") as handle:
                    handle.write("FORBIDDEN_TEST_OBSERVATION,must,not,parse\n")
            pd.DataFrame({"time": time_values, "T": t/100, "P": t/100+950, "U": t/200+50}).to_csv(root/"weather_central.csv", index=False)
            with (root/"weather_central.csv").open("a") as handle:
                handle.write("FORBIDDEN_TEST_OBSERVATION,must,not,parse\n")
            for name in ("adj.csv", "distance.csv"):
                (root/name).write_text("001,002\n", encoding="utf-8")
            (root/"inf.csv").write_text("TAZID\n001\n002\n", encoding="utf-8")
            observed = {"forbidden_read": 0, "forbidden_parse": 0, "parser_prefixes": 0}
            original_open, original_read = Path.open, pd.read_csv
            observations = {"volume.csv", "e_price.csv", "s_price.csv", "weather_central.csv"}

            class PrefixOnlyFile:
                def __init__(self, handle):
                    self.handle, self.rows = handle, 0
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    self.handle.close()
                def __iter__(self):
                    return self
                def __next__(self):
                    if self.rows >= stop + 1:
                        observed["forbidden_read"] += 1
                        raise AssertionError("text reader attempted test row")
                    line = next(self.handle)
                    self.rows += 1
                    return line
                def readline(self, *args):
                    return next(self)
                def read(self, *args):
                    raise AssertionError("unbounded observation text read")

            def safe_open(path, mode="r", *args, **kwargs):
                handle = original_open(path, mode, *args, **kwargs)
                if path.parent == root and path.name in observations and "b" not in mode:
                    return PrefixOnlyFile(handle)
                return handle

            def safe_parse(source, *args, **kwargs):
                if isinstance(source, (str, Path)) and Path(source).name in observations:
                    observed["forbidden_parse"] += 1
                    raise AssertionError("parser received unrestricted observation path")
                if isinstance(source, StringIO):
                    observed["parser_prefixes"] += 1
                    self.assertNotIn("FORBIDDEN_TEST_OBSERVATION", source.getvalue())
                return original_read(source, *args, **kwargs)

            with mock.patch.object(urban, "EXPECTED_NODES", 2), \
                    mock.patch.object(Path, "open", safe_open), \
                    mock.patch.object(pd, "read_csv", side_effect=safe_parse):
                raw = urban.UrbanEVRawData.load(
                    root, train_validation_fold=6,
                    expected_node_hash=urban.sequence_sha256(node_ids))
                bundle = urban.UrbanEVFoldPreprocessor(raw).fit_transform(6, "F4")
                for h in (3, 6, 9, 12):
                    for split in ("train", "validation"):
                        view = TemporalRegionDataset(bundle, split=split, label_horizon=h)
                        view[0]
                        view[len(view)-1]
                    with self.assertRaisesRegex(ValueError, "forbidden"):
                        TemporalRegionDataset(bundle, split="test", label_horizon=h)
            self.assertEqual(observed["forbidden_read"], 0)
            self.assertEqual(observed["forbidden_parse"], 0)
            self.assertEqual(observed["parser_prefixes"], 5)
            self.assertEqual(raw.num_timestamps, stop)
            self.assertEqual(bundle.features.shape, (stop, 2, 11))
            np.testing.assert_allclose(bundle.preprocessing_state.volume_mean,
                                       [np.mean(t[:3475]+1), np.mean(2*t[:3475]+5)])
            self.assertEqual(raw.timestamp_order_sha256, EXPECTED_TIMESTAMP_ORDER_SHA256)
            self.assertEqual(len(raw.file_sha256), 7)
            with self.assertRaisesRegex(ValueError, "fold"):
                urban.UrbanEVFoldPreprocessor(raw).fit_transform(5, "F4")


class UrbanEVDefaultRestrictedCompatibilityTests(unittest.TestCase):
    """Production CSV parser/preprocessor on full synthetic frozen-shape inputs."""
    STAT_FIELDS = (
        "volume_mean", "volume_scale", "e_price_min", "e_price_range",
        "e_price_safe_range", "s_price_min", "s_price_range", "s_price_safe_range",
        "weather_mean", "weather_scale",
    )

    @classmethod
    def setUpClass(cls):
        import tempfile
        from urbanev_synthetic_fixture import write_synthetic_urbanev
        cls.temporary = tempfile.TemporaryDirectory(prefix="urban-default-prefix-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.reference = write_synthetic_urbanev(cls.root / "clean")
        cls.full_raw = UrbanEVRawData.load(cls.root / "clean")
        cls.prefix_raw = UrbanEVRawData.load(cls.root / "clean", train_validation_fold=6)
        cls.full = UrbanEVFoldPreprocessor(cls.full_raw).fit_transform(6, "F4")
        cls.prefix = UrbanEVFoldPreprocessor(cls.prefix_raw).fit_transform(6, "F4")

    def _assert_statistics(self, state, reference, train_end):
        expected = {}
        for source, prefix in (("volume", "volume"), ("weather_central", "weather")):
            values = reference[source][:train_end]
            scale = np.std(values, axis=0, ddof=0)
            expected[prefix + "_mean"] = np.mean(values, axis=0)
            expected[prefix + "_scale"] = np.where(scale == 0, 1.0, scale)
        for source in ("e_price", "s_price"):
            values = reference[source][:train_end]
            low, high = np.min(values, axis=0), np.max(values, axis=0)
            expected[source + "_min"] = low
            expected[source + "_range"] = high - low
            expected[source + "_safe_range"] = np.where(high == low, 1.0, high - low)
        self.assertEqual(set(expected), set(self.STAT_FIELDS))
        self.assertEqual(state.train_start_idx, 0)
        self.assertEqual(state.train_end_idx, train_end)
        for field, wanted in expected.items():
            with self.subTest(statistic=field):
                actual = getattr(state, field)
                self.assertEqual(actual.shape, wanted.shape)
                np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-12)

    def _assert_stats_equal(self, left, right):
        for field in self.STAT_FIELDS:
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))

    def _assert_shared_values(self, left, right, stop=3909):
        self.assertEqual(left.node_ids, right.node_ids)
        self.assertEqual(left.feature_names, right.feature_names)
        self.assertEqual(left.target_idx, right.target_idx)
        np.testing.assert_array_equal(left.raw.timestamps[:stop], right.raw.timestamps[:stop])
        for field in ("volume", "e_price", "s_price", "weather_central", "calendar"):
            np.testing.assert_array_equal(getattr(left.raw, field)[:stop], getattr(right.raw, field)[:stop])
        np.testing.assert_array_equal(left.features[:stop], right.features[:stop])
        np.testing.assert_array_equal(left.target[:stop], right.target[:stop])

    def test_default_and_restricted_csv_share_train_validation_contract(self):
        import torch
        from utils.temporal_region_dataset import TemporalRegionDataset
        full, prefix = self.full, self.prefix
        self.assertIsNone(self.full_raw.restricted_fold)
        self.assertEqual(self.prefix_raw.restricted_fold.fold, 6)
        self.assertEqual(self.full_raw.num_timestamps, 4344)
        self.assertEqual(self.prefix_raw.num_timestamps, 3909)
        self._assert_shared_values(full, prefix)
        self._assert_stats_equal(full.preprocessing_state, prefix.preprocessing_state)
        self.assertEqual(full.node_ids, self.reference["node_ids"])
        for field in ("volume", "e_price", "s_price", "weather_central"):
            np.testing.assert_array_equal(getattr(self.full_raw, field), self.reference[field])
        for horizon in (3, 6, 9, 12):
            for split in ("train", "validation"):
                with self.subTest(horizon=horizon, split=split):
                    left = TemporalRegionDataset(full, split=split, label_horizon=horizon)
                    right = TemporalRegionDataset(prefix, split=split, label_horizon=horizon)
                    self.assertEqual(len(left), len(right))
                    self.assertEqual(full.split_slice(split), prefix.split_slice(split))
                    np.testing.assert_array_equal(full.split_timestamps(split), prefix.split_timestamps(split))
                    for index in (0, 274, 275, len(left)-275, len(left)-1):
                        x, y = left[index]
                        other_x, other_y = right[index]
                        self.assertTrue(torch.equal(x, other_x))
                        self.assertTrue(torch.equal(y, other_y))
                        meta = left.metadata(index)
                        self.assertEqual(meta, right.metadata(index))
                        self.assertEqual(meta["label_idx"], meta["window_start_idx"] + 12 + horizon - 1)
                        node = meta["node_position"]
                        self.assertEqual(meta["node_id"], full.node_ids[node])
                        for bundle in (full, prefix):
                            restored = bundle.inverse_transform_target(y, node_position=node)
                            self.assertAlmostEqual(float(restored.item()), self.reference["volume"][meta["label_idx"], node], places=5)
                        if index == len(left)-1:
                            self.assertEqual(meta["label_idx"], full.split_slice(split).stop-1)
            self.assertGreater(len(TemporalRegionDataset(full, split="test", label_horizon=horizon)), 0)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                TemporalRegionDataset(prefix, split="test", label_horizon=horizon)
        positions = np.asarray([0, 10, 274])
        values = self.reference["volume"][np.asarray([10, 20, 30]), positions, None]
        for bundle in (full, prefix):
            z = bundle.transform_target(values, node_position=positions)
            np.testing.assert_allclose(bundle.inverse_transform_target(z, node_position=positions), values, rtol=0, atol=1e-10)
        print("URBAN_DEFAULT_PREFIX shared_values=exact horizons=4 nodes=275 first_last_labels=passed")

    def test_csv_statistics_constant_prices_and_independent_folds(self):
        processor = UrbanEVFoldPreprocessor(self.full_raw)
        five = processor.fit_transform(5, "F4")
        before = {field: getattr(five.preprocessing_state, field).copy() for field in self.STAT_FIELDS}
        before_features = five.features.copy()
        six = processor.fit_transform(6, "F4")
        for bundle in (five, six, self.prefix):
            self._assert_statistics(bundle.preprocessing_state, self.reference, bundle.fold_definition.n_train)
        for field in self.STAT_FIELDS:
            np.testing.assert_array_equal(getattr(five.preprocessing_state, field), before[field])
            self.assertFalse(np.shares_memory(getattr(five.preprocessing_state, field), getattr(six.preprocessing_state, field)))
        np.testing.assert_array_equal(five.features, before_features)
        self.assertNotEqual(five.preprocessing_state_fingerprint, six.preprocessing_state_fingerprint)
        self.assertFalse(np.array_equal(five.preprocessing_state.volume_mean, six.preprocessing_state.volume_mean))
        state = six.preprocessing_state
        self.assertEqual(state.e_price_range[0], 0.0)
        self.assertEqual(state.e_price_safe_range[0], 1.0)
        self.assertEqual(state.s_price_range[2], 0.0)
        self.assertEqual(state.s_price_safe_range[2], 1.0)
        for bundle in (six, self.prefix):
            self.assertTrue(np.all(bundle.features[:3475, 0, 1] == 0))
            self.assertTrue(np.all(bundle.features[:3475, 2, 2] == 0))
            self.assertEqual(bundle.features[3475, 0, 1], 2)
            self.assertEqual(bundle.features[3475, 2, 2], -2)
            self.assertTrue(np.isfinite(bundle.features).all())
        print("URBAN_SCALER statistics=10 train_only_reference=passed independent_folds=5,6 constant_price_no_clip=passed")

    def test_csv_test_tail_pollution_preserves_shared_train_validation(self):
        from urbanev_synthetic_fixture import write_synthetic_urbanev
        root = self.root / "test-polluted"
        write_synthetic_urbanev(root, perturb="test")
        for restricted in (False, True):
            with self.subTest(restricted=restricted):
                raw = UrbanEVRawData.load(root, **({"train_validation_fold": 6} if restricted else {}))
                bundle = UrbanEVFoldPreprocessor(raw).fit_transform(6, "F4")
                self._assert_shared_values(self.full, bundle)
                self._assert_stats_equal(self.full.preprocessing_state, bundle.preprocessing_state)
                self.assertNotEqual(raw.data_fingerprint, self.full_raw.data_fingerprint)
                if not restricted:
                    self.assertFalse(np.array_equal(bundle.features[3909:], self.full.features[3909:]))

    def test_csv_validation_pollution_preserves_train_fit_and_transform(self):
        from urbanev_synthetic_fixture import write_synthetic_urbanev
        root = self.root / "validation-polluted"
        reference = write_synthetic_urbanev(root, perturb="validation")
        for restricted in (False, True):
            with self.subTest(restricted=restricted):
                raw = UrbanEVRawData.load(root, **({"train_validation_fold": 6} if restricted else {}))
                bundle = UrbanEVFoldPreprocessor(raw).fit_transform(6, "F4")
                self._assert_shared_values(self.full, bundle, stop=3475)
                self._assert_stats_equal(self.full.preprocessing_state, bundle.preprocessing_state)
                self._assert_statistics(bundle.preprocessing_state, reference, 3475)
                self.assertFalse(np.array_equal(bundle.features[3475:3909], self.full.features[3475:3909]))
                self.assertNotEqual(raw.data_fingerprint, self.full_raw.data_fingerprint)

if __name__ == "__main__":
    unittest.main()
