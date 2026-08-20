"""Fold-train-only preprocessing and future-contamination tests."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from utils.dataloader_urbanev import (
    UrbanEVFoldPreprocessor,
    UrbanEVRawData,
    build_fold_definition,
)
from utils.graph_window_dataset import GraphWindowDataset


def _synthetic_raw(*, contaminate_future: bool = False) -> UrbanEVRawData:
    timestamps = pd.date_range(
        "2022-09-01 00:00:00", "2023-02-28 23:00:00", freq="h"
    )
    time = np.arange(len(timestamps), dtype=np.float64)[:, None]
    volume = 50.0 + time * np.asarray([[1.0, 2.0, 4.0]])
    e_price = np.column_stack(
        (
            np.full(len(timestamps), 5.0),
            1.0 + (time[:, 0] % 24.0) / 24.0,
            3.0 + time[:, 0] / 1000.0,
        )
    )
    s_price = np.column_stack(
        (
            (time[:, 0] % 7.0) / 7.0,
            2.0 + time[:, 0] / 2000.0,
            np.zeros(len(timestamps)),
        )
    )
    weather = np.column_stack(
        (
            20.0 + np.sin(time[:, 0] / 24.0),
            760.0 + np.cos(time[:, 0] / 48.0),
            50.0 + (time[:, 0] % 30.0),
        )
    )
    if contaminate_future:
        train_end = build_fold_definition(timestamps, 6).n_train
        volume[train_end:] += 1.0e9
        e_price[train_end:] += 1.0e6
        s_price[train_end:] -= 1.0e6
        weather[train_end:] += 1.0e8
    return UrbanEVRawData.from_arrays(
        timestamps=timestamps,
        node_ids=("001", "010", "100"),
        volume=volume,
        e_price=e_price,
        s_price=s_price,
        weather_central=weather,
    )


class FoldScalerNoLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "data" / "UrbanEV" / "data"
        cls.actual_bundle = UrbanEVFoldPreprocessor(
            UrbanEVRawData.load(root)
        ).fit_transform(6, "F4")

    def test_future_pollution_cannot_change_fitted_scalers(self):
        clean = UrbanEVFoldPreprocessor(_synthetic_raw()).fit_transform(6, "F4")
        polluted = UrbanEVFoldPreprocessor(
            _synthetic_raw(contaminate_future=True)
        ).fit_transform(6, "F4")
        clean_state = clean.preprocessing_state
        polluted_state = polluted.preprocessing_state
        self.assertEqual(clean_state.train_start_idx, 0)
        self.assertEqual(clean_state.train_end_idx, 3475)
        for field in (
            "volume_mean",
            "volume_scale",
            "e_price_min",
            "e_price_range",
            "e_price_safe_range",
            "s_price_min",
            "s_price_range",
            "s_price_safe_range",
            "weather_mean",
            "weather_scale",
        ):
            with self.subTest(field=field):
                np.testing.assert_array_equal(
                    getattr(clean_state, field), getattr(polluted_state, field)
                )

    def test_statistics_axes_and_population_variance_are_explicit(self):
        raw = _synthetic_raw()
        state = UrbanEVFoldPreprocessor(raw).fit_transform(
            6, "F4"
        ).preprocessing_state
        train = slice(0, 3475)
        self.assertEqual(state.volume_mean.shape, (3,))
        self.assertEqual(state.volume_scale.shape, (3,))
        np.testing.assert_allclose(state.volume_mean, raw.volume[train].mean(axis=0))
        np.testing.assert_allclose(
            state.volume_scale, raw.volume[train].std(axis=0, ddof=0)
        )
        self.assertEqual(state.e_price_min.shape, (3,))
        self.assertEqual(state.e_price_range.shape, (3,))
        np.testing.assert_allclose(state.e_price_min, raw.e_price[train].min(axis=0))
        np.testing.assert_allclose(
            state.e_price_range,
            raw.e_price[train].max(axis=0) - raw.e_price[train].min(axis=0),
        )
        self.assertEqual(state.s_price_min.shape, (3,))
        self.assertEqual(state.weather_mean.shape, (3,))
        self.assertEqual(state.weather_scale.shape, (3,))
        np.testing.assert_allclose(
            state.weather_mean, raw.weather_central[train].mean(axis=0)
        )
        np.testing.assert_allclose(
            state.weather_scale, raw.weather_central[train].std(axis=0, ddof=0)
        )

    def test_constant_prices_are_safe_zero_on_train_and_not_clipped(self):
        raw = _synthetic_raw()
        e_price = raw.e_price.copy()
        s_price = raw.s_price.copy()
        train_end = build_fold_definition(raw.timestamps, 6).n_train
        e_price[train_end:, 0] = 7.0
        s_price[train_end:, 2] = -2.0
        changed = UrbanEVRawData.from_arrays(
            timestamps=raw.timestamps,
            node_ids=raw.node_ids,
            volume=raw.volume,
            e_price=e_price,
            s_price=s_price,
            weather_central=raw.weather_central,
        )
        bundle = UrbanEVFoldPreprocessor(changed).fit_transform(6, "F4")
        state = bundle.preprocessing_state
        self.assertEqual(state.e_price_range[0], 0.0)
        self.assertEqual(state.e_price_safe_range[0], 1.0)
        self.assertEqual(state.s_price_range[2], 0.0)
        self.assertEqual(state.s_price_safe_range[2], 1.0)
        self.assertTrue(np.all(bundle.features[:train_end, 0, 1] == 0.0))
        self.assertTrue(np.all(bundle.features[:train_end, 2, 2] == 0.0))
        self.assertEqual(bundle.features[train_end, 0, 1], 2.0)
        self.assertEqual(bundle.features[train_end, 2, 2], -2.0)
        self.assertTrue(np.isfinite(bundle.features).all())
        self.assertGreater(bundle.features[train_end, 0, 1], 1.0)
        self.assertLess(bundle.features[train_end, 2, 2], 0.0)

    def test_same_fold_all_horizons_share_state_and_folds_are_independent(self):
        raw = _synthetic_raw()
        preprocessor = UrbanEVFoldPreprocessor(raw)
        fingerprints = []
        for horizon in (3, 6, 9, 12):
            bundle = preprocessor.fit_transform(6, "F4")
            GraphWindowDataset(bundle, split="train", label_horizon=horizon)
            fingerprints.append(bundle.preprocessing_state_fingerprint)
        self.assertEqual(len(set(fingerprints)), 1)
        fold_five = preprocessor.fit_transform(5, "F4")
        fold_six = preprocessor.fit_transform(6, "F4")
        self.assertNotEqual(
            fold_five.preprocessing_state_fingerprint,
            fold_six.preprocessing_state_fingerprint,
        )
        self.assertFalse(
            np.array_equal(
                fold_five.preprocessing_state.volume_mean,
                fold_six.preprocessing_state.volume_mean,
            )
        )

    def test_target_transform_inverse_round_trip_numpy_and_torch(self):
        bundle = self.actual_bundle
        raw_graph = bundle.raw.volume[:32]
        transformed_graph = bundle.transform_target(raw_graph)
        restored_graph = bundle.inverse_transform_target(transformed_graph)
        np.testing.assert_allclose(restored_graph, raw_graph, rtol=0.0, atol=1e-10)
        positions = np.asarray([0, 10, 274])
        raw_temporal = np.asarray(
            [
                [bundle.raw.volume[10, 0]],
                [bundle.raw.volume[20, 10]],
                [bundle.raw.volume[30, 274]],
            ],
            dtype=np.float64,
        )
        transformed_temporal = bundle.transform_target(
            raw_temporal, node_position=positions
        )
        restored_temporal = bundle.inverse_transform_target(
            transformed_temporal, node_position=positions
        )
        np.testing.assert_allclose(
            restored_temporal, raw_temporal, rtol=0.0, atol=1e-10
        )
        graph_tensor = torch.from_numpy(raw_graph[:2]).float().unsqueeze(-1)
        transformed_tensor = bundle.transform_target(graph_tensor)
        restored_tensor = bundle.inverse_transform_target(transformed_tensor)
        self.assertLess(
            torch.max(torch.abs(restored_tensor - graph_tensor)).item(), 1e-3
        )


if __name__ == "__main__":
    unittest.main()
