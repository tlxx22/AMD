"""Frozen split-local label offsets and all 72 UrbanEV window counts."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from utils.dataloader_urbanev import UrbanEVFoldBundle, UrbanEVFoldPreprocessor, UrbanEVRawData
from utils.graph_window_dataset import GraphWindowDataset
from utils.temporal_region_dataset import TemporalRegionDataset


EXPECTED_WINDOW_COUNTS = {
    1: {
        "train": {3: 562, 6: 559, 9: 556, 12: 553},
        "validation": {3: 58, 6: 55, 9: 52, 12: 49},
        "test": {3: 58, 6: 55, 9: 52, 12: 49},
    },
    2: {
        "train": {3: 1157, 6: 1154, 9: 1151, 12: 1148},
        "validation": {3: 132, 6: 129, 9: 126, 12: 123},
        "test": {3: 133, 6: 130, 9: 127, 12: 124},
    },
    3: {
        "train": {3: 1733, 6: 1730, 9: 1727, 12: 1724},
        "validation": {3: 204, 6: 201, 9: 198, 12: 195},
        "test": {3: 205, 6: 202, 9: 199, 12: 196},
    },
    4: {
        "train": {3: 2328, 6: 2325, 9: 2322, 12: 2319},
        "validation": {3: 278, 6: 275, 9: 272, 12: 269},
        "test": {3: 280, 6: 277, 9: 274, 12: 271},
    },
    5: {
        "train": {3: 2923, 6: 2920, 9: 2917, 12: 2914},
        "validation": {3: 353, 6: 350, 9: 347, 12: 344},
        "test": {3: 354, 6: 351, 9: 348, 12: 345},
    },
    6: {
        "train": {3: 3461, 6: 3458, 9: 3455, 12: 3452},
        "validation": {3: 420, 6: 417, 9: 414, 12: 411},
        "test": {3: 421, 6: 418, 9: 415, 12: 412},
    },
}


class TargetOffsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "data" / "UrbanEV" / "data"
        cls.preprocessor = UrbanEVFoldPreprocessor(UrbanEVRawData.load(root))
        cls.bundles: dict[tuple[int, str], UrbanEVFoldBundle] = {}

    @classmethod
    def bundle(cls, fold: int, preset: str) -> UrbanEVFoldBundle:
        key = (fold, preset)
        if key not in cls.bundles:
            cls.bundles[key] = cls.preprocessor.fit_transform(fold, preset)
        return cls.bundles[key]

    def test_label_formula_first_last_and_inclusive_final_label(self):
        bundle = self.bundle(6, "F0")
        for horizon in (3, 6, 9, 12):
            for split in ("train", "validation", "test"):
                with self.subTest(horizon=horizon, split=split):
                    dataset = GraphWindowDataset(
                        bundle, split=split, label_horizon=horizon, history_len=12
                    )
                    split_target = bundle.split_target(split)
                    split_timestamps = bundle.split_timestamps(split)
                    first_x, first_y = dataset[0]
                    first_meta = dataset.metadata(0)
                    first_label_position = 12 + horizon - 1
                    self.assertEqual(first_x.shape, (12, len(bundle.node_ids), 1))
                    self.assertTrue(
                        torch.equal(
                            first_y,
                            torch.from_numpy(
                                split_target[first_label_position, :, None]
                            ),
                        )
                    )
                    self.assertEqual(
                        first_meta["label_timestamp"],
                        split_timestamps[first_label_position].isoformat(),
                    )
                    self.assertEqual(
                        first_meta["window_start_idx"], bundle.split_slice(split).start
                    )

                    last_position = len(dataset) - 1
                    _, last_y = dataset[last_position]
                    last_meta = dataset.metadata(last_position)
                    self.assertTrue(
                        torch.equal(last_y, torch.from_numpy(split_target[-1, :, None]))
                    )
                    self.assertEqual(
                        last_meta["label_timestamp"], split_timestamps[-1].isoformat()
                    )
                    self.assertEqual(
                        last_meta["label_idx"], bundle.split_slice(split).stop - 1
                    )
                    self.assertLess(
                        last_meta["label_idx"], bundle.split_slice(split).stop
                    )

    def test_all_72_report_counts_match_both_dataset_views(self):
        checked = 0
        for fold, split_counts in EXPECTED_WINDOW_COUNTS.items():
            bundle = self.bundle(fold, "F0")
            for split, horizon_counts in split_counts.items():
                for horizon, expected_windows in horizon_counts.items():
                    with self.subTest(fold=fold, split=split, horizon=horizon):
                        graph = GraphWindowDataset(
                            bundle, split=split, label_horizon=horizon
                        )
                        temporal = TemporalRegionDataset(
                            bundle, split=split, label_horizon=horizon
                        )
                        self.assertEqual(len(graph), expected_windows)
                        self.assertEqual(len(temporal), expected_windows * 275)
                        checked += 1
        self.assertEqual(checked, 72)

    def test_invalid_horizon_is_rejected(self):
        bundle = self.bundle(1, "F0")
        with self.assertRaisesRegex(ValueError, "label_horizon"):
            GraphWindowDataset(bundle, split="train", label_horizon=1)


if __name__ == "__main__":
    unittest.main()
