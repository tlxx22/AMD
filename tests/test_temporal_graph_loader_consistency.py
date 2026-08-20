"""Temporal/graph view parity, ordering, batching, and AMDEnhanced outputs."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch
from torch.utils.data import DataLoader, Subset

from models.tsAMD_enhanced import AMDEnhanced
from utils.dataloader_graph import (
    flatten_graph_batch,
    flatten_graph_targets,
    restore_node_batch,
)
from utils.dataloader_urbanev import UrbanEVFoldBundle, UrbanEVFoldPreprocessor, UrbanEVRawData
from utils.graph_window_dataset import GraphWindowDataset
from utils.temporal_region_dataset import TemporalRegionDataset


EXPECTED_PRESETS = {
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
    "F4": (
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
    ),
}


def _stack_temporal_window(
    dataset: TemporalRegionDataset, window_position: int
) -> tuple[torch.Tensor, torch.Tensor]:
    node_count = len(dataset.node_ids)
    samples = [
        dataset[window_position * node_count + node] for node in range(node_count)
    ]
    return (
        torch.stack([sample[0] for sample in samples], dim=0),
        torch.stack([sample[1] for sample in samples], dim=0),
    )


def _capture_rng():
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class TemporalGraphLoaderConsistencyTests(unittest.TestCase):
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

    def test_feature_presets_preserve_canonical_order_and_target(self):
        for preset, expected in EXPECTED_PRESETS.items():
            with self.subTest(preset=preset):
                bundle = self.bundle(1, preset)
                self.assertEqual(bundle.feature_names, expected)
                self.assertEqual(len(set(bundle.feature_names)), len(bundle.feature_names))
                self.assertEqual(bundle.target_idx, 0)
                self.assertEqual(bundle.target_idx, bundle.feature_names.index("volume"))
                self.assertEqual(bundle.features.shape[-1], len(bundle.feature_names))

    def test_graph_window_equals_canonical_temporal_stack_and_batch_shapes(self):
        bundle = self.bundle(1, "F4")
        graph = GraphWindowDataset(bundle, split="validation", label_horizon=6)
        temporal = TemporalRegionDataset(bundle, split="validation", label_horizon=6)
        for window_position in (0, len(graph) - 1):
            with self.subTest(window_position=window_position):
                graph_x, graph_y = graph[window_position]
                temporal_x, temporal_y = _stack_temporal_window(
                    temporal, window_position
                )
                flat_graph_x = flatten_graph_batch(graph_x.unsqueeze(0))
                self.assertTrue(torch.equal(flat_graph_x, temporal_x))
                self.assertTrue(
                    torch.equal(flatten_graph_targets(graph_y.unsqueeze(0)), temporal_y)
                )
                self.assertTrue(torch.equal(graph_x, temporal_x.permute(1, 0, 2)))
                self.assertTrue(torch.equal(graph_y, temporal_y))

        temporal_batch_x, temporal_batch_y = next(
            iter(DataLoader(temporal, batch_size=16, shuffle=False))
        )
        graph_batch_x, graph_batch_y = next(
            iter(DataLoader(graph, batch_size=2, shuffle=False))
        )
        self.assertEqual(temporal_batch_x.shape, (16, 12, 11))
        self.assertEqual(temporal_batch_y.shape, (16, 1))
        self.assertEqual(graph_batch_x.shape, (2, 12, 275, 11))
        self.assertEqual(graph_batch_y.shape, (2, 275, 1))
        calendar = graph_batch_x[0, :, :, 6:]
        self.assertTrue(
            torch.equal(calendar, calendar[:, :1, :].expand_as(calendar))
        )

    def test_train_shuffle_changes_only_order_and_eval_order_is_deterministic(self):
        bundle = self.bundle(1, "F0")
        train = TemporalRegionDataset(
            bundle, split="train", label_horizon=3, return_metadata=True
        )
        selected_indices = list(range(2 * len(train.node_ids)))
        shuffled_loader = DataLoader(
            Subset(train, selected_indices),
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(20240820),
        )
        observed: list[tuple[int, int]] = []
        for batch_x, batch_y, metadata in shuffled_loader:
            for position in range(batch_x.shape[0]):
                window = int(metadata["window_position"][position])
                node = int(metadata["node_position"][position])
                canonical_index = window * len(train.node_ids) + node
                expected_x, expected_y, expected_metadata = train[canonical_index]
                self.assertTrue(torch.equal(batch_x[position], expected_x))
                self.assertTrue(torch.equal(batch_y[position], expected_y))
                self.assertEqual(expected_metadata["node_id"], train.node_ids[node])
                observed.append((window, node))
        expected_identity = [
            divmod(index, len(train.node_ids)) for index in selected_indices
        ]
        self.assertEqual(sorted(observed), sorted(expected_identity))
        self.assertNotEqual(observed, expected_identity)

        for split in ("validation", "test"):
            evaluation = TemporalRegionDataset(
                bundle,
                split=split,
                label_horizon=3,
                return_metadata=True,
            )
            _, _, metadata = next(
                iter(DataLoader(evaluation, batch_size=8, shuffle=False))
            )
            self.assertEqual(metadata["window_position"].tolist(), [0] * 8)
            self.assertEqual(metadata["node_position"].tolist(), list(range(8)))
            self.assertEqual(list(metadata["node_id"]), list(bundle.node_ids[:8]))

    def test_amd_enhanced_y_time_and_state_source_match_both_views(self):
        original_rng = _capture_rng()
        try:
            bundle = self.bundle(1, "F0")
            graph = GraphWindowDataset(bundle, split="validation", label_horizon=3)
            temporal = TemporalRegionDataset(
                bundle, split="validation", label_horizon=3
            )
            graph_x, graph_y = graph[0]
            graph_input = flatten_graph_batch(graph_x.unsqueeze(0))
            temporal_input, temporal_y = _stack_temporal_window(temporal, 0)
            graph_target = flatten_graph_targets(graph_y.unsqueeze(0))
            raw_x_error = torch.max(torch.abs(graph_input - temporal_input)).item()
            raw_y_error = torch.max(torch.abs(graph_target - temporal_y)).item()
            self.assertLessEqual(raw_x_error, 1e-6)
            self.assertLessEqual(raw_y_error, 1e-6)

            torch.manual_seed(20240820)
            model = AMDEnhanced(
                input_shape=(12, 1),
                pred_len=1,
                n_block=0,
                dropout=0.0,
                patch=12,
                k=0,
                c=2,
                alpha=0.0,
                target_slice=slice(0, None),
                norm=False,
                layernorm=False,
                target_idx=0,
                teb_context_dim=3,
            ).eval()
            shared_rng = _capture_rng()
            _restore_rng(shared_rng)
            with torch.no_grad():
                graph_pred, graph_moe, graph_state = model(
                    graph_input, return_state_source=True
                )
            _restore_rng(shared_rng)
            with torch.no_grad():
                temporal_pred, temporal_moe, temporal_state = model(
                    temporal_input, return_state_source=True
                )

            graph_y_time = restore_node_batch(
                graph_pred[:, :, bundle.target_idx], batch_size=1, node_count=275
            )
            temporal_y_time = restore_node_batch(
                temporal_pred[:, :, bundle.target_idx], batch_size=1, node_count=275
            )
            graph_state_restored = restore_node_batch(
                graph_state, batch_size=1, node_count=275
            )
            temporal_state_restored = restore_node_batch(
                temporal_state, batch_size=1, node_count=275
            )
            pred_error = torch.max(
                torch.abs(graph_y_time - temporal_y_time)
            ).item()
            state_error = torch.max(
                torch.abs(graph_state_restored - temporal_state_restored)
            ).item()
            moe_error = torch.max(torch.abs(graph_moe - temporal_moe)).item()
            self.assertEqual(graph_y_time.shape, (1, 275, 1))
            self.assertEqual(graph_state_restored.shape, (1, 275, 27))
            self.assertTrue(
                torch.equal(
                    graph_state_restored[..., -3:],
                    torch.zeros_like(graph_state_restored[..., -3:]),
                )
            )
            self.assertLessEqual(pred_error, 1e-6)
            self.assertLessEqual(state_error, 1e-6)
            self.assertLessEqual(moe_error, 1e-6)
            print(
                "M1 dual-interface parity "
                f"raw_x_max_abs={raw_x_error:.9g} "
                f"raw_y_max_abs={raw_y_error:.9g} "
                f"y_time_max_abs={pred_error:.9g} "
                f"state_source_max_abs={state_error:.9g} "
                f"moe_loss_max_abs={moe_error:.9g}"
            )
        finally:
            _restore_rng(original_rng)


if __name__ == "__main__":
    unittest.main()
