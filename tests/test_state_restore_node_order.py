"""Window-major/node-major flatten and restore invariants."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from utils.dataloader_graph import (
    flatten_graph_batch,
    flatten_graph_targets,
    restore_graph_batch,
    restore_node_batch,
    restore_temporal_samples,
)
from utils.dataloader_urbanev import UrbanEVFoldPreprocessor, UrbanEVRawData
from utils.graph_window_dataset import GraphWindowDataset


class StateRestoreNodeOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "data" / "UrbanEV" / "data"
        cls.bundle = UrbanEVFoldPreprocessor(UrbanEVRawData.load(root)).fit_transform(
            1, "F0"
        )

    def test_flatten_restore_is_reversible_and_window_node_major(self):
        batch_size, history_len, node_count, channels = 2, 3, 4, 2
        graph_x = torch.arange(
            batch_size * history_len * node_count * channels, dtype=torch.float32
        ).reshape(batch_size, history_len, node_count, channels)
        flattened = flatten_graph_batch(graph_x)
        self.assertEqual(
            flattened.shape, (batch_size * node_count, history_len, channels)
        )
        for batch in range(batch_size):
            for node in range(node_count):
                flat_index = batch * node_count + node
                self.assertTrue(
                    torch.equal(flattened[flat_index], graph_x[batch, :, node, :])
                )
        restored = restore_graph_batch(
            flattened, batch_size=batch_size, node_count=node_count
        )
        self.assertTrue(torch.equal(restored, graph_x))

    def test_restore_y_time_state_source_and_target_shapes(self):
        batch_size, node_count = 3, 5
        y_time = torch.arange(batch_size * node_count, dtype=torch.float32).reshape(
            -1, 1
        )
        state_source = torch.arange(
            batch_size * node_count * 27, dtype=torch.float32
        ).reshape(batch_size * node_count, 27)
        target = torch.arange(
            batch_size * node_count, dtype=torch.float32
        ).reshape(batch_size, node_count, 1)
        self.assertEqual(
            restore_node_batch(
                y_time, batch_size=batch_size, node_count=node_count
            ).shape,
            (batch_size, node_count, 1),
        )
        self.assertEqual(
            restore_node_batch(
                state_source, batch_size=batch_size, node_count=node_count
            ).shape,
            (batch_size, node_count, 27),
        )
        self.assertTrue(torch.equal(flatten_graph_targets(target), y_time))

    def test_actual_node_ids_remain_strings_and_canonical(self):
        dataset = GraphWindowDataset(
            self.bundle, split="train", label_horizon=3
        )
        self.assertEqual(dataset.node_ids, self.bundle.raw.node_ids)
        self.assertEqual(len(dataset.node_ids), 275)
        self.assertTrue(all(isinstance(node_id, str) for node_id in dataset.node_ids))
        self.assertEqual(dataset.node_ids[0], "102")

    def test_restore_rejects_wrong_dimensions_and_shuffled_temporal_samples(self):
        with self.assertRaisesRegex(ValueError, r"B\*N"):
            restore_node_batch(torch.zeros(9, 2), batch_size=2, node_count=5)
        with self.assertRaisesRegex(ValueError, "positive"):
            restore_node_batch(torch.zeros(10, 2), batch_size=2, node_count=0)
        with self.assertRaisesRegex(ValueError, "node_x"):
            restore_graph_batch(torch.zeros(10, 3), batch_size=2, node_count=5)

        values = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        canonical_windows = [4, 4, 4, 5, 5, 5]
        canonical_nodes = [0, 1, 2, 0, 1, 2]
        restored = restore_temporal_samples(
            values,
            window_positions=canonical_windows,
            node_positions=canonical_nodes,
            node_count=3,
        )
        self.assertEqual(restored.shape, (2, 3, 2))
        with self.assertRaisesRegex(ValueError, "canonical"):
            restore_temporal_samples(
                values[[1, 0, 2, 3, 4, 5]],
                window_positions=[4, 4, 4, 5, 5, 5],
                node_positions=[1, 0, 2, 0, 1, 2],
                node_count=3,
            )
        with self.assertRaisesRegex(ValueError, "one window"):
            restore_temporal_samples(
                values,
                window_positions=[4, 5, 4, 5, 5, 5],
                node_positions=canonical_nodes,
                node_count=3,
            )


if __name__ == "__main__":
    unittest.main()
