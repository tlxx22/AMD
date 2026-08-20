"""Node-wise temporal view over a shared UrbanEV fold bundle."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from utils.dataloader_urbanev import (
    UrbanEVFoldBundle,
    validate_label_horizon,
    window_count,
)


class TemporalRegionDataset(Dataset):
    """Expose one ``[history, features]`` sample per window and region.

    Samples are deterministically ordered window-major, then by the canonical
    node order inherited from ``volume.csv``.  The Dataset itself never
    shuffles; callers may use a shuffled training DataLoader.
    """

    def __init__(
        self,
        bundle: UrbanEVFoldBundle,
        *,
        split: str,
        label_horizon: int,
        history_len: int = 12,
        return_metadata: bool = False,
    ) -> None:
        validate_label_horizon(label_horizon)
        if history_len <= 0:
            raise ValueError("history_len must be positive")
        self.bundle = bundle
        self.split = split
        self.label_horizon = label_horizon
        self.history_len = history_len
        self.return_metadata = return_metadata
        self._split_slice = bundle.split_slice(split)
        self._features = bundle.split_features(split)
        self._target = bundle.split_target(split)
        self._timestamps = bundle.split_timestamps(split)
        self.window_count = window_count(
            len(self._timestamps), label_horizon, history_len
        )
        self.node_ids = bundle.node_ids
        self.feature_names = bundle.feature_names
        self.target_idx = bundle.target_idx

    def __len__(self) -> int:
        return self.window_count * len(self.node_ids)

    def _identity(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(f"TemporalRegionDataset index out of range: {index}")
        return divmod(index, len(self.node_ids))

    def __getitem__(self, index: int) -> Any:
        window_position, node_position = self._identity(index)
        history_end = window_position + self.history_len
        label_position = history_end + self.label_horizon - 1
        x = torch.from_numpy(
            self._features[window_position:history_end, node_position, :]
        )
        y = torch.tensor(
            [self._target[label_position, node_position]], dtype=torch.float32
        )
        if self.return_metadata:
            return x, y, self.metadata(index)
        return x, y

    def metadata(self, index: int) -> dict[str, Any]:
        """Return stable sample identity without changing the default batch API."""

        window_position, node_position = self._identity(index)
        local_history_end = window_position + self.history_len - 1
        local_label = window_position + self.history_len + self.label_horizon - 1
        split_start = int(self._split_slice.start)
        return {
            "fold": self.bundle.fold_definition.fold,
            "split": self.split,
            "horizon": self.label_horizon,
            "window_position": window_position,
            "window_start_idx": split_start + window_position,
            "window_start_timestamp": self._timestamps[window_position].isoformat(),
            "history_end_timestamp": self._timestamps[local_history_end].isoformat(),
            "label_idx": split_start + local_label,
            "label_timestamp": self._timestamps[local_label].isoformat(),
            "node_position": node_position,
            "node_id": self.node_ids[node_position],
        }
