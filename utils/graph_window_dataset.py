"""Whole-city graph-window view over a shared UrbanEV fold bundle."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from utils.dataloader_urbanev import (
    UrbanEVFoldBundle,
    validate_label_horizon,
    window_count,
)


class GraphWindowDataset(Dataset):
    """Expose every canonical node for one time window as a single sample.

    This M1 Dataset deliberately contains no adjacency matrix or graph message
    passing; it is only a deterministic time/node tensor view.
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
        return self.window_count

    def _validate_index(self, index: int) -> None:
        if index < 0 or index >= len(self):
            raise IndexError(f"GraphWindowDataset index out of range: {index}")

    def __getitem__(self, index: int) -> Any:
        self._validate_index(index)
        history_end = index + self.history_len
        label_position = history_end + self.label_horizon - 1
        x = torch.from_numpy(self._features[index:history_end, :, :])
        y = torch.from_numpy(self._target[label_position, :, None])
        if self.return_metadata:
            return x, y, self.metadata(index)
        return x, y

    def metadata(self, index: int) -> dict[str, Any]:
        """Return stable graph-window identity."""

        self._validate_index(index)
        local_history_end = index + self.history_len - 1
        local_label = index + self.history_len + self.label_horizon - 1
        split_start = int(self._split_slice.start)
        return {
            "fold": self.bundle.fold_definition.fold,
            "split": self.split,
            "horizon": self.label_horizon,
            "window_position": index,
            "window_start_idx": split_start + index,
            "window_start_timestamp": self._timestamps[index].isoformat(),
            "history_end_timestamp": self._timestamps[local_history_end].isoformat(),
            "label_idx": split_start + local_label,
            "label_timestamp": self._timestamps[local_label].isoformat(),
        }
