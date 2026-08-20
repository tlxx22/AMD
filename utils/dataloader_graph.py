"""Pure tensor flatten/restore helpers for M1 graph/temporal parity."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def flatten_graph_batch(graph_x: torch.Tensor) -> torch.Tensor:
    """Convert ``[B,T,N,C]`` to window-major/node-major ``[B*N,T,C]``."""

    if graph_x.ndim != 4:
        raise ValueError(f"graph_x must have shape [B,T,N,C], got {tuple(graph_x.shape)}")
    batch_size, history_len, node_count, feature_count = graph_x.shape
    if batch_size <= 0 or node_count <= 0:
        raise ValueError("graph_x batch and node dimensions must be positive")
    return (
        graph_x.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(batch_size * node_count, history_len, feature_count)
    )


def flatten_graph_targets(graph_y: torch.Tensor) -> torch.Tensor:
    """Convert ``[B,N,...]`` to the matching ``[B*N,...]`` order."""

    if graph_y.ndim < 2:
        raise ValueError(f"graph_y must have shape [B,N,...], got {tuple(graph_y.shape)}")
    batch_size, node_count = graph_y.shape[:2]
    if batch_size <= 0 or node_count <= 0:
        raise ValueError("graph_y batch and node dimensions must be positive")
    return graph_y.contiguous().reshape(batch_size * node_count, *graph_y.shape[2:])


def restore_node_batch(
    node_values: torch.Tensor, *, batch_size: int, node_count: int
) -> torch.Tensor:
    """Restore window-major/node-major values from ``[B*N,...]`` to ``[B,N,...]``."""

    if node_values.ndim < 1:
        raise ValueError("node_values must have at least one dimension")
    if batch_size <= 0 or node_count <= 0:
        raise ValueError("batch_size and node_count must be positive")
    expected = batch_size * node_count
    if node_values.shape[0] != expected:
        raise ValueError(
            f"leading dimension must equal B*N={expected}, got {node_values.shape[0]}"
        )
    return node_values.contiguous().reshape(batch_size, node_count, *node_values.shape[1:])


def restore_graph_batch(
    node_x: torch.Tensor, *, batch_size: int, node_count: int
) -> torch.Tensor:
    """Invert :func:`flatten_graph_batch`, restoring ``[B,T,N,C]``."""

    if node_x.ndim != 3:
        raise ValueError(f"node_x must have shape [B*N,T,C], got {tuple(node_x.shape)}")
    restored = restore_node_batch(
        node_x, batch_size=batch_size, node_count=node_count
    )
    return restored.permute(0, 2, 1, 3).contiguous()


def restore_temporal_samples(
    values: torch.Tensor,
    *,
    window_positions: Sequence[int] | torch.Tensor,
    node_positions: Sequence[int] | torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    """Restore temporal samples only after validating canonical sample order.

    Arbitrarily shuffled temporal samples are rejected instead of being
    silently reshaped into a corrupt graph window.
    """

    windows = torch.as_tensor(window_positions, dtype=torch.long)
    nodes = torch.as_tensor(node_positions, dtype=torch.long)
    if windows.ndim != 1 or nodes.ndim != 1:
        raise ValueError("window_positions and node_positions must be one-dimensional")
    if len(windows) != values.shape[0] or len(nodes) != values.shape[0]:
        raise ValueError("identity lengths must match the values leading dimension")
    if node_count <= 0 or values.shape[0] % node_count != 0:
        raise ValueError("sample count must be divisible by the positive node_count")
    batch_size = values.shape[0] // node_count
    window_grid = windows.reshape(batch_size, node_count)
    node_grid = nodes.reshape(batch_size, node_count)
    expected_nodes = torch.arange(node_count, dtype=torch.long).expand(batch_size, -1)
    if not torch.equal(node_grid.cpu(), expected_nodes):
        raise ValueError(
            "temporal samples are not in canonical window-major/node-major order"
        )
    if not torch.all(window_grid == window_grid[:, :1]):
        raise ValueError("each restored group must contain exactly one window position")
    group_windows = window_grid[:, 0]
    if batch_size > 1 and not torch.all(group_windows[1:] > group_windows[:-1]):
        raise ValueError("window groups must be in strictly increasing order")
    return restore_node_batch(values, batch_size=batch_size, node_count=node_count)
