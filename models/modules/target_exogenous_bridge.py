"""Lightweight target--exogenous bridge for AMD hidden representations.

The v1 bridge deliberately keeps only a global target-conditioned query,
shared historical-variate projection, one cross-attention operation, and an
outer residual. It is not a TimeXer encoder and does not use future observed
exogenous variables.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn


TARGET_EXOGENOUS = "target_exogenous"
PARALLEL_MULTIVARIATE = "parallel_multivariate"
SUPPORTED_TASK_MODES = (TARGET_EXOGENOUS, PARALLEL_MULTIVARIATE)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _ordered_indices(value: object, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be an ordered iterable of integers")
    result = tuple(value)
    for index in result:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                f"{name} must contain only non-bool integers, got {index!r}"
            )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate indices")
    return result


class TargetExogenousBridge(nn.Module):
    """Global target-conditioned residual bridge.

    Task mode, target index, and auxiliary order are construction-time
    scientific configuration rather than forward-time choices.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        feature_num: int,
        task_mode: str,
        target_idx: int,
        aux_idx: Iterable[int] | None,
        context_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        gamma_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.seq_len = _positive_int(seq_len, "seq_len")
        self.feature_num = _positive_int(feature_num, "feature_num")
        self.context_dim = _positive_int(context_dim, "context_dim")
        self.num_heads = _positive_int(num_heads, "num_heads")

        if task_mode not in SUPPORTED_TASK_MODES:
            raise ValueError(
                f"task_mode must be one of {SUPPORTED_TASK_MODES}, got {task_mode!r}"
            )
        if (
            isinstance(target_idx, bool)
            or not isinstance(target_idx, int)
            or not 0 <= target_idx < self.feature_num
        ):
            raise ValueError(
                "target_idx must index one input feature, "
                f"got {target_idx!r} for feature_num={self.feature_num}"
            )
        if self.context_dim % self.num_heads != 0:
            raise ValueError(
                "context_dim must be divisible by num_heads, "
                f"got {self.context_dim} and {self.num_heads}"
            )
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout!r}")
        if (
            isinstance(gamma_init, bool)
            or not isinstance(gamma_init, (int, float))
            or not math.isfinite(gamma_init)
            or float(gamma_init) != 1e-3
        ):
            raise ValueError(
                f"gamma_init is fixed at 1e-3, got {gamma_init!r}"
            )

        ordered_aux = _ordered_indices(aux_idx, "aux_idx")
        out_of_range = [
            index for index in ordered_aux if not 0 <= index < self.feature_num
        ]
        if out_of_range:
            raise ValueError(
                f"aux_idx contains out-of-range indices {out_of_range} "
                f"for feature_num={self.feature_num}"
            )
        if target_idx in ordered_aux:
            raise ValueError("aux_idx must exclude target_idx")
        if task_mode == TARGET_EXOGENOUS and not ordered_aux:
            raise ValueError("TEB requires at least one auxiliary variable.")
        if task_mode == PARALLEL_MULTIVARIATE:
            if self.feature_num < 2:
                raise ValueError("Parallel TEB requires at least two variables.")
            if ordered_aux:
                raise ValueError(
                    "parallel_multivariate uses all other variables; aux_idx must be empty"
                )

        self.task_mode = task_mode
        self.target_idx = target_idx
        self.aux_idx = ordered_aux
        self.dropout = float(dropout)

        self.query_projection = nn.Linear(
            self.seq_len, self.context_dim, bias=True
        )
        self.query_norm = nn.LayerNorm(self.context_dim, eps=1e-5)
        self.exogenous_projection = nn.Linear(
            self.seq_len, self.context_dim, bias=True
        )
        self.exogenous_norm = nn.LayerNorm(self.context_dim, eps=1e-5)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.context_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=True,
            batch_first=True,
        )
        self.output_projection = nn.Linear(
            self.context_dim, self.seq_len, bias=True
        )
        self.gamma_teb = nn.Parameter(torch.tensor(float(gamma_init)))

    def _validate_inputs(
        self,
        hidden: torch.Tensor,
        normalized_input: torch.Tensor,
        need_weights: bool,
    ) -> None:
        if not isinstance(need_weights, bool):
            raise TypeError(
                f"need_weights must be bool, got {type(need_weights).__name__}"
            )
        if not torch.is_tensor(hidden) or not torch.is_tensor(normalized_input):
            raise TypeError("hidden and normalized_input must be torch.Tensor instances")
        if hidden.ndim != 3:
            raise ValueError(
                "TEB hidden must have shape [batch, variable, time], "
                f"got {tuple(hidden.shape)}"
            )
        if normalized_input.ndim != 3:
            raise ValueError(
                "TEB normalized_input must have shape [batch, time, variable], "
                f"got {tuple(normalized_input.shape)}"
            )
        expected_hidden = (self.feature_num, self.seq_len)
        if hidden.shape[0] <= 0 or tuple(hidden.shape[1:]) != expected_hidden:
            raise ValueError(
                f"TEB hidden expects [batch, {self.feature_num}, {self.seq_len}], "
                f"got {tuple(hidden.shape)}"
            )
        if tuple(normalized_input.shape) != (
            hidden.shape[0],
            self.seq_len,
            self.feature_num,
        ):
            raise ValueError(
                "TEB normalized_input batch/time/variable dimensions must match hidden; "
                f"got hidden={tuple(hidden.shape)}, "
                f"normalized_input={tuple(normalized_input.shape)}"
            )
        if not hidden.is_floating_point() or not normalized_input.is_floating_point():
            raise TypeError("TEB inputs must use a floating-point dtype")
        if hidden.dtype != normalized_input.dtype:
            raise TypeError(
                "TEB inputs must use the same dtype, "
                f"got {hidden.dtype} and {normalized_input.dtype}"
            )
        if hidden.device != normalized_input.device:
            raise ValueError(
                "TEB inputs must be on the same device, "
                f"got {hidden.device} and {normalized_input.device}"
            )

    def _single_target(
        self,
        hidden: torch.Tensor,
        normalized_input: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        target_hidden = hidden[:, self.target_idx, :]
        query = self.query_norm(self.query_projection(target_hidden)).unsqueeze(1)
        auxiliary = normalized_input[:, :, self.aux_idx].transpose(1, 2)
        exogenous = self.exogenous_norm(self.exogenous_projection(auxiliary))
        context, weights = self.cross_attention(
            query,
            exogenous,
            exogenous,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        exo_context = context.squeeze(1)
        delta_target = self.output_projection(exo_context)
        target_hidden_out = target_hidden + self.gamma_teb * delta_target
        hidden_out = hidden.clone()
        hidden_out[:, self.target_idx, :] = target_hidden_out
        return hidden_out, exo_context, weights

    def _parallel(
        self,
        hidden: torch.Tensor,
        normalized_input: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        query = self.query_norm(self.query_projection(hidden))
        exogenous = self.exogenous_norm(
            self.exogenous_projection(normalized_input.transpose(1, 2))
        )
        diagonal_mask = torch.eye(
            self.feature_num,
            dtype=torch.bool,
            device=hidden.device,
        )
        context_all, weights = self.cross_attention(
            query,
            exogenous,
            exogenous,
            attn_mask=diagonal_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        delta_all = self.output_projection(context_all)
        hidden_out = hidden + self.gamma_teb * delta_all
        exo_context = context_all[:, self.target_idx, :]
        return hidden_out, exo_context, weights

    def forward(
        self,
        hidden: torch.Tensor,
        normalized_input: torch.Tensor,
        *,
        need_weights: bool = False,
    ):
        """Return hidden/context and optional unaveraged per-head weights."""

        self._validate_inputs(hidden, normalized_input, need_weights)
        if self.task_mode == TARGET_EXOGENOUS:
            result = self._single_target(hidden, normalized_input, need_weights)
        else:
            result = self._parallel(hidden, normalized_input, need_weights)
        hidden_out, exo_context, weights = result
        if need_weights:
            return hidden_out, exo_context, weights
        return hidden_out, exo_context


__all__ = [
    "PARALLEL_MULTIVARIATE",
    "SUPPORTED_TASK_MODES",
    "TARGET_EXOGENOUS",
    "TargetExogenousBridge",
]
