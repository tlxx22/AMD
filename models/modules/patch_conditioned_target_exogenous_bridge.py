"""Patch-conditioned target--exogenous bridge for AMD.

T2 keeps TimeXer's hierarchy (target patches, a target global token, and
whole-series exogenous variate tokens) while using one lightweight direct
cross-attention.  It is intentionally independent from the frozen Global TEB
v1 implementation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.target_exogenous_bridge import (
    PARALLEL_MULTIVARIATE,
    SUPPORTED_TASK_MODES,
    TARGET_EXOGENOUS,
)


PATCH_CONDITIONED_V1 = "patch_conditioned_v1"
RIGHT_ZERO_CROP = "right_zero_crop"
FIXED_SINUSOIDAL = "fixed_sinusoidal"


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


def _fixed_sinusoidal_position(num_patches: int, context_dim: int) -> torch.Tensor:
    """Return deterministic sinusoidal positions with shape [1,N,d]."""

    position = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, context_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / context_dim)
    )
    encoding = torch.zeros(num_patches, context_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequency)
    odd_width = encoding[:, 1::2].shape[1]
    if odd_width:
        encoding[:, 1::2] = torch.cos(position * frequency[:odd_width])
    return encoding.unsqueeze(0)


class PatchConditionedTargetExogenousBridge(nn.Module):
    """Direct patch-query bridge with shared variable-independent parameters."""

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
        dropout: float,
        patch_size: int,
        gamma_init: float = 1e-3,
        padding_policy: str = RIGHT_ZERO_CROP,
        position_policy: str = FIXED_SINUSOIDAL,
    ) -> None:
        super().__init__()
        self.seq_len = _positive_int(seq_len, "seq_len")
        self.feature_num = _positive_int(feature_num, "feature_num")
        self.context_dim = _positive_int(context_dim, "context_dim")
        self.num_heads = _positive_int(num_heads, "num_heads")
        self.patch_size = _positive_int(patch_size, "patch_size")

        if self.patch_size > self.seq_len:
            raise ValueError(
                "patch_size must not exceed seq_len, "
                f"got {self.patch_size} for seq_len={self.seq_len}"
            )
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
            raise ValueError(f"gamma_init is fixed at 1e-3, got {gamma_init!r}")
        if padding_policy != RIGHT_ZERO_CROP:
            raise ValueError(
                f"padding_policy must be {RIGHT_ZERO_CROP!r}, got {padding_policy!r}"
            )
        if position_policy != FIXED_SINUSOIDAL:
            raise ValueError(
                f"position_policy must be {FIXED_SINUSOIDAL!r}, got {position_policy!r}"
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
        self.padding_policy = padding_policy
        self.position_policy = position_policy
        self.num_patches = math.ceil(self.seq_len / self.patch_size)
        self.pad_len = self.num_patches * self.patch_size - self.seq_len

        self.patch_query_projection = nn.Linear(
            self.patch_size, self.context_dim, bias=True
        )
        self.patch_query_norm = nn.LayerNorm(self.context_dim, eps=1e-5)
        self.global_query_projection = nn.Linear(
            self.seq_len, self.context_dim, bias=True
        )
        self.global_query_norm = nn.LayerNorm(self.context_dim, eps=1e-5)
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
        self.patch_output_projection = nn.Linear(
            self.context_dim, self.patch_size, bias=True
        )
        self.gamma_teb = nn.Parameter(torch.tensor(float(gamma_init)))
        self.register_buffer(
            "fixed_sinusoidal_position",
            _fixed_sinusoidal_position(self.num_patches, self.context_dim),
            persistent=False,
        )

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
        if hidden.shape[0] <= 0 or tuple(hidden.shape[1:]) != (
            self.feature_num,
            self.seq_len,
        ):
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

    def _patchify(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pad_len:
            hidden = F.pad(hidden, (0, self.pad_len), mode="constant", value=0.0)
        return hidden.reshape(*hidden.shape[:-1], self.num_patches, self.patch_size)

    def _patch_queries(self, patches: torch.Tensor) -> torch.Tensor:
        projected = self.patch_query_projection(patches)
        position = self.fixed_sinusoidal_position.to(
            device=projected.device,
            dtype=projected.dtype,
        )
        if projected.ndim == 4:
            position = position.unsqueeze(1)
        return self.patch_query_norm(projected + position)

    def _single_target(
        self,
        hidden: torch.Tensor,
        normalized_input: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        target_hidden = hidden[:, self.target_idx, :]
        patch_query = self._patch_queries(self._patchify(target_hidden))
        global_query = self.global_query_norm(
            self.global_query_projection(target_hidden)
        ).unsqueeze(1)
        query = torch.cat((patch_query, global_query), dim=1)
        auxiliary = normalized_input[:, :, self.aux_idx].transpose(1, 2)
        exogenous = self.exogenous_norm(self.exogenous_projection(auxiliary))
        context, weights = self.cross_attention(
            query,
            exogenous,
            exogenous,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        patch_context = context[:, : self.num_patches, :]
        exo_context = context[:, self.num_patches, :]
        delta_patch = self.patch_output_projection(patch_context)
        delta_target = delta_patch.reshape(hidden.shape[0], -1)[:, : self.seq_len]
        hidden_out = hidden.clone()
        hidden_out[:, self.target_idx, :] = (
            target_hidden + self.gamma_teb * delta_target
        )
        return hidden_out, exo_context, weights

    def _parallel_attention_mask(self, device: torch.device) -> torch.Tensor:
        owner = torch.arange(self.feature_num, device=device).repeat_interleave(
            self.num_patches + 1
        )
        keys = torch.arange(self.feature_num, device=device)
        return keys.unsqueeze(0).eq(owner.unsqueeze(1))

    def _parallel(
        self,
        hidden: torch.Tensor,
        normalized_input: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = hidden.shape[0]
        patch_query = self._patch_queries(self._patchify(hidden))
        global_query = self.global_query_norm(
            self.global_query_projection(hidden)
        ).unsqueeze(2)
        query_by_variable = torch.cat((patch_query, global_query), dim=2)
        query = query_by_variable.reshape(
            batch_size,
            self.feature_num * (self.num_patches + 1),
            self.context_dim,
        )
        exogenous = self.exogenous_norm(
            self.exogenous_projection(normalized_input.transpose(1, 2))
        )
        context_flat, weights = self.cross_attention(
            query,
            exogenous,
            exogenous,
            attn_mask=self._parallel_attention_mask(hidden.device),
            need_weights=need_weights,
            average_attn_weights=False,
        )
        context = context_flat.reshape(
            batch_size,
            self.feature_num,
            self.num_patches + 1,
            self.context_dim,
        )
        patch_context = context[:, :, : self.num_patches, :]
        global_context = context[:, :, self.num_patches, :]
        delta_patch = self.patch_output_projection(patch_context)
        delta_all = delta_patch.reshape(batch_size, self.feature_num, -1)[
            :, :, : self.seq_len
        ]
        hidden_out = hidden + self.gamma_teb * delta_all
        exo_context = global_context[:, self.target_idx, :]
        return hidden_out, exo_context, weights

    def forward(
        self,
        hidden: torch.Tensor,
        normalized_input: torch.Tensor,
        *,
        need_weights: bool = False,
    ):
        """Return refined hidden/context and optional per-head weights."""

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
    "FIXED_SINUSOIDAL",
    "PATCH_CONDITIONED_V1",
    "RIGHT_ZERO_CROP",
    "PatchConditionedTargetExogenousBridge",
]
