"""Global-mediated patch-conditioned target--exogenous bridge for AMD.

T2G is a single-factor extension of the frozen T2 bridge.  It keeps T2's
direct patch-to-exogenous cross-attention unchanged, residualizes only the
global cross-attention response, and injects that global bridge into the raw
patch attention response through an identity-initialized scalar-per-patch
gate.  No target patch residual is present.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn

from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    RIGHT_ZERO_CROP,
    PatchConditionedTargetExogenousBridge,
)
from models.modules.target_exogenous_bridge import TARGET_EXOGENOUS


GLOBAL_MEDIATED_PATCH_V1 = "global_mediated_patch_v1"
GLOBAL_RESIDUAL_CONTRACT = "query_plus_attention_post_layernorm"
PATCH_ATTENTION_RESIDUAL_NONE = "none"
GLOBAL_GATE_SCALAR_PER_PATCH = "scalar_per_patch"
GLOBAL_GATE_INPUT_CONTRACT = "patch_attention_and_global_bridge"
GLOBAL_GATE_IDENTITY_INIT = "identity"


class GlobalMediatedPatchTargetExogenousBridge(
    PatchConditionedTargetExogenousBridge
):
    """T2 patch attention plus a lightweight global-mediated interaction."""

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
        global_residual: str = GLOBAL_RESIDUAL_CONTRACT,
        patch_attention_residual: str = PATCH_ATTENTION_RESIDUAL_NONE,
        global_gate: str = GLOBAL_GATE_SCALAR_PER_PATCH,
        global_gate_input: str = GLOBAL_GATE_INPUT_CONTRACT,
        global_gate_init: str = GLOBAL_GATE_IDENTITY_INIT,
        beta_global_init: float = 1e-3,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            feature_num=feature_num,
            task_mode=task_mode,
            target_idx=target_idx,
            aux_idx=aux_idx,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
            patch_size=patch_size,
            gamma_init=gamma_init,
            padding_policy=padding_policy,
            position_policy=position_policy,
        )
        expected = {
            "global_residual": (global_residual, GLOBAL_RESIDUAL_CONTRACT),
            "patch_attention_residual": (
                patch_attention_residual,
                PATCH_ATTENTION_RESIDUAL_NONE,
            ),
            "global_gate": (global_gate, GLOBAL_GATE_SCALAR_PER_PATCH),
            "global_gate_input": (
                global_gate_input,
                GLOBAL_GATE_INPUT_CONTRACT,
            ),
            "global_gate_init": (global_gate_init, GLOBAL_GATE_IDENTITY_INIT),
        }
        mismatches = [
            f"{name}={actual!r} (expected {required!r})"
            for name, (actual, required) in expected.items()
            if actual != required
        ]
        if mismatches:
            raise ValueError("T2G contract mismatch: " + ", ".join(mismatches))
        if (
            isinstance(beta_global_init, bool)
            or not isinstance(beta_global_init, (int, float))
            or not math.isfinite(beta_global_init)
            or float(beta_global_init) != 1e-3
        ):
            raise ValueError(
                "beta_global_init is fixed at 1e-3, "
                f"got {beta_global_init!r}"
            )

        self.global_residual = global_residual
        self.patch_attention_residual = patch_attention_residual
        self.global_gate = global_gate
        self.global_gate_input = global_gate_input
        self.global_gate_init = global_gate_init

        self.global_bridge_norm = nn.LayerNorm(self.context_dim, eps=1e-5)
        self.global_injection_gate = nn.Linear(
            2 * self.context_dim, 1, bias=True
        )
        nn.init.zeros_(self.global_injection_gate.weight)
        nn.init.zeros_(self.global_injection_gate.bias)
        self.beta_global = nn.Parameter(torch.tensor(float(beta_global_init)))

    def _global_mediated_patch_context(
        self,
        patch_attention: torch.Tensor,
        global_query: torch.Tensor,
        global_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused patch response, residualized global bridge, and gate."""

        global_bridge = self.global_bridge_norm(global_query + global_attention)
        if patch_attention.ndim == 3:
            global_for_patch = global_bridge.expand(
                -1, self.num_patches, -1
            )
        else:
            global_for_patch = global_bridge.expand(
                -1, -1, self.num_patches, -1
            )
        gate_input = torch.cat((patch_attention, global_for_patch), dim=-1)
        gate = 2.0 * torch.sigmoid(self.global_injection_gate(gate_input))
        fused = patch_attention + self.beta_global * gate * global_for_patch
        return fused, global_bridge.squeeze(-2), gate

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
        attention, weights = self.cross_attention(
            query,
            exogenous,
            exogenous,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        patch_attention = attention[:, : self.num_patches, :]
        global_attention = attention[:, self.num_patches :, :]
        fused_patch, exo_context, _ = self._global_mediated_patch_context(
            patch_attention,
            global_query,
            global_attention,
        )
        delta_patch = self.patch_output_projection(fused_patch)
        delta_target = delta_patch.reshape(hidden.shape[0], -1)[:, : self.seq_len]
        hidden_out = hidden.clone()
        hidden_out[:, self.target_idx, :] = (
            target_hidden + self.gamma_teb * delta_target
        )
        return hidden_out, exo_context, weights

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
        attention_flat, weights = self.cross_attention(
            query,
            exogenous,
            exogenous,
            attn_mask=self._parallel_attention_mask(hidden.device),
            need_weights=need_weights,
            average_attn_weights=False,
        )
        attention = attention_flat.reshape(
            batch_size,
            self.feature_num,
            self.num_patches + 1,
            self.context_dim,
        )
        patch_attention = attention[:, :, : self.num_patches, :]
        global_attention = attention[:, :, self.num_patches :, :]
        fused_patch, global_bridge, _ = self._global_mediated_patch_context(
            patch_attention,
            global_query,
            global_attention,
        )
        delta_patch = self.patch_output_projection(fused_patch)
        delta_all = delta_patch.reshape(batch_size, self.feature_num, -1)[
            :, :, : self.seq_len
        ]
        hidden_out = hidden + self.gamma_teb * delta_all
        exo_context = global_bridge[:, self.target_idx, :]
        return hidden_out, exo_context, weights


__all__ = [
    "GLOBAL_GATE_IDENTITY_INIT",
    "GLOBAL_GATE_INPUT_CONTRACT",
    "GLOBAL_GATE_SCALAR_PER_PATCH",
    "GLOBAL_MEDIATED_PATCH_V1",
    "GLOBAL_RESIDUAL_CONTRACT",
    "PATCH_ATTENTION_RESIDUAL_NONE",
    "GlobalMediatedPatchTargetExogenousBridge",
]
