"""Selective patch-conditioned target--exogenous bridge for AMD.

T3 is a single-factor extension of the frozen T2 bridge.  It preserves T2's
patch/global queries, exogenous variate tokens, vectorized cross-attention,
owner mask, and state-only global context.  Its only structural addition is
an identity-initialized scalar confidence gate applied after the shared patch
output projection, so the gate controls the complete patch residual including
the projection bias.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    RIGHT_ZERO_CROP,
    PatchConditionedTargetExogenousBridge,
)


SELECTIVE_PATCH_V1 = "selective_patch_v1"
PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION = (
    "scalar_per_patch_post_projection"
)
PATCH_GATE_INPUT_QUERY_AND_ATTENTION = "query_and_attention_response"
PATCH_GATE_ACTIVATION_TWO_SIGMOID = "two_sigmoid"
PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY = "explicit_zero_identity"
GLOBAL_PREDICTION_ROLE_STATE_ONLY = "state_only_forecast_disconnected"


class SelectivePatchTargetExogenousBridge(
    PatchConditionedTargetExogenousBridge
):
    """T2 patch attention with a post-projection scalar confidence gate."""

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
        patch_confidence_gate: str = (
            PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION
        ),
        patch_gate_input: str = PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
        patch_gate_activation: str = PATCH_GATE_ACTIVATION_TWO_SIGMOID,
        patch_gate_init: str = PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
        global_prediction_role: str = GLOBAL_PREDICTION_ROLE_STATE_ONLY,
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
            "patch_confidence_gate": (
                patch_confidence_gate,
                PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION,
            ),
            "patch_gate_input": (
                patch_gate_input,
                PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
            ),
            "patch_gate_activation": (
                patch_gate_activation,
                PATCH_GATE_ACTIVATION_TWO_SIGMOID,
            ),
            "patch_gate_init": (
                patch_gate_init,
                PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
            ),
            "global_prediction_role": (
                global_prediction_role,
                GLOBAL_PREDICTION_ROLE_STATE_ONLY,
            ),
        }
        mismatches = [
            f"{name}={actual!r} (expected {required!r})"
            for name, (actual, required) in expected.items()
            if actual != required
        ]
        if mismatches:
            raise ValueError("T3 contract mismatch: " + ", ".join(mismatches))

        self.patch_confidence_gate = patch_confidence_gate
        self.patch_gate_input = patch_gate_input
        self.patch_gate_activation = patch_gate_activation
        self.patch_gate_init = patch_gate_init
        self.global_prediction_role = global_prediction_role

        # Explicit Parameters preserve T2's RNG stream: their construction
        # performs no random initialization and their exact initial gate is 1.
        self.patch_confidence_gate_weight = nn.Parameter(
            torch.zeros(1, 2 * self.context_dim)
        )
        self.patch_confidence_gate_bias = nn.Parameter(torch.zeros(1))

    def compute_patch_confidence_gate(
        self,
        patch_query: torch.Tensor,
        patch_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scalar-per-patch gate with shape [..., N, 1]."""

        gate_input = torch.cat((patch_query, patch_attention), dim=-1)
        logits = F.linear(
            gate_input,
            self.patch_confidence_gate_weight,
            self.patch_confidence_gate_bias,
        )
        return 2.0 * torch.sigmoid(logits)

    def compute_raw_patch_delta(
        self,
        patch_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Return the ungated output-projected patch residual."""

        return self.patch_output_projection(patch_attention)

    def compute_effective_patch_delta(
        self,
        patch_query: torch.Tensor,
        patch_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Return the post-projection confidence-gated patch residual."""

        gate = self.compute_patch_confidence_gate(patch_query, patch_attention)
        return gate * self.compute_raw_patch_delta(patch_attention)

    def compute_patch_delta_components(
        self,
        patch_query: torch.Tensor,
        patch_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw delta, gate, and effective delta for diagnostics."""

        raw_delta = self.compute_raw_patch_delta(patch_attention)
        gate = self.compute_patch_confidence_gate(patch_query, patch_attention)
        return raw_delta, gate, gate * raw_delta

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
        patch_attention = context[:, : self.num_patches, :]
        exo_context = context[:, self.num_patches, :]
        _, _, effective_delta = self.compute_patch_delta_components(
            patch_query, patch_attention
        )
        delta_target = effective_delta.reshape(hidden.shape[0], -1)[
            :, : self.seq_len
        ]
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
        patch_attention = context[:, :, : self.num_patches, :]
        global_context = context[:, :, self.num_patches, :]
        _, _, effective_delta = self.compute_patch_delta_components(
            patch_query, patch_attention
        )
        delta_all = effective_delta.reshape(batch_size, self.feature_num, -1)[
            :, :, : self.seq_len
        ]
        hidden_out = hidden + self.gamma_teb * delta_all
        exo_context = global_context[:, self.target_idx, :]
        return hidden_out, exo_context, weights


__all__ = [
    "GLOBAL_PREDICTION_ROLE_STATE_ONLY",
    "PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION",
    "PATCH_GATE_ACTIVATION_TWO_SIGMOID",
    "PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY",
    "PATCH_GATE_INPUT_QUERY_AND_ATTENTION",
    "SELECTIVE_PATCH_V1",
    "SelectivePatchTargetExogenousBridge",
]
