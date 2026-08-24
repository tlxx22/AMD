from collections.abc import Iterable, Mapping
import math

import torch

from models.modules.modern_conv_refinement import (
    PeakPreservingModernConvRefinement,
)
from models.modules.target_exogenous_bridge import (
    PARALLEL_MULTIVARIATE,
    SUPPORTED_TASK_MODES,
    TARGET_EXOGENOUS,
    TargetExogenousBridge,
)
from models.tsAMD import AMD


def _ordered_aux_idx(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("AMDEnhanced aux_idx must be an ordered iterable of integers")
    result = tuple(value)
    for index in result:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                "AMDEnhanced aux_idx must contain only non-bool integers, "
                f"got {index!r}"
            )
    if len(set(result)) != len(result):
        raise ValueError("AMDEnhanced aux_idx must not contain duplicate indices")
    return result


class AMDEnhanced(AMD):
    """AMD with optional PMCR, TEB, and fixed-width raw state source."""

    def __init__(
        self,
        input_shape,
        pred_len,
        n_block,
        dropout,
        patch,
        k,
        c,
        alpha,
        target_slice,
        norm=True,
        layernorm=True,
        *,
        target_idx,
        teb_context_dim,
        task_mode=None,
        aux_idx=(),
        use_pmcr=False,
        pmcr_hidden_dim=None,
        pmcr_kernel_small=None,
        pmcr_kernel_large=None,
        pmcr_dropout=0.1,
        pmcr_gamma_init=1e-3,
        use_teb=False,
        teb_heads=4,
        teb_dropout=0.1,
        teb_gamma_init=1e-3,
    ):
        super().__init__(
            input_shape=input_shape,
            pred_len=pred_len,
            n_block=n_block,
            dropout=dropout,
            patch=patch,
            k=k,
            c=c,
            alpha=alpha,
            target_slice=target_slice,
            norm=norm,
            layernorm=layernorm,
        )

        if (
            isinstance(target_idx, bool)
            or not isinstance(target_idx, int)
            or not 0 <= target_idx < self.feature_num
        ):
            raise ValueError(
                "AMDEnhanced target_idx must index one input feature, "
                f"got {target_idx!r} for feature_num={self.feature_num}"
            )
        if (
            isinstance(teb_context_dim, bool)
            or not isinstance(teb_context_dim, int)
            or teb_context_dim <= 0
        ):
            raise ValueError(
                "AMDEnhanced teb_context_dim must be a positive integer, "
                f"got {teb_context_dim!r}"
            )
        if task_mode is not None and task_mode not in SUPPORTED_TASK_MODES:
            raise ValueError(
                f"AMDEnhanced task_mode must be one of {SUPPORTED_TASK_MODES} "
                f"or None for the frozen legacy path, got {task_mode!r}"
            )
        if task_mode is not None and target_slice is not None:
            raise ValueError(
                "AMDEnhanced formal task modes require target_slice=None; "
                "target_idx is the unique target selector"
            )

        ordered_aux = _ordered_aux_idx(aux_idx)
        out_of_range = [
            index for index in ordered_aux if not 0 <= index < self.feature_num
        ]
        if out_of_range:
            raise ValueError(
                f"AMDEnhanced aux_idx contains out-of-range indices {out_of_range}"
            )
        if target_idx in ordered_aux:
            raise ValueError("AMDEnhanced aux_idx must exclude target_idx")
        if task_mode == PARALLEL_MULTIVARIATE and ordered_aux:
            raise ValueError(
                "parallel_multivariate uses all other variables; aux_idx must be empty"
            )

        if not isinstance(use_pmcr, bool):
            raise TypeError(
                "AMDEnhanced use_pmcr must be bool, "
                f"got {type(use_pmcr).__name__}"
            )
        if not isinstance(use_teb, bool):
            raise TypeError(
                "AMDEnhanced use_teb must be bool, "
                f"got {type(use_teb).__name__}"
            )
        if task_mode is None and use_teb:
            raise ValueError("AMDEnhanced use_teb=True requires an explicit task_mode")
        if task_mode is None and ordered_aux:
            raise ValueError(
                "AMDEnhanced legacy task_mode=None requires aux_idx to be empty"
            )
        if use_teb and (
            isinstance(teb_heads, bool)
            or not isinstance(teb_heads, int)
            or teb_heads <= 0
        ):
            raise ValueError(
                f"AMDEnhanced teb_heads must be a positive integer, got {teb_heads!r}"
            )
        if use_teb and teb_context_dim % teb_heads != 0:
            raise ValueError(
                "AMDEnhanced teb_context_dim must be divisible by teb_heads, "
                f"got {teb_context_dim} and {teb_heads}"
            )
        if use_teb and (
            isinstance(teb_dropout, bool)
            or not isinstance(teb_dropout, (int, float))
            or not math.isfinite(teb_dropout)
            or not 0 <= teb_dropout < 1
        ):
            raise ValueError(
                f"AMDEnhanced teb_dropout must satisfy 0 <= dropout < 1, "
                f"got {teb_dropout!r}"
            )
        if (
            isinstance(teb_gamma_init, bool)
            or not isinstance(teb_gamma_init, (int, float))
            or not math.isfinite(teb_gamma_init)
            or float(teb_gamma_init) != 1e-3
        ):
            raise ValueError(
                "AMDEnhanced teb_gamma_init is fixed at 1e-3, "
                f"got {teb_gamma_init!r}"
            )

        self.target_idx = target_idx
        self.teb_context_dim = teb_context_dim
        self.task_mode = task_mode
        self.aux_idx = ordered_aux
        self.use_pmcr = use_pmcr
        self.use_teb = use_teb

        self.pmcr = None
        if self.use_pmcr:
            required = {
                "pmcr_hidden_dim": pmcr_hidden_dim,
                "pmcr_kernel_small": pmcr_kernel_small,
                "pmcr_kernel_large": pmcr_kernel_large,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "AMDEnhanced use_pmcr=True requires explicit "
                    + ", ".join(missing)
                )
            if (
                isinstance(pmcr_kernel_large, bool)
                or not isinstance(pmcr_kernel_large, int)
                or pmcr_kernel_large > self.seq_len
            ):
                raise ValueError(
                    "AMDEnhanced requires pmcr_kernel_large <= seq_len, "
                    f"got {pmcr_kernel_large!r} for seq_len={self.seq_len}"
                )
            self.pmcr = PeakPreservingModernConvRefinement(
                hidden_dim=pmcr_hidden_dim,
                kernel_small=pmcr_kernel_small,
                kernel_large=pmcr_kernel_large,
                dropout=pmcr_dropout,
                gamma_init=pmcr_gamma_init,
            )

        self.teb = None
        if self.use_teb:
            self.teb = TargetExogenousBridge(
                seq_len=self.seq_len,
                feature_num=self.feature_num,
                task_mode=self.task_mode,
                target_idx=self.target_idx,
                aux_idx=self.aux_idx,
                context_dim=self.teb_context_dim,
                num_heads=teb_heads,
                dropout=teb_dropout,
                gamma_init=teb_gamma_init,
            )

    def _state_key_groups(self):
        current_keys = set(self.state_dict())
        pmcr_keys = {key for key in current_keys if key.startswith("pmcr.")}
        teb_keys = {key for key in current_keys if key.startswith("teb.")}
        backbone_keys = current_keys - pmcr_keys - teb_keys
        return current_keys, backbone_keys, pmcr_keys, teb_keys

    def load_enhancement_state_dict(self, state_dict, *, source_kind):
        """Load a declared baseline/PMCR-only/TEB-only source exactly.

        The complete incoming key set and tensor shapes are validated before
        any parameter is modified. Same-structure resume must use ordinary
        load_state_dict(strict=True) instead.
        """

        if not isinstance(state_dict, Mapping):
            raise TypeError("enhancement source state_dict must be a mapping")
        if source_kind not in {"baseline", "pmcr_only", "teb_only"}:
            raise ValueError(
                "source_kind must be one of baseline, pmcr_only, teb_only, "
                f"got {source_kind!r}"
            )

        current_state = self.state_dict()
        current_keys, backbone_keys, pmcr_keys, teb_keys = self._state_key_groups()
        if source_kind == "baseline":
            expected_source_keys = backbone_keys
        elif source_kind == "pmcr_only":
            if not pmcr_keys:
                raise RuntimeError(
                    "source_kind='pmcr_only' requires PMCR in the target model"
                )
            expected_source_keys = backbone_keys | pmcr_keys
        else:
            if not teb_keys:
                raise RuntimeError(
                    "source_kind='teb_only' requires TEB in the target model"
                )
            expected_source_keys = backbone_keys | teb_keys

        incoming_keys = set(state_dict)
        missing_source = sorted(expected_source_keys - incoming_keys)
        unexpected = sorted(incoming_keys - expected_source_keys)
        if missing_source or unexpected:
            raise RuntimeError(
                "enhancement checkpoint key contract failed: "
                f"source_kind={source_kind!r}, "
                f"missing_source_keys={missing_source}, "
                f"missing_non_pmcr={missing_source}, "
                f"unexpected={unexpected}, "
                f"allowed_missing_pmcr={sorted(pmcr_keys - expected_source_keys)}, "
                f"allowed_missing_teb={sorted(teb_keys - expected_source_keys)}"
            )

        metadata_errors = []
        for key in sorted(expected_source_keys):
            incoming = state_dict[key]
            expected = current_state[key]
            if not torch.is_tensor(incoming):
                metadata_errors.append(f"{key}: not a tensor")
            elif incoming.shape != expected.shape:
                metadata_errors.append(
                    f"{key}: shape {tuple(incoming.shape)} != {tuple(expected.shape)}"
                )
        if metadata_errors:
            raise RuntimeError(
                "enhancement checkpoint tensor contract failed before loading: "
                + "; ".join(metadata_errors)
            )

        if incoming_keys == current_keys:
            return self.load_state_dict(state_dict, strict=True)

        completed_state = current_state.copy()
        completed_state.update(state_dict)
        return self.load_state_dict(completed_state, strict=True)

    def load_amd_backbone_state_dict(self, state_dict):
        """Initialize enabled enhancements from the frozen AMD backbone."""

        return self.load_enhancement_state_dict(
            state_dict,
            source_kind="baseline",
        )

    def forward(self, x, return_state_source=False):
        """Run AMD with optional local/exogenous refinements and raw state source."""

        if not isinstance(return_state_source, bool):
            raise TypeError(
                "AMDEnhanced return_state_source must be bool, "
                f"got {type(return_state_source).__name__}"
            )

        if not torch.is_tensor(x):
            raise TypeError("AMD input must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(
                f"AMD expects [batch, sequence, feature], got shape {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError("AMD requires a non-empty batch")
        if x.shape[1] != self.seq_len or x.shape[2] != self.feature_num:
            raise ValueError(
                f"AMD expects [batch, {self.seq_len}, {self.feature_num}], "
                f"got {tuple(x.shape)}"
            )
        if self.training and self._uses_batch_norm and x.shape[0] < 2:
            raise ValueError(
                "AMD DDI internal normalization uses BatchNorm1d and requires "
                "training batch size >= 2"
            )

        normalized_input = self.rev_norm(x, "norm") if self.norm else x
        x_ch = torch.transpose(normalized_input, 1, 2)

        # Frozen paper-close inter-module connection:
        #   x_ch -> MDM(u_mdm) -> DDI(v); AMS(v_final, u_mdm).
        u_mdm = self.pastmixing(x_ch)
        v = u_mdm
        for fc_block in self.fc_blocks:
            v = fc_block(v)

        if self.use_pmcr:
            v = self.pmcr(v)
        v_local = v

        exo_context = v_local.new_zeros(
            (v_local.shape[0], self.teb_context_dim)
        )
        v_final = v_local
        if self.use_teb:
            v_final, exo_context = self.teb(
                hidden=v_local,
                normalized_input=normalized_input,
            )

        pred_all_norm, moe_loss = self.moe(v_final, u_mdm)
        pred_all_norm = torch.transpose(pred_all_norm, 1, 2)

        if self.task_mode is None:
            prediction = pred_all_norm
            if self.norm:
                prediction = self.rev_norm(
                    prediction, "denorm", self.target_slice
                )
            if self.target_slice:
                prediction = prediction[:, :, self.target_slice]
        else:
            pred_all = pred_all_norm
            if self.norm:
                pred_all = self.rev_norm(
                    pred_all_norm, "denorm", slice(None)
                )
            if self.task_mode == TARGET_EXOGENOUS:
                prediction = pred_all[
                    :, :, self.target_idx : self.target_idx + 1
                ]
            else:
                prediction = pred_all

        if not return_state_source:
            return prediction, moe_loss

        state_source = torch.cat(
            (
                v_final[:, self.target_idx, :],
                u_mdm[:, self.target_idx, :],
                exo_context,
            ),
            dim=-1,
        )
        return prediction, moe_loss, state_source
