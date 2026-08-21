from collections.abc import Mapping

import torch

from models.modules.modern_conv_refinement import (
    PeakPreservingModernConvRefinement,
)
from models.tsAMD import AMD


class AMDEnhanced(AMD):
    """AMD extension point with optional PMCR and raw state-source output."""

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
        use_pmcr=False,
        pmcr_hidden_dim=None,
        pmcr_kernel_small=None,
        pmcr_kernel_large=None,
        pmcr_dropout=0.1,
        pmcr_gamma_init=1e-3,
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

        self.target_idx = target_idx
        self.teb_context_dim = teb_context_dim

        if not isinstance(use_pmcr, bool):
            raise TypeError(
                "AMDEnhanced use_pmcr must be bool, "
                f"got {type(use_pmcr).__name__}"
            )
        self.use_pmcr = use_pmcr
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

    def load_amd_backbone_state_dict(self, state_dict):
        """Load frozen AMD weights with an exact PMCR-only missing allowlist.

        Complete PMCR checkpoints must continue to use ``load_state_dict`` with
        ``strict=True``. This importer is solely for initializing the enhanced
        model from a frozen AMD checkpoint.
        """

        if not isinstance(state_dict, Mapping):
            raise TypeError("AMD backbone state_dict must be a mapping")

        current_state = self.state_dict()
        current_keys = set(current_state)
        pmcr_keys = {key for key in current_keys if key.startswith("pmcr.")}
        backbone_keys = current_keys - pmcr_keys
        incoming_keys = set(state_dict)
        missing_backbone = sorted(backbone_keys - incoming_keys)
        unexpected = sorted(incoming_keys - backbone_keys)
        missing_pmcr = pmcr_keys - incoming_keys

        if missing_backbone or unexpected or missing_pmcr != pmcr_keys:
            raise RuntimeError(
                "AMD backbone checkpoint key contract failed: "
                f"missing_non_pmcr={missing_backbone}, "
                f"unexpected={unexpected}, "
                f"allowed_missing_pmcr={sorted(pmcr_keys)}"
            )

        if not pmcr_keys:
            return self.load_state_dict(state_dict, strict=True)

        completed_state = current_state.copy()
        completed_state.update(state_dict)
        return self.load_state_dict(completed_state, strict=True)

    def forward(self, x, return_state_source=False):
        """Run the frozen AMD path and optionally return its state source."""

        if not isinstance(return_state_source, bool):
            raise TypeError(
                "AMDEnhanced return_state_source must be bool, "
                f"got {type(return_state_source).__name__}"
            )

        # The validation and prediction path below deliberately mirrors
        # amd_reproduced_baseline_v1:models/tsAMD.py.
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

        if self.norm:
            x = self.rev_norm(x, "norm")

        x = torch.transpose(x, 1, 2)

        # Frozen paper-close inter-module connection:
        #   x_ch -> MDM(u_mdm) -> DDI(v); AMS(v, u_mdm).
        u_mdm = self.pastmixing(x)
        v = u_mdm
        for fc_block in self.fc_blocks:
            v = fc_block(v)

        if self.use_pmcr:
            v = self.pmcr(v)

        # TEB does not exist in M2. Its fixed-width context is therefore a
        # deterministic zero tensor inherited from v's dtype and device.
        exo_context = v.new_zeros((v.shape[0], self.teb_context_dim))

        x, moe_loss = self.moe(v, u_mdm)
        x = torch.transpose(x, 1, 2)

        if self.norm:
            x = self.rev_norm(x, "denorm", self.target_slice)

        if self.target_slice:
            x = x[:, :, self.target_slice]

        if not return_state_source:
            return x, moe_loss

        state_source = torch.cat(
            (
                v[:, self.target_idx, :],
                u_mdm[:, self.target_idx, :],
                exo_context,
            ),
            dim=-1,
        )
        return x, moe_loss, state_source
