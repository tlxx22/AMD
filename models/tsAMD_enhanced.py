import torch

from models.tsAMD import AMD


class AMDEnhanced(AMD):
    """M0-B pass-through AMD with an optional raw state-source output."""

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

        # TEB does not exist in M0-B. Its fixed-width context is therefore a
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
