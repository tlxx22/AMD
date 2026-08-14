import math

import torch
import torch.nn as nn

from models.common import RevIN
from models.common import DDI
from models.common import MDM
from models.tsmoe import AMS


class AMD(nn.Module):
    """Implementation of AMD."""

    def __init__(self, input_shape, pred_len, n_block, dropout, patch, k, c, alpha, target_slice, norm=True, layernorm=True):
        super(AMD, self).__init__()

        if not isinstance(input_shape, (tuple, list, torch.Size)) or len(input_shape) != 2:
            raise ValueError(
                "AMD input_shape must be (seq_len, feature_num), "
                f"got {input_shape!r}"
            )
        seq_len, feature_num = input_shape
        for name, value in (("seq_len", seq_len), ("feature_num", feature_num)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"AMD {name} must be a positive integer, got {value!r}")
        if isinstance(n_block, bool) or not isinstance(n_block, int) or n_block < 0:
            raise ValueError(f"AMD n_block must be a non-negative integer, got {n_block!r}")
        if isinstance(patch, bool) or not isinstance(patch, int) or patch <= 0:
            raise ValueError(f"AMD patch must be a positive integer, got {patch!r}")
        if patch > seq_len or seq_len % patch != 0:
            raise ValueError(
                "AMD requires patch <= seq_len and seq_len divisible by patch, "
                f"got seq_len={seq_len}, patch={patch}"
            )
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or alpha < 0
        ):
            raise ValueError(f"AMD alpha must be finite and non-negative, got {alpha!r}")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError(f"AMD dropout must satisfy 0 <= dropout < 1, got {dropout!r}")
        if not isinstance(norm, bool):
            raise TypeError(f"AMD norm must be bool, got {type(norm).__name__}")
        if not isinstance(layernorm, bool):
            raise TypeError(f"AMD layernorm must be bool, got {type(layernorm).__name__}")

        self.seq_len = seq_len
        self.feature_num = feature_num
        # DDI's released internal norm1/norm2 remain BatchNorm1d. The
        # layernorm-controlled MDM/DDI entry normalizers are true LayerNorm.
        self._uses_batch_norm = n_block > 0 and seq_len > patch

        self.target_slice = target_slice
        self.norm = norm

        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])

        self.pastmixing = MDM(input_shape, k=k, c=c, layernorm=layernorm)

        self.fc_blocks = nn.ModuleList([DDI(input_shape, dropout=dropout, patch=patch, alpha=alpha, layernorm=layernorm)
                                        for _ in range(n_block)])

        self.moe = AMS(input_shape, pred_len, ff_dim=2048, dropout=dropout, num_experts=8, top_k=2)

    def forward(self, x):
        # [batch_size, seq_len, feature_num]

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

        # layer norm
        if self.norm:
            x = self.rev_norm(x, 'norm')
        # [batch_size, seq_len, feature_num]

        # [batch_size, seq_len, feature_num]
        x = torch.transpose(x, 1, 2)
        # [batch_size, feature_num, seq_len]

        # Paper-aligned inter-module connection:
        #   X --MDM--> U --DDI--> representation consumed by the experts.
        # U is also retained as the selector/time embedding input to AMS.
        u = self.pastmixing(x)
        ddi_output = u

        for fc_block in self.fc_blocks:
            ddi_output = fc_block(ddi_output)

        # MOE
        x, moe_loss = self.moe(ddi_output, u)  # seq_len -> pred_len

        # [batch_size, feature_num, pred_len]
        x = torch.transpose(x, 1, 2)
        # [batch_size, pred_len, feature_num]

        if self.norm:
            x = self.rev_norm(x, 'denorm', self.target_slice)
        # [batch_size, pred_len, feature_num]

        if self.target_slice:
            x = x[:, :, self.target_slice]

        return x, moe_loss
