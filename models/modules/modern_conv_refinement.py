"""ModernTCN-inspired, variable-independent temporal refinement for AMD.

This module intentionally retains only the large/small temporal depthwise
convolutions, ConvFFN1, structural reparameterization, and outer residual
needed by PMCR. It does not implement ModernTCN's patch stem, multi-stage
backbone, ConvFFN2, or prediction head.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_kernel_pair(kernel_small: object, kernel_large: object) -> tuple[int, int]:
    small = _positive_int(kernel_small, "kernel_small")
    large = _positive_int(kernel_large, "kernel_large")
    if small % 2 == 0 or large % 2 == 0:
        raise ValueError(
            "kernel_small and kernel_large must both be odd, "
            f"got {small} and {large}"
        )
    if large <= small:
        raise ValueError(
            "kernel_large must be strictly greater than kernel_small, "
            f"got {large} <= {small}"
        )
    return small, large


class ReparamLargeKernelDWConv(nn.Module):
    """Two temporal depthwise branches that can be fused for deployment."""

    def __init__(
        self,
        channels: int,
        kernel_small: int,
        kernel_large: int,
        *,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.kernel_small, self.kernel_large = _validate_kernel_pair(
            kernel_small, kernel_large
        )
        if not isinstance(deploy, bool):
            raise TypeError(f"deploy must be bool, got {type(deploy).__name__}")

        self.deploy = deploy
        if deploy:
            self.reparam_branch = self._make_conv(self.kernel_large)
            self._initialize_deploy_branch()
        else:
            self.large_branch = self._make_conv(self.kernel_large)
            self.small_branch = self._make_conv(self.kernel_small)
            self._initialize_training_branches()

    def _make_conv(self, kernel_size: int) -> nn.Conv1d:
        return nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=self.channels,
            bias=True,
            padding_mode="zeros",
        )

    def _initialize_training_branches(self) -> None:
        branch_scale = 1.0 / math.sqrt(2.0)
        with torch.no_grad():
            for branch in (self.large_branch, self.small_branch):
                nn.init.kaiming_uniform_(
                    branch.weight, mode="fan_in", nonlinearity="linear"
                )
                branch.weight.mul_(branch_scale)
                nn.init.zeros_(branch.bias)

    def _initialize_deploy_branch(self) -> None:
        # Direct deploy construction exists for strict deploy-checkpoint loads.
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.reparam_branch.weight,
                mode="fan_in",
                nonlinearity="linear",
            )
            nn.init.zeros_(self.reparam_branch.bias)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not torch.is_tensor(x):
            raise TypeError("ReparamLargeKernelDWConv input must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(
                "ReparamLargeKernelDWConv expects [batch, channel, time], "
                f"got shape {tuple(x.shape)}"
            )
        if any(size <= 0 for size in x.shape):
            raise ValueError(
                "ReparamLargeKernelDWConv requires non-empty batch/channel/time "
                f"dimensions, got {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"ReparamLargeKernelDWConv expects {self.channels} channels, "
                f"got {x.shape[1]}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        if self.deploy:
            return self.reparam_branch(x)
        return self.large_branch(x) + self.small_branch(x)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fused kernel and bias without changing this module."""

        if self.deploy:
            return self.reparam_branch.weight.clone(), self.reparam_branch.bias.clone()

        pad_each_side = (self.kernel_large - self.kernel_small) // 2
        small_kernel = F.pad(
            self.small_branch.weight,
            (pad_each_side, pad_each_side),
            mode="constant",
            value=0.0,
        )
        kernel = self.large_branch.weight + small_kernel
        bias = self.large_branch.bias + self.small_branch.bias
        return kernel, bias

    def switch_to_deploy(self) -> "ReparamLargeKernelDWConv":
        """Fuse the temporal branches in place; repeated calls are safe."""

        if self.deploy:
            return self

        kernel, bias = self.get_equivalent_kernel_bias()
        reference = self.large_branch.weight
        reparam_branch = self._make_conv(self.kernel_large).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        reparam_branch.train(self.training)
        with torch.no_grad():
            reparam_branch.weight.copy_(kernel)
            reparam_branch.bias.copy_(bias)

        self.reparam_branch = reparam_branch
        del self.large_branch
        del self.small_branch
        self.deploy = True
        return self

    def to_deploy(self) -> "ReparamLargeKernelDWConv":
        """Return an eval deployed deep copy, leaving this module untouched."""

        deployed = copy.deepcopy(self)
        deployed.eval()
        deployed.switch_to_deploy()
        return deployed


class PeakPreservingModernConvRefinement(nn.Module):
    """Variable-independent PMCR operating on [batch, variable, time]."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_small: int,
        kernel_large: int,
        dropout: float = 0.1,
        gamma_init: float = 1e-3,
        *,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        if hidden_dim < 2:
            raise ValueError(f"hidden_dim must be at least 2, got {hidden_dim}")
        kernel_small, kernel_large = _validate_kernel_pair(
            kernel_small, kernel_large
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
            or gamma_init == 0
        ):
            raise ValueError(
                f"gamma_init must be finite and non-zero, got {gamma_init!r}"
            )

        self.hidden_dim = hidden_dim
        self.kernel_small = kernel_small
        self.kernel_large = kernel_large

        self.input_projection = nn.Conv1d(1, hidden_dim, kernel_size=1, bias=True)
        self.temporal_conv = ReparamLargeKernelDWConv(
            channels=hidden_dim,
            kernel_small=kernel_small,
            kernel_large=kernel_large,
            deploy=deploy,
        )
        self.feature_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.ffn_expand = nn.Conv1d(
            hidden_dim, 2 * hidden_dim, kernel_size=1, bias=True
        )
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.ffn_reduce = nn.Conv1d(
            2 * hidden_dim, hidden_dim, kernel_size=1, bias=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.output_projection = nn.Conv1d(
            hidden_dim, 1, kernel_size=1, bias=True
        )
        self.gamma_pmcr = nn.Parameter(torch.tensor(float(gamma_init)))

        self._initialize_pointwise_and_norm()

    def _initialize_pointwise_and_norm(self) -> None:
        with torch.no_grad():
            for projection in (
                self.input_projection,
                self.ffn_expand,
                self.ffn_reduce,
                self.output_projection,
            ):
                nn.init.xavier_uniform_(projection.weight)
                nn.init.zeros_(projection.bias)
            nn.init.ones_(self.feature_norm.weight)
            nn.init.zeros_(self.feature_norm.bias)

    @staticmethod
    def _validate_hidden(hidden: torch.Tensor) -> None:
        if not torch.is_tensor(hidden):
            raise TypeError("PMCR input must be a torch.Tensor")
        if hidden.ndim != 3:
            raise ValueError(
                f"PMCR expects [batch, variable, time], got shape {tuple(hidden.shape)}"
            )
        if any(size <= 0 for size in hidden.shape):
            raise ValueError(
                "PMCR requires non-empty batch/variable/time dimensions, "
                f"got {tuple(hidden.shape)}"
            )

    def compute_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the unscaled variable-independent temporal residual."""

        self._validate_hidden(hidden)
        batch_size, variable_count, time_length = hidden.shape
        value = hidden.reshape(batch_size * variable_count, 1, time_length)
        value = self.input_projection(value)
        value = self.temporal_conv(value)
        value = self.feature_norm(value.transpose(1, 2)).transpose(1, 2)
        value = self.ffn_expand(value)
        value = self.activation(value)
        value = self.dropout1(value)
        value = self.ffn_reduce(value)
        value = self.dropout2(value)
        value = self.output_projection(value)
        return value.reshape(batch_size, variable_count, time_length)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        delta = self.compute_delta(hidden)
        return hidden + self.gamma_pmcr * delta

    @property
    def deploy(self) -> bool:
        return self.temporal_conv.deploy

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.temporal_conv.get_equivalent_kernel_bias()

    def switch_to_deploy(self) -> "PeakPreservingModernConvRefinement":
        self.temporal_conv.switch_to_deploy()
        return self

    def to_deploy(self) -> "PeakPreservingModernConvRefinement":
        """Return an eval deployed deep copy of the complete PMCR block."""

        deployed = copy.deepcopy(self)
        deployed.eval()
        deployed.switch_to_deploy()
        return deployed
