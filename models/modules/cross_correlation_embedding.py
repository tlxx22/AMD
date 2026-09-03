"""CrossLinear-inspired cross-correlation embedding for AMD.

This is an independent implementation of the retained cross-correlation
embedding idea. It deliberately excludes CrossLinear normalization,
patch/position embeddings, and forecasting head.
"""

from collections.abc import Iterable
import math

import torch
from torch import nn
from torch.nn import functional as F


TARGET_EXOGENOUS = "target_exogenous"
PARALLEL_MULTIVARIATE = "parallel_multivariate"
SUPPORTED_TASK_MODES = (TARGET_EXOGENOUS, PARALLEL_MULTIVARIATE)

ZERO_SAME = "zero_same"
ORDERED_AUX_THEN_TARGET = "ordered_aux_then_target"
FEATURE_SCHEMA_ORDER = "feature_schema_order"
IDENTITY_RESIDUAL_DELTA_V1 = "identity_residual_delta_v1"
SIGMOID_LOGIT_PLUS_RHO = "sigmoid_logit_plus_rho"
REVIN_REUSE_NO_INTERNAL_NORMALIZATION = "reuse_amd_revin_no_internal_normalization"
LEGACY_WIDTH_COMPATIBILITY_ZERO = "legacy_width_compatibility_zero"
CCE_INSERTION_POINT = "after_revin_before_mdm"
EARLY_CCE_ARCHITECTURE = "crosslinear_inspired_observation_space_cce_v1"
EARLY_CCE_INPUT_REPRESENTATION = "amd_revin_normalized_x_ch"
LATE_CCE_ARCHITECTURE = "crosslinear_inspired_hidden_state_late_cce_v1"
LATE_CCE_INSERTION_POINT = "post_pmcr_pre_ams"
LATE_CCE_INPUT_REPRESENTATION = "amd_hidden_v_local"
CCE_SOURCE_IMPORT_CONTRACT_VERSION = "cce_source_import_contract_v1"


def _ordered_indices(value, *, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be an ordered iterable of integers")
    result = tuple(value)
    for index in result:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"{name} must contain only non-bool integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate indices")
    return result


class CrossCorrelationEmbedding(nn.Module):
    """Identity-preserving k=3 cross-variable residual embedding.

    Inputs and outputs use AMD channel-first representation [B, C, T].
    The learnable convolution is represented directly as zero-initialized
    delta parameters so construction consumes no random numbers.
    """

    def __init__(
        self,
        *,
        feature_num: int,
        task_mode: str,
        target_idx: int,
        aux_idx=(),
        kernel_size: int = 3,
        lambda_init: float = 0.1,
        padding_policy: str = ZERO_SAME,
        input_order_policy: str | None = None,
        parameterization_policy: str = IDENTITY_RESIDUAL_DELTA_V1,
    ):
        super().__init__()

        if (
            isinstance(feature_num, bool)
            or not isinstance(feature_num, int)
            or feature_num <= 0
        ):
            raise ValueError("feature_num must be a positive integer")
        if task_mode not in SUPPORTED_TASK_MODES:
            raise ValueError(
                f"task_mode must be one of {SUPPORTED_TASK_MODES}, got {task_mode!r}"
            )
        if (
            isinstance(target_idx, bool)
            or not isinstance(target_idx, int)
            or not 0 <= target_idx < feature_num
        ):
            raise ValueError(
                f"target_idx must index one of {feature_num} features, "
                f"got {target_idx!r}"
            )
        ordered_aux = _ordered_indices(aux_idx, name="aux_idx")
        out_of_range = [index for index in ordered_aux if not 0 <= index < feature_num]
        if out_of_range:
            raise ValueError(f"aux_idx contains out-of-range indices {out_of_range}")
        if target_idx in ordered_aux:
            raise ValueError("aux_idx must exclude target_idx")
        if (
            isinstance(kernel_size, bool)
            or not isinstance(kernel_size, int)
            or kernel_size != 3
        ):
            raise ValueError("CCE v1 kernel_size is fixed at 3")
        if (
            isinstance(lambda_init, bool)
            or not isinstance(lambda_init, (int, float))
            or not math.isfinite(lambda_init)
            or float(lambda_init) != 0.1
        ):
            raise ValueError("CCE v1 lambda_init is fixed at 0.1")
        if padding_policy != ZERO_SAME:
            raise ValueError(f"CCE v1 padding_policy is fixed at {ZERO_SAME!r}")
        if parameterization_policy != IDENTITY_RESIDUAL_DELTA_V1:
            raise ValueError(
                "CCE v1 parameterization_policy is fixed at "
                f"{IDENTITY_RESIDUAL_DELTA_V1!r}"
            )

        expected_order = (
            ORDERED_AUX_THEN_TARGET
            if task_mode == TARGET_EXOGENOUS
            else FEATURE_SCHEMA_ORDER
        )
        if input_order_policy is None:
            input_order_policy = expected_order
        if input_order_policy != expected_order:
            raise ValueError(
                f"{task_mode} requires input_order_policy={expected_order!r}, "
                f"got {input_order_policy!r}"
            )
        if task_mode == TARGET_EXOGENOUS:
            if not ordered_aux:
                raise ValueError("target_exogenous CCE requires non-empty aux_idx")
            source_idx = (*ordered_aux, target_idx)
            weight_shape = (1, len(source_idx), kernel_size)
        else:
            if ordered_aux:
                raise ValueError(
                    "parallel_multivariate CCE uses feature schema order; "
                    "aux_idx must be empty"
                )
            if feature_num < 2:
                raise ValueError(
                    "parallel_multivariate CCE requires at least two variables"
                )
            source_idx = tuple(range(feature_num))
            weight_shape = (feature_num, feature_num, kernel_size)

        self.feature_num = feature_num
        self.task_mode = task_mode
        self.target_idx = target_idx
        self.aux_idx = ordered_aux
        self.source_idx = source_idx
        self.kernel_size = kernel_size
        self.lambda_init = float(lambda_init)
        self.padding_policy = padding_policy
        self.input_order_policy = input_order_policy
        self.parameterization_policy = parameterization_policy
        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.groups = 1
        self.bias = True

        # Direct zero allocation is intentional: unlike nn.Conv1d
        # construction followed by zeroing, this does not consume RNG.
        self.delta_weight = nn.Parameter(torch.zeros(weight_shape))
        self.delta_bias = nn.Parameter(torch.zeros(weight_shape[0]))
        self.rho = nn.Parameter(torch.zeros(()))

    def effective_lambda(self) -> torch.Tensor:
        """Return sigmoid(logit(0.1) + rho), exactly 0.1 when rho is zero."""

        lambda_zero = self.rho.new_tensor(self.lambda_init)
        logit_zero = torch.log(lambda_zero) - torch.log1p(-lambda_zero)
        stable = torch.sigmoid(logit_zero + self.rho)
        # Correct only the one-ULP float64 round-trip at rho==0.  The detached
        # correction preserves sigmoid derivative and bounded behavior.
        exact_at_zero = stable + (lambda_zero - stable).detach()
        return torch.where(
            self.rho.detach() == 0,
            exact_at_zero,
            stable,
        )

    def _validate_input(self, x_ch: torch.Tensor) -> None:
        if not torch.is_tensor(x_ch):
            raise TypeError("CCE input must be a torch.Tensor")
        if x_ch.ndim != 3:
            raise ValueError(
                f"CCE expects [batch, channel, time], got {tuple(x_ch.shape)}"
            )
        if x_ch.shape[0] <= 0 or x_ch.shape[2] <= 0:
            raise ValueError("CCE requires non-empty batch and time dimensions")
        if x_ch.shape[1] != self.feature_num:
            raise ValueError(
                f"CCE expects {self.feature_num} channels, got {x_ch.shape[1]}"
            )
        if not x_ch.is_floating_point():
            raise TypeError("CCE input must use a floating dtype")
        if (
            x_ch.dtype != self.delta_weight.dtype
            or x_ch.device != self.delta_weight.device
        ):
            raise ValueError(
                "CCE input dtype/device must match CCE parameters, got "
                f"{x_ch.dtype}/{x_ch.device} and "
                f"{self.delta_weight.dtype}/{self.delta_weight.device}"
            )

    def compute_ungated_delta(self, x_ch: torch.Tensor) -> torch.Tensor:
        """Return the raw convolutional delta before applying lambda."""

        self._validate_input(x_ch)
        if self.task_mode == TARGET_EXOGENOUS:
            source_index = torch.tensor(
                self.source_idx,
                dtype=torch.long,
                device=x_ch.device,
            )
            conv_input = x_ch.index_select(1, source_index)
        else:
            conv_input = x_ch
        return F.conv1d(
            conv_input,
            self.delta_weight,
            self.delta_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def equivalent_crosslinear_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return selector-identity plus gated delta in module input order."""

        weight = self.effective_lambda() * self.delta_weight
        bias = self.effective_lambda() * self.delta_bias
        identity = torch.zeros_like(weight)
        center = self.kernel_size // 2
        if self.task_mode == TARGET_EXOGENOUS:
            # target is always the final source by the locked input order.
            identity[0, len(self.source_idx) - 1, center] = 1
        else:
            diagonal = torch.arange(self.feature_num, device=identity.device)
            identity[diagonal, diagonal, center] = 1
        return identity + weight, bias

    def forward(self, x_ch: torch.Tensor) -> torch.Tensor:
        delta = self.compute_ungated_delta(x_ch)
        gated_delta = self.effective_lambda() * delta
        if self.task_mode == PARALLEL_MULTIVARIATE:
            return x_ch + gated_delta

        target_selector = F.one_hot(
            torch.tensor(self.target_idx, device=x_ch.device),
            num_classes=self.feature_num,
        ).to(dtype=x_ch.dtype)
        target_residual = gated_delta * target_selector.view(1, -1, 1)
        # Preserve the direct identity path for bit-exact first-step gradients.
        return x_ch + target_residual

    def extra_repr(self) -> str:
        return (
            f"feature_num={self.feature_num}, task_mode={self.task_mode!r}, "
            f"target_idx={self.target_idx}, aux_idx={self.aux_idx}, "
            f"kernel_size={self.kernel_size}, lambda_init={self.lambda_init}"
        )
