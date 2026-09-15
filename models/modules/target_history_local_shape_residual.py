"""Target-only signed-history residual; M4 THLS, distinct from PMCR/P2."""
from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn

from models.modules.modern_conv_refinement import ReparamLargeKernelDWConv

IMPLEMENTATION_VARIANT = "amd-m4-target-history-local-shape-v1"
DEVELOPMENT_PROTOCOL = "m4_target_history_local_shape_two_arm_from_scratch_v1"
ETTM1_DEVELOPMENT_PROTOCOL = "m4_thls_ettm1_full_horizon_safety_two_arm_from_scratch_v1"
CONTROL_ABLATION_ID = "M4_THLS_CONTROL"
ABLATION_ID = "M4_THLS"
STRUCTURE_CONTRACT_VERSION = "target_history_signed_diff_residual_v1"
INITIALIZATION_POLICY = "matched_amd_and_isolated_target_history_local_shape_v1"
MODULE_CONNECTION = "X->RevIN(z)->MDM(U)->DDI; target+=THLS(z_target); AMS_selector<-U"


def configuration(seq_len=12, kernel_small=3, kernel_large=7):
    """Documented structure only; this does not authorize a training workload."""
    return {
        "structure_contract_version": STRUCTURE_CONTRACT_VERSION,
        "input_representation": "original_revin_target_history_pre_mdm_ddi",
        "insertion_point": "post_last_ddi_target_add_before_ams",
        "feature_order": ["y", "signed_d1", "signed_d2"],
        "difference_boundaries": "d1_first_zero_d2_first_two_zero",
        "seq_len": seq_len, "hidden_dim": 8,
        "kernel_small": kernel_small, "kernel_large": kernel_large,
        "padding": "zeros_same", "stride": 1, "dilation": 1,
        "normalization": "feature_wise_layernorm", "layernorm_eps": 1e-5,
        "ffn_ratio": 2, "ffn_internal_residual": False,
        "gelu_approximate": "none", "dropout": 0.1,
        "eta_init": 1e-3, "eta_constraint": "none",
        "projection_init": "xavier_uniform_gain1_bias0",
        "output_projection_zero_init": False, "model_form": "train",
    }


def interface_contract(enabled, *, seq_len=12, kernel_small=3, kernel_large=7):
    return {
        "contract_version": STRUCTURE_CONTRACT_VERSION,
        "initialization_policy": INITIALIZATION_POLICY,
        "run_seed": 2024, "train_generator_seed": 2024,
        "local_shape_init_seed": 2024, "branch_instantiated": enabled,
        "configuration": configuration(seq_len, kernel_small, kernel_large), "model_form": "train",
        "input_reorder": "none", "future_observed_covariates": False,
        "loss_scope": "specified_target_full_model_pred_len",
        "metric_scope": "all_valid_target_elements",
        "metric_space": "train-standardized",
        "metric_aggregation": "sum_SSE_SAE_divide_Q",
        "evaluation_tail": "keep_all",
        "best_selection": "finite_validation_mse_strict_decrease_earlier_tie",
    }


class TargetHistoryLocalShapeResidual(nn.Module):
    """Return eta*r, not y+eta*r. Sample sharing does not mix samples."""

    def __init__(self, seq_len, kernel_small, kernel_large, *, deploy=False):
        super().__init__()
        if type(seq_len) is not int or seq_len < 3:
            raise ValueError("THLS seq_len must be an integer >= 3")
        if (type(kernel_small) is not int or type(kernel_large) is not int
                or not 0 < kernel_small < kernel_large <= seq_len
                or kernel_small % 2 != 1 or kernel_large % 2 != 1):
            raise ValueError("THLS requires explicit odd small < large <= seq_len kernels")
        if type(deploy) is not bool:
            raise TypeError("deploy must be bool")
        self.seq_len = seq_len
        self.input_projection = nn.Conv1d(3, 8, 1, bias=True)
        self.temporal_conv = ReparamLargeKernelDWConv(
            8, kernel_small, kernel_large, deploy=deploy)
        self.feature_norm = nn.LayerNorm(8, eps=1e-5)
        self.ffn_expand = nn.Conv1d(8, 16, 1, bias=True)
        self.activation = nn.GELU(approximate="none")
        self.dropout1 = nn.Dropout(0.1)
        self.ffn_reduce = nn.Conv1d(16, 8, 1, bias=True)
        self.dropout2 = nn.Dropout(0.1)
        self.output_projection = nn.Conv1d(8, 1, 1, bias=True)
        self.eta = nn.Parameter(torch.tensor(1e-3))
        for layer in (self.input_projection, self.ffn_expand,
                      self.ffn_reduce, self.output_projection):
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        nn.init.ones_(self.feature_norm.weight)
        nn.init.zeros_(self.feature_norm.bias)

    @property
    def deploy(self):
        return self.temporal_conv.deploy

    def compute_features(self, y):
        if not torch.is_tensor(y) or not y.is_floating_point():
            raise TypeError("THLS expects a floating target history tensor")
        if y.ndim != 2 or y.shape[0] == 0 or y.shape[1] != self.seq_len:
            raise ValueError(f"THLS expects nonempty [B,{self.seq_len}] target history")
        d1 = torch.cat((torch.zeros_like(y[:, :1]), y[:, 1:] - y[:, :-1]), 1)
        d2 = torch.cat((torch.zeros_like(y[:, :2]),
                        y[:, 2:] - 2*y[:, 1:-1] + y[:, :-2]), 1)
        return torch.stack((y, d1, d2), dim=1)

    def _delta_from_features(self, features):
        value = self.temporal_conv(self.input_projection(features))
        value = self.feature_norm(value.transpose(1, 2)).transpose(1, 2)
        value = self.dropout1(self.activation(self.ffn_expand(value)))
        value = self.dropout2(self.ffn_reduce(value))
        return self.output_projection(value)

    def compute_delta(self, y):
        return self._delta_from_features(self.compute_features(y))

    def compute_components(self, y):
        features = self.compute_features(y)
        delta = self._delta_from_features(features)
        return {"features": features, "delta": delta, "residual": self.eta*delta}

    def forward(self, y):
        return self.eta*self.compute_delta(y)

    def switch_to_deploy(self):
        self.temporal_conv.switch_to_deploy()
        return self

    def to_deploy(self):
        result = deepcopy(self).eval()
        return result.switch_to_deploy()

    def load_state_dict(self, state_dict, strict=True):
        if strict is not True:
            raise ValueError("THLS restore requires strict=True and matching model form")
        if not isinstance(state_dict, Mapping):
            raise TypeError("THLS state must be a mapping")
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise RuntimeError("THLS state key/model form mismatch before loading")
        for key, value in state_dict.items():
            if (not torch.is_tensor(value) or value.shape != expected[key].shape
                    or value.dtype != expected[key].dtype
                    or not bool(torch.isfinite(value).all())):
                raise RuntimeError(f"THLS tensor metadata/finite mismatch before loading: {key}")
        snapshot = {key: value.detach().clone() for key, value in expected.items()}
        try:
            return super().load_state_dict(state_dict, strict=True)
        except BaseException:
            super().load_state_dict(snapshot, strict=True)
            raise
