"""Project-local, variable-independent residual gate around the unchanged PMCR v1."""
from collections.abc import Mapping
import copy

import torch
from torch import nn

from models.modules.modern_conv_refinement import PeakPreservingModernConvRefinement

GATE_CONTRACT_VERSION = "pmcr_p2_absdiff_bounded_gate_v1"


def gate_configuration():
    """Fresh, serializable C-only scientific metadata; no module construction."""
    return {
        "input": "v_ddi[B,C,T]", "minimum_time_length": 3,
        "differences": "backward_D1_zero_first_D2_zero_first_two",
        "absolute_differences": True,
        "normalization": "a/(a+mean_t(a)+epsilon)",
        "epsilon": 1e-6, "statistics_axis": "time", "statistics_detached": False,
        "fold": "B*C_independent_shared_parameters",
        "conv1": {"in_channels": 2, "out_channels": 4, "kernel": 3,
                  "padding": 1, "padding_mode": "replicate", "bias": True},
        "activation": "GELU(approximate=none)",
        "conv2": {"in_channels": 4, "out_channels": 1, "kernel": 1,
                  "padding": 0, "bias": True},
        "output": "1+tanh(z)", "floating_range": [0, 2],
        "conv1_initialization": "xavier_uniform_gain1_bias0",
        "conv2_initialization": "weight0_bias0",
        "parameter_count": 33, "convolution_MAC_per_BCT": 28,
        "body_namespace": "pmcr_p2.body", "gate_namespace": "pmcr_p2.gate",
        "deploy": "fuse_body_temporal_branches_only_dynamic_gate_retained",
    }


class LocalChangeGate(nn.Module):
    """Shared bounded gate; statistics never mix samples or hidden variables."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 4, 3, padding=1, padding_mode="replicate", bias=True)
        self.activation = nn.GELU(approximate="none")
        self.conv2 = nn.Conv1d(4, 1, 1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight, gain=1.0)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    @staticmethod
    def _validate_hidden(hidden):
        if not torch.is_tensor(hidden):
            raise TypeError("P2 input must be a torch.Tensor")
        if hidden.ndim != 3 or any(size <= 0 for size in hidden.shape):
            raise ValueError("P2 expects non-empty [batch, variable, time]")
        if hidden.shape[-1] < 3:
            raise ValueError("P2 requires time length >= 3")
        if not hidden.is_floating_point():
            raise TypeError("P2 input must have a floating dtype")

    @staticmethod
    def compute_features(hidden):
        LocalChangeGate._validate_hidden(hidden)
        d1 = torch.cat((torch.zeros_like(hidden[..., :1]),
                        hidden[..., 1:] - hidden[..., :-1]), dim=-1)
        d2 = torch.cat((torch.zeros_like(hidden[..., :2]),
                        d1[..., 2:] - d1[..., 1:-1]), dim=-1)
        absolute = torch.stack((d1.abs(), d2.abs()), dim=2)
        return absolute / (absolute + absolute.mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, hidden):
        features = self.compute_features(hidden)
        batch, variables, _, length = features.shape
        logits = self.conv2(self.activation(self.conv1(
            features.reshape(batch * variables, 2, length))))
        return (1 + torch.tanh(logits)).reshape(batch, variables, length)


class LocalChangeGatedPMCR(nn.Module):
    """One fresh v1 body plus a dynamic gate, with isolated CPU construction."""

    def __init__(self, hidden_dim, kernel_small, kernel_large, dropout=0.1,
                 gamma_init=1e-3, *, body_init_seed=2024, gate_init_seed=2025,
                 deploy=False):
        super().__init__()
        if (type(body_init_seed) is not int or body_init_seed != 2024
                or type(gate_init_seed) is not int or gate_init_seed != 2025):
            raise ValueError("P2 body/gate initialization seeds must be 2024/2025")
        if hidden_dim != 8 or dropout != 0.1 or gamma_init != 1e-3:
            raise ValueError("P2 preserves the d=8/dropout=.1/gamma_init=1e-3 body")
        # CPU default generator only: neither current nor lazy CUDA seeds change.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(body_init_seed)
            self.body = PeakPreservingModernConvRefinement(
                hidden_dim, kernel_small, kernel_large, dropout, gamma_init,
                deploy=deploy)
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(gate_init_seed)
            self.gate = LocalChangeGate()

    def compute_base_delta(self, hidden):
        LocalChangeGate._validate_hidden(hidden)
        return self.body.compute_delta(hidden)

    def compute_gate(self, hidden):
        return self.gate(hidden)

    def compute_delta(self, hidden):
        return self.compute_gate(hidden) * self.compute_base_delta(hidden)

    def compute_components(self, hidden):
        # Exactly one body/dropout sample; every returned tensor retains autograd.
        base_delta = self.compute_base_delta(hidden)
        gate = self.compute_gate(hidden)
        effective_delta = gate * base_delta
        residual = self.body.gamma_pmcr * effective_delta
        return {"base_delta": base_delta, "gate": gate,
                "effective_delta": effective_delta, "residual": residual,
                "output": hidden + residual}

    def forward(self, hidden):
        return self.compute_components(hidden)["output"]

    @property
    def deploy(self):
        return self.body.deploy

    def switch_to_deploy(self):
        self.body.switch_to_deploy()
        return self

    def to_deploy(self):
        deployed = copy.deepcopy(self)
        deployed.eval()
        deployed.switch_to_deploy()
        return deployed

    def load_state_dict(self, state_dict, strict=True):
        if strict is not True:
            raise ValueError("P2 restore requires strict=True")
        if not isinstance(state_dict, Mapping):
            raise TypeError("P2 state_dict must be a mapping")
        current = self.state_dict()
        if set(state_dict) != set(current):
            raise RuntimeError("P2 checkpoint key/train-deploy form mismatch before loading")
        for key, expected in current.items():
            value = state_dict[key]
            if (not torch.is_tensor(value) or value.shape != expected.shape
                    or value.dtype != expected.dtype):
                raise RuntimeError("P2 checkpoint tensor mismatch before loading: " + key)
        snapshot = {key: value.detach().clone() for key, value in current.items()}
        try:
            return super().load_state_dict(state_dict, strict=True)
        except BaseException:
            super().load_state_dict(snapshot, strict=True)
            raise
