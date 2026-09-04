"""Sonnet-inspired joint-wavelet MVCA target residual for AMD.

This is an independent implementation of the formula and retained behavior
audited for the M4 Sonnet S2 candidate.  It intentionally excludes Sonnet's
RevIN, Koopman operator, decoder, forecasting head, multi-block wrapper, and
the official repository's paper-undefined var_attn trace.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import torch
from torch import nn


TARGET_EXOGENOUS = "target_exogenous"

SONNET_IMPLEMENTATION_VARIANT = (
    "el-amd-m4-sonnet-mvca-wavelet-residual-v1"
)
SONNET_CONTROL_ABLATION_ID = "M4_SONNET_MVCA_CONTROL"
SONNET_CANDIDATE_ABLATION_ID = "M4_SONNET_MVCA"
SONNET_ARCHITECTURE_IDENTITY = (
    "sonnet_inspired_joint_wavelet_mvca_target_residual_v1"
)
SONNET_INPUT_IDENTITY = "amd_revin_normalized_target_exogenous"
SONNET_INSERTION_IDENTITY = "after_revin_before_mdm"
SONNET_DEVELOPMENT_PROTOCOL_ID = "m4_sonnet_mvca_from_scratch_pair_v1"

SONNET_D_MODEL = 64
SONNET_N_ATOMS = 8
SONNET_ALPHA = 0.5
SONNET_EPSILON = 1e-6
SONNET_ATTENTION_DROPOUT = 0.1
SONNET_GAMMA_INIT = 1e-3

SONNET_RAW_SOURCE_ORDER = "ordered_aux_then_target_target_last"
SONNET_LATENT_CONCAT_ORDER = "exogenous_then_target"
SONNET_WAVELET_TIME_GRID = "linspace_zero_one_inclusive_runtime_T"
SONNET_WAVELET_PARAMETERIZATION = (
    "unconstrained_standard_normal_d_by_K_by_three"
)
SONNET_FFT_AXIS_POLICY = "latent_last_dimension_d"
SONNET_COHERENCE_DENOMINATOR_POLICY = (
    "mean_psd_product_plus_epsilon_no_clamp"
)
SONNET_COHERENCE_SCALE_POLICY = "divide_by_sqrt_d"
SONNET_SOFTMAX_AXIS_POLICY = "time_dimension_T"
SONNET_VALUE_POLICY = "broadcast_time_weight_times_V_no_time_reduction"
SONNET_VAR_ATTN_POLICY = "deleted_no_replacement"
SONNET_RECONSTRUCTION_POLICY = "same_atoms_sum_over_K_no_koopman"
SONNET_READOUT_POLICY = "linear_d_to_one_xavier_uniform_zero_bias"
SONNET_RESIDUAL_POLICY = "target_only_unconstrained_global_scalar_gate"
SONNET_NORMALIZATION_POLICY = "reuse_amd_revin_no_internal_normalization"
SONNET_MODULE_INIT_SEED_POLICY = (
    "equal_main_run_seed_isolated_cpu_cuda_rng_restore"
)
SONNET_STATE_CONTEXT_POLICY = "deterministic_zero_placeholder_no_sonnet_context"
SONNET_LICENSE_STATUS = "license_text_missing_classifier_only"

SONNET_RETAINED_COMPONENTS = (
    "joint_embedding",
    "learnable_wavelet",
    "paper_defined_mvca",
    "same_atom_reconstruction",
    "minimal_linear_readout",
    "target_only_gated_residual",
)
SONNET_DELETED_COMPONENTS = (
    "sonnet_revin",
    "koopman_operator",
    "sonnet_decoder",
    "sonnet_forecasting_head",
    "multiblock_downsampling_wrapper",
    "horizon_dependent_head",
    "exogenous_context_pooling_or_head",
    "paper_appendix_d8_feature_head_split",
    "official_var_attn",
    "parallel_multivariate_sonnet_path",
)
SONNET_SOURCE_IDENTITY = {
    "paper_title": (
        "Sonnet: Spectral Operator Neural Network for Multivariable "
        "Time Series Forecasting"
    ),
    "authors": ["Yuxuan Shu", "Vasileios Lampos"],
    "conference": "AAAI 2026",
    "year": 2026,
    "doi": "10.1609/aaai.v40i30.39736",
    "pdf_sha256": (
        "b076e6fed68448d3c3382c96f6f6985a988ea019ef3c470353780385c4011079"
    ),
    "official_repo_url": "https://github.com/ClaudiaShu/Sonnet.git",
    "official_repo_commit": "bf3d4801d34c5e7261718490f287c6fb15cadfdb",
    "core_source_sha256": {
        "sonnet/mts_model/models/Sonnet.py": (
            "be4fd33b9d1eb4a4f09be0a325a8aa87d5efd5d754e184606fc8a5808769b684"
        ),
        "sonnet/mts_model/layers/RevIN.py": (
            "0139409a58e57aca7c7e5423346db3f9224c6e871fecead418797ec4977e756b"
        ),
        "sonnet/lightning/lightning_module.py": (
            "f25e2e9ee1d12444eabf4ad6616c14f4f77c9bdad9a6886091193c6eca744d62"
        ),
        "configs/model/sonnet.yaml": (
            "329463667b7bb4aa80cf1a7761c3ac6adc7091c8a2cfda63545e69fd2f756346"
        ),
        "setup.py": (
            "85a9f7773200d374a04ad42006a78efbf580c5680602367150b09ebd9979dcc7"
        ),
    },
    "license_status": SONNET_LICENSE_STATUS,
    "implementation_policy": (
        "independent_from_paper_formula_and_audited_behavior_no_source_copy"
    ),
}


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _ordered_indices(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be an ordered iterable of integers")
    result = tuple(value)
    for index in result:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"{name} must contain only non-bool integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate indices")
    return result


def _feature_schema(value: object, feature_num: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("feature_schema must be an ordered iterable of names")
    result = tuple(value)
    if len(result) != feature_num:
        raise ValueError(
            f"feature_schema must contain {feature_num} names, got {len(result)}"
        )
    if any(not isinstance(name, str) or not name for name in result):
        raise TypeError("feature_schema must contain only non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("feature_schema must not contain duplicate names")
    return result


class PaperDefinedMVCA(nn.Module):
    """Coherence weighting over time for each wavelet atom.

    The FFT is over the final latent dimension.  The resulting scalar
    coherence at every [batch, atom, time] position is normalized across
    time and broadcasts over V; there is no T-by-T matrix or time reduction.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = SONNET_D_MODEL,
        n_atoms: int = SONNET_N_ATOMS,
        epsilon: float = SONNET_EPSILON,
        attention_dropout: float = SONNET_ATTENTION_DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.n_atoms = _positive_int(n_atoms, "n_atoms")
        if self.hidden_dim != SONNET_D_MODEL:
            raise ValueError(f"MVCA hidden_dim is fixed at {SONNET_D_MODEL}")
        if self.n_atoms != SONNET_N_ATOMS:
            raise ValueError(f"MVCA n_atoms is fixed at {SONNET_N_ATOMS}")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon)
            or float(epsilon) != SONNET_EPSILON
        ):
            raise ValueError(f"MVCA epsilon is fixed at {SONNET_EPSILON}")
        if (
            isinstance(attention_dropout, bool)
            or not isinstance(attention_dropout, (int, float))
            or not math.isfinite(attention_dropout)
            or float(attention_dropout) != SONNET_ATTENTION_DROPOUT
        ):
            raise ValueError(
                f"MVCA attention_dropout is fixed at {SONNET_ATTENTION_DROPOUT}"
            )
        self.epsilon = float(epsilon)
        self.attention_dropout_probability = float(attention_dropout)

        self.qkv_projection = nn.Linear(
            self.hidden_dim, 3 * self.hidden_dim, bias=True
        )
        self.attention_dropout = nn.Dropout(self.attention_dropout_probability)
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
        )
        self.output_projection = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=True
        )

    def _validate_input(self, projected: torch.Tensor) -> None:
        if not torch.is_tensor(projected):
            raise TypeError("MVCA input must be a torch.Tensor")
        if projected.ndim != 4:
            raise ValueError(
                "MVCA requires wavelet-space [B,K,T,d], "
                f"got {tuple(projected.shape)}"
            )
        if projected.shape[0] <= 0 or projected.shape[2] <= 0:
            raise ValueError("MVCA requires non-empty batch and time dimensions")
        if projected.shape[1] != self.n_atoms or projected.shape[3] != self.hidden_dim:
            raise ValueError(
                f"MVCA expects [B,{self.n_atoms},T,{self.hidden_dim}], "
                f"got {tuple(projected.shape)}"
            )
        if not projected.is_floating_point():
            raise TypeError("MVCA input must use a real floating dtype")
        parameter = self.qkv_projection.weight
        if projected.dtype != parameter.dtype or projected.device != parameter.device:
            raise ValueError(
                "MVCA input dtype/device must match its parameters, got "
                f"{projected.dtype}/{projected.device} and "
                f"{parameter.dtype}/{parameter.device}"
            )

    def project_qkv(
        self, projected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_input(projected)
        return self.qkv_projection(projected).chunk(3, dim=-1)

    def coherence(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return coherence and the complex spectra used to construct it."""

        if query.shape != key.shape:
            raise ValueError("MVCA query and key shapes must match")
        query_frequency = torch.fft.rfft(query, dim=-1)
        key_frequency = torch.fft.rfft(key, dim=-1)
        cross_spectral_density = (
            query_frequency * torch.conj(key_frequency)
        ).mean(dim=-1)
        query_power = (
            query_frequency * torch.conj(query_frequency)
        ).mean(dim=-1).real
        key_power = (
            key_frequency * torch.conj(key_frequency)
        ).mean(dim=-1).real
        coherence = cross_spectral_density.abs().square() / (
            query_power * key_power + self.epsilon
        )
        return coherence, query_frequency, key_frequency, cross_spectral_density

    def forward_with_diagnostics(
        self, projected: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query, key, value = self.project_qkv(projected)
        coherence, query_frequency, key_frequency, cross_spectral_density = (
            self.coherence(query, key)
        )
        logits = coherence / math.sqrt(self.hidden_dim)
        probabilities = torch.softmax(logits, dim=2)
        attention = self.attention_dropout(probabilities)
        weighted_value = attention.unsqueeze(-1) * value
        hidden = weighted_value + self.residual_mlp(weighted_value)
        output = self.output_projection(hidden)
        return output, {
            "query": query,
            "key": key,
            "value": value,
            "query_frequency": query_frequency,
            "key_frequency": key_frequency,
            "cross_spectral_density": cross_spectral_density,
            "coherence": coherence,
            "logits": logits,
            "probabilities": probabilities,
            "attention": attention,
            "weighted_value": weighted_value,
            "hidden": hidden,
        }

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_diagnostics(projected)
        return output


class SonnetMVCATargetResidual(nn.Module):
    """Target-only Sonnet S2 residual over AMD's RevIN-normalized input."""

    def __init__(
        self,
        *,
        seq_len: int,
        feature_num: int,
        task_mode: str,
        target_idx: int,
        ordered_aux_idx: Iterable[int],
        feature_schema: Iterable[str],
        schema_fingerprint: str,
        d_model: int = SONNET_D_MODEL,
        n_atoms: int = SONNET_N_ATOMS,
        alpha: float = SONNET_ALPHA,
        epsilon: float = SONNET_EPSILON,
        attention_dropout: float = SONNET_ATTENTION_DROPOUT,
        gamma_init: float = SONNET_GAMMA_INIT,
    ) -> None:
        super().__init__()
        self.seq_len = _positive_int(seq_len, "seq_len")
        self.feature_num = _positive_int(feature_num, "feature_num")
        self.d_model = _positive_int(d_model, "d_model")
        self.n_atoms = _positive_int(n_atoms, "n_atoms")
        if task_mode != TARGET_EXOGENOUS:
            raise ValueError("Sonnet S2 supports only task_mode='target_exogenous'")
        if (
            isinstance(target_idx, bool)
            or not isinstance(target_idx, int)
            or not 0 <= target_idx < self.feature_num
        ):
            raise ValueError(
                f"target_idx must index one of {self.feature_num} features"
            )
        aux_idx = _ordered_indices(ordered_aux_idx, "ordered_aux_idx")
        if not aux_idx:
            raise ValueError("Sonnet S2 requires non-empty ordered_aux_idx")
        if target_idx in aux_idx:
            raise ValueError("ordered_aux_idx must exclude target_idx")
        out_of_range = [
            index for index in aux_idx if not 0 <= index < self.feature_num
        ]
        if out_of_range:
            raise ValueError(
                f"ordered_aux_idx contains out-of-range indices {out_of_range}"
            )
        schema = _feature_schema(feature_schema, self.feature_num)
        if not isinstance(schema_fingerprint, str) or not schema_fingerprint:
            raise ValueError("schema_fingerprint must be a non-empty string")
        if self.d_model != SONNET_D_MODEL:
            raise ValueError(f"Sonnet S2 d_model is fixed at {SONNET_D_MODEL}")
        if self.n_atoms != SONNET_N_ATOMS:
            raise ValueError(f"Sonnet S2 n_atoms is fixed at {SONNET_N_ATOMS}")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or float(alpha) != SONNET_ALPHA
        ):
            raise ValueError(f"Sonnet S2 alpha is fixed at {SONNET_ALPHA}")
        if (
            isinstance(gamma_init, bool)
            or not isinstance(gamma_init, (int, float))
            or not math.isfinite(gamma_init)
            or float(gamma_init) != SONNET_GAMMA_INIT
        ):
            raise ValueError(
                f"Sonnet S2 gamma_init is fixed at {SONNET_GAMMA_INIT}"
            )

        self.task_mode = task_mode
        self.target_idx = target_idx
        self.ordered_aux_idx = aux_idx
        self.source_idx = (*aux_idx, target_idx)
        self.feature_schema = schema
        self.schema_fingerprint = schema_fingerprint
        self.alpha = float(alpha)
        self.aux_embedding_dim = int(self.alpha * self.d_model)
        self.target_embedding_dim = self.d_model - self.aux_embedding_dim
        if self.aux_embedding_dim != 32 or self.target_embedding_dim != 32:
            raise ValueError("Sonnet S2 latent allocation is fixed at 32/32")

        self.aux_embedding = nn.Linear(
            len(self.ordered_aux_idx), self.aux_embedding_dim, bias=True
        )
        self.target_embedding = nn.Linear(
            1, self.target_embedding_dim, bias=True
        )
        self.freq_params = nn.Parameter(
            torch.randn(self.d_model, self.n_atoms, 3)
        )
        self.mvca = PaperDefinedMVCA(
            hidden_dim=self.d_model,
            n_atoms=self.n_atoms,
            epsilon=epsilon,
            attention_dropout=attention_dropout,
        )
        self.readout = nn.Linear(self.d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.gamma_sonnet = nn.Parameter(torch.tensor(float(gamma_init)))

    def _validate_input(self, normalized_input: torch.Tensor) -> None:
        if not torch.is_tensor(normalized_input):
            raise TypeError("Sonnet S2 input must be a torch.Tensor")
        if normalized_input.ndim != 3:
            raise ValueError(
                "Sonnet S2 expects [batch,time,feature], "
                f"got {tuple(normalized_input.shape)}"
            )
        expected = (self.seq_len, self.feature_num)
        if normalized_input.shape[0] <= 0 or tuple(normalized_input.shape[1:]) != expected:
            raise ValueError(
                f"Sonnet S2 expects [B,{self.seq_len},{self.feature_num}], "
                f"got {tuple(normalized_input.shape)}"
            )
        if not normalized_input.is_floating_point():
            raise TypeError("Sonnet S2 input must use a floating dtype")
        parameter = self.freq_params
        if normalized_input.dtype != parameter.dtype or normalized_input.device != parameter.device:
            raise ValueError(
                "Sonnet S2 input dtype/device must match its parameters, got "
                f"{normalized_input.dtype}/{normalized_input.device} and "
                f"{parameter.dtype}/{parameter.device}"
            )

    def gather_sources(
        self, normalized_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(normalized_input)
        index = torch.tensor(
            self.ordered_aux_idx, dtype=torch.long, device=normalized_input.device
        )
        auxiliary = normalized_input.index_select(2, index)
        target = normalized_input[:, :, self.target_idx : self.target_idx + 1]
        return auxiliary, target

    def joint_embedding(
        self, normalized_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        auxiliary, target = self.gather_sources(normalized_input)
        auxiliary_embedding = self.aux_embedding(auxiliary)
        target_embedding = self.target_embedding(target)
        embedding = torch.cat(
            (auxiliary_embedding, target_embedding), dim=-1
        )
        return embedding, auxiliary_embedding, target_embedding

    def wavelet_atoms(self, reference: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(reference):
            raise TypeError("wavelet atom reference must be a torch.Tensor")
        time = torch.linspace(
            0.0,
            1.0,
            steps=self.seq_len,
            dtype=reference.dtype,
            device=reference.device,
        ).view(1, 1, self.seq_len)
        time_squared = time.square()
        w_alpha = self.freq_params[:, :, 0].unsqueeze(-1)
        w_beta = self.freq_params[:, :, 1].unsqueeze(-1)
        w_gamma = self.freq_params[:, :, 2].unsqueeze(-1)
        return torch.exp(-w_alpha * time_squared) * torch.cos(
            w_beta * time + w_gamma * time_squared
        )

    def wavelet_transform(
        self, embedding: torch.Tensor, atoms: torch.Tensor
    ) -> torch.Tensor:
        expected_embedding = (self.seq_len, self.d_model)
        expected_atoms = (self.d_model, self.n_atoms, self.seq_len)
        if embedding.ndim != 3 or tuple(embedding.shape[1:]) != expected_embedding:
            raise ValueError(
                f"joint embedding must be [B,{self.seq_len},{self.d_model}]"
            )
        if tuple(atoms.shape) != expected_atoms:
            raise ValueError(
                f"wavelet atoms must be {expected_atoms}, got {tuple(atoms.shape)}"
            )
        return embedding.unsqueeze(1) * atoms.permute(1, 2, 0).unsqueeze(0)

    def compute_delta_with_diagnostics(
        self, normalized_input: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedding, auxiliary_embedding, target_embedding = self.joint_embedding(
            normalized_input
        )
        atoms = self.wavelet_atoms(embedding)
        projected = self.wavelet_transform(embedding, atoms)
        mvca_output, mvca_diagnostics = self.mvca.forward_with_diagnostics(projected)
        reconstructed = (
            mvca_output * atoms.permute(1, 2, 0).unsqueeze(0)
        ).sum(dim=1)
        delta = self.readout(reconstructed)
        diagnostics = {
            "embedding": embedding,
            "auxiliary_embedding": auxiliary_embedding,
            "target_embedding": target_embedding,
            "atoms": atoms,
            "projected": projected,
            "mvca_output": mvca_output,
            "reconstructed": reconstructed,
            "delta": delta,
            **{f"mvca_{key}": value for key, value in mvca_diagnostics.items()},
        }
        return delta, diagnostics

    def compute_delta(self, normalized_input: torch.Tensor) -> torch.Tensor:
        delta, _ = self.compute_delta_with_diagnostics(normalized_input)
        return delta

    def forward_with_diagnostics(
        self, normalized_input: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta, diagnostics = self.compute_delta_with_diagnostics(normalized_input)
        target = normalized_input[
            :, :, self.target_idx : self.target_idx + 1
        ]
        gated_delta = self.gamma_sonnet * delta
        output = normalized_input.clone()
        output[:, :, self.target_idx : self.target_idx + 1] = target + gated_delta
        diagnostics["gated_delta"] = gated_delta
        return output, diagnostics

    def forward(self, normalized_input: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_diagnostics(normalized_input)
        return output

    def extra_repr(self) -> str:
        return (
            f"seq_len={self.seq_len}, feature_num={self.feature_num}, "
            f"target_idx={self.target_idx}, ordered_aux_idx={self.ordered_aux_idx}, "
            f"d_model={self.d_model}, n_atoms={self.n_atoms}, alpha={self.alpha}"
        )
