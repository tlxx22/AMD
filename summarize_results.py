"""Build deterministic summaries from completed AMD run artifacts only."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import uuid
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch

from models.modules.cross_correlation_embedding import (
    CCE_INSERTION_POINT,
    FEATURE_SCHEMA_ORDER,
    IDENTITY_RESIDUAL_DELTA_V1,
    LEGACY_WIDTH_COMPATIBILITY_ZERO,
    LATE_CCE_ARCHITECTURE,
    LATE_CCE_INPUT_REPRESENTATION,
    LATE_CCE_INSERTION_POINT,
    ORDERED_AUX_THEN_TARGET,
    REVIN_REUSE_NO_INTERNAL_NORMALIZATION,
    SIGMOID_LOGIT_PLUS_RHO,
    ZERO_SAME,
)
from models.modules import sonnet_mvca_target_residual as sonnet_spec
from utils.feature_schema import CANONICAL_FEATURE_NAMES

IMPLEMENTATION_VARIANT = "AMD-paper-norm-wd-ddi-v1"
ENHANCED_IMPLEMENTATION_VARIANT = "el-amd-pmcr-teb-v1"
T2_IMPLEMENTATION_VARIANT = "el-amd-m4-t2-patch-teb-v1"
T2G_IMPLEMENTATION_VARIANT = "el-amd-m4-t2g-global-mediated-patch-teb-v1"
T3_IMPLEMENTATION_VARIANT = "el-amd-m4-t3-selective-patch-teb-v1"
CCE_IMPLEMENTATION_VARIANT = "el-amd-m4-crosslinear-cce-v1"
LATE_CCE_IMPLEMENTATION_VARIANT = "el-amd-m4-crosslinear-late-cce-v1"
SONNET_IMPLEMENTATION_VARIANT = sonnet_spec.SONNET_IMPLEMENTATION_VARIANT
SUPPORTED_IMPLEMENTATION_VARIANTS = (
    IMPLEMENTATION_VARIANT,
    ENHANCED_IMPLEMENTATION_VARIANT,
    T2_IMPLEMENTATION_VARIANT,
    T2G_IMPLEMENTATION_VARIANT,
    T3_IMPLEMENTATION_VARIANT,
    CCE_IMPLEMENTATION_VARIANT,
    LATE_CCE_IMPLEMENTATION_VARIANT,
    SONNET_IMPLEMENTATION_VARIANT,
)
ENHANCED_ARTIFACT_SCHEMA_VERSION = 2
TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION = "target_exogenous_schema_v1"
ENHANCED_CHECKSUM_FILES = (
    "best.pt", "last.pt", "config.resolved.json", "history.jsonl",
    "metrics.json", "manifest.json", "sys.argv.json", "command.txt",
    "stdout.log", "stderr.log", "train.log", "source_fingerprint.json",
    "data_fingerprint.json",
)
SCHEMA_VERSION = 1
METRIC_SPACE = "train-standardized"
STANDARD_TRAINING_PROTOCOL = "standard_from_scratch"
CCE_DEVELOPMENT_PROTOCOL = "m4_crosslinear_cce_from_scratch_pair_v1"
CCE_CONTROL_ABLATION_ID = "M4_CCE_CONTROL"
CCE_CANDIDATE_ABLATION_ID = "M4_CCE"
LATE_CCE_DEVELOPMENT_PROTOCOL = "m4_crosslinear_late_cce_from_scratch_pair_v1"
LATE_CCE_CONTROL_ABLATION_ID = "M4_LATE_CCE_CONTROL"
LATE_CCE_CANDIDATE_ABLATION_ID = "M4_LATE_CCE"
SONNET_DEVELOPMENT_PROTOCOL = sonnet_spec.SONNET_DEVELOPMENT_PROTOCOL_ID
SONNET_CONTROL_ABLATION_ID = sonnet_spec.SONNET_CONTROL_ABLATION_ID
SONNET_CANDIDATE_ABLATION_ID = sonnet_spec.SONNET_CANDIDATE_ABLATION_ID
TRAIN_VALIDATION_TEST = "train_validation_test"
TRAIN_VALIDATION_ONLY = "train_validation_only"
M4_DEVELOPMENT_CANDIDATE = "m4_development_candidate"
T2_ADAPTER_TRAINING_PROTOCOL = "m4_t2_u1_warmstart_frozen_adapter_v1"
U1_CONTINUATION_TRAINING_PROTOCOL = "m4_u1_matched_budget_continuation_v1"
WARM_START_TRAINING_PROTOCOLS = (
    T2_ADAPTER_TRAINING_PROTOCOL,
    U1_CONTINUATION_TRAINING_PROTOCOL,
)
WARM_START_CONTRACT_VERSION = "warm_start_contract_v1"
SOURCE_COMPATIBILITY_PROOF_VERSION = "source_compatibility_proof_v1"
T2_ADAPTER_ABLATION_ID = "M4_T2_ADAPTER"
U1_CONTINUATION_ABLATION_ID = "M4_U1_CONTINUATION"
CROSSLINEAR_SOURCE_IDENTITY = {
    "paper_title": (
        "CrossLinear: Plug-and-Play Cross-Correlation Embedding for "
        "Time Series Forecasting with Exogenous Variables"
    ),
    "conference": "KDD 2025",
    "doi": "10.1145/3711896.3736899",
    "pdf_sha256": (
        "45557c426ca8bfa88f35ec41f09fd87ab864c9a382eef1c659c2296a4a1b0152"
    ),
    "official_repo_url": "https://github.com/mumiao2000/CrossLinear.git",
    "official_repo_commit": "d22366e2f59ced560a02b2b1c7cc673e3c02a13f",
    "official_model_sha256": (
        "a062ac97231c55384c621f27981b8225bb87822f50704df201b381dd8e037593"
    ),
    "retained_component": "cross_correlation_embedding_only",
}
M4_U1_SOURCE_COMMIT = "be2185c3382ec42c7287e4bcc9b2cad5c07fdbad"
M4_U1_SOURCE_FINGERPRINT = (
    "bffb7f1975f4f4f9448e44576bc626a0e82c75e54902fda4800847c89611065e"
)
M4_U1_DATA_FINGERPRINT = (
    "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"
)
M4_U1_SCHEMA_FINGERPRINT = (
    "f6dd94841b5d9d0b7515b19e0ff1876bf6476068054eacdc02ac6fcab3f084dc"
)
M4_U1_SOURCE_IDENTITIES = {
    96: {
        "run_id": "20260901T095811.286299Z-6e33fa77",
        "config_hash": "fa2c4da41f34eca232907e4d6462305cb8ef3ef15fc8996f7c67baa0411ddb2d",
        "comparison_config_hash": "4fbf51cca6fa7bad95bc8e35ddfc416d6dc45a78c331aead19e007f6d24ef74b",
        "best_epoch": 7,
        "checkpoint_sha256": "66458be335ac7948889156bf6a7a91af7221f3b75838a1522fd701e8e78b42d0",
    },
    192: {
        "run_id": "20260901T100147.203364Z-f03509fd",
        "config_hash": "1585152dbf1ff74d935f7404f8b2699881b85c7224252371e939db585f2611d0",
        "comparison_config_hash": "771cdff549663c52cc213c5cfaf9ed731362ce5b544457776bf7653ff0c950bf",
        "best_epoch": 3,
        "checkpoint_sha256": "f8b0308578b10f09ade232cf2c6ac2e7826b1e28ceb994c2a321d44c4563be6a",
    },
    336: {
        "run_id": "20260901T100520.627934Z-0c9f399c",
        "config_hash": "45ea9453083d6fb38381d03ba3a8455191e28e6221a2c9f86d0dee72ba8e8ff5",
        "comparison_config_hash": "8ccb82795987bd3df127f1cd087c8da7ff2cae53ee1d8c282fd565e730392d4b",
        "best_epoch": 3,
        "checkpoint_sha256": "89da2c854dce4e124c09575835d52e313bf3a6aeab2b556a060c7174709cb530",
    },
    720: {
        "run_id": "20260901T100834.831317Z-5f71979f",
        "config_hash": "93ddc59a0879b435a4cf4742e76c75460c254283ca00b2138164c4fe735d51cd",
        "comparison_config_hash": "3f5f973cd636b205f813ba4589845c5a5ffb91fdf86a379726dd2cc6d039c291",
        "best_epoch": 7,
        "checkpoint_sha256": "a13126522bb2cf8f5871c46272222242b8ebb6077145658558199c9473ba109b",
    },
}
SOURCE_COMPATIBILITY_CRITICAL_FILES = (
    "models/tsAMD.py",
    "models/common.py",
    "models/tsmoe.py",
    "models/tsAMD_enhanced.py",
    "models/modules/__init__.py",
    "models/modules/patch_conditioned_target_exogenous_bridge.py",
    "utils/dataloader.py",
)
EXPECTED_MODEL_CONTRACT = {
    "entry_normalization_impl": "torch_layernorm_last_dim_sequence",
    "entry_normalization_scope": "mdm_and_ddi_entries_controlled_by_layernorm_flag",
    "ddi_internal_normalization_impl": (
        "released_batchnorm1d_norm1_and_norm2_when_alpha_gt_0"
    ),
    "ddi_hidden_rule": "max(32,2**ceil(log2(feature_count)))_when_alpha_gt_0",
    "module_connection": "X->MDM(U)->DDI; AMS_selector<-U",
    "selector_mode": "horizon_shared_dense_emphasis",
}
EXPECTED_OPTIMIZATION_CONTRACT = {
    "optimizer": "Adam",
    "weight_decay": 1e-7,
}
RUN_FIELDS = (
    "implementation_variant", "training_protocol_id",
    "development_protocol_id", "ablation_id",
    "dataset_id", "task_mode", "target",
    "label_horizon", "fold", "seq_len", "pred_len", "seed",
    "target_exogenous_schema_contract",
    "run_id", "best_epoch", "val_mse", "val_mae", "test_mse", "test_mae",
    "parameter_count", "train_epochs", "duration_seconds", "config_hash",
    "comparison_config_hash", "data_sha256", "completed_at", "artifact_dir",
)
AGGREGATE_FIELDS = (
    "implementation_variant", "training_protocol_id",
    "development_protocol_id", "ablation_id",
    "dataset_id", "seq_len", "pred_len",
    "comparison_config_hash", "seed_count", "seeds",
    "val_mse_mean", "val_mse_sample_std", "val_mae_mean", "val_mae_sample_std",
    "test_mse_mean", "test_mse_sample_std", "test_mae_mean", "test_mae_sample_std",
)
SONNET_TEST_RUN_FIELDS = RUN_FIELDS + (
    "evaluation_policy", "artifact_purpose",
)
SONNET_VALIDATION_RUN_FIELDS = tuple(
    field for field in SONNET_TEST_RUN_FIELDS
    if field not in {"test_mse", "test_mae"}
)
SONNET_TEST_AGGREGATE_FIELDS = AGGREGATE_FIELDS + (
    "evaluation_policy", "artifact_purpose",
)
SONNET_VALIDATION_AGGREGATE_FIELDS = tuple(
    field for field in SONNET_TEST_AGGREGATE_FIELDS
    if not field.startswith("test_")
)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _stable_hash(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return value


def _comparison_hash(resolved_config, path, train_epochs):
    scientific = deepcopy(resolved_config.get("scientific_config"))
    if not isinstance(scientific, dict):
        raise ValueError(f"missing scientific_config in {path}")
    try:
        del scientific["execution"]["seed"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing scientific_config.execution.seed in {path}") from exc
    if scientific.get("implementation_variant") == SONNET_IMPLEMENTATION_VARIANT:
        try:
            del scientific["model"]["sonnet_mvca"]["module_init_seed"]["seed"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"missing Sonnet module initialization seed in {path}"
            ) from exc
    protocol = scientific.get("training_protocol")
    protocol_id = (
        protocol.get("training_protocol_id")
        if isinstance(protocol, dict)
        else STANDARD_TRAINING_PROTOCOL
    )
    if protocol_id in WARM_START_TRAINING_PROTOCOLS:
        # Adapter seed is scientific, but comparison grouping removes it and
        # duplicate identity adds the path seed separately.  Runtime outcomes
        # completed_epochs/best_epoch never enter this object.
        protocol.pop("adapter_seed", None)
    else:
        # Preserve the exact historical comparison hash for standard runs.
        scientific["completed_train_epochs"] = int(train_epochs)
    return _stable_hash(scientific)


def _validate_variant_contract(scientific, run_dir):
    """Reject self-consistent artifacts that do not implement this variant."""

    model = scientific.get("model")
    optimization = scientific.get("optimization")
    if not isinstance(model, dict) or not isinstance(optimization, dict):
        raise ValueError(f"missing model/optimization contract in {run_dir}")
    for field, expected in EXPECTED_MODEL_CONTRACT.items():
        if model.get(field) != expected:
            raise ValueError(
                f"model contract mismatch for {field} in {run_dir}: "
                f"expected {expected!r}, got {model.get(field)!r}"
            )
    for field, expected in EXPECTED_OPTIMIZATION_CONTRACT.items():
        if optimization.get(field) != expected:
            raise ValueError(
                f"optimization contract mismatch for {field} in {run_dir}: "
                f"expected {expected!r}, got {optimization.get(field)!r}"
            )


def _validate_cce_variant_contract(
    scientific, run_dir, implementation_variant=CCE_IMPLEMENTATION_VARIANT
):
    is_late = implementation_variant == LATE_CCE_IMPLEMENTATION_VARIANT
    control_ablation = (
        LATE_CCE_CONTROL_ABLATION_ID if is_late else CCE_CONTROL_ABLATION_ID
    )
    candidate_ablation = (
        LATE_CCE_CANDIDATE_ABLATION_ID if is_late else CCE_CANDIDATE_ABLATION_ID
    )
    development_protocol = (
        LATE_CCE_DEVELOPMENT_PROTOCOL if is_late else CCE_DEVELOPMENT_PROTOCOL
    )
    model = scientific.get("model")
    dataset = scientific.get("dataset")
    experiment = scientific.get("experiment")
    optimization = scientific.get("optimization")
    if not all(
        isinstance(value, dict)
        for value in (model, dataset, experiment, optimization)
    ):
        raise ValueError(f"CCE scientific contract is incomplete: {run_dir}")
    if scientific.get("training_protocol") is not None:
        raise ValueError(f"CCE must use standard from-scratch identity: {run_dir}")

    ablation_id = experiment.get("ablation_id")
    if ablation_id not in {control_ablation, candidate_ablation}:
        raise ValueError(f"CCE ablation identity mismatch: {run_dir}")
    enabled = ablation_id == candidate_ablation
    mode = dataset.get("task_mode")
    feature_names = dataset.get("feature_names")
    aux_idx = dataset.get("aux_idx")
    target_idx = dataset.get("target_idx")
    if (
        mode not in {"target_exogenous", "parallel_multivariate"}
        or not isinstance(feature_names, list)
        or len(feature_names) < 1
        or any(not isinstance(name, str) or not name for name in feature_names)
        or len(set(feature_names)) != len(feature_names)
        or not isinstance(aux_idx, list)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in aux_idx)
        or isinstance(target_idx, bool)
        or not isinstance(target_idx, int)
        or not 0 <= target_idx < len(feature_names)
    ):
        raise ValueError(f"CCE feature/index contract is invalid: {run_dir}")
    if len(set(aux_idx)) != len(aux_idx):
        raise ValueError(f"CCE aux_idx contains duplicates: {run_dir}")
    if target_idx in aux_idx or any(
        not 0 <= index < len(feature_names) for index in aux_idx
    ):
        raise ValueError(f"CCE aux_idx is invalid: {run_dir}")

    if mode == "target_exogenous":
        if (
            dataset.get("feature_type") != "MS"
            or not aux_idx
            or dataset.get("target") != dataset.get("target_feature_name")
        ):
            raise ValueError(f"CCE target_exogenous contract mismatch: {run_dir}")
        input_order = ORDERED_AUX_THEN_TARGET
        source_idx = [*aux_idx, target_idx]
    else:
        if (
            dataset.get("feature_type") != "M"
            or dataset.get("target") != "all"
            or aux_idx
            or len(feature_names) < 2
        ):
            raise ValueError(f"CCE parallel contract mismatch: {run_dir}")
        input_order = FEATURE_SCHEMA_ORDER
        source_idx = list(range(len(feature_names)))

    expected_cce = {
        "source": deepcopy(CROSSLINEAR_SOURCE_IDENTITY),
        "enabled": enabled,
        "insertion_point": LATE_CCE_INSERTION_POINT if is_late else CCE_INSERTION_POINT,
        "mode": mode,
        "input_order_policy": input_order,
        "source_idx": source_idx,
        "kernel": {
            "size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "padding_policy": ZERO_SAME,
            "bias": True,
        },
        "lambda": {
            "transform": SIGMOID_LOGIT_PLUS_RHO,
            "raw_parameter": "rho",
            "raw_init": 0.0,
            "effective_init": 0.1,
            "scope": "global_shared_scalar",
        },
        "parameterization_policy": IDENTITY_RESIDUAL_DELTA_V1,
        "normalization_reuse_policy": REVIN_REUSE_NO_INTERNAL_NORMALIZATION,
        "state_zero_placeholder_policy": LEGACY_WIDTH_COMPATIBILITY_ZERO,
        "excluded_crosslinear_components": [
            "normalization",
            "patch_embedding",
            "positional_embedding",
            "forecasting_head",
        ],
    }
    if is_late:
        expected_cce.update({
            "cce_architecture": LATE_CCE_ARCHITECTURE,
            "cce_insertion_point": LATE_CCE_INSERTION_POINT,
            "cce_input_representation": LATE_CCE_INPUT_REPRESENTATION,
        })
    mismatches = {}
    if model.get("cce") != expected_cce:
        mismatches["cce"] = (expected_cce, model.get("cce"))
    expected_switches = (enabled, False, False)
    observed_switches = (
        model.get("use_cce"),
        model.get("use_pmcr"),
        model.get("use_teb"),
    )
    if observed_switches != expected_switches:
        mismatches["module_switches"] = (expected_switches, observed_switches)
    if (
        model.get("norm") is not True
        or model.get("target_idx") != target_idx
        or model.get("module_connection")
        != (
            "X->RevIN->MDM(U)->DDI->PMCR?->LateCCE?; "
            "AMS(experts=LateCCE?,selector=U)"
            if is_late
            else "X->RevIN->CCE?->MDM(U)->DDI; AMS_selector<-U"
        )
    ):
        mismatches["model_route"] = ("locked CCE route", model)
    if (
        optimization.get("optimizer") != "Adam"
        or optimization.get("learning_rate") != 3e-5
        or optimization.get("weight_decay") != 1e-7
    ):
        mismatches["optimization"] = ("Adam/3e-5/1e-7", optimization)
    if experiment.get("development_protocol_id") != development_protocol:
        mismatches["development_protocol_id"] = (
            development_protocol,
            experiment.get("development_protocol_id"),
        )
    if mismatches:
        raise ValueError(f"unsupported CCE contract {mismatches}: {run_dir}")
    return {
        "development_protocol_id": development_protocol,
        "ablation_id": ablation_id,
        "task_mode": mode,
        "feature_names": feature_names,
        "target_idx": target_idx,
        "aux_idx": aux_idx,
        "schema_fingerprint": dataset.get("schema_fingerprint"),
        "cce": expected_cce,
    }


def _expected_sonnet_model_contract(dataset, execution, enabled):
    d_model = sonnet_spec.SONNET_D_MODEL
    n_atoms = sonnet_spec.SONNET_N_ATOMS
    aux_idx = dataset["aux_idx"]
    target_idx = dataset["target_idx"]
    return {
        "source": deepcopy(sonnet_spec.SONNET_SOURCE_IDENTITY),
        "enabled": enabled,
        "architecture_identity": sonnet_spec.SONNET_ARCHITECTURE_IDENTITY,
        "input_identity": sonnet_spec.SONNET_INPUT_IDENTITY,
        "insertion_identity": sonnet_spec.SONNET_INSERTION_IDENTITY,
        "retained_components": list(sonnet_spec.SONNET_RETAINED_COMPONENTS),
        "deleted_components": list(sonnet_spec.SONNET_DELETED_COMPONENTS),
        "raw_source_order": {
            "policy": sonnet_spec.SONNET_RAW_SOURCE_ORDER,
            "indices": [*aux_idx, target_idx],
            "target_position": "last",
        },
        "joint_embedding": {
            "aux_input_dim": len(aux_idx),
            "aux_output_dim": 32,
            "target_input_dim": 1,
            "target_output_dim": 32,
            "d_model": d_model,
            "alpha": sonnet_spec.SONNET_ALPHA,
            "alpha_parameter_policy": "fixed_hyperparameter_not_nn_parameter",
            "aux_bias": True,
            "target_bias": True,
            "latent_concat_order": sonnet_spec.SONNET_LATENT_CONCAT_ORDER,
            "post_embedding_normalization": None,
            "post_embedding_dropout": None,
        },
        "learnable_wavelet": {
            "n_atoms": n_atoms,
            "freq_params_shape": [d_model, n_atoms, 3],
            "freq_params_initialization": "standard_normal",
            "freq_params_constraint": "none",
            "time_grid": sonnet_spec.SONNET_WAVELET_TIME_GRID,
            "atom_formula": "exp(-w_alpha*t^2)*cos(w_beta*t+w_gamma*t^2)",
            "projection": "elementwise_B_by_K_by_T_by_d",
            "padding": None,
            "crop": None,
            "time_compression": None,
        },
        "mvca": {
            "hidden_width": d_model,
            "qkv": "single_linear_d_to_3d_bias_true",
            "fft_operator": "torch.fft.rfft",
            "fft_axis_policy": sonnet_spec.SONNET_FFT_AXIS_POLICY,
            "cross_spectral_density": "mean_frequency_Qf_times_conj_Kf",
            "power_spectral_density": "mean_frequency_self_product_real",
            "epsilon": sonnet_spec.SONNET_EPSILON,
            "denominator_policy": (
                sonnet_spec.SONNET_COHERENCE_DENOMINATOR_POLICY
            ),
            "hard_clamp": False,
            "scale_policy": sonnet_spec.SONNET_COHERENCE_SCALE_POLICY,
            "softmax_axis_policy": sonnet_spec.SONNET_SOFTMAX_AXIS_POLICY,
            "attention_dropout": sonnet_spec.SONNET_ATTENTION_DROPOUT,
            "attention_dropout_position": "after_softmax",
            "value_policy": sonnet_spec.SONNET_VALUE_POLICY,
            "time_attention_matrix": False,
            "time_reduction": False,
            "residual_mlp": "linear_d_d_gelu_linear_d_d_all_bias_true",
            "output_projection": "linear_d_d_bias_true_nonzero_default_init",
            "var_attn_policy": sonnet_spec.SONNET_VAR_ATTN_POLICY,
        },
        "reconstruction": {
            "policy": sonnet_spec.SONNET_RECONSTRUCTION_POLICY,
            "output_shape": "B_by_T_by_d",
            "koopman": False,
        },
        "readout": {
            "policy": sonnet_spec.SONNET_READOUT_POLICY,
            "shape": "linear_d_to_one_bias_true",
            "weight_initialization": "xavier_uniform",
            "bias_initialization": 0.0,
        },
        "residual": {
            "policy": sonnet_spec.SONNET_RESIDUAL_POLICY,
            "gamma_name": "gamma_sonnet",
            "gamma_scope": "global_shared_scalar",
            "gamma_constraint": "none",
            "gamma_init": sonnet_spec.SONNET_GAMMA_INIT,
            "writeback": "target_only",
            "non_target_policy": "bitwise_unchanged",
            "off_policy": "not_instantiated_exact_amd_identity",
            "on_identity": "near_identity_not_exact_identity",
        },
        "normalization_policy": sonnet_spec.SONNET_NORMALIZATION_POLICY,
        "state_context_policy": sonnet_spec.SONNET_STATE_CONTEXT_POLICY,
        "module_init_seed": {
            "policy": sonnet_spec.SONNET_MODULE_INIT_SEED_POLICY,
            "seed": execution["seed"],
        },
    }


def _validate_sonnet_variant_contract(scientific, run_dir):
    model = scientific.get("model")
    dataset = scientific.get("dataset")
    experiment = scientific.get("experiment")
    execution = scientific.get("execution")
    optimization = scientific.get("optimization")
    evaluation = scientific.get("evaluation")
    training_protocol = scientific.get("training_protocol")
    if not all(
        isinstance(value, dict)
        for value in (
            model,
            dataset,
            experiment,
            execution,
            optimization,
            evaluation,
            training_protocol,
        )
    ):
        raise ValueError(f"Sonnet scientific contract is incomplete: {run_dir}")

    expected_training = {
        "training_protocol_id": STANDARD_TRAINING_PROTOCOL,
        "warm_start_contract_version": None,
        "initialization_policy": "matched_standard_from_scratch",
        "source_checkpoint": None,
        "source_importer": None,
        "optimizer_state_policy": "fresh",
        "parameter_scope": "all_model_parameters_trainable",
    }
    if training_protocol != expected_training:
        raise ValueError(f"Sonnet training protocol mismatch: {run_dir}")
    if scientific.get("source_lineage") is not None or scientific.get(
        "source_compatibility_proof"
    ) is not None:
        raise ValueError(f"Sonnet must not carry source checkpoint lineage: {run_dir}")

    ablation_id = experiment.get("ablation_id")
    if ablation_id not in {
        SONNET_CONTROL_ABLATION_ID,
        SONNET_CANDIDATE_ABLATION_ID,
    }:
        raise ValueError(f"Sonnet ablation identity mismatch: {run_dir}")
    enabled = ablation_id == SONNET_CANDIDATE_ABLATION_ID
    if (
        experiment.get("development_protocol_id")
        != SONNET_DEVELOPMENT_PROTOCOL
        or experiment.get("artifact_purpose")
        != M4_DEVELOPMENT_CANDIDATE
    ):
        raise ValueError(f"Sonnet development identity mismatch: {run_dir}")

    dataset_id = dataset.get("id")
    label_horizon = dataset.get("label_horizon")
    if dataset_id == "ETTm1":
        expected_policy = TRAIN_VALIDATION_TEST
        expected_features = [
            "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"
        ]
        expected_dataset = {
            "feature_type": "MS",
            "target": "OT",
            "target_feature_name": "OT",
            "target_idx": 6,
            "target_indices": [6],
            "aux_idx": [0, 1, 2, 3, 4, 5],
            "aux_feature_names": expected_features[:-1],
            "feature_names": expected_features,
            "feature_preset": None,
            "fold": "official",
            "model_pred_len": label_horizon,
            "artifact_horizon": label_horizon,
        }
        if label_horizon not in {96, 192, 336, 720}:
            raise ValueError(f"Sonnet ETTm1 horizon mismatch: {run_dir}")
        expected_seq_len = 512
        expected_pred_len = label_horizon
        expected_batch = 32
    elif dataset_id == "UrbanEV":
        expected_policy = TRAIN_VALIDATION_ONLY
        expected_features = list(CANONICAL_FEATURE_NAMES)
        expected_dataset = {
            "feature_type": "MS",
            "target": "volume",
            "target_feature_name": "volume",
            "target_idx": 0,
            "target_indices": [0],
            "aux_idx": list(range(1, 11)),
            "aux_feature_names": expected_features[1:],
            "feature_names": expected_features,
            "feature_preset": "F4",
            "fold": 6,
            "model_pred_len": 1,
            "artifact_horizon": label_horizon,
        }
        if label_horizon not in {3, 6, 9, 12}:
            raise ValueError(f"Sonnet UrbanEV horizon mismatch: {run_dir}")
        expected_seq_len = 12
        expected_pred_len = 1
        expected_batch = 128
    else:
        raise ValueError(f"Sonnet dataset identity mismatch: {run_dir}")
    dataset_mismatches = [
        field
        for field, expected in expected_dataset.items()
        if dataset.get(field) != expected
    ]
    if dataset_mismatches:
        raise ValueError(
            f"Sonnet dataset contract mismatch {dataset_mismatches}: {run_dir}"
        )
    expected_evaluation = {
        "evaluation_policy": expected_policy,
        "artifact_purpose": M4_DEVELOPMENT_CANDIDATE,
        "test_access_policy": (
            "forbidden"
            if expected_policy == TRAIN_VALIDATION_ONLY
            else "development_only"
        ),
    }
    if (
        evaluation != expected_evaluation
        or experiment.get("evaluation_policy") != expected_policy
    ):
        raise ValueError(f"Sonnet evaluation contract mismatch: {run_dir}")
    if (
        execution.get("seed") != 2024
        or model.get("seq_len") != expected_seq_len
        or model.get("pred_len") != expected_pred_len
        or model.get("model_pred_len") != expected_pred_len
        or model.get("norm") is not True
        or model.get("target_idx") != dataset["target_idx"]
        or model.get("use_sonnet_mvca") is not enabled
        or model.get("use_cce") is not False
        or model.get("use_pmcr") is not False
        or model.get("use_teb") is not False
        or model.get("module_connection")
        != (
            "X->RevIN->SonnetTargetResidual?->MDM(U)->DDI; "
            "AMS(experts=DDI,selector=U)"
        )
    ):
        raise ValueError(f"Sonnet AMD route/switch contract mismatch: {run_dir}")
    expected_optimization = {
        "optimizer": "Adam",
        "learning_rate": 3e-5,
        "weight_decay": 1e-7,
        "batch_size": expected_batch,
        "train_drop_last": True,
        "validation_drop_last": False,
    }
    if optimization != expected_optimization:
        raise ValueError(f"Sonnet optimization contract mismatch: {run_dir}")

    expected_sonnet = _expected_sonnet_model_contract(
        dataset, execution, enabled
    )
    if model.get("sonnet_mvca") != expected_sonnet:
        raise ValueError(f"Sonnet model/source contract mismatch: {run_dir}")
    return {
        "development_protocol_id": SONNET_DEVELOPMENT_PROTOCOL,
        "ablation_id": ablation_id,
        "architecture_identity": sonnet_spec.SONNET_ARCHITECTURE_IDENTITY,
        "input_identity": sonnet_spec.SONNET_INPUT_IDENTITY,
        "insertion_identity": sonnet_spec.SONNET_INSERTION_IDENTITY,
        "task_mode": "target_exogenous",
        "feature_names": expected_features,
        "target_idx": dataset["target_idx"],
        "aux_idx": dataset["aux_idx"],
        "schema_fingerprint": dataset["schema_fingerprint"],
        "seq_len": expected_seq_len,
        "evaluation_policy": expected_policy,
        "artifact_purpose": M4_DEVELOPMENT_CANDIDATE,
        "sonnet_mvca": expected_sonnet,
    }


def _validate_enhanced_variant_contract(scientific, implementation_variant, run_dir):
    """Keep legacy TEB, warm-start, and CCE artifact identities distinct."""

    model = scientific.get("model")
    experiment = scientific.get("experiment")
    if not isinstance(model, dict) or not isinstance(experiment, dict):
        raise ValueError(f"enhanced variant contract is incomplete: {run_dir}")
    if implementation_variant == SONNET_IMPLEMENTATION_VARIANT:
        return _validate_sonnet_variant_contract(scientific, run_dir)
    if implementation_variant in {
        CCE_IMPLEMENTATION_VARIANT,
        LATE_CCE_IMPLEMENTATION_VARIANT,
    }:
        return _validate_cce_variant_contract(
            scientific, run_dir, implementation_variant=implementation_variant
        )
    teb = model.get("teb")
    if not isinstance(teb, dict):
        raise ValueError(f"enhanced TEB contract is missing: {run_dir}")
    training_protocol = scientific.get("training_protocol")
    protocol_id = (
        training_protocol.get("training_protocol_id")
        if isinstance(training_protocol, dict)
        else STANDARD_TRAINING_PROTOCOL
    )

    patch_fields = {
        "architecture",
        "patch_size",
        "patch_padding",
        "patch_position",
        "target_selection_policy",
    }
    t2g_fields = {
        "global_residual",
        "patch_attention_residual",
        "global_gate",
        "global_gate_input",
        "global_gate_init",
        "beta_global_init",
    }
    t3_fields = {
        "patch_confidence_gate",
        "patch_gate_input",
        "patch_gate_activation",
        "patch_gate_init",
        "global_prediction_role",
    }
    if implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT:
        unexpected = sorted((patch_fields | t2g_fields | t3_fields) & set(teb))
        if unexpected:
            raise ValueError(
                f"Global TEB v1 artifact contains candidate fields "
                f"{unexpected}: {run_dir}"
            )
        if protocol_id == U1_CONTINUATION_TRAINING_PROTOCOL:
            if (
                experiment.get("ablation_id") != U1_CONTINUATION_ABLATION_ID
                or model.get("use_pmcr") is not False
                or model.get("use_teb") is not False
            ):
                raise ValueError(
                    f"continuation U1 structure/ablation mismatch: {run_dir}"
                )
        elif protocol_id != STANDARD_TRAINING_PROTOCOL:
            raise ValueError(
                f"unsupported protocol for Global v1 artifact: {protocol_id}"
            )
        return None

    expected = {
        "architecture": {
            T2_IMPLEMENTATION_VARIANT: "patch_conditioned_v1",
            T2G_IMPLEMENTATION_VARIANT: "global_mediated_patch_v1",
            T3_IMPLEMENTATION_VARIANT: "selective_patch_v1",
        }[implementation_variant],
        "context_dim": 32,
        "heads": 4,
        "dropout": 0.1,
        "gamma_init": 1e-3,
        "patch_padding": "right_zero_crop",
        "patch_position": "fixed_sinusoidal",
        "target_selection_policy": "full_denorm_then_task_select",
    }
    if implementation_variant == T2G_IMPLEMENTATION_VARIANT:
        expected.update({
            "global_residual": "query_plus_attention_post_layernorm",
            "patch_attention_residual": "none",
            "global_gate": "scalar_per_patch",
            "global_gate_input": "patch_attention_and_global_bridge",
            "global_gate_init": "identity",
            "beta_global_init": 1e-3,
        })
    elif implementation_variant == T3_IMPLEMENTATION_VARIANT:
        expected.update({
            "patch_confidence_gate": "scalar_per_patch_post_projection",
            "patch_gate_input": "query_and_attention_response",
            "patch_gate_activation": "two_sigmoid",
            "patch_gate_init": "explicit_zero_identity",
            "global_prediction_role": "state_only_forecast_disconnected",
        })
    mismatches = {
        field: (expected_value, teb.get(field))
        for field, expected_value in expected.items()
        if teb.get(field) != expected_value
    }
    patch_size = teb.get("patch_size")
    seq_len = model.get("seq_len")
    if (
        isinstance(patch_size, bool)
        or not isinstance(patch_size, int)
        or isinstance(seq_len, bool)
        or not isinstance(seq_len, int)
        or not 0 < patch_size <= seq_len
    ):
        mismatches["patch_size"] = ("0 < patch_size <= seq_len", patch_size)
    if implementation_variant == T2_IMPLEMENTATION_VARIANT:
        unexpected = sorted((t2g_fields | t3_fields) & set(teb))
        if unexpected:
            mismatches["unexpected_candidate_fields"] = ([], unexpected)
    elif implementation_variant == T2G_IMPLEMENTATION_VARIANT:
        unexpected = sorted(t3_fields & set(teb))
        if unexpected:
            mismatches["unexpected_t3_fields"] = ([], unexpected)
    else:
        unexpected = sorted(t2g_fields & set(teb))
        if unexpected:
            mismatches["unexpected_t2g_fields"] = ([], unexpected)

    if model.get("use_pmcr") is not False or model.get("use_teb") is not True:
        mismatches["module_switches"] = ((False, True), (model.get("use_pmcr"), model.get("use_teb")))
    expected_ablation = {
        T2_IMPLEMENTATION_VARIANT: "M4_T2",
        T2G_IMPLEMENTATION_VARIANT: "M4_T2G",
        T3_IMPLEMENTATION_VARIANT: "M4_T3",
    }[implementation_variant]
    if (
        implementation_variant == T2_IMPLEMENTATION_VARIANT
        and protocol_id == T2_ADAPTER_TRAINING_PROTOCOL
    ):
        expected_ablation = T2_ADAPTER_ABLATION_ID
    if experiment.get("ablation_id") != expected_ablation:
        mismatches["ablation_id"] = (expected_ablation, experiment.get("ablation_id"))
    if mismatches:
        label = {
            T2_IMPLEMENTATION_VARIANT: "T2",
            T2G_IMPLEMENTATION_VARIANT: "T2G",
            T3_IMPLEMENTATION_VARIANT: "T3",
        }[implementation_variant]
        raise ValueError(
            f"unsupported {label} patch config {mismatches}: {run_dir}"

        )
    dataset = scientific.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"enhanced dataset contract is missing: {run_dir}")
    contract = {
        "ablation_id": experiment.get("ablation_id"),
        "teb_architecture": teb.get("architecture"),
        "teb_patch_size": teb.get("patch_size"),
        "teb_patch_padding": teb.get("patch_padding"),
        "teb_patch_position": teb.get("patch_position"),
        "teb_context_dim": teb.get("context_dim"),
        "teb_heads": teb.get("heads"),
        "teb_dropout": teb.get("dropout"),
        "teb_gamma_init": teb.get("gamma_init"),
        "seq_len": model.get("seq_len"),
        "task_mode": dataset.get("task_mode"),
        "target_idx": dataset.get("target_idx"),
        "aux_idx": dataset.get("aux_idx"),
        "schema_fingerprint": dataset.get("schema_fingerprint"),
        "target_selection_policy": teb.get("target_selection_policy"),
    }
    if implementation_variant == T2G_IMPLEMENTATION_VARIANT:
        contract.update({
            "teb_global_residual": teb.get("global_residual"),
            "teb_patch_attention_residual": teb.get("patch_attention_residual"),
            "teb_global_gate": teb.get("global_gate"),
            "teb_global_gate_input": teb.get("global_gate_input"),
            "teb_global_gate_init": teb.get("global_gate_init"),
            "teb_beta_global_init": teb.get("beta_global_init"),
        })
    elif implementation_variant == T3_IMPLEMENTATION_VARIANT:
        contract.update({
            "teb_patch_confidence_gate": teb.get("patch_confidence_gate"),
            "teb_patch_gate_input": teb.get("patch_gate_input"),
            "teb_patch_gate_activation": teb.get("patch_gate_activation"),
            "teb_patch_gate_init": teb.get("patch_gate_init"),
            "teb_global_prediction_role": teb.get("global_prediction_role"),
        })
    return contract


def _warm_start_protocol_expected(protocol_id):
    adapter = protocol_id == T2_ADAPTER_TRAINING_PROTOCOL
    if not adapter and protocol_id != U1_CONTINUATION_TRAINING_PROTOCOL:
        raise ValueError(f"unsupported warm-start protocol {protocol_id!r}")
    return {
        "training_protocol_id": protocol_id,
        "warm_start_contract_version": WARM_START_CONTRACT_VERSION,
        "initialization_policy": (
            "source_u1_amd_plus_fresh_t2"
            if adapter else "source_u1_same_structure"
        ),
        "backbone_parameter_policy": "frozen" if adapter else "trainable",
        "backbone_buffer_policy": "frozen" if adapter else "train_updates",
        "backbone_module_mode": "eval" if adapter else "train",
        "adapter_module_mode": "train" if adapter else None,
        "adapter_trainable_scope": (
            "forecast_connected_t2_only" if adapter else None
        ),
        "adapter_trainable_parameter_names": (
            [
                "teb.gamma_teb",
                "teb.patch_query_projection.weight",
                "teb.patch_query_projection.bias",
                "teb.patch_query_norm.weight",
                "teb.patch_query_norm.bias",
                "teb.exogenous_projection.weight",
                "teb.exogenous_projection.bias",
                "teb.exogenous_norm.weight",
                "teb.exogenous_norm.bias",
                "teb.cross_attention.in_proj_weight",
                "teb.cross_attention.in_proj_bias",
                "teb.cross_attention.out_proj.weight",
                "teb.cross_attention.out_proj.bias",
                "teb.patch_output_projection.weight",
                "teb.patch_output_projection.bias",
            ]
            if adapter else None
        ),
        "adapter_trainable_tensor_count": 15 if adapter else None,
        "adapter_trainable_parameter_count": 22881 if adapter else None,
        "global_query_parameter_policy": "frozen" if adapter else None,
        "global_query_parameter_names": (
            [
                "teb.global_query_projection.weight",
                "teb.global_query_projection.bias",
                "teb.global_query_norm.weight",
                "teb.global_query_norm.bias",
            ]
            if adapter else None
        ),
        "global_query_tensor_count": 4 if adapter else None,
        "global_query_parameter_count": 16480 if adapter else None,
        "optimizer_state_policy": "fresh",
        "optimizer_parameter_scope": (
            "exact_forecast_connected_t2_only" if adapter else "all_amd_parameters"
        ),
        "optimizer_parameter_names": (
            [
                "teb.gamma_teb",
                "teb.patch_query_projection.weight",
                "teb.patch_query_projection.bias",
                "teb.patch_query_norm.weight",
                "teb.patch_query_norm.bias",
                "teb.exogenous_projection.weight",
                "teb.exogenous_projection.bias",
                "teb.exogenous_norm.weight",
                "teb.exogenous_norm.bias",
                "teb.cross_attention.in_proj_weight",
                "teb.cross_attention.in_proj_bias",
                "teb.cross_attention.out_proj.weight",
                "teb.cross_attention.out_proj.bias",
                "teb.patch_output_projection.weight",
                "teb.patch_output_projection.bias",
            ]
            if adapter else None
        ),
        "optimizer_name": "Adam",
        "adapter_learning_rate": 3e-5 if adapter else None,
        "adapter_weight_decay": 0.0 if adapter else None,
        "adapter_seed": 2024 if adapter else None,
        "continuation_learning_rate": 3e-5 if not adapter else None,
        "continuation_weight_decay": 1e-7 if not adapter else None,
        "teb_constructor_gamma_init": 1e-3 if adapter else None,
        "gamma_initialization_policy": (
            "zero_after_fresh_t2_initialization" if adapter else None
        ),
        "effective_teb_gamma_init": 0.0 if adapter else None,
        "epoch_zero_selection_policy": "included_strict_improvement",
        "epoch_zero_checkpoint_role": "source_equivalent_initialization",
        "max_adapter_epochs": 10 if adapter else None,
        "max_continuation_epochs": 10 if not adapter else None,
        "stopping_policy": "fixed_budget_no_early_stopping",
        "training_objective_policy": (
            "prediction_mse_plus_frozen_selector_auxiliary"
            if adapter else "prediction_mse_plus_selector_auxiliary"
        ),
    }


def _read_history_records(path):
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read history {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid history JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"history record must be an object: {path}")
        records.append(record)
    return records


def _stable_lineage(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} source lineage is missing")
    path = value.get("source_artifact_path")
    if path is not None and (not isinstance(path, str) or not path):
        raise ValueError(f"{label} source artifact path is invalid")
    stable = deepcopy(value)
    stable.pop("source_artifact_path", None)
    return stable


def _validate_warm_start_artifact(
    scientific,
    config,
    manifest,
    metrics,
    run_dir,
    implementation_variant,
):
    """Validate protocol, lineage, epoch-zero, history, and checkpoint sealing."""

    scientific_protocol = scientific.get("training_protocol")
    protocol_id = (
        scientific_protocol.get("training_protocol_id")
        if isinstance(scientific_protocol, dict)
        else STANDARD_TRAINING_PROTOCOL
    )
    if protocol_id == STANDARD_TRAINING_PROTOCOL:
        if implementation_variant == SONNET_IMPLEMENTATION_VARIANT:
            expected_default = {
                "training_protocol_id": STANDARD_TRAINING_PROTOCOL,
                "warm_start_contract_version": None,
                "initialization_policy": "matched_standard_from_scratch",
                "source_checkpoint": None,
                "source_importer": None,
                "optimizer_state_policy": "fresh",
                "parameter_scope": "all_model_parameters_trainable",
            }
            if scientific_protocol != expected_default:
                raise ValueError(
                    f"Sonnet scientific training protocol mismatch: {run_dir}"
                )
        else:
            expected_default = {
                "training_protocol_id": STANDARD_TRAINING_PROTOCOL,
                "warm_start_contract_version": None,
            }
        for label, document in (
            ("config", config), ("manifest", manifest), ("metrics", metrics)
        ):
            observed = document.get("training_protocol")
            if (
                implementation_variant == SONNET_IMPLEMENTATION_VARIANT
                and observed != expected_default
            ) or (
                implementation_variant != SONNET_IMPLEMENTATION_VARIANT
                and observed is not None
                and observed != expected_default
            ):
                raise ValueError(
                    f"standard artifact carries a warm-start protocol: {run_dir}"
                )
            if document.get("source_lineage") is not None:
                raise ValueError(
                    f"standard artifact carries source lineage: {run_dir}"
                )
            if document.get("source_compatibility_proof") is not None:
                raise ValueError(
                    f"standard artifact carries compatibility proof: {run_dir}"
                )
        return {
            "training_protocol_id": STANDARD_TRAINING_PROTOCOL,
            "warm_start": False,
        }

    if protocol_id not in WARM_START_TRAINING_PROTOCOLS:
        raise ValueError(f"unsupported training protocol in {run_dir}: {protocol_id}")
    expected_protocol = _warm_start_protocol_expected(protocol_id)
    if scientific_protocol != expected_protocol:
        raise ValueError(f"warm-start scientific protocol mismatch: {run_dir}")
    for label, document in (
        ("config", config), ("manifest", manifest), ("metrics", metrics)
    ):
        if document.get("training_protocol") != expected_protocol:
            raise ValueError(f"{label} warm-start protocol mismatch: {run_dir}")

    if protocol_id == T2_ADAPTER_TRAINING_PROTOCOL:
        if implementation_variant != T2_IMPLEMENTATION_VARIANT:
            raise ValueError(f"adapter is not a T2 artifact: {run_dir}")
    elif implementation_variant != ENHANCED_IMPLEMENTATION_VARIANT:
        raise ValueError(f"continuation is not an AMD-Concat artifact: {run_dir}")

    scientific_lineage = _stable_lineage(
        scientific.get("source_lineage"), "scientific"
    )
    lineage_documents = [
        ("config", config.get("source_lineage")),
        ("manifest", manifest.get("source_lineage")),
        ("metrics", metrics.get("source_lineage")),
    ]
    for label, lineage in lineage_documents:
        if _stable_lineage(lineage, label) != scientific_lineage:
            raise ValueError(f"{label} source lineage mismatch: {run_dir}")
    required_lineage = {
        "source_run_id",
        "source_implementation_variant",
        "source_ablation_id",
        "source_checkpoint_role",
        "source_checkpoint_sha256",
        "source_config_hash",
        "source_comparison_config_hash",
        "source_commit",
        "source_executable_fingerprint",
        "source_data_fingerprint",
        "source_best_epoch",
        "source_task_mode",
        "source_feature_type",
        "source_target",
        "source_target_idx",
        "source_target_indices",
        "source_aux_idx",
        "source_target_exogenous_schema_version",
        "source_schema_fingerprint",
    }
    if set(scientific_lineage) != required_lineage:
        raise ValueError(f"source lineage field set mismatch: {run_dir}")
    if (
        scientific_lineage["source_implementation_variant"]
        != ENHANCED_IMPLEMENTATION_VARIANT
        or scientific_lineage["source_ablation_id"] != "U1"
        or scientific_lineage["source_checkpoint_role"] != "best"
        or scientific_lineage["source_task_mode"] != "target_exogenous"
        or scientific_lineage["source_feature_type"] != "MS"
        or scientific_lineage["source_target"] != "OT"
        or scientific_lineage["source_target_idx"] != 6
        or scientific_lineage["source_target_indices"] != [6]
        or scientific_lineage["source_aux_idx"] != [0, 1, 2, 3, 4, 5]
        or scientific_lineage["source_target_exogenous_schema_version"]
        != TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION
    ):
        raise ValueError(f"source lineage identity mismatch: {run_dir}")
    dataset = scientific.get("dataset")
    model = scientific.get("model")
    experiment = scientific.get("experiment")
    execution = scientific.get("execution")
    if not all(
        isinstance(value, dict)
        for value in (dataset, model, experiment, execution)
    ):
        raise ValueError(f"warm-start scientific task contract is incomplete: {run_dir}")
    horizon = dataset.get("artifact_horizon")
    expected_source = M4_U1_SOURCE_IDENTITIES.get(horizon)
    observed_source = {
        "run_id": scientific_lineage.get("source_run_id"),
        "config_hash": scientific_lineage.get("source_config_hash"),
        "comparison_config_hash": scientific_lineage.get(
            "source_comparison_config_hash"
        ),
        "best_epoch": scientific_lineage.get("source_best_epoch"),
        "checkpoint_sha256": scientific_lineage.get(
            "source_checkpoint_sha256"
        ),
    }
    if expected_source is None or observed_source != expected_source:
        raise ValueError(f"locked U1 source identity mismatch: {run_dir}")
    expected_task = {
        "dataset_id": "ETTm1",
        "task_mode": "target_exogenous",
        "feature_type": "MS",
        "target": "OT",
        "target_idx": 6,
        "target_indices": [6],
        "aux_idx": [0, 1, 2, 3, 4, 5],
        "seq_len": 512,
        "model_pred_len": horizon,
        "label_horizon": horizon,
        "seed": 2024,
        "source_commit": M4_U1_SOURCE_COMMIT,
        "source_fingerprint": M4_U1_SOURCE_FINGERPRINT,
        "source_data": M4_U1_DATA_FINGERPRINT,
        "source_schema": M4_U1_SCHEMA_FINGERPRINT,
    }
    observed_task = {
        "dataset_id": dataset.get("id"),
        "task_mode": dataset.get("task_mode"),
        "feature_type": dataset.get("feature_type"),
        "target": dataset.get("target"),
        "target_idx": dataset.get("target_idx"),
        "target_indices": dataset.get("target_indices"),
        "aux_idx": dataset.get("aux_idx"),
        "seq_len": model.get("seq_len"),
        "model_pred_len": model.get("model_pred_len"),
        "label_horizon": dataset.get("label_horizon"),
        "seed": execution.get("seed"),
        "source_commit": scientific_lineage.get("source_commit"),
        "source_fingerprint": scientific_lineage.get(
            "source_executable_fingerprint"
        ),
        "source_data": scientific_lineage.get("source_data_fingerprint"),
        "source_schema": scientific_lineage.get("source_schema_fingerprint"),
    }
    if observed_task != expected_task:
        raise ValueError(f"warm-start task/source contract mismatch: {run_dir}")
    if protocol_id == T2_ADAPTER_TRAINING_PROTOCOL:
        if (
            experiment.get("ablation_id") != T2_ADAPTER_ABLATION_ID
            or model.get("use_pmcr") is not False
            or model.get("use_teb") is not True
            or model.get("teb", {}).get("architecture")
            != "patch_conditioned_v1"
            or model.get("teb", {}).get("patch_size") != 32
        ):
            raise ValueError(f"adapter model/ablation contract mismatch: {run_dir}")
    elif (
        experiment.get("ablation_id") != U1_CONTINUATION_ABLATION_ID
        or model.get("use_pmcr") is not False
        or model.get("use_teb") is not False
    ):
        raise ValueError(f"continuation model/ablation contract mismatch: {run_dir}")

    for field in (
        "source_checkpoint_sha256", "source_config_hash",
        "source_comparison_config_hash", "source_executable_fingerprint",
        "source_data_fingerprint", "source_schema_fingerprint",
    ):
        value = scientific_lineage[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"invalid source lineage digest {field}: {run_dir}")

    proof = scientific.get("source_compatibility_proof")
    if not isinstance(proof, dict):
        raise ValueError(f"source compatibility proof is missing: {run_dir}")
    for label, document in (
        ("config", config), ("manifest", manifest), ("metrics", metrics)
    ):
        if document.get("source_compatibility_proof") != proof:
            raise ValueError(f"{label} compatibility proof mismatch: {run_dir}")
    expected_proof_fields = {
        "contract_version",
        "source_executable_fingerprint",
        "current_executable_fingerprint",
        "global_fingerprint_equal",
        "critical_files",
        "source_state_key_count",
        "target_state_key_count",
        "mapped_key_count",
        "allowed_missing_keys",
        "unexpected_keys",
        "shape_mismatches",
        "dtype_mismatches",
    }
    if set(proof) != expected_proof_fields:
        raise ValueError(f"compatibility proof field set mismatch: {run_dir}")
    if (
        proof["contract_version"] != SOURCE_COMPATIBILITY_PROOF_VERSION
        or proof["source_executable_fingerprint"]
        != scientific_lineage["source_executable_fingerprint"]
        or proof["current_executable_fingerprint"]
        != config.get("source", {}).get("sha256")
        or proof["global_fingerprint_equal"]
        != (
            proof["source_executable_fingerprint"]
            == proof["current_executable_fingerprint"]
        )
    ):
        raise ValueError(f"compatibility proof fingerprint mismatch: {run_dir}")
    critical = proof.get("critical_files")
    if (
        not isinstance(critical, list)
        or [item.get("path") for item in critical if isinstance(item, dict)]
        != list(SOURCE_COMPATIBILITY_CRITICAL_FILES)
        or any(
            item.get("source_sha256") != item.get("current_sha256")
            for item in critical if isinstance(item, dict)
        )
    ):
        raise ValueError(f"compatibility critical file proof mismatch: {run_dir}")
    adapter = protocol_id == T2_ADAPTER_TRAINING_PROTOCOL
    expected_counts = (60, 79, 60, 19) if adapter else (60, 60, 60, 0)
    observed_counts = (
        proof.get("source_state_key_count"),
        proof.get("target_state_key_count"),
        proof.get("mapped_key_count"),
        len(proof.get("allowed_missing_keys", [])),
    )
    if (
        observed_counts != expected_counts
        or proof.get("unexpected_keys") != []
        or proof.get("shape_mismatches") != []
        or proof.get("dtype_mismatches") != []
        or (
            adapter and any(
                not key.startswith("teb.")
                for key in proof.get("allowed_missing_keys", [])
            )
        )
    ):
        raise ValueError(f"compatibility state mapping mismatch: {run_dir}")

    history = _read_history_records(Path(run_dir) / "history.jsonl")
    completed_epochs = metrics.get("completed_epochs")
    best_epoch = metrics.get("best_epoch")
    maximum_epochs = (
        expected_protocol["max_adapter_epochs"]
        if adapter else expected_protocol["max_continuation_epochs"]
    )
    if (
        isinstance(completed_epochs, bool)
        or not isinstance(completed_epochs, int)
        or not 0 <= completed_epochs <= maximum_epochs
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 0 <= best_epoch <= completed_epochs
        or metrics.get("train_epochs") != maximum_epochs
        or config.get("run", {}).get("train_epochs") != maximum_epochs
        or manifest.get("completed_epoch") != completed_epochs
        or manifest.get("completed_epochs") != completed_epochs
    ):
        raise ValueError(f"warm-start epoch metadata mismatch: {run_dir}")
    if [record.get("epoch") for record in history] != list(
        range(1, completed_epochs + 1)
    ):
        raise ValueError(f"warm-start history contains invalid/epoch-zero rows: {run_dir}")

    initialization = metrics.get("initialization_validation")
    if not isinstance(initialization, dict):
        raise ValueError(f"initialization validation is missing: {run_dir}")
    for field in ("mse", "mae"):
        _finite_number(initialization.get(field), f"epoch-zero {field}")
    expected_role = "epoch_zero_initialization" if best_epoch == 0 else "trained_epoch"
    if (
        metrics.get("epoch_zero_in_best_selection") is not True
        or manifest.get("epoch_zero_in_best_selection") is not True
        or metrics.get("best_checkpoint_role") != expected_role
        or manifest.get("best_checkpoint_role") != expected_role
        or manifest.get("initialization_validation") != initialization
        or config.get("run", {}).get("initialization_validation") != initialization
    ):
        raise ValueError(f"warm-start epoch-zero lifecycle mismatch: {run_dir}")

    try:
        best_checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu")
        last_checkpoint = torch.load(Path(run_dir) / "last.pt", map_location="cpu")
    except Exception as exc:
        raise ValueError(f"cannot read warm-start checkpoints: {run_dir}") from exc
    if not isinstance(best_checkpoint, dict) or not isinstance(last_checkpoint, dict):
        raise ValueError(f"warm-start checkpoints must be dictionaries: {run_dir}")
    for label, checkpoint in (
        ("best", best_checkpoint), ("last", last_checkpoint)
    ):
        if (
            checkpoint.get("training_protocol") != expected_protocol
            or _stable_lineage(
                checkpoint.get("source_lineage"), f"{label} checkpoint"
            ) != scientific_lineage
            or checkpoint.get("source_compatibility_proof") != proof
            or checkpoint.get("initialization_validation") != initialization
            or checkpoint.get("epoch_zero_in_best_selection") is not True
        ):
            raise ValueError(f"{label} checkpoint warm-start metadata mismatch: {run_dir}")
    if (
        best_checkpoint.get("best_epoch") != best_epoch
        or best_checkpoint.get("best_checkpoint_role") != expected_role
        or best_checkpoint.get("checkpoint_role") != expected_role
        or last_checkpoint.get("completed_epoch") != completed_epochs
        or last_checkpoint.get("completed_epochs") != completed_epochs
        or last_checkpoint.get("best_epoch") != best_epoch
        or last_checkpoint.get("best_checkpoint_role") != expected_role
        or last_checkpoint.get("checkpoint_role")
        != ("epoch_zero_initialization" if completed_epochs == 0 else "last_trained_epoch")
    ):
        raise ValueError(f"warm-start checkpoint role/epoch mismatch: {run_dir}")
    return {
        "training_protocol_id": protocol_id,
        "warm_start": True,
        "completed_epochs": completed_epochs,
        "best_checkpoint_role": expected_role,
    }


def _validate_cce_checkpoints(
    scientific,
    implementation_variant,
    config,
    manifest,
    metrics,
    run_dir,
):
    """Reject checkpoint metadata or state that spoofs the sealed CCE identity."""

    model = scientific["model"]
    dataset = scientific["dataset"]
    enabled = model["use_cce"]
    feature_count = len(dataset["feature_names"])
    if dataset["task_mode"] == "target_exogenous":
        weight_shape = (1, len(dataset["aux_idx"]) + 1, 3)
        bias_shape = (1,)
    else:
        weight_shape = (feature_count, feature_count, 3)
        bias_shape = (feature_count,)
    expected_cce_keys = (
        {"cce.delta_weight", "cce.delta_bias", "cce.rho"}
        if enabled
        else set()
    )

    try:
        checkpoints = {
            role: torch.load(Path(run_dir) / f"{role}.pt", map_location="cpu")
            for role in ("best", "last")
        }
    except Exception as exc:
        raise ValueError(f"cannot read CCE checkpoints: {run_dir}") from exc
    for role, checkpoint in checkpoints.items():
        if not isinstance(checkpoint, dict):
            raise ValueError(f"CCE {role} checkpoint must be a dictionary: {run_dir}")
        resolved = checkpoint.get("resolved_config")
        if (
            checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("artifact_schema_version")
            != ENHANCED_ARTIFACT_SCHEMA_VERSION
            or checkpoint.get("implementation_variant")
            != implementation_variant
            or checkpoint.get("config_hash") != metrics.get("config_hash")
            or checkpoint.get("data_sha256") != metrics.get("data_sha256")
            or not isinstance(resolved, dict)
            or resolved.get("config_hash") != metrics.get("config_hash")
            or resolved.get("scientific_config") != scientific
        ):
            raise ValueError(
                f"CCE {role} checkpoint scientific identity mismatch: {run_dir}"
            )
        state = checkpoint.get("model_state")
        if not isinstance(state, dict):
            raise ValueError(f"CCE {role} checkpoint has no model state: {run_dir}")
        cce_keys = {key for key in state if key.startswith("cce.")}
        forbidden = sorted(
            key for key in state
            if key.startswith("teb.") or key.startswith("pmcr.")
        )
        if cce_keys != expected_cce_keys or forbidden:
            raise ValueError(
                f"CCE {role} checkpoint module key mismatch: "
                f"cce={sorted(cce_keys)}, forbidden={forbidden}: {run_dir}"
            )
        if enabled:
            weight = state["cce.delta_weight"]
            bias = state["cce.delta_bias"]
            rho = state["cce.rho"]
            if (
                not all(torch.is_tensor(value) for value in (weight, bias, rho))
                or tuple(weight.shape) != weight_shape
                or tuple(bias.shape) != bias_shape
                or tuple(rho.shape) != ()
                or not weight.is_floating_point()
                or bias.dtype != weight.dtype
                or rho.dtype != weight.dtype
                or not all(
                    bool(torch.isfinite(value).all())
                    for value in (weight, bias, rho)
                )
            ):
                raise ValueError(
                    f"CCE {role} checkpoint tensor contract mismatch: {run_dir}"
                )
    if manifest.get("candidate_contract", {}).get("cce") != model.get("cce"):
        raise ValueError(f"CCE manifest/checkpoint source contract mismatch: {run_dir}")


def _validate_sonnet_checkpoints(scientific, config, manifest, metrics, run_dir):
    """Validate sealed Sonnet checkpoint identity and exact module state scope."""

    model = scientific["model"]
    dataset = scientific["dataset"]
    enabled = model["use_sonnet_mvca"]
    aux_count = len(dataset["aux_idx"])
    expected_shapes = (
        {
            "sonnet_mvca.gamma_sonnet": (),
            "sonnet_mvca.freq_params": (64, 8, 3),
            "sonnet_mvca.aux_embedding.weight": (32, aux_count),
            "sonnet_mvca.aux_embedding.bias": (32,),
            "sonnet_mvca.target_embedding.weight": (32, 1),
            "sonnet_mvca.target_embedding.bias": (32,),
            "sonnet_mvca.mvca.qkv_projection.weight": (192, 64),
            "sonnet_mvca.mvca.qkv_projection.bias": (192,),
            "sonnet_mvca.mvca.residual_mlp.0.weight": (64, 64),
            "sonnet_mvca.mvca.residual_mlp.0.bias": (64,),
            "sonnet_mvca.mvca.residual_mlp.2.weight": (64, 64),
            "sonnet_mvca.mvca.residual_mlp.2.bias": (64,),
            "sonnet_mvca.mvca.output_projection.weight": (64, 64),
            "sonnet_mvca.mvca.output_projection.bias": (64,),
            "sonnet_mvca.readout.weight": (1, 64),
            "sonnet_mvca.readout.bias": (1,),
        }
        if enabled
        else {}
    )
    expected_protocol = scientific["training_protocol"]
    expected_evaluation = scientific["evaluation"]
    try:
        checkpoints = {
            role: torch.load(Path(run_dir) / f"{role}.pt", map_location="cpu")
            for role in ("best", "last")
        }
    except Exception as exc:
        raise ValueError(f"cannot read Sonnet checkpoints: {run_dir}") from exc
    for role, checkpoint in checkpoints.items():
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Sonnet {role} checkpoint must be a dictionary: {run_dir}"
            )
        resolved = checkpoint.get("resolved_config")
        if (
            checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("artifact_schema_version")
            != ENHANCED_ARTIFACT_SCHEMA_VERSION
            or checkpoint.get("implementation_variant")
            != SONNET_IMPLEMENTATION_VARIANT
            or checkpoint.get("config_hash") != metrics.get("config_hash")
            or checkpoint.get("data_sha256") != metrics.get("data_sha256")
            or checkpoint.get("evaluation_policy")
            != expected_evaluation["evaluation_policy"]
            or checkpoint.get("artifact_purpose")
            != expected_evaluation["artifact_purpose"]
            or checkpoint.get("training_protocol") != expected_protocol
            or not isinstance(resolved, dict)
            or resolved.get("config_hash") != metrics.get("config_hash")
            or resolved.get("scientific_config") != scientific
            or resolved.get("evaluation_policy")
            != expected_evaluation["evaluation_policy"]
            or resolved.get("artifact_purpose")
            != expected_evaluation["artifact_purpose"]
        ):
            raise ValueError(
                f"Sonnet {role} checkpoint scientific identity mismatch: {run_dir}"
            )

        states = [("model_state", checkpoint.get("model_state"))]
        if role == "last":
            states.append(
                ("best_model_state", checkpoint.get("best_model_state"))
            )
        for state_name, state in states:
            if not isinstance(state, dict):
                raise ValueError(
                    f"Sonnet {role} checkpoint has no {state_name}: {run_dir}"
                )
            sonnet_keys = {
                key for key in state if key.startswith("sonnet_mvca.")
            }
            forbidden = sorted(
                key
                for key in state
                if key.startswith(("teb.", "cce.", "pmcr.", "xlinear."))
            )
            if sonnet_keys != set(expected_shapes) or forbidden:
                raise ValueError(
                    f"Sonnet {role} {state_name} module key mismatch: "
                    f"sonnet={sorted(sonnet_keys)}, forbidden={forbidden}: {run_dir}"
                )
            for key, shape in expected_shapes.items():
                tensor = state[key]
                if (
                    not torch.is_tensor(tensor)
                    or tuple(tensor.shape) != shape
                    or tensor.dtype != torch.float32
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ValueError(
                        f"Sonnet {role} {state_name} tensor mismatch "
                        f"for {key}: {run_dir}"
                    )
    if (
        manifest.get("candidate_contract", {}).get("sonnet_mvca")
        != model.get("sonnet_mvca")
    ):
        raise ValueError(
            f"Sonnet manifest/checkpoint source contract mismatch: {run_dir}"
        )


_TARGET_EXOGENOUS_SCHEMA_FIELDS = {
    "contract_version",
    "feature_type",
    "feature_names",
    "target_feature_name",
    "target_idx",
    "target_indices",
    "aux_idx",
    "aux_feature_names",
    "schema_fingerprint",
}


def _validate_target_exogenous_schema(scientific, manifest, run_dir):
    dataset = scientific.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"enhanced dataset contract is missing: {run_dir}")
    task_mode = dataset.get("task_mode")
    version = dataset.get("target_exogenous_schema_contract_version")
    observed = manifest.get("target_exogenous_schema")

    if task_mode != "target_exogenous":
        if version is not None or observed is not None:
            raise ValueError(
                "parallel/non-target artifact must not carry "
                f"target_exogenous_schema_v1: {run_dir}"
            )
        return "not_applicable"

    if version is None:
        if observed is not None:
            raise ValueError(
                "manifest has target_exogenous schema but config has no version: "
                f"{run_dir}"
            )
        return "legacy"
    if version != TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION:
        raise ValueError(
            f"unsupported target_exogenous schema version {version!r}: {run_dir}"
        )

    expected = {
        "contract_version": version,
        "feature_type": dataset.get("feature_type"),
        "feature_names": dataset.get("feature_names"),
        "target_feature_name": dataset.get("target_feature_name"),
        "target_idx": dataset.get("target_idx"),
        "target_indices": dataset.get("target_indices"),
        "aux_idx": dataset.get("aux_idx"),
        "aux_feature_names": dataset.get("aux_feature_names"),
        "schema_fingerprint": dataset.get("schema_fingerprint"),
    }
    if not isinstance(observed, dict):
        raise ValueError(
            f"target_exogenous_schema_v1 manifest block is missing: {run_dir}"
        )
    if set(observed) != _TARGET_EXOGENOUS_SCHEMA_FIELDS:
        raise ValueError(
            f"target_exogenous schema field set mismatch: {run_dir}"
        )
    feature_names = expected["feature_names"]
    target_idx = expected["target_idx"]
    target_indices = expected["target_indices"]
    aux_idx = expected["aux_idx"]
    if (
        expected["feature_type"] != "MS"
        or dataset.get("target") != expected["target_feature_name"]
        or not isinstance(feature_names, list)
        or not feature_names
        or any(not isinstance(name, str) or not name for name in feature_names)
        or len(set(feature_names)) != len(feature_names)
        or isinstance(target_idx, bool)
        or not isinstance(target_idx, int)
        or not 0 <= target_idx < len(feature_names)
        or target_indices != [target_idx]
        or expected["target_feature_name"] != feature_names[target_idx]
        or not isinstance(aux_idx, list)
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in aux_idx
        )
        or len(aux_idx) != len(set(aux_idx))
        or any(not 0 <= index < len(feature_names) for index in aux_idx)
        or target_idx in aux_idx
        or expected["aux_feature_names"]
        != [feature_names[index] for index in aux_idx]
        or not isinstance(expected["schema_fingerprint"], str)
        or not expected["schema_fingerprint"]
    ):
        raise ValueError(
            f"target_exogenous scientific schema is invalid: {run_dir}"
        )
    if observed != expected:
        raise ValueError(
            "target_exogenous config/manifest schema mismatch: "
            f"expected {expected!r}, got {observed!r}: {run_dir}"
        )
    return version



def _load_legacy_completed_runs(artifact_root):
    """Return validated run rows; failed/running/foreign variants are ignored."""

    variant_root = Path(artifact_root).resolve() / IMPLEMENTATION_VARIANT
    if not variant_root.exists():
        return []

    rows = []
    seen_run_ids = set()
    for metrics_path in sorted(variant_root.rglob("metrics.json")):
        metrics = _read_json(metrics_path)
        if metrics.get("status") != "completed":
            continue
        if metrics.get("implementation_variant") != IMPLEMENTATION_VARIANT:
            raise ValueError(f"variant mismatch in {metrics_path}")

        run_dir = metrics_path.parent.resolve()
        manifest_path = run_dir / "manifest.json"
        config_path = run_dir / "config.resolved.json"
        required_artifacts = (
            manifest_path,
            config_path,
            run_dir / "best.pt",
            run_dir / "last.pt",
            run_dir / "history.jsonl",
        )
        missing = [path.name for path in required_artifacts if not path.is_file()]
        if missing:
            raise ValueError(
                f"completed run is missing required artifacts {missing}: {run_dir}"
            )
        manifest = _read_json(manifest_path)
        config = _read_json(config_path)
        if not all(isinstance(value, dict) for value in (metrics, manifest, config)):
            raise ValueError(f"run metadata must contain JSON objects: {run_dir}")
        for label, value, path in (
            ("metrics", metrics, metrics_path),
            ("manifest", manifest, manifest_path),
            ("config", config, config_path),
        ):
            if value.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{label} schema version mismatch: {path}")
        if manifest.get("status") != "completed":
            raise ValueError(f"metrics are completed but manifest is not: {run_dir}")
        if manifest.get("implementation_variant") != IMPLEMENTATION_VARIANT:
            raise ValueError(f"manifest variant mismatch: {manifest_path}")
        if config.get("implementation_variant") != IMPLEMENTATION_VARIANT:
            raise ValueError(f"config variant mismatch: {config_path}")

        run_id = metrics.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"missing run_id in {metrics_path}")
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate run_id {run_id!r}")
        seen_run_ids.add(run_id)
        if manifest.get("run_id") != run_id or run_dir.name != run_id:
            raise ValueError(f"run_id/path mismatch in {run_dir}")
        expected_artifact_dir = str(run_dir)
        if (
            manifest.get("artifact_dir") != expected_artifact_dir
            or metrics.get("artifact_dir") != expected_artifact_dir
        ):
            raise ValueError(f"artifact path mismatch in {run_dir}")
        if manifest.get("metrics_file") != metrics_path.name:
            raise ValueError(f"manifest metrics filename mismatch in {run_dir}")

        config_hash = metrics.get("config_hash")
        data_sha256 = metrics.get("data_sha256")
        scientific = config.get("scientific_config")
        if not isinstance(scientific, dict):
            raise ValueError(f"missing scientific_config in {config_path}")
        if scientific.get("implementation_variant") != IMPLEMENTATION_VARIANT:
            raise ValueError(f"scientific config variant mismatch in {run_dir}")
        if not isinstance(config_hash, str) or _stable_hash(scientific) != config_hash:
            raise ValueError(f"scientific config hash mismatch in {run_dir}")
        _validate_variant_contract(scientific, run_dir)
        if config.get("config_hash") != config_hash:
            raise ValueError(f"config hash mismatch in {run_dir}")
        if manifest.get("config_hash") != config_hash:
            raise ValueError(f"manifest config hash mismatch in {run_dir}")
        if manifest.get("data_sha256") != data_sha256:
            raise ValueError(f"data hash mismatch in {run_dir}")
        dataset_config = scientific.get("dataset")
        model_config = scientific.get("model")
        execution_config = scientific.get("execution")
        run_config = config.get("run")
        if not all(
            isinstance(value, dict)
            for value in (dataset_config, model_config, execution_config, run_config)
        ):
            raise ValueError(f"incomplete resolved configuration in {run_dir}")
        if dataset_config.get("sha256") != data_sha256:
            raise ValueError(f"scientific data hash mismatch in {run_dir}")
        recorded_run_dir = run_config.get("run_dir")
        if (
            not isinstance(recorded_run_dir, str)
            or Path(recorded_run_dir).resolve() != run_dir
        ):
            raise ValueError(f"resolved run path mismatch in {run_dir}")

        train_epochs = int(metrics["train_epochs"])
        best_epoch = int(metrics["best_epoch"])
        if train_epochs <= 0 or not 1 <= best_epoch <= train_epochs:
            raise ValueError(f"invalid epoch metadata in {metrics_path}")
        if int(run_config.get("train_epochs", -1)) != train_epochs:
            raise ValueError(f"resolved training budget mismatch in {run_dir}")
        if int(manifest.get("completed_epoch", -1)) != train_epochs:
            raise ValueError(f"manifest completed epoch mismatch in {run_dir}")
        if int(manifest.get("best_epoch", -1)) != best_epoch:
            raise ValueError(f"manifest best epoch mismatch in {run_dir}")
        expected_identity = (
            str(dataset_config.get("id")),
            int(model_config.get("seq_len", -1)),
            int(model_config.get("pred_len", -1)),
            int(execution_config.get("seed", -1)),
        )
        observed_identity = (
            str(metrics.get("dataset_id")),
            int(metrics.get("seq_len", -1)),
            int(metrics.get("pred_len", -1)),
            int(metrics.get("seed", -1)),
        )
        if observed_identity != expected_identity:
            raise ValueError(f"run identity mismatch in {run_dir}")
        if (
            metrics.get("metric_space") != METRIC_SPACE
            or execution_config.get("metric_space") != METRIC_SPACE
        ):
            raise ValueError(f"metric space mismatch in {run_dir}")

        validation = metrics.get("best_validation")
        test = metrics.get("test")
        if not isinstance(validation, dict) or not isinstance(test, dict):
            raise ValueError(f"missing validation/test metrics in {metrics_path}")
        val_mse = _finite_number(validation.get("mse"), f"{run_id} val_mse")
        val_mae = _finite_number(validation.get("mae"), f"{run_id} val_mae")
        test_mse = _finite_number(test.get("mse"), f"{run_id} test_mse")
        test_mae = _finite_number(test.get("mae"), f"{run_id} test_mae")
        for field, observed, expected in (
            ("best_validation_mse", manifest.get("best_validation_mse"), val_mse),
            ("test_mse", manifest.get("test_mse"), test_mse),
            ("test_mae", manifest.get("test_mae"), test_mae),
        ):
            if _finite_number(observed, f"{run_id} manifest {field}") != expected:
                raise ValueError(f"manifest {field} mismatch in {run_dir}")
        row = {
            "implementation_variant": IMPLEMENTATION_VARIANT,
            "training_protocol_id": STANDARD_TRAINING_PROTOCOL,
            "development_protocol_id": None,
            "ablation_id": None,
            "dataset_id": str(metrics["dataset_id"]),
            "seq_len": int(metrics["seq_len"]),
            "pred_len": int(metrics["pred_len"]),
            "seed": int(metrics["seed"]),
            "run_id": run_id,
            "best_epoch": best_epoch,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "parameter_count": int(metrics["parameter_count"]),
            "train_epochs": train_epochs,
            "duration_seconds": _finite_number(
                metrics.get("duration_seconds"), f"{run_id} duration_seconds"
            ),
            "config_hash": str(config_hash),
            "comparison_config_hash": _comparison_hash(
                config, config_path, metrics["train_epochs"]
            ),
            "data_sha256": str(data_sha256),
            "completed_at": str(metrics["completed_at"]),
            "artifact_dir": str(run_dir),
        }
        rows.append(row)

    rows.sort(key=lambda item: (
        item["dataset_id"], item["seq_len"], item["pred_len"], item["seed"],
        item["run_id"],
    ))
    return rows


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_enhanced_checksums(run_dir):
    run_dir = Path(run_dir)
    checksum_path = run_dir / "checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError(f"enhanced completed run has no checksums.sha256: {run_dir}")
    observed = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid enhanced checksum line in {checksum_path}: {line!r}")
        candidate = Path(name)
        if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name != name:
            raise ValueError(f"unsafe enhanced checksum path: {name!r}")
        if name in observed:
            raise ValueError(f"duplicate enhanced checksum entry: {name}")
        observed[name] = digest
    expected = set(ENHANCED_CHECKSUM_FILES)
    if set(observed) != expected:
        raise ValueError(
            "enhanced checksum file set mismatch: "
            f"missing={sorted(expected - set(observed))}, "
            f"unexpected={sorted(set(observed) - expected)}"
        )
    for name, expected_digest in observed.items():
        path = run_dir / name
        if not path.is_file():
            raise ValueError(f"enhanced checksum target is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected_digest:
            raise ValueError(
                f"enhanced checksum mismatch for {name}: "
                f"{actual} != {expected_digest}"
            )
    allowed_files = expected | {"checksums.sha256"}
    if (run_dir / ".run.lock").exists():
        raise ValueError(f"enhanced completed artifact retains .run.lock: {run_dir}")
    extras = sorted(
        path.name for path in run_dir.iterdir()
        if path.name not in allowed_files
    )
    if extras:
        raise ValueError(f"unexpected entries in enhanced completed artifact: {extras}")
    return observed


def _test_result_paths(document, *, allow_access_policy=False):
    paths = []

    def walk(value, prefix=()):
        if isinstance(value, dict):
            for key, child in value.items():
                current = (*prefix, str(key))
                is_test_field = key == "test" or str(key).startswith("test_")
                allowed = (
                    allow_access_policy
                    and not prefix
                    and key == "test_access_policy"
                )
                if is_test_field and not allowed:
                    paths.append(".".join(current))
                walk(child, current)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*prefix, str(index)))

    walk(document)
    return paths


def _validate_sonnet_evaluation_artifact(
    scientific, config, manifest, metrics, run_dir
):
    evaluation = scientific.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"Sonnet evaluation block is missing: {run_dir}")
    policy = evaluation.get("evaluation_policy")
    purpose = evaluation.get("artifact_purpose")
    access = evaluation.get("test_access_policy")
    if (
        purpose != M4_DEVELOPMENT_CANDIDATE
        or policy not in {TRAIN_VALIDATION_TEST, TRAIN_VALIDATION_ONLY}
    ):
        raise ValueError(f"Sonnet evaluation identity mismatch: {run_dir}")
    if (
        config.get("evaluation_policy") != policy
        or config.get("artifact_purpose") != purpose
        or manifest.get("evaluation_policy") != policy
        or manifest.get("artifact_purpose") != purpose
        or manifest.get("test_access_policy") != access
        or metrics.get("evaluation_policy") != policy
        or metrics.get("artifact_purpose") != purpose
    ):
        raise ValueError(
            f"Sonnet evaluation metadata mismatch across artifact: {run_dir}"
        )

    if policy == TRAIN_VALIDATION_ONLY:
        if access != "forbidden":
            raise ValueError(
                f"validation-only test access must be forbidden: {run_dir}"
            )
        metric_test_paths = _test_result_paths(metrics)
        manifest_test_paths = _test_result_paths(
            manifest, allow_access_policy=True
        )
        if metric_test_paths or manifest_test_paths:
            raise ValueError(
                "validation-only artifact contains test result fields "
                f"metrics={metric_test_paths}, manifest={manifest_test_paths}: "
                f"{run_dir}"
            )
        forbidden_log_markers = ("final test", "test_mse", "test_mae")
        for name in ("stdout.log", "stderr.log", "train.log"):
            text = (Path(run_dir) / name).read_text(
                encoding="utf-8", errors="replace"
            ).casefold()
            if any(marker in text for marker in forbidden_log_markers):
                raise ValueError(
                    f"validation-only artifact log contains test output: "
                    f"{name}: {run_dir}"
                )
    else:
        if access != "development_only":
            raise ValueError(
                f"ETTm1 test access must be development_only: {run_dir}"
            )
        if not isinstance(metrics.get("test"), dict):
            raise ValueError(f"test-inclusive metrics are missing: {run_dir}")
        if "test_mse" not in manifest or "test_mae" not in manifest:
            raise ValueError(
                f"test-inclusive manifest metrics are missing: {run_dir}"
            )
    return policy


def _enhanced_path_identity(variant_root, run_dir):
    relative = run_dir.relative_to(variant_root)
    parts = relative.parts
    if len(parts) != 7:
        raise ValueError(
            "enhanced artifact path must be dataset/task/target/horizon/fold/seed/run_id: "
            f"{run_dir}"
        )
    dataset_id, task_mode, target, horizon_part, fold_part, seed_part, run_id = parts
    if not horizon_part.startswith("horizon_"):
        raise ValueError(f"enhanced artifact horizon component is invalid: {run_dir}")
    if not fold_part.startswith("fold_"):
        raise ValueError(f"enhanced artifact fold component is invalid: {run_dir}")
    if not seed_part.startswith("seed_"):
        raise ValueError(f"enhanced artifact seed component is invalid: {run_dir}")
    try:
        horizon = int(horizon_part[len("horizon_"):])
        seed = int(seed_part[len("seed_"):])
    except ValueError as exc:
        raise ValueError(f"enhanced artifact horizon/seed is not an integer: {run_dir}") from exc
    return {
        "dataset_id": dataset_id,
        "task_mode": task_mode,
        "target": target,
        "artifact_horizon": horizon,
        "fold": fold_part[len("fold_"):],
        "seed": seed,
        "run_id": run_id,
    }


def _load_enhanced_completed_runs(artifact_root, implementation_variant):
    variant_root = Path(artifact_root).resolve() / implementation_variant
    if not variant_root.exists():
        return []

    candidate_dirs = {
        path.parent.resolve()
        for pattern in ("manifest.json", "metrics.json")
        for path in variant_root.rglob(pattern)
        if not any(
            part.startswith(".") and part.endswith(".staging")
            for part in path.relative_to(variant_root).parts
        )
    }
    rows = []
    seen_run_ids = set()
    seen_scientific_seed = set()
    for run_dir in sorted(candidate_dirs):
        identity = _enhanced_path_identity(variant_root, run_dir)
        _verify_enhanced_checksums(run_dir)
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.resolved.json"
        manifest = _read_json(manifest_path)
        metrics = _read_json(metrics_path)
        config = _read_json(config_path)
        if not all(isinstance(value, dict) for value in (manifest, metrics, config)):
            raise ValueError(f"enhanced metadata must contain JSON objects: {run_dir}")
        for label, value in (
            ("manifest", manifest),
            ("metrics", metrics),
            ("config", config),
        ):
            if value.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"enhanced {label} schema version mismatch: {run_dir}")
            if (
                value.get("artifact_schema_version")
                != ENHANCED_ARTIFACT_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"enhanced {label} artifact schema version mismatch: {run_dir}"
                )
            if value.get("implementation_variant") != implementation_variant:
                raise ValueError(f"enhanced {label} variant mismatch: {run_dir}")
        if manifest.get("status") != "completed" or metrics.get("status") != "completed":
            raise ValueError(f"enhanced final artifact is not completed: {run_dir}")

        run_id = identity["run_id"]
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate enhanced run_id {run_id!r}")
        seen_run_ids.add(run_id)
        if manifest.get("run_id") != run_id or metrics.get("run_id") != run_id:
            raise ValueError(f"enhanced run_id/path mismatch: {run_dir}")
        expected_dir = str(run_dir)
        run_config = config.get("run")
        if not isinstance(run_config, dict):
            raise ValueError(f"enhanced resolved config has no run object: {run_dir}")
        if any(
            value != expected_dir
            for value in (
                manifest.get("artifact_dir"),
                metrics.get("artifact_dir"),
                run_config.get("run_dir"),
            )
        ):
            raise ValueError(f"enhanced artifact path identity mismatch: {run_dir}")

        scientific = config.get("scientific_config")
        if not isinstance(scientific, dict):
            raise ValueError(f"enhanced config has no scientific_config: {run_dir}")
        config_hash = metrics.get("config_hash")
        if not isinstance(config_hash, str) or _stable_hash(scientific) != config_hash:
            raise ValueError(f"enhanced scientific config hash mismatch: {run_dir}")
        if (
            config.get("config_hash") != config_hash
            or manifest.get("config_hash") != config_hash
        ):
            raise ValueError(f"enhanced config/manifest hash mismatch: {run_dir}")
        if scientific.get("implementation_variant") != implementation_variant:
            raise ValueError(f"enhanced scientific variant mismatch: {run_dir}")

        dataset = scientific.get("dataset")
        model = scientific.get("model")
        execution = scientific.get("execution")
        optimization = scientific.get("optimization")
        experiment = scientific.get("experiment")
        if not all(
            isinstance(value, dict)
            for value in (dataset, model, execution, optimization, experiment)
        ):
            raise ValueError(f"enhanced scientific contract is incomplete: {run_dir}")
        if model.get("model_class") != "AMDEnhanced":
            raise ValueError(f"enhanced model_class mismatch: {run_dir}")
        target_schema_identity = _validate_target_exogenous_schema(
            scientific, manifest, run_dir
        )
        expected_candidate_contract = _validate_enhanced_variant_contract(
            scientific, implementation_variant, run_dir
        )
        observed_candidate_contract = manifest.get("candidate_contract")
        if observed_candidate_contract != expected_candidate_contract:
            raise ValueError(
                "enhanced manifest candidate contract mismatch: "
                f"expected {expected_candidate_contract!r}, "
                f"got {observed_candidate_contract!r}: {run_dir}"
            )
        evaluation_policy = (
            _validate_sonnet_evaluation_artifact(
                scientific, config, manifest, metrics, run_dir
            )
            if implementation_variant == SONNET_IMPLEMENTATION_VARIANT
            else TRAIN_VALIDATION_TEST
        )
        protocol_info = _validate_warm_start_artifact(
            scientific,
            config,
            manifest,
            metrics,
            run_dir,
            implementation_variant,
        )
        if implementation_variant in {
            CCE_IMPLEMENTATION_VARIANT,
            LATE_CCE_IMPLEMENTATION_VARIANT,
        }:
            _validate_cce_checkpoints(
                scientific,
                implementation_variant,
                config,
                manifest,
                metrics,
                run_dir,
            )
        elif implementation_variant == SONNET_IMPLEMENTATION_VARIANT:
            _validate_sonnet_checkpoints(
                scientific, config, manifest, metrics, run_dir
            )
        expected_weight_decay = (
            0.0
            if protocol_info["training_protocol_id"]
            == T2_ADAPTER_TRAINING_PROTOCOL
            else 1e-7
        )
        if (
            optimization.get("optimizer") != "Adam"
            or optimization.get("weight_decay") != expected_weight_decay
        ):
            raise ValueError(f"enhanced optimization contract mismatch: {run_dir}")

        observed_identity = {
            "dataset_id": str(dataset.get("id")),
            "task_mode": dataset.get("task_mode"),
            "target": dataset.get("target"),
            "artifact_horizon": dataset.get("artifact_horizon"),
            "fold": str(dataset.get("fold")),
            "seed": execution.get("seed"),
            "run_id": run_id,
        }
        if observed_identity != identity:
            raise ValueError(
                f"enhanced config/path identity mismatch: "
                f"{observed_identity!r} != {identity!r}"
            )
        for document_name, document in (("manifest", manifest), ("metrics", metrics)):
            document_identity = {
                "task_mode": document.get("task_mode"),
                "target": document.get("target"),
                "artifact_horizon": document.get("artifact_horizon"),
                "fold": str(document.get("fold")),
                "seed": document.get("seed"),
            }
            expected = {
                key: identity[key]
                for key in ("task_mode", "target", "artifact_horizon", "fold", "seed")
            }
            if document_identity != expected:
                raise ValueError(
                    f"enhanced {document_name}/path identity mismatch: {run_dir}"
                )
        if experiment.get("artifact_horizon") != identity["artifact_horizon"]:
            raise ValueError(f"enhanced experiment horizon mismatch: {run_dir}")
        if model.get("model_pred_len") != metrics.get("model_pred_len"):
            raise ValueError(f"enhanced model_pred_len mismatch: {run_dir}")
        if dataset.get("label_horizon") != metrics.get("label_horizon"):
            raise ValueError(f"enhanced label_horizon mismatch: {run_dir}")

        data_sha256 = metrics.get("data_sha256")
        if (
            not isinstance(data_sha256, str)
            or dataset.get("sha256") != data_sha256
            or manifest.get("data_sha256") != data_sha256
        ):
            raise ValueError(f"enhanced data fingerprint mismatch: {run_dir}")
        train_epochs = int(metrics.get("train_epochs", -1))
        if (
            implementation_variant == SONNET_IMPLEMENTATION_VARIANT
            and train_epochs != 10
        ):
            raise ValueError(f"Sonnet train_epochs must equal 10: {run_dir}")
        best_epoch = int(metrics.get("best_epoch", -1))
        if protocol_info["warm_start"]:
            completed_epochs = protocol_info["completed_epochs"]
            if (
                train_epochs <= 0
                or not 0 <= best_epoch <= completed_epochs
                or int(run_config.get("train_epochs", -1)) != train_epochs
                or int(manifest.get("best_epoch", -1)) != best_epoch
            ):
                raise ValueError(
                    f"enhanced warm-start epoch metadata is invalid: {run_dir}"
                )
        else:
            if train_epochs <= 0 or not 1 <= best_epoch <= train_epochs:
                raise ValueError(f"enhanced epoch metadata is invalid: {run_dir}")
            if (
                int(run_config.get("train_epochs", -1)) != train_epochs
                or int(manifest.get("completed_epoch", -1)) != train_epochs
                or int(manifest.get("best_epoch", -1)) != best_epoch
            ):
                raise ValueError(
                    f"enhanced manifest/config epoch mismatch: {run_dir}"
                )

        validation = metrics.get("best_validation")
        if not isinstance(validation, dict):
            raise ValueError(f"enhanced validation metrics are missing: {run_dir}")
        val_mse = _finite_number(validation.get("mse"), f"{run_id} val_mse")
        val_mae = _finite_number(validation.get("mae"), f"{run_id} val_mae")
        if _finite_number(
            manifest.get("best_validation_mse"),
            f"{run_id} manifest best_validation_mse",
        ) != val_mse:
            raise ValueError(
                f"enhanced manifest best_validation_mse mismatch: {run_dir}"
            )
        test_mse = test_mae = None
        if evaluation_policy == TRAIN_VALIDATION_TEST:
            test = metrics.get("test")
            if not isinstance(test, dict):
                raise ValueError(f"enhanced test metrics are missing: {run_dir}")
            test_mse = _finite_number(test.get("mse"), f"{run_id} test_mse")
            test_mae = _finite_number(test.get("mae"), f"{run_id} test_mae")
            for field, observed, expected in (
                ("test_mse", manifest.get("test_mse"), test_mse),
                ("test_mae", manifest.get("test_mae"), test_mae),
            ):
                if _finite_number(
                    observed, f"{run_id} manifest {field}"
                ) != expected:
                    raise ValueError(
                        f"enhanced manifest {field} mismatch: {run_dir}"
                    )

        comparison_hash = _comparison_hash(config, config_path, train_epochs)
        duplicate_identity = (comparison_hash, int(identity["seed"]))
        if duplicate_identity in seen_scientific_seed:
            raise ValueError(
                "multiple enhanced completed runs exist for the same scientific "
                "identity and seed; no run was selected automatically"
            )
        seen_scientific_seed.add(duplicate_identity)
        row = {
            "implementation_variant": implementation_variant,
            "training_protocol_id": protocol_info["training_protocol_id"],
            "development_protocol_id": experiment.get("development_protocol_id"),
            "ablation_id": experiment.get("ablation_id"),
            "dataset_id": identity["dataset_id"],
            "task_mode": identity["task_mode"],
            "target": identity["target"],
            "label_horizon": metrics.get("label_horizon"),
            "fold": identity["fold"],
            "seq_len": int(metrics.get("seq_len", -1)),
            "pred_len": int(metrics.get("pred_len", -1)),
            "target_exogenous_schema_contract": target_schema_identity,
            "seed": int(identity["seed"]),
            "run_id": run_id,
            "best_epoch": best_epoch,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "parameter_count": int(metrics.get("parameter_count", -1)),
            "train_epochs": train_epochs,
            "duration_seconds": _finite_number(
                metrics.get("duration_seconds"), f"{run_id} duration_seconds"
            ),
            "config_hash": config_hash,
            "comparison_config_hash": comparison_hash,
            "data_sha256": data_sha256,
            "completed_at": str(metrics.get("completed_at")),
            "artifact_dir": str(run_dir),
        }
        if evaluation_policy == TRAIN_VALIDATION_TEST:
            row.update({"test_mse": test_mse, "test_mae": test_mae})
        if implementation_variant == SONNET_IMPLEMENTATION_VARIANT:
            row.update({
                "evaluation_policy": evaluation_policy,
                "artifact_purpose": M4_DEVELOPMENT_CANDIDATE,
            })
        rows.append(row)
    if implementation_variant == SONNET_IMPLEMENTATION_VARIANT:
        policies = {row["evaluation_policy"] for row in rows}
        if len(policies) > 1:
            raise ValueError(
                "Sonnet summarizer refuses mixed evaluation_policy artifacts"
            )
    rows.sort(
        key=lambda item: (
            item["dataset_id"],
            item["task_mode"],
            item["target"],
            str(item["label_horizon"]),
            str(item["fold"]),
            item["seed"],
            item["run_id"],
        )
    )
    return rows


def load_completed_runs(
    artifact_root,
    implementation_variant=IMPLEMENTATION_VARIANT,
):
    """Load one explicitly supported variant without mutating artifacts."""

    if implementation_variant not in SUPPORTED_IMPLEMENTATION_VARIANTS:
        raise ValueError(
            f"unsupported implementation variant: {implementation_variant!r}; "
            f"expected one of {SUPPORTED_IMPLEMENTATION_VARIANTS}"
        )
    if implementation_variant == IMPLEMENTATION_VARIANT:
        return _load_legacy_completed_runs(artifact_root)
    return _load_enhanced_completed_runs(
        artifact_root, implementation_variant
    )



def aggregate_runs(rows):
    """Aggregate paired seed runs, refusing ambiguous duplicate successful seeds."""

    groups = defaultdict(list)
    for row in rows:
        key = (
            row["implementation_variant"], row["dataset_id"], row["seq_len"],
            row["pred_len"], row["comparison_config_hash"],
        )
        groups[key].append(row)

    aggregates = []
    for key, group in sorted(groups.items()):
        category_identities = {
            (
                row.get("training_protocol_id", STANDARD_TRAINING_PROTOCOL),
                row.get("development_protocol_id"),
                row.get("ablation_id"),
            )
            for row in group
        }
        if len(category_identities) != 1:
            raise ValueError(
                "one comparison group contains mixed protocol/ablation identities"
            )
        training_protocol_id, development_protocol_id, ablation_id = next(
            iter(category_identities)
        )
        seeds = [row["seed"] for row in group]
        if len(seeds) != len(set(seeds)):
            duplicates = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
            raise ValueError(
                "multiple completed runs exist for the same scientific config and seed "
                f"{duplicates}; no run was selected automatically"
            )

        def mean_and_std(field):
            values = [row[field] for row in group]
            return statistics.mean(values), (
                statistics.stdev(values) if len(values) > 1 else ""
            )

        val_mse_mean, val_mse_std = mean_and_std("val_mse")
        val_mae_mean, val_mae_std = mean_and_std("val_mae")
        policies = {
            row.get("evaluation_policy", TRAIN_VALIDATION_TEST)
            for row in group
        }
        if len(policies) != 1:
            raise ValueError(
                "one comparison group contains mixed evaluation_policy values"
            )
        evaluation_policy = next(iter(policies))
        aggregate = {
            "implementation_variant": key[0],
            "training_protocol_id": training_protocol_id,
            "development_protocol_id": development_protocol_id,
            "ablation_id": ablation_id,
            "dataset_id": key[1],
            "seq_len": key[2],
            "pred_len": key[3],
            "comparison_config_hash": key[4],
            "seed_count": len(seeds),
            "seeds": ";".join(str(seed) for seed in sorted(seeds)),
            "val_mse_mean": val_mse_mean,
            "val_mse_sample_std": val_mse_std,
            "val_mae_mean": val_mae_mean,
            "val_mae_sample_std": val_mae_std,
        }
        if evaluation_policy == TRAIN_VALIDATION_TEST:
            test_mse_mean, test_mse_std = mean_and_std("test_mse")
            test_mae_mean, test_mae_std = mean_and_std("test_mae")
            aggregate.update({
                "test_mse_mean": test_mse_mean,
                "test_mse_sample_std": test_mse_std,
                "test_mae_mean": test_mae_mean,
                "test_mae_sample_std": test_mae_std,
            })
        if key[0] == SONNET_IMPLEMENTATION_VARIANT:
            aggregate.update({
                "evaluation_policy": evaluation_policy,
                "artifact_purpose": M4_DEVELOPMENT_CANDIDATE,
            })
        aggregates.append(aggregate)
    return aggregates


def _atomic_write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_summaries(
    artifact_root,
    output_dir,
    implementation_variant=IMPLEMENTATION_VARIANT,
):
    rows = load_completed_runs(
        artifact_root,
        implementation_variant=implementation_variant,
    )
    aggregates = aggregate_runs(rows)
    output_dir = Path(output_dir).resolve()
    run_path = output_dir / f"{implementation_variant}.csv"
    aggregate_path = output_dir / f"{implementation_variant}-aggregate.csv"
    run_fields = RUN_FIELDS
    aggregate_fields = AGGREGATE_FIELDS
    if implementation_variant == SONNET_IMPLEMENTATION_VARIANT:
        policies = {row["evaluation_policy"] for row in rows}
        if len(policies) > 1:
            raise ValueError(
                "Sonnet summarizer refuses mixed evaluation_policy artifacts"
            )
        policy = (
            next(iter(policies)) if policies else TRAIN_VALIDATION_TEST
        )
        if policy == TRAIN_VALIDATION_ONLY:
            run_fields = SONNET_VALIDATION_RUN_FIELDS
            aggregate_fields = SONNET_VALIDATION_AGGREGATE_FIELDS
        else:
            run_fields = SONNET_TEST_RUN_FIELDS
            aggregate_fields = SONNET_TEST_AGGREGATE_FIELDS
    _atomic_write_csv(run_path, run_fields, rows)
    _atomic_write_csv(aggregate_path, aggregate_fields, aggregates)
    return run_path, aggregate_path, len(rows), len(aggregates)


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=f"Summarize completed {IMPLEMENTATION_VARIANT} artifacts"
    )
    parser.add_argument("--artifact_root", default=str(root / "artifacts"))
    parser.add_argument("--output_dir", default=str(root / "summaries"))
    parser.add_argument(
        "--implementation_variant",
        default=IMPLEMENTATION_VARIANT,
        choices=SUPPORTED_IMPLEMENTATION_VARIANTS,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_path, aggregate_path, run_count, group_count = write_summaries(
        args.artifact_root,
        args.output_dir,
        implementation_variant=args.implementation_variant,
    )
    print(f"wrote {run_count} run(s) to {run_path}")
    print(f"wrote {group_count} aggregate group(s) to {aggregate_path}")


if __name__ == "__main__":
    main()
