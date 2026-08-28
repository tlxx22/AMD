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


IMPLEMENTATION_VARIANT = "AMD-paper-norm-wd-ddi-v1"
ENHANCED_IMPLEMENTATION_VARIANT = "el-amd-pmcr-teb-v1"
T2_IMPLEMENTATION_VARIANT = "el-amd-m4-t2-patch-teb-v1"
SUPPORTED_IMPLEMENTATION_VARIANTS = (
    IMPLEMENTATION_VARIANT,
    ENHANCED_IMPLEMENTATION_VARIANT,
    T2_IMPLEMENTATION_VARIANT,
)
ENHANCED_ARTIFACT_SCHEMA_VERSION = 2
ENHANCED_CHECKSUM_FILES = (
    "best.pt", "last.pt", "config.resolved.json", "history.jsonl",
    "metrics.json", "manifest.json", "sys.argv.json", "command.txt",
    "stdout.log", "stderr.log", "train.log", "source_fingerprint.json",
    "data_fingerprint.json",
)
SCHEMA_VERSION = 1
METRIC_SPACE = "train-standardized"
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
    "implementation_variant", "dataset_id", "task_mode", "target",
    "label_horizon", "fold", "seq_len", "pred_len", "seed",
    "run_id", "best_epoch", "val_mse", "val_mae", "test_mse", "test_mae",
    "parameter_count", "train_epochs", "duration_seconds", "config_hash",
    "comparison_config_hash", "data_sha256", "completed_at", "artifact_dir",
)
AGGREGATE_FIELDS = (
    "implementation_variant", "dataset_id", "seq_len", "pred_len",
    "comparison_config_hash", "seed_count", "seeds",
    "val_mse_mean", "val_mse_sample_std", "val_mae_mean", "val_mae_sample_std",
    "test_mse_mean", "test_mse_sample_std", "test_mae_mean", "test_mae_sample_std",
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
    # The resume compatibility hash intentionally permits increasing the target
    # epoch count.  Completed runs with different training budgets must still
    # remain separate comparison groups.
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


def _validate_enhanced_variant_contract(scientific, implementation_variant, run_dir):
    """Keep Global v1 and T2 candidate artifacts distinct and explicit."""

    model = scientific.get("model")
    experiment = scientific.get("experiment")
    if not isinstance(model, dict) or not isinstance(experiment, dict):
        raise ValueError(f"enhanced variant contract is incomplete: {run_dir}")
    teb = model.get("teb")
    if not isinstance(teb, dict):
        raise ValueError(f"enhanced TEB contract is missing: {run_dir}")

    patch_fields = {
        "architecture",
        "patch_size",
        "patch_padding",
        "patch_position",
        "target_selection_policy",
    }
    if implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT:
        unexpected = sorted(patch_fields & set(teb))
        if unexpected:
            raise ValueError(
                f"Global TEB v1 artifact contains T2 patch fields {unexpected}: {run_dir}"
            )
        return None

    expected = {
        "architecture": "patch_conditioned_v1",
        "context_dim": 32,
        "heads": 4,
        "dropout": 0.1,
        "gamma_init": 1e-3,
        "patch_padding": "right_zero_crop",
        "patch_position": "fixed_sinusoidal",
        "target_selection_policy": "full_denorm_then_task_select",
    }
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
    if model.get("use_pmcr") is not False or model.get("use_teb") is not True:
        mismatches["module_switches"] = ((False, True), (model.get("use_pmcr"), model.get("use_teb")))
    if experiment.get("ablation_id") != "M4_T2":
        mismatches["ablation_id"] = ("M4_T2", experiment.get("ablation_id"))
    if mismatches:
        raise ValueError(f"unsupported T2 patch config {mismatches}: {run_dir}")

    dataset = scientific.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"enhanced dataset contract is missing: {run_dir}")
    return {
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
        if (
            optimization.get("optimizer") != "Adam"
            or optimization.get("weight_decay") != 1e-7
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
        best_epoch = int(metrics.get("best_epoch", -1))
        if train_epochs <= 0 or not 1 <= best_epoch <= train_epochs:
            raise ValueError(f"enhanced epoch metadata is invalid: {run_dir}")
        if (
            int(run_config.get("train_epochs", -1)) != train_epochs
            or int(manifest.get("completed_epoch", -1)) != train_epochs
            or int(manifest.get("best_epoch", -1)) != best_epoch
        ):
            raise ValueError(f"enhanced manifest/config epoch mismatch: {run_dir}")

        validation = metrics.get("best_validation")
        test = metrics.get("test")
        if not isinstance(validation, dict) or not isinstance(test, dict):
            raise ValueError(f"enhanced validation/test metrics are missing: {run_dir}")
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
                raise ValueError(f"enhanced manifest {field} mismatch: {run_dir}")

        comparison_hash = _comparison_hash(config, config_path, train_epochs)
        duplicate_identity = (comparison_hash, int(identity["seed"]))
        if duplicate_identity in seen_scientific_seed:
            raise ValueError(
                "multiple enhanced completed runs exist for the same scientific "
                "identity and seed; no run was selected automatically"
            )
        seen_scientific_seed.add(duplicate_identity)
        rows.append({
            "implementation_variant": implementation_variant,
            "dataset_id": identity["dataset_id"],
            "task_mode": identity["task_mode"],
            "target": identity["target"],
            "label_horizon": metrics.get("label_horizon"),
            "fold": identity["fold"],
            "seq_len": int(metrics.get("seq_len", -1)),
            "pred_len": int(metrics.get("pred_len", -1)),
            "seed": int(identity["seed"]),
            "run_id": run_id,
            "best_epoch": best_epoch,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
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
        })
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
        test_mse_mean, test_mse_std = mean_and_std("test_mse")
        test_mae_mean, test_mae_std = mean_and_std("test_mae")
        aggregates.append({
            "implementation_variant": key[0],
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
            "test_mse_mean": test_mse_mean,
            "test_mse_sample_std": test_mse_std,
            "test_mae_mean": test_mae_mean,
            "test_mae_sample_std": test_mae_std,
        })
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
    _atomic_write_csv(run_path, RUN_FIELDS, rows)
    _atomic_write_csv(aggregate_path, AGGREGATE_FIELDS, aggregates)
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
