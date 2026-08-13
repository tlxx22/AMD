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
    "implementation_variant", "dataset_id", "seq_len", "pred_len", "seed",
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


def load_completed_runs(artifact_root):
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


def write_summaries(artifact_root, output_dir):
    rows = load_completed_runs(artifact_root)
    aggregates = aggregate_runs(rows)
    output_dir = Path(output_dir).resolve()
    run_path = output_dir / f"{IMPLEMENTATION_VARIANT}.csv"
    aggregate_path = output_dir / f"{IMPLEMENTATION_VARIANT}-aggregate.csv"
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_path, aggregate_path, run_count, group_count = write_summaries(
        args.artifact_root, args.output_dir
    )
    print(f"wrote {run_count} run(s) to {run_path}")
    print(f"wrote {group_count} aggregate group(s) to {aggregate_path}")


if __name__ == "__main__":
    main()
