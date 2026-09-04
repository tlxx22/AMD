import csv
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

import main as runner
import summarize_results as summary


class SummaryTests(unittest.TestCase):
    @staticmethod
    def _make_run(root, run_id, seed, status="completed", test_mse=0.4):
        run_dir = (
            Path(root) / summary.IMPLEMENTATION_VARIANT / "toy" / "sl4_pl2"
            / f"seed{seed}" / run_id
        )
        run_dir.mkdir(parents=True)
        scientific = {
            "implementation_variant": summary.IMPLEMENTATION_VARIANT,
            "dataset": {"id": "toy", "sha256": "data-hash"},
            "execution": {
                "seed": seed,
                "device": "cpu",
                "metric_space": summary.METRIC_SPACE,
            },
            "model": {
                "seq_len": 4,
                "pred_len": 2,
                **summary.EXPECTED_MODEL_CONTRACT,
            },
            "optimization": {
                "batch_size": 8,
                **summary.EXPECTED_OPTIMIZATION_CONTRACT,
            },
        }
        config_hash = summary._stable_hash(scientific)
        config = {
            "schema_version": summary.SCHEMA_VERSION,
            "implementation_variant": summary.IMPLEMENTATION_VARIANT,
            "config_hash": config_hash,
            "scientific_config": scientific,
            "run": {"run_dir": str(run_dir.resolve()), "train_epochs": 2},
        }
        manifest = {
            "schema_version": summary.SCHEMA_VERSION,
            "implementation_variant": summary.IMPLEMENTATION_VARIANT,
            "run_id": run_id,
            "status": status,
            "config_hash": config_hash,
            "data_sha256": "data-hash",
            "artifact_dir": str(run_dir.resolve()),
            "completed_epoch": 2,
            "best_epoch": 2,
            "metrics_file": "metrics.json",
            "best_validation_mse": 0.3 + seed % 2 / 10,
            "test_mse": test_mse,
            "test_mae": 0.25,
        }
        metrics = {
            "schema_version": summary.SCHEMA_VERSION,
            "implementation_variant": summary.IMPLEMENTATION_VARIANT,
            "run_id": run_id,
            "status": status,
            "dataset_id": "toy",
            "seq_len": 4,
            "pred_len": 2,
            "seed": seed,
            "best_epoch": 2,
            "best_validation": {"mse": 0.3 + seed % 2 / 10, "mae": 0.2},
            "test": {"mse": test_mse, "mae": 0.25},
            "parameter_count": 10,
            "train_epochs": 2,
            "duration_seconds": 1.0,
            "metric_space": summary.METRIC_SPACE,
            "config_hash": config_hash,
            "data_sha256": "data-hash",
            "completed_at": "2026-01-01T00:00:00Z",
            "artifact_dir": str(run_dir.resolve()),
        }
        for filename, value in (
            ("config.resolved.json", config),
            ("manifest.json", manifest),
            ("metrics.json", metrics),
        ):
            (run_dir / filename).write_text(json.dumps(value), encoding="utf-8")
        for filename in ("best.pt", "last.pt", "history.jsonl"):
            (run_dir / filename).write_bytes(b"fixture")
        return run_dir
    @staticmethod
    def _write_enhanced_checksums(run_dir):
        lines = [
            f"{summary._sha256_file(run_dir / name)}  {name}"
            for name in summary.ENHANCED_CHECKSUM_FILES
        ]
        (run_dir / "checksums.sha256").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def _make_enhanced_run(
        cls,
        root,
        run_id,
        seed=2024,
        status="completed",
        implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
        patch_size=3,
        legacy_schema=False,
    ):
        run_dir = (
            Path(root)
            / implementation_variant
            / "UrbanEV"
            / "target_exogenous"
            / "volume"
            / "horizon_3"
            / "fold_1"
            / f"seed_{seed}"
            / run_id
        )
        run_dir.mkdir(parents=True)
        is_t2 = implementation_variant == summary.T2_IMPLEMENTATION_VARIANT
        is_t2g = implementation_variant == summary.T2G_IMPLEMENTATION_VARIANT
        is_t3 = implementation_variant == summary.T3_IMPLEMENTATION_VARIANT
        dataset = {
            "id": "UrbanEV",
            "sha256": "data-hash",
            "task_mode": "target_exogenous",
            "target": "volume",
            "feature_type": "MS",
            "feature_names": ["volume", "speed", "temperature"],
            "target_feature_name": "volume",
            "target_idx": 0,
            "aux_idx": [1, 2],
            "aux_feature_names": ["speed", "temperature"],
            "schema_fingerprint": "fixture-schema",
            "fold": 1,
            "label_horizon": 3,
            "model_pred_len": 1,
            "artifact_horizon": 3,
        }
        target_schema = {
            "contract_version": summary.TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION,
            "feature_type": "MS",
            "feature_names": ["volume", "speed", "temperature"],
            "target_feature_name": "volume",
            "target_idx": 0,
            "target_indices": [0],
            "aux_idx": [1, 2],
            "aux_feature_names": ["speed", "temperature"],
            "schema_fingerprint": "fixture-schema",
        }
        if not legacy_schema:
            dataset.update({
                "target_indices": [0],
                "target_exogenous_schema_contract_version": (
                    summary.TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION
                ),
            })
        teb = {
            "context_dim": 32,
            "heads": 4,
            "dropout": 0.1,
            "gamma_init": 1e-3,
            "query_policy": "linear_full_sequence_then_feature_layernorm",
            "projector_policy": "shared_linear_full_sequence_then_feature_layernorm",
            "variable_identity_embedding": False,
            "output_dropout": False,
            "query_residual": False,
            "post_attention_ffn": False,
        }
        if is_t2 or is_t2g or is_t3:
            teb.update({
                "architecture": (
                    "selective_patch_v1"
                    if is_t3
                    else (
                        "global_mediated_patch_v1"
                        if is_t2g
                        else "patch_conditioned_v1"
                    )
                ),
                "patch_size": patch_size,
                "patch_padding": "right_zero_crop",
                "patch_position": "fixed_sinusoidal",
                "target_selection_policy": "full_denorm_then_task_select",
            })
            if is_t2g:
                teb.update({
                    "global_residual": "query_plus_attention_post_layernorm",
                    "patch_attention_residual": "none",
                    "global_gate": "scalar_per_patch",
                    "global_gate_input": "patch_attention_and_global_bridge",
                    "global_gate_init": "identity",
                    "beta_global_init": 1e-3,
                })
            elif is_t3:
                teb.update({
                    "patch_confidence_gate": "scalar_per_patch_post_projection",
                    "patch_gate_input": "query_and_attention_response",
                    "patch_gate_activation": "two_sigmoid",
                    "patch_gate_init": "explicit_zero_identity",
                    "global_prediction_role": "state_only_forecast_disconnected",
                })
        scientific = {
            "implementation_variant": implementation_variant,
            "source_sha256": "source-hash",
            "dataset": dataset,
            "model": {
                "model_class": "AMDEnhanced",
                "seq_len": 12,
                "pred_len": 1,
                "model_pred_len": 1,
                "use_pmcr": False,
                "use_teb": True,
                "teb": teb,
            },
            "optimization": {
                "optimizer": "Adam",
                "weight_decay": 1e-7,
                "batch_size": 4,
            },
            "execution": {
                "seed": seed,
                "device": "cpu",
                "metric_space": summary.METRIC_SPACE,
            },
            "experiment": {
                "task_mode": "target_exogenous",
                "target": "volume",
                "fold": 1,
                "label_horizon": 3,
                "model_pred_len": 1,
                "artifact_horizon": 3,
                "ablation_id": (
                    "M4_T3"
                    if is_t3
                    else (
                        "M4_T2G" if is_t2g else ("M4_T2" if is_t2 else "U2")
                    )
                ),
            },
        }
        config_hash = summary._stable_hash(scientific)
        common = {
            "schema_version": summary.SCHEMA_VERSION,
            "artifact_schema_version": summary.ENHANCED_ARTIFACT_SCHEMA_VERSION,
            "implementation_variant": implementation_variant,
        }
        config = {
            **common,
            "config_hash": config_hash,
            "scientific_config": scientific,
            "run": {
                "run_dir": str(run_dir.resolve()),
                "train_epochs": 1,
            },
        }
        manifest = {
            **common,
            "run_id": run_id,
            "status": status,
            "config_hash": config_hash,
            "data_sha256": "data-hash",
            "artifact_dir": str(run_dir.resolve()),
            "completed_epoch": 1,
            "best_epoch": 1,
            "best_validation_mse": 0.3,
            "test_mse": 0.4,
            "test_mae": 0.25,
            "task_mode": "target_exogenous",
            "target": "volume",
            "artifact_horizon": 3,
            "fold": 1,
            "seed": seed,
        }
        if not legacy_schema:
            manifest["target_exogenous_schema"] = target_schema
        if is_t2 or is_t2g or is_t3:
            manifest["candidate_contract"] = {
                "ablation_id": (
                    "M4_T3"
                    if is_t3
                    else ("M4_T2G" if is_t2g else "M4_T2")
                ),
                "teb_architecture": (
                    "selective_patch_v1"
                    if is_t3
                    else (
                        "global_mediated_patch_v1"
                        if is_t2g
                        else "patch_conditioned_v1"
                    )
                ),
                "teb_patch_size": patch_size,
                "teb_patch_padding": "right_zero_crop",
                "teb_patch_position": "fixed_sinusoidal",
                "teb_context_dim": 32,
                "teb_heads": 4,
                "teb_dropout": 0.1,
                "teb_gamma_init": 1e-3,
                "seq_len": 12,
                "task_mode": "target_exogenous",
                "target_idx": 0,
                "aux_idx": [1, 2],
                "schema_fingerprint": "fixture-schema",
                "target_selection_policy": "full_denorm_then_task_select",
            }
            if is_t2g:
                manifest["candidate_contract"].update({
                    "teb_global_residual": "query_plus_attention_post_layernorm",
                    "teb_patch_attention_residual": "none",
                    "teb_global_gate": "scalar_per_patch",
                    "teb_global_gate_input": "patch_attention_and_global_bridge",
                    "teb_global_gate_init": "identity",
                    "teb_beta_global_init": 1e-3,
                })
            elif is_t3:
                manifest["candidate_contract"].update({
                    "teb_patch_confidence_gate": "scalar_per_patch_post_projection",
                    "teb_patch_gate_input": "query_and_attention_response",
                    "teb_patch_gate_activation": "two_sigmoid",
                    "teb_patch_gate_init": "explicit_zero_identity",
                    "teb_global_prediction_role": "state_only_forecast_disconnected",
                })
        metrics = {
            **common,
            "run_id": run_id,
            "status": status,
            "dataset_id": "UrbanEV",
            "task_mode": "target_exogenous",
            "target": "volume",
            "seq_len": 12,
            "pred_len": 1,
            "label_horizon": 3,
            "model_pred_len": 1,
            "artifact_horizon": 3,
            "fold": 1,
            "seed": seed,
            "best_epoch": 1,
            "best_validation": {"mse": 0.3, "mae": 0.2},
            "test": {"mse": 0.4, "mae": 0.25},
            "parameter_count": 10,
            "train_epochs": 1,
            "duration_seconds": 1.0,
            "metric_space": summary.METRIC_SPACE,
            "config_hash": config_hash,
            "data_sha256": "data-hash",
            "completed_at": "2026-08-24T00:00:00Z",
            "artifact_dir": str(run_dir.resolve()),
        }
        documents = {
            "config.resolved.json": config,
            "manifest.json": manifest,
            "metrics.json": metrics,
            "sys.argv.json": {"argv": ["main.py"]},
            "source_fingerprint.json": {"sha256": "source-hash"},
            "data_fingerprint.json": {"sha256": "data-hash"},
        }
        for name, value in documents.items():
            (run_dir / name).write_text(
                json.dumps(value, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        for name in ("best.pt", "last.pt"):
            (run_dir / name).write_bytes(b"checkpoint")
        (run_dir / "history.jsonl").write_text("{}\n", encoding="utf-8")
        (run_dir / "command.txt").write_text("python main.py\n", encoding="utf-8")
        (run_dir / "stdout.log").write_text("completed\n", encoding="utf-8")
        (run_dir / "stderr.log").write_text("\n", encoding="utf-8")
        (run_dir / "train.log").write_text("completed\n", encoding="utf-8")
        cls._write_enhanced_checksums(run_dir)
        return run_dir

    def test_only_completed_runs_are_summarized_with_sample_std(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            output = root / "summaries"
            self._make_run(artifacts, "run-a", 2024, test_mse=0.4)
            self._make_run(artifacts, "run-b", 2025, test_mse=0.6)
            self._make_run(artifacts, "failed", 2026, status="failed", test_mse=99)
            run_path, aggregate_path, run_count, group_count = summary.write_summaries(
                artifacts, output
            )
            self.assertEqual((run_count, group_count), (2, 1))
            with run_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["seed"] for row in rows], ["2024", "2025"])
            with aggregate_path.open(encoding="utf-8", newline="") as handle:
                aggregate = next(csv.DictReader(handle))
            self.assertEqual(aggregate["seed_count"], "2")
            self.assertEqual(aggregate["seeds"], "2024;2025")
            self.assertAlmostEqual(float(aggregate["test_mse_mean"]), 0.5)
            self.assertAlmostEqual(
                float(aggregate["test_mse_sample_std"]), 0.14142135623730948
            )

    def test_duplicate_completed_seed_is_rejected_without_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            self._make_run(artifacts, "run-a", 2024)
            self._make_run(artifacts, "run-b", 2024)
            rows = summary.load_completed_runs(artifacts)
            with self.assertRaisesRegex(ValueError, "no run was selected automatically"):
                summary.aggregate_runs(rows)

    def test_inconsistent_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            run_dir = self._make_run(artifacts, "run-a", 2024)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["config_hash"] = "different"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest config hash mismatch"):
                summary.load_completed_runs(artifacts)

    def test_tampered_scientific_config_is_rejected_even_if_hash_field_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            run_dir = self._make_run(artifacts, "run-a", 2024)
            config_path = run_dir / "config.resolved.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["scientific_config"]["model"]["seq_len"] = 999
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scientific config hash mismatch"):
                summary.load_completed_runs(artifacts)

    def test_self_consistent_wrong_variant_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            run_dir = self._make_run(artifacts, "run-a", 2024)
            config_path = run_dir / "config.resolved.json"
            metrics_path = run_dir / "metrics.json"
            manifest_path = run_dir / "manifest.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            config["scientific_config"]["optimization"]["weight_decay"] = 1e-9
            wrong_hash = summary._stable_hash(config["scientific_config"])
            config["config_hash"] = wrong_hash
            metrics["config_hash"] = wrong_hash
            manifest["config_hash"] = wrong_hash
            config_path.write_text(json.dumps(config), encoding="utf-8")
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "optimization contract mismatch for weight_decay"
            ):
                summary.load_completed_runs(artifacts)

    def test_missing_artifact_or_manifest_best_epoch_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            run_dir = self._make_run(artifacts, "run-a", 2024)
            (run_dir / "best.pt").unlink()
            with self.assertRaisesRegex(ValueError, "missing required artifacts"):
                summary.load_completed_runs(artifacts)

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            run_dir = self._make_run(artifacts, "run-a", 2024)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["best_epoch"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest best epoch mismatch"):
                summary.load_completed_runs(artifacts)
    def test_enhanced_schema_v2_validates_checksums_and_ignores_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            run_dir = self._make_enhanced_run(artifacts, "run-a")
            staging = run_dir.parent / ".interrupted.staging"
            staging.mkdir()
            (staging / "manifest.json").write_text(
                json.dumps({"status": "running"}),
                encoding="utf-8",
            )
            rows = summary.load_completed_runs(
                artifacts,
                implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "run-a")
            self.assertEqual(rows[0]["label_horizon"], 3)
            self.assertEqual(rows[0]["fold"], "1")
            self.assertEqual(
                rows[0]["target_exogenous_schema_contract"],
                summary.TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION,
            )

            output = Path(directory) / "summaries"
            run_path, aggregate_path, run_count, group_count = summary.write_summaries(
                artifacts,
                output,
                implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
            )
            self.assertEqual((run_count, group_count), (1, 1))
            self.assertEqual(
                run_path.name,
                f"{summary.ENHANCED_IMPLEMENTATION_VARIANT}.csv",
            )
            self.assertTrue(aggregate_path.is_file())

    def test_target_schema_legacy_is_accepted_and_explicitly_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            self._make_enhanced_run(
                artifacts, "legacy", legacy_schema=True
            )
            rows = summary.load_completed_runs(
                artifacts,
                implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["target_exogenous_schema_contract"], "legacy"
            )

    def test_target_schema_missing_and_misplaced_blocks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "missing"
            run_dir = self._make_enhanced_run(artifacts, "v1")
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("target_exogenous_schema")
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            self._write_enhanced_checksums(run_dir)
            with self.assertRaisesRegex(ValueError, "manifest block is missing"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
                )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "misplaced"
            run_dir = self._make_enhanced_run(
                artifacts, "legacy", legacy_schema=True
            )
            manifest_path = run_dir / "manifest.json"
            config = json.loads(
                (run_dir / "config.resolved.json").read_text(encoding="utf-8")
            )
            dataset = config["scientific_config"]["dataset"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["target_exogenous_schema"] = {
                "contract_version": (
                    summary.TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION
                ),
                "feature_type": dataset["feature_type"],
                "feature_names": dataset["feature_names"],
                "target_feature_name": dataset["target_feature_name"],
                "target_idx": dataset["target_idx"],
                "target_indices": [dataset["target_idx"]],
                "aux_idx": dataset["aux_idx"],
                "aux_feature_names": dataset["aux_feature_names"],
                "schema_fingerprint": dataset["schema_fingerprint"],
            }
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            self._write_enhanced_checksums(run_dir)
            with self.assertRaisesRegex(ValueError, "config has no version"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
                )

    def test_target_schema_tamper_and_parallel_conflict_are_rejected(self):
        mutations = {
            "aux_order": lambda block: block.update({
                "aux_idx": list(reversed(block["aux_idx"])),
                "aux_feature_names": list(reversed(block["aux_feature_names"])),
            }),
            "target_indices": lambda block: block.update({
                "target_indices": [2],
            }),
            "aux_names_order": lambda block: block.update({
                "aux_feature_names": list(reversed(block["aux_feature_names"])),
            }),
            "feature_order": lambda block: block.update({
                "feature_names": [
                    block["feature_names"][0],
                    block["feature_names"][2],
                    block["feature_names"][1],
                ],
            }),
            "target_idx": lambda block: block.update({
                "target_idx": 1,
                "target_indices": [1],
                "target_feature_name": block["feature_names"][1],
            }),
            "fingerprint": lambda block: block.update({
                "schema_fingerprint": "tampered",
            }),
        }
        for name, mutate in mutations.items():
            with self.subTest(tamper=name), tempfile.TemporaryDirectory() as directory:
                artifacts = Path(directory) / "artifacts"
                run_dir = self._make_enhanced_run(artifacts, "v1")
                manifest_path = run_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest["target_exogenous_schema"])
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self._write_enhanced_checksums(run_dir)
                with self.assertRaisesRegex(ValueError, "schema mismatch"):
                    summary.load_completed_runs(
                        artifacts,
                        implementation_variant=(
                            summary.ENHANCED_IMPLEMENTATION_VARIANT
                        ),
                    )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "version"
            run_dir = self._make_enhanced_run(artifacts, "v1")
            config_path = run_dir / "config.resolved.json"
            manifest_path = run_dir / "manifest.json"
            metrics_path = run_dir / "metrics.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config["scientific_config"]["dataset"][
                "target_exogenous_schema_contract_version"
            ] = "unsupported-v2"
            config_hash = summary._stable_hash(config["scientific_config"])
            config["config_hash"] = config_hash
            manifest["config_hash"] = config_hash
            metrics["config_hash"] = config_hash
            for path, value in (
                (config_path, config), (manifest_path, manifest),
                (metrics_path, metrics),
            ):
                path.write_text(
                    json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
                )
            self._write_enhanced_checksums(run_dir)
            with self.assertRaisesRegex(ValueError, "unsupported.*schema version"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
                )

        with self.assertRaisesRegex(ValueError, "must not carry"):
            summary._validate_target_exogenous_schema(
                {"dataset": {
                    "task_mode": "parallel_multivariate",
                    "target_exogenous_schema_contract_version": (
                        summary.TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION
                    ),
                }},
                {"target_exogenous_schema": {}},
                Path("/tmp/parallel-fixture"),
            )

    def test_enhanced_tamper_missing_checksum_and_noncompleted_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "tamper"
            run_dir = self._make_enhanced_run(artifacts, "run-a")
            with (run_dir / "metrics.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
                )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "missing"
            run_dir = self._make_enhanced_run(artifacts, "run-a")
            (run_dir / "checksums.sha256").unlink()
            with self.assertRaisesRegex(ValueError, "no checksums"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
                )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "running"
            self._make_enhanced_run(artifacts, "run-a", status="running")
            with self.assertRaisesRegex(ValueError, "not completed"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
                )

    def test_enhanced_identity_mismatch_and_duplicate_success_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "identity"
            run_dir = self._make_enhanced_run(artifacts, "run-a")
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["target"] = "wrong"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._write_enhanced_checksums(run_dir)
            with self.assertRaisesRegex(ValueError, "manifest/path identity mismatch"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
                )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "duplicate"
            self._make_enhanced_run(artifacts, "run-a", seed=2024)
            self._make_enhanced_run(artifacts, "run-b", seed=2024)
            with self.assertRaisesRegex(ValueError, "no run was selected automatically"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
                )

    def test_t2_candidate_is_summarized_separately_from_global_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            self._make_enhanced_run(artifacts, "global")
            t2_dir = self._make_enhanced_run(
                artifacts,
                "t2",
                implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
            )
            rows = summary.load_completed_runs(
                artifacts,
                implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["implementation_variant"],
                summary.T2_IMPLEMENTATION_VARIANT,
            )
            self.assertEqual(rows[0]["run_id"], "t2")
            config = json.loads(
                (t2_dir / "config.resolved.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                config["scientific_config"]["model"]["teb"]["patch_size"],
                3,
            )

    def test_t2g_candidate_is_separate_and_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            self._make_enhanced_run(artifacts, "global")
            self._make_enhanced_run(
                artifacts,
                "t2",
                implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
            )
            t2g_dir = self._make_enhanced_run(
                artifacts,
                "t2g",
                implementation_variant=summary.T2G_IMPLEMENTATION_VARIANT,
            )
            rows = summary.load_completed_runs(
                artifacts,
                implementation_variant=summary.T2G_IMPLEMENTATION_VARIANT,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "t2g")
            config = json.loads(
                (t2g_dir / "config.resolved.json").read_text(encoding="utf-8")
            )
            teb = config["scientific_config"]["model"]["teb"]
            self.assertEqual(teb["architecture"], "global_mediated_patch_v1")
            self.assertEqual(teb["global_residual"], "query_plus_attention_post_layernorm")
            self.assertEqual(teb["patch_attention_residual"], "none")
            self.assertEqual(teb["global_gate"], "scalar_per_patch")
            self.assertEqual(teb["beta_global_init"], 1e-3)

    def test_t2g_contract_tamper_and_duplicate_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "tamper"
            run_dir = self._make_enhanced_run(
                artifacts,
                "t2g",
                implementation_variant=summary.T2G_IMPLEMENTATION_VARIANT,
            )
            config_path = run_dir / "config.resolved.json"
            manifest_path = run_dir / "manifest.json"
            metrics_path = run_dir / "metrics.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config["scientific_config"]["model"]["teb"]["global_gate"] = "vector"
            config_hash = summary._stable_hash(config["scientific_config"])
            config["config_hash"] = config_hash
            manifest["config_hash"] = config_hash
            metrics["config_hash"] = config_hash
            for path, value in (
                (config_path, config),
                (manifest_path, manifest),
                (metrics_path, metrics),
            ):
                path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            self._write_enhanced_checksums(run_dir)
            with self.assertRaisesRegex(ValueError, "unsupported T2G patch config"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.T2G_IMPLEMENTATION_VARIANT,
                )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "duplicates"
            for run_id in ("t2g-a", "t2g-b"):
                self._make_enhanced_run(
                    artifacts,
                    run_id,
                    implementation_variant=summary.T2G_IMPLEMENTATION_VARIANT,
                )
            with self.assertRaisesRegex(ValueError, "no run was selected automatically"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.T2G_IMPLEMENTATION_VARIANT,
                )

    def test_t3_candidate_is_exact_and_rejects_tamper_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "exact"
            self._make_enhanced_run(artifacts, "global")
            self._make_enhanced_run(
                artifacts,
                "t2",
                implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
            )
            self._make_enhanced_run(
                artifacts,
                "t2g",
                implementation_variant=summary.T2G_IMPLEMENTATION_VARIANT,
            )
            t3_dir = self._make_enhanced_run(
                artifacts,
                "t3",
                implementation_variant=summary.T3_IMPLEMENTATION_VARIANT,
            )
            rows = summary.load_completed_runs(
                artifacts,
                implementation_variant=summary.T3_IMPLEMENTATION_VARIANT,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "t3")
            config = json.loads(
                (t3_dir / "config.resolved.json").read_text(encoding="utf-8")
            )
            teb = config["scientific_config"]["model"]["teb"]
            self.assertEqual(teb["architecture"], "selective_patch_v1")
            self.assertEqual(
                teb["patch_confidence_gate"],
                "scalar_per_patch_post_projection",
            )
            self.assertEqual(
                teb["patch_gate_input"], "query_and_attention_response"
            )
            self.assertEqual(teb["patch_gate_activation"], "two_sigmoid")
            self.assertEqual(teb["patch_gate_init"], "explicit_zero_identity")
            self.assertEqual(
                teb["global_prediction_role"],
                "state_only_forecast_disconnected",
            )
            self.assertFalse({
                "global_residual", "patch_attention_residual", "global_gate",
                "global_gate_input", "global_gate_init", "beta_global_init",
            } & set(teb))

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "tamper"
            run_dir = self._make_enhanced_run(
                artifacts,
                "t3",
                implementation_variant=summary.T3_IMPLEMENTATION_VARIANT,
            )
            config_path = run_dir / "config.resolved.json"
            manifest_path = run_dir / "manifest.json"
            metrics_path = run_dir / "metrics.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config["scientific_config"]["model"]["teb"][
                "patch_gate_activation"
            ] = "relu"
            config_hash = summary._stable_hash(config["scientific_config"])
            config["config_hash"] = config_hash
            manifest["config_hash"] = config_hash
            metrics["config_hash"] = config_hash
            for path, value in (
                (config_path, config),
                (manifest_path, manifest),
                (metrics_path, metrics),
            ):
                path.write_text(
                    json.dumps(value, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            self._write_enhanced_checksums(run_dir)
            with self.assertRaisesRegex(ValueError, "unsupported T3 patch config"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.T3_IMPLEMENTATION_VARIANT,
                )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "duplicates"
            for run_id in ("t3-a", "t3-b"):
                self._make_enhanced_run(
                    artifacts,
                    run_id,
                    implementation_variant=summary.T3_IMPLEMENTATION_VARIANT,
                )
            with self.assertRaisesRegex(
                ValueError, "no run was selected automatically"
            ):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.T3_IMPLEMENTATION_VARIANT,
                )

    def test_t2_unsupported_patch_contract_and_manifest_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "unsupported"
            run_dir = self._make_enhanced_run(
                artifacts,
                "t2",
                implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
            )
            config_path = run_dir / "config.resolved.json"
            manifest_path = run_dir / "manifest.json"
            metrics_path = run_dir / "metrics.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config["scientific_config"]["model"]["teb"]["patch_position"] = "learnable"
            config_hash = summary._stable_hash(config["scientific_config"])
            config["config_hash"] = config_hash
            manifest["config_hash"] = config_hash
            metrics["config_hash"] = config_hash
            config_path.write_text(
                json.dumps(config, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(metrics, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._write_enhanced_checksums(run_dir)
            with self.assertRaisesRegex(ValueError, "unsupported T2 patch config"):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
                )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "manifest"
            run_dir = self._make_enhanced_run(
                artifacts,
                "t2",
                implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
            )
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["candidate_contract"]["teb_patch_size"] = 4
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._write_enhanced_checksums(run_dir)
            with self.assertRaisesRegex(
                ValueError,
                "manifest candidate contract mismatch",
            ):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
                )

    def test_duplicate_t2_scientific_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "duplicates"
            for run_id in ("t2-a", "t2-b"):
                self._make_enhanced_run(
                    artifacts,
                    run_id,
                    implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
                )
            with self.assertRaisesRegex(
                ValueError,
                "no run was selected automatically",
            ):
                summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
                )

    def test_unsupported_variant_is_an_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported implementation variant"):
                summary.load_completed_runs(
                    Path(directory),
                    implementation_variant="unknown-variant",
                )


class WarmStartSummaryTests(unittest.TestCase):
    """Permanent sealed-lineage and warm epoch-zero summarizer tests."""

    FEATURE_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    AUX_IDX = [0, 1, 2, 3, 4, 5]
    T2_MISSING = [
        "teb.gamma_teb",
        "teb.patch_query_projection.weight",
        "teb.patch_query_projection.bias",
        "teb.patch_query_norm.weight",
        "teb.patch_query_norm.bias",
        "teb.global_query_projection.weight",
        "teb.global_query_projection.bias",
        "teb.global_query_norm.weight",
        "teb.global_query_norm.bias",
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

    @staticmethod
    def _reseal(run_dir):
        SummaryTests._write_enhanced_checksums(run_dir)

    @classmethod
    def _make_warm_run(
        cls,
        root,
        run_id,
        *,
        protocol=summary.T2_ADAPTER_TRAINING_PROTOCOL,
        completed_epochs=2,
        best_epoch=0,
        source_path="/machine-a/sealed-u1",
    ):
        adapter = protocol == summary.T2_ADAPTER_TRAINING_PROTOCOL
        variant = (
            summary.T2_IMPLEMENTATION_VARIANT
            if adapter else summary.ENHANCED_IMPLEMENTATION_VARIANT
        )
        horizon = 96
        source_identity = summary.M4_U1_SOURCE_IDENTITIES[horizon]
        run_dir = (
            Path(root) / variant / "ETTm1" / "target_exogenous" / "OT"
            / "horizon_96" / "fold_official" / "seed_2024" / run_id
        )
        run_dir.mkdir(parents=True)
        target_schema = {
            "contract_version": summary.TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION,
            "feature_type": "MS",
            "feature_names": cls.FEATURE_NAMES,
            "target_feature_name": "OT",
            "target_idx": 6,
            "target_indices": [6],
            "aux_idx": cls.AUX_IDX,
            "aux_feature_names": cls.FEATURE_NAMES[:6],
            "schema_fingerprint": summary.M4_U1_SCHEMA_FINGERPRINT,
        }
        lineage = {
            "source_artifact_path": source_path,
            "source_run_id": source_identity["run_id"],
            "source_implementation_variant": summary.ENHANCED_IMPLEMENTATION_VARIANT,
            "source_ablation_id": "U1",
            "source_checkpoint_role": "best",
            "source_checkpoint_sha256": source_identity["checkpoint_sha256"],
            "source_config_hash": source_identity["config_hash"],
            "source_comparison_config_hash": source_identity[
                "comparison_config_hash"
            ],
            "source_commit": summary.M4_U1_SOURCE_COMMIT,
            "source_executable_fingerprint": summary.M4_U1_SOURCE_FINGERPRINT,
            "source_data_fingerprint": summary.M4_U1_DATA_FINGERPRINT,
            "source_best_epoch": source_identity["best_epoch"],
            "source_task_mode": "target_exogenous",
            "source_feature_type": "MS",
            "source_target": "OT",
            "source_target_idx": 6,
            "source_target_indices": [6],
            "source_aux_idx": cls.AUX_IDX,
            "source_target_exogenous_schema_version": (
                summary.TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION
            ),
            "source_schema_fingerprint": summary.M4_U1_SCHEMA_FINGERPRINT,
        }
        stable_lineage = deepcopy(lineage)
        stable_lineage.pop("source_artifact_path")
        current_fingerprint = "c" * 64
        missing = cls.T2_MISSING if adapter else []
        proof = {
            "contract_version": summary.SOURCE_COMPATIBILITY_PROOF_VERSION,
            "source_executable_fingerprint": summary.M4_U1_SOURCE_FINGERPRINT,
            "current_executable_fingerprint": current_fingerprint,
            "global_fingerprint_equal": False,
            "critical_files": [
                {"path": path, "source_sha256": "d" * 64,
                 "current_sha256": "d" * 64}
                for path in summary.SOURCE_COMPATIBILITY_CRITICAL_FILES
            ],
            "source_state_key_count": 60,
            "target_state_key_count": 79 if adapter else 60,
            "mapped_key_count": 60,
            "allowed_missing_keys": missing,
            "unexpected_keys": [],
            "shape_mismatches": [],
            "dtype_mismatches": [],
        }
        training_protocol = summary._warm_start_protocol_expected(protocol)
        teb = {
            "context_dim": 32,
            "heads": 4,
            "dropout": 0.1,
            "gamma_init": 1e-3,
            "query_policy": "linear_full_sequence_then_feature_layernorm",
            "projector_policy": "shared_linear_full_sequence_then_feature_layernorm",
            "variable_identity_embedding": False,
            "output_dropout": False,
            "query_residual": False,
            "post_attention_ffn": False,
        }
        if adapter:
            teb.update({
                "architecture": "patch_conditioned_v1",
                "patch_size": 32,
                "patch_padding": "right_zero_crop",
                "patch_position": "fixed_sinusoidal",
                "target_selection_policy": "full_denorm_then_task_select",
            })
        dataset = {
            "id": "ETTm1",
            "sha256": summary.M4_U1_DATA_FINGERPRINT,
            "task_mode": "target_exogenous",
            "feature_type": "MS",
            "target": "OT",
            "feature_names": cls.FEATURE_NAMES,
            "target_feature_name": "OT",
            "target_idx": 6,
            "target_indices": [6],
            "aux_idx": cls.AUX_IDX,
            "aux_feature_names": cls.FEATURE_NAMES[:6],
            "schema_fingerprint": summary.M4_U1_SCHEMA_FINGERPRINT,
            "fold": "official",
            "label_horizon": horizon,
            "model_pred_len": horizon,
            "artifact_horizon": horizon,
            "target_exogenous_schema_contract_version": (
                summary.TARGET_EXOGENOUS_SCHEMA_CONTRACT_VERSION
            ),
        }
        scientific = {
            "implementation_variant": variant,
            "source_sha256": current_fingerprint,
            "dataset": dataset,
            "model": {
                "model_class": "AMDEnhanced",
                "seq_len": 512,
                "pred_len": horizon,
                "model_pred_len": horizon,
                "use_pmcr": False,
                "use_teb": adapter,
                "teb": teb,
            },
            "optimization": {
                "optimizer": "Adam",
                "learning_rate": 3e-5,
                "weight_decay": 0.0 if adapter else 1e-7,
                "batch_size": 128,
            },
            "execution": {
                "seed": 2024,
                "device": "cpu",
                "metric_space": summary.METRIC_SPACE,
            },
            "experiment": {
                "task_mode": "target_exogenous",
                "target": "OT",
                "fold": "official",
                "label_horizon": horizon,
                "model_pred_len": horizon,
                "artifact_horizon": horizon,
                "ablation_id": (
                    summary.T2_ADAPTER_ABLATION_ID
                    if adapter else summary.U1_CONTINUATION_ABLATION_ID
                ),
            },
            "training_protocol": training_protocol,
            "source_lineage": stable_lineage,
            "source_compatibility_proof": proof,
        }
        config_hash = summary._stable_hash(scientific)
        common = {
            "schema_version": summary.SCHEMA_VERSION,
            "artifact_schema_version": summary.ENHANCED_ARTIFACT_SCHEMA_VERSION,
            "implementation_variant": variant,
        }
        initialization = {"mse": 0.3, "mae": 0.2, "num_elements": 1,
                          "num_batches": 1}
        best_role = "epoch_zero_initialization" if best_epoch == 0 else "trained_epoch"
        config = {
            **common,
            "config_hash": config_hash,
            "scientific_config": scientific,
            "run": {
                "run_dir": str(run_dir.resolve()),
                "train_epochs": 10,
                "initialization_validation": initialization,
                "epoch_zero_in_best_selection": True,
                "epoch_zero_checkpoint_role": "source_equivalent_initialization",
            },
            "source": {"sha256": current_fingerprint},
            "training_protocol": training_protocol,
            "source_lineage": lineage,
            "source_compatibility_proof": proof,
        }
        manifest = {
            **common,
            "run_id": run_id,
            "status": "completed",
            "config_hash": config_hash,
            "data_sha256": summary.M4_U1_DATA_FINGERPRINT,
            "artifact_dir": str(run_dir.resolve()),
            "completed_epoch": completed_epochs,
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "best_validation_mse": 0.3 if best_epoch == 0 else 0.25,
            "test_mse": 0.4,
            "test_mae": 0.25,
            "task_mode": "target_exogenous",
            "target": "OT",
            "artifact_horizon": horizon,
            "fold": "official",
            "seed": 2024,
            "target_exogenous_schema": target_schema,
            "training_protocol": training_protocol,
            "source_lineage": lineage,
            "source_compatibility_proof": proof,
            "initialization_validation": initialization,
            "epoch_zero_in_best_selection": True,
            "best_checkpoint_role": best_role,
        }
        if adapter:
            manifest["candidate_contract"] = {
                "ablation_id": summary.T2_ADAPTER_ABLATION_ID,
                "teb_architecture": "patch_conditioned_v1",
                "teb_patch_size": 32,
                "teb_patch_padding": "right_zero_crop",
                "teb_patch_position": "fixed_sinusoidal",
                "teb_context_dim": 32,
                "teb_heads": 4,
                "teb_dropout": 0.1,
                "teb_gamma_init": 1e-3,
                "seq_len": 512,
                "task_mode": "target_exogenous",
                "target_idx": 6,
                "aux_idx": cls.AUX_IDX,
                "schema_fingerprint": summary.M4_U1_SCHEMA_FINGERPRINT,
                "target_selection_policy": "full_denorm_then_task_select",
            }
        metrics = {
            **common,
            "run_id": run_id,
            "status": "completed",
            "dataset_id": "ETTm1",
            "task_mode": "target_exogenous",
            "target": "OT",
            "seq_len": 512,
            "pred_len": horizon,
            "label_horizon": horizon,
            "model_pred_len": horizon,
            "artifact_horizon": horizon,
            "fold": "official",
            "seed": 2024,
            "best_epoch": best_epoch,
            "best_validation": {"mse": 0.3 if best_epoch == 0 else 0.25,
                                "mae": 0.2},
            "test": {"mse": 0.4, "mae": 0.25},
            "parameter_count": 1,
            "train_epochs": 10,
            "completed_epochs": completed_epochs,
            "duration_seconds": 1.0,
            "metric_space": summary.METRIC_SPACE,
            "config_hash": config_hash,
            "data_sha256": summary.M4_U1_DATA_FINGERPRINT,
            "completed_at": "2026-09-01T00:00:00Z",
            "artifact_dir": str(run_dir.resolve()),
            "training_protocol_id": protocol,
            "warm_start_contract_version": summary.WARM_START_CONTRACT_VERSION,
            "training_protocol": training_protocol,
            "source_lineage": lineage,
            "source_compatibility_proof": proof,
            "initialization_validation": initialization,
            "epoch_zero_in_best_selection": True,
            "best_checkpoint_role": best_role,
        }
        best_checkpoint = {
            "training_protocol": training_protocol,
            "source_lineage": lineage,
            "source_compatibility_proof": proof,
            "initialization_validation": initialization,
            "epoch_zero_in_best_selection": True,
            "best_epoch": best_epoch,
            "best_checkpoint_role": best_role,
            "checkpoint_role": best_role,
        }
        last_checkpoint = {
            **best_checkpoint,
            "completed_epoch": completed_epochs,
            "completed_epochs": completed_epochs,
            "checkpoint_role": (
                "epoch_zero_initialization"
                if completed_epochs == 0 else "last_trained_epoch"
            ),
        }
        for name, value in {
            "config.resolved.json": config,
            "manifest.json": manifest,
            "metrics.json": metrics,
            "sys.argv.json": {"argv": ["main.py"]},
            "source_fingerprint.json": {"sha256": current_fingerprint},
            "data_fingerprint.json": {"sha256": summary.M4_U1_DATA_FINGERPRINT},
        }.items():
            (run_dir / name).write_text(
                json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
            )
        torch.save(best_checkpoint, run_dir / "best.pt")
        torch.save(last_checkpoint, run_dir / "last.pt")
        (run_dir / "history.jsonl").write_text(
            "".join(json.dumps({"epoch": epoch}) + "\n"
                    for epoch in range(1, completed_epochs + 1)),
            encoding="utf-8",
        )
        (run_dir / "command.txt").write_text("python main.py\n", encoding="utf-8")
        (run_dir / "stdout.log").write_text("completed\n", encoding="utf-8")
        (run_dir / "stderr.log").write_text("\n", encoding="utf-8")
        (run_dir / "train.log").write_text("completed\n", encoding="utf-8")
        cls._reseal(run_dir)
        return run_dir

    def test_valid_adapter_epoch_zero_and_trained_best_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, best_epoch in (("epoch-zero", 0), ("trained", 1)):
                artifacts = root / run_id
                self._make_warm_run(artifacts, run_id, best_epoch=best_epoch)
                rows = summary.load_completed_runs(
                    artifacts,
                    implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["best_epoch"], best_epoch)
                self.assertEqual(rows[0]["training_protocol_id"],
                                 summary.T2_ADAPTER_TRAINING_PROTOCOL)

    def test_valid_continuation_is_separate_from_standard_u1(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "continuation"
            self._make_warm_run(
                artifacts, "continuation",
                protocol=summary.U1_CONTINUATION_TRAINING_PROTOCOL,
                best_epoch=1,
            )
            rows = summary.load_completed_runs(
                artifacts,
                implementation_variant=summary.ENHANCED_IMPLEMENTATION_VARIANT,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["training_protocol_id"],
                             summary.U1_CONTINUATION_TRAINING_PROTOCOL)

    def test_lineage_and_compatibility_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "lineage"
            run_dir = self._make_warm_run(artifacts, "tampered")
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["source_lineage"]["source_checkpoint_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest) + "\n")
            self._reseal(run_dir)
            with self.assertRaisesRegex(ValueError, "lineage mismatch"):
                summary.load_completed_runs(
                    artifacts, implementation_variant=summary.T2_IMPLEMENTATION_VARIANT
                )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "proof"
            run_dir = self._make_warm_run(artifacts, "tampered")
            metrics_path = run_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["source_compatibility_proof"]["critical_files"][0][
                "current_sha256"
            ] = "0" * 64
            metrics_path.write_text(json.dumps(metrics) + "\n")
            self._reseal(run_dir)
            with self.assertRaisesRegex(ValueError, "proof mismatch"):
                summary.load_completed_runs(
                    artifacts, implementation_variant=summary.T2_IMPLEMENTATION_VARIANT
                )

    def test_epoch_zero_role_and_fake_history_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "role"
            run_dir = self._make_warm_run(artifacts, "bad-role")
            best_path = run_dir / "best.pt"
            best = torch.load(best_path, map_location="cpu")
            best["checkpoint_role"] = "trained_epoch"
            torch.save(best, best_path)
            self._reseal(run_dir)
            with self.assertRaisesRegex(ValueError, "role/epoch mismatch"):
                summary.load_completed_runs(
                    artifacts, implementation_variant=summary.T2_IMPLEMENTATION_VARIANT
                )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "history"
            run_dir = self._make_warm_run(artifacts, "fake-zero")
            history = run_dir / "history.jsonl"
            history.write_text('{"epoch": 0}\n' + history.read_text())
            self._reseal(run_dir)
            with self.assertRaisesRegex(ValueError, "epoch-zero rows"):
                summary.load_completed_runs(
                    artifacts, implementation_variant=summary.T2_IMPLEMENTATION_VARIANT
                )

    def test_completed_epochs_and_absolute_source_path_do_not_evade_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "duplicates"
            self._make_warm_run(
                artifacts, "run-a", completed_epochs=2,
                source_path="/machine-a/source",
            )
            self._make_warm_run(
                artifacts, "run-b", completed_epochs=3,
                source_path="/machine-b/source",
            )
            with self.assertRaisesRegex(ValueError, "no run was selected automatically"):
                summary.load_completed_runs(
                    artifacts, implementation_variant=summary.T2_IMPLEMENTATION_VARIANT
                )

    def test_standard_best_epoch_zero_and_parallel_warm_spoof_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "standard"
            run_dir = SummaryTests._make_enhanced_run(
                artifacts, "standard-zero",
                implementation_variant=summary.T2_IMPLEMENTATION_VARIANT,
            )
            for filename in ("manifest.json", "metrics.json"):
                path = run_dir / filename
                value = json.loads(path.read_text())
                value["best_epoch"] = 0
                path.write_text(json.dumps(value) + "\n")
            self._reseal(run_dir)
            with self.assertRaisesRegex(ValueError, "epoch metadata is invalid"):
                summary.load_completed_runs(
                    artifacts, implementation_variant=summary.T2_IMPLEMENTATION_VARIANT
                )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "parallel"
            run_dir = self._make_warm_run(artifacts, "parallel-spoof")
            config_path = run_dir / "config.resolved.json"
            config = json.loads(config_path.read_text())
            config["scientific_config"]["dataset"]["task_mode"] = "parallel_multivariate"
            config["scientific_config"]["dataset"]["target"] = "all"
            config["scientific_config"]["experiment"]["task_mode"] = "parallel_multivariate"
            new_hash = summary._stable_hash(config["scientific_config"])
            config["config_hash"] = new_hash
            config_path.write_text(json.dumps(config) + "\n")
            for filename in ("manifest.json", "metrics.json"):
                path = run_dir / filename
                value = json.loads(path.read_text())
                value["config_hash"] = new_hash
                path.write_text(json.dumps(value) + "\n")
            self._reseal(run_dir)
            with self.assertRaises(ValueError):
                summary.load_completed_runs(
                    artifacts, implementation_variant=summary.T2_IMPLEMENTATION_VARIANT
                )



class SonnetSummaryContractTests(unittest.TestCase):
    FEATURES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")

    @classmethod
    def _args(cls, *, enabled=True):
        return runner.prepare_args(runner.parse_args([
            "--implementation_variant", runner.SONNET_IMPLEMENTATION_VARIANT,
            "--data", str(runner.ROOT / "data" / "ETTm1.csv"),
            "--dataset_id", "ETTm1",
            "--device", "cpu",
            "--progress", "false",
            "--feature_type", "MS",
            "--target", "OT",
            "--task_mode", "target_exogenous",
            "--target_idx", "6",
            "--aux_idx", "0", "1", "2", "3", "4", "5",
            "--feature_names", *cls.FEATURES,
            "--target_feature_name", "OT",
            "--aux_feature_names", *cls.FEATURES[:-1],
            "--schema_fingerprint", runner.stable_hash(cls.FEATURES),
            "--fold", "official",
            "--label_horizon", "96",
            "--seq_len", "512",
            "--pred_len", "96",
            "--batch_size", "32",
            "--train_epochs", "10",
            "--learning_rate", "0.00003",
            "--ablation_id", (
                runner.SONNET_CANDIDATE_ABLATION_ID
                if enabled else runner.SONNET_CONTROL_ABLATION_ID
            ),
            "--use_sonnet_mvca", str(enabled).lower(),
            "--use_pmcr", "false",
            "--use_teb", "false",
            "--development_protocol_id", runner.SONNET_DEVELOPMENT_PROTOCOL,
            "--training_protocol_id", runner.STANDARD_TRAINING_PROTOCOL,
        ]))

    @classmethod
    def _documents(cls, *, enabled=True, run_dir=Path("/tmp/sonnet-summary")):
        args = cls._args(enabled=enabled)
        preprocessing = {
            "feature_type": "MS",
            "columns": list(cls.FEATURES),
            "resolved_target": "OT",
            "target_indices": [6],
            "feature_schema_fingerprint": args.schema_fingerprint,
        }
        schema = runner._build_target_exogenous_schema_contract(
            args, preprocessing
        )
        training = runner._training_protocol_block(args)
        scientific = runner._scientific_config(
            args,
            "data-sha",
            "source-sha",
            preprocessing,
            torch.device("cpu"),
            {"python": "test"},
            schema,
            training,
        )
        config_hash = runner.stable_hash(scientific)
        config = runner._resolved_config(
            args,
            scientific,
            config_hash,
            run_dir,
            {"sha256": "source-sha"},
            {"python": "test"},
            training,
        )
        manifest = {
            "candidate_contract": runner._sonnet_candidate_contract(args),
            "evaluation_policy": args.evaluation_policy,
            "artifact_purpose": args.artifact_purpose,
            "test_access_policy": "development_only",
        }
        metrics = {
            "config_hash": config_hash,
            "data_sha256": "data-sha",
            "evaluation_policy": args.evaluation_policy,
            "artifact_purpose": args.artifact_purpose,
        }
        return args, scientific, config, manifest, metrics

    def test_exact_source_math_route_and_policy_tamper_are_rejected(self):
        _, scientific, _, _, _ = self._documents()
        expected = summary._validate_sonnet_variant_contract(
            scientific, Path("/tmp/sonnet-summary")
        )
        self.assertEqual(
            expected["sonnet_mvca"],
            scientific["model"]["sonnet_mvca"],
        )
        mutations = (
            ("source commit", ("source", "official_repo_commit"), "wrong"),
            ("FFT axis", ("mvca", "fft_axis_policy"), "time_dimension_T"),
            ("denominator", ("mvca", "denominator_policy"), "clamped"),
            ("scale", ("mvca", "scale_policy"), "multiply_sqrt_K"),
        )
        for label, path, value in mutations:
            with self.subTest(label=label):
                changed = deepcopy(scientific)
                sonnet = changed["model"]["sonnet_mvca"]
                if path[0] == "source":
                    sonnet[path[0]][path[1]] = value
                else:
                    sonnet[path[0]][path[1]] = value
                with self.assertRaisesRegex(ValueError, "model/source contract"):
                    summary._validate_sonnet_variant_contract(
                        changed, Path("/tmp/sonnet-summary")
                    )
        changed = deepcopy(scientific)
        changed["model"]["sonnet_mvca"]["insertion_identity"] = "after_ddi"
        with self.assertRaisesRegex(ValueError, "model/source contract"):
            summary._validate_sonnet_variant_contract(
                changed, Path("/tmp/sonnet-summary")
            )
        changed = deepcopy(scientific)
        changed["dataset"]["aux_idx"] = [5, 4, 3, 2, 1, 0]
        with self.assertRaisesRegex(ValueError, "dataset contract"):
            summary._validate_sonnet_variant_contract(
                changed, Path("/tmp/sonnet-summary")
            )

    def test_validation_only_rejects_every_test_result_surface(self):
        scientific = {
            "evaluation": {
                "evaluation_policy": summary.TRAIN_VALIDATION_ONLY,
                "artifact_purpose": summary.M4_DEVELOPMENT_CANDIDATE,
                "test_access_policy": "forbidden",
            }
        }
        config = {
            "evaluation_policy": summary.TRAIN_VALIDATION_ONLY,
            "artifact_purpose": summary.M4_DEVELOPMENT_CANDIDATE,
        }
        manifest = {
            "evaluation_policy": summary.TRAIN_VALIDATION_ONLY,
            "artifact_purpose": summary.M4_DEVELOPMENT_CANDIDATE,
            "test_access_policy": "forbidden",
        }
        metrics = {
            "evaluation_policy": summary.TRAIN_VALIDATION_ONLY,
            "artifact_purpose": summary.M4_DEVELOPMENT_CANDIDATE,
            "best_validation": {"mse": 1.0, "mae": 1.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in ("stdout.log", "stderr.log", "train.log"):
                (run_dir / name).write_text("", encoding="utf-8")
            self.assertEqual(
                summary._validate_sonnet_evaluation_artifact(
                    scientific, config, manifest, metrics, run_dir
                ),
                summary.TRAIN_VALIDATION_ONLY,
            )
            changed = deepcopy(metrics)
            changed["test"] = {"mse": 1.0}
            with self.assertRaisesRegex(ValueError, "test result fields"):
                summary._validate_sonnet_evaluation_artifact(
                    scientific, config, manifest, changed, run_dir
                )
            changed = deepcopy(manifest)
            changed["test_mae"] = 1.0
            with self.assertRaisesRegex(ValueError, "test result fields"):
                summary._validate_sonnet_evaluation_artifact(
                    scientific, config, changed, metrics, run_dir
                )
            (run_dir / "train.log").write_text(
                "test_mse=1.0", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "log contains test output"):
                summary._validate_sonnet_evaluation_artifact(
                    scientific, config, manifest, metrics, run_dir
                )
        self.assertNotIn("test_mse", summary.SONNET_VALIDATION_RUN_FIELDS)
        self.assertNotIn(
            "test_mse_mean", summary.SONNET_VALIDATION_AGGREGATE_FIELDS
        )

    def test_checkpoint_scope_is_exact_and_cross_source_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            args, scientific, config, manifest, metrics = self._documents(
                run_dir=run_dir
            )
            module = summary.sonnet_spec.SonnetMVCATargetResidual(
                seq_len=512,
                feature_num=7,
                task_mode="target_exogenous",
                target_idx=6,
                ordered_aux_idx=range(6),
                feature_schema=self.FEATURES,
                schema_fingerprint=args.schema_fingerprint,
            )
            state = {
                f"sonnet_mvca.{key}": value.detach().clone()
                for key, value in module.state_dict().items()
            }
            common = {
                "schema_version": summary.SCHEMA_VERSION,
                "artifact_schema_version": summary.ENHANCED_ARTIFACT_SCHEMA_VERSION,
                "implementation_variant": summary.SONNET_IMPLEMENTATION_VARIANT,
                "config_hash": metrics["config_hash"],
                "data_sha256": metrics["data_sha256"],
                "evaluation_policy": summary.TRAIN_VALIDATION_TEST,
                "artifact_purpose": summary.M4_DEVELOPMENT_CANDIDATE,
                "training_protocol": scientific["training_protocol"],
                "resolved_config": config,
                "model_state": state,
            }
            torch.save(common, run_dir / "best.pt")
            last = deepcopy(common)
            last["best_model_state"] = deepcopy(state)
            torch.save(last, run_dir / "last.pt")
            summary._validate_sonnet_checkpoints(
                scientific, config, manifest, metrics, run_dir
            )

            broken = torch.load(run_dir / "best.pt", map_location="cpu")
            broken["model_state"]["sonnet_mvca.freq_params"] = broken[
                "model_state"
            ]["sonnet_mvca.freq_params"].double()
            torch.save(broken, run_dir / "best.pt")
            with self.assertRaisesRegex(ValueError, "tensor mismatch"):
                summary._validate_sonnet_checkpoints(
                    scientific, config, manifest, metrics, run_dir
                )

            torch.save(common, run_dir / "best.pt")
            broken = torch.load(run_dir / "last.pt", map_location="cpu")
            broken["model_state"]["teb.forbidden"] = torch.zeros(1)
            torch.save(broken, run_dir / "last.pt")
            with self.assertRaisesRegex(ValueError, "module key mismatch"):
                summary._validate_sonnet_checkpoints(
                    scientific, config, manifest, metrics, run_dir
                )

    def test_aggregate_rejects_mixed_evaluation_policy(self):
        base = {
            "implementation_variant": summary.SONNET_IMPLEMENTATION_VARIANT,
            "training_protocol_id": summary.STANDARD_TRAINING_PROTOCOL,
            "development_protocol_id": summary.SONNET_DEVELOPMENT_PROTOCOL,
            "ablation_id": summary.SONNET_CANDIDATE_ABLATION_ID,
            "dataset_id": "fixture",
            "seq_len": 12,
            "pred_len": 1,
            "comparison_config_hash": "same",
            "val_mse": 1.0,
            "val_mae": 1.0,
        }
        rows = [
            {**base, "seed": 1, "evaluation_policy": summary.TRAIN_VALIDATION_ONLY},
            {**base, "seed": 2, "evaluation_policy": summary.TRAIN_VALIDATION_TEST},
        ]
        with self.assertRaisesRegex(ValueError, "mixed evaluation_policy"):
            summary.aggregate_runs(rows)


if __name__ == "__main__":
    unittest.main()
