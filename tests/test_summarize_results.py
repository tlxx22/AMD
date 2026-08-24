import csv
import json
import tempfile
import unittest
from pathlib import Path

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
    ):
        run_dir = (
            Path(root)
            / summary.ENHANCED_IMPLEMENTATION_VARIANT
            / "UrbanEV"
            / "target_exogenous"
            / "volume"
            / "horizon_3"
            / "fold_1"
            / f"seed_{seed}"
            / run_id
        )
        run_dir.mkdir(parents=True)
        scientific = {
            "implementation_variant": summary.ENHANCED_IMPLEMENTATION_VARIANT,
            "source_sha256": "source-hash",
            "dataset": {
                "id": "UrbanEV",
                "sha256": "data-hash",
                "task_mode": "target_exogenous",
                "target": "volume",
                "fold": 1,
                "label_horizon": 3,
                "model_pred_len": 1,
                "artifact_horizon": 3,
            },
            "model": {
                "model_class": "AMDEnhanced",
                "seq_len": 12,
                "pred_len": 1,
                "model_pred_len": 1,
                "use_pmcr": False,
                "use_teb": True,
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
                "ablation_id": "U2",
            },
        }
        config_hash = summary._stable_hash(scientific)
        common = {
            "schema_version": summary.SCHEMA_VERSION,
            "artifact_schema_version": summary.ENHANCED_ARTIFACT_SCHEMA_VERSION,
            "implementation_variant": summary.ENHANCED_IMPLEMENTATION_VARIANT,
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

    def test_unsupported_variant_is_an_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported implementation variant"):
                summary.load_completed_runs(
                    Path(directory),
                    implementation_variant="unknown-variant",
                )


if __name__ == "__main__":
    unittest.main()
