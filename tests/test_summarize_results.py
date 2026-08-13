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
            "model": {"seq_len": 4, "pred_len": 2},
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


if __name__ == "__main__":
    unittest.main()
