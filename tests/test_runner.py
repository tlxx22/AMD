import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

import main as runner
import summarize_results as summary


class _ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x), x.new_zeros(())


class _TinyCheckpointModel(torch.nn.Module):
    """One-parameter stand-in used to make selection tests unambiguous."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))


def _assert_nested_equal(test_case, left, right, path="root"):
    if torch.is_tensor(left) or torch.is_tensor(right):
        test_case.assertTrue(
            torch.is_tensor(left) and torch.is_tensor(right), path
        )
        test_case.assertTrue(torch.equal(left, right), path)
    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        test_case.assertTrue(
            isinstance(left, np.ndarray) and isinstance(right, np.ndarray), path
        )
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict) or isinstance(right, dict):
        test_case.assertTrue(isinstance(left, dict) and isinstance(right, dict), path)
        test_case.assertEqual(left.keys(), right.keys(), path)
        for key in left:
            _assert_nested_equal(test_case, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        test_case.assertEqual(type(left), type(right), path)
        test_case.assertEqual(len(left), len(right), path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_equal(
                test_case, left_item, right_item, f"{path}[{index}]"
            )
    else:
        test_case.assertEqual(left, right, path)


def _runtime_metadata():
    return {
        "python": "test",
        "numpy": "test",
        "pandas": "test",
        "scipy": "test",
        "scikit_learn": "test",
        "torch": torch.__version__,
        "torch_cuda": None,
        "cudnn": None,
        "device": "cpu",
        "device_name": None,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }


def _write_resume_fixture(run_dir, completed_epoch=1):
    """Create the smallest internally consistent resumable run on disk."""

    run_dir.mkdir(parents=True)
    scientific_config = {
        "dataset": {"id": "fixture", "sha256": "data-fixture"},
        "model": {"seq_len": 4, "pred_len": 2},
        "execution": {"seed": 123},
    }
    config_hash = runner.stable_hash(scientific_config)
    data_sha256 = "data-fixture"
    resolved_config = {
        "schema_version": runner.SCHEMA_VERSION,
        "implementation_variant": runner.IMPLEMENTATION_VARIANT,
        "config_hash": config_hash,
        "scientific_config": scientific_config,
        "run": {"run_dir": str(run_dir.resolve()), "train_epochs": completed_epoch + 1},
    }
    manifest = {
        "schema_version": runner.SCHEMA_VERSION,
        "implementation_variant": runner.IMPLEMENTATION_VARIANT,
        "run_id": run_dir.name,
        "status": "failed",
        "config_hash": config_hash,
        "data_sha256": data_sha256,
        "artifact_dir": str(run_dir.resolve()),
        "completed_epoch": completed_epoch,
    }
    generator = torch.Generator().manual_seed(123)
    checkpoint = {
        "schema_version": runner.SCHEMA_VERSION,
        "implementation_variant": runner.IMPLEMENTATION_VARIANT,
        "config_hash": config_hash,
        "data_sha256": data_sha256,
        "resolved_config": resolved_config,
        "completed_epoch": completed_epoch,
        "history": [{"epoch": epoch} for epoch in range(1, completed_epoch + 1)],
        "model_state": {},
        "optimizer_state": {},
        "best_model_state": {},
        "best_epoch": 1,
        "best_mse": 0.5,
        "best_val_metrics": {"mse": 0.5, "mae": 0.4},
        "train_generator_state": generator.get_state(),
        "rng_state": runner.capture_rng_state(),
    }
    runner.atomic_write_json(run_dir / "manifest.json", manifest)
    runner.atomic_write_json(run_dir / "config.resolved.json", resolved_config)
    runner.atomic_torch_save(run_dir / "last.pt", checkpoint)
    return config_hash, data_sha256


class RunnerUnitTests(unittest.TestCase):
    def test_source_fingerprint_recurses_models_in_stable_relative_order(self):
        files = {
            "main.py": "# runner\n",
            "models/top.py": "TOP = 1\n",
            "models/modules/nested.py": "NESTED = 2\n",
            "utils/helper.py": "HELPER = 3\n",
        }

        def build(root, ordered_paths):
            for relative in ordered_paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(files[relative], encoding="utf-8")

        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_root = Path(first_dir)
            second_root = Path(second_dir)
            build(first_root, list(files))
            build(second_root, list(reversed(files)))

            original = runner.source_fingerprint(first_root)
            self.assertEqual(original, runner.source_fingerprint(second_root))

            nested = first_root / "models/modules/nested.py"
            nested.write_text("NESTED = 99\n", encoding="utf-8")
            self.assertNotEqual(original, runner.source_fingerprint(first_root))

            before_non_python = runner.source_fingerprint(first_root)
            (first_root / "models/modules/ignored.txt").write_text(
                "not executable Python", encoding="utf-8"
            )
            self.assertEqual(before_non_python, runner.source_fingerprint(first_root))

    def test_strict_boolean_parser(self):
        for value in ("true", "TRUE", "yes", "on", "1"):
            self.assertIs(runner.str2bool(value), True)
        for value in ("false", "FALSE", "no", "off", "0"):
            self.assertIs(runner.str2bool(value), False)
        with self.assertRaisesRegex(Exception, "expected a boolean"):
            runner.str2bool("maybe")

        args = runner.parse_args(["--norm", "False", "--layernorm", "False"])
        self.assertIs(args.norm, False)
        self.assertIs(args.layernorm, False)

    def test_global_metrics_weight_elements_not_batches(self):
        batches = [
            (torch.ones(128, 1, 1), torch.zeros(128, 1, 1)),
            (torch.ones(1, 1, 1), torch.full((1, 1, 1), 10.0)),
        ]
        metrics = runner.evaluate(
            _ZeroModel(), batches, torch.device("cpu"), show_progress=False
        )
        self.assertAlmostEqual(metrics["mse"], 100.0 / 129.0)
        self.assertAlmostEqual(metrics["mae"], 10.0 / 129.0)
        self.assertEqual(metrics["num_elements"], 129)
        self.assertEqual(metrics["num_batches"], 2)

    def test_best_selection_is_strict(self):
        self.assertTrue(runner.should_update_best(0.2, 0.3))
        self.assertFalse(runner.should_update_best(0.2, 0.2))
        self.assertFalse(runner.should_update_best(float("nan"), 0.3))

    def test_non_finite_numeric_arguments_are_rejected(self):
        for option, value in (
            ("--alpha", "nan"),
            ("--alpha", "inf"),
            ("--dropout", "nan"),
            ("--learning_rate", "inf"),
        ):
            with self.subTest(option=option, value=value):
                args = runner.parse_args([option, value])
                with self.assertRaises(ValueError):
                    runner.prepare_args(args)

    def test_paper_weight_decay_and_model_contract_are_resolved(self):
        args = runner.prepare_args(
            runner.parse_args(["--weight_decay", "0.0000001"])
        )
        self.assertEqual(args.weight_decay, 1e-7)
        scientific = runner._scientific_config(
            args,
            data_sha256="data-sha",
            source_sha256="source-sha",
            preprocessing={},
            device=torch.device("cpu"),
            environment=_runtime_metadata(),
        )
        self.assertEqual(scientific["optimization"]["weight_decay"], 1e-7)
        self.assertEqual(
            scientific["model"]["entry_normalization_impl"],
            "torch_layernorm_last_dim_sequence",
        )
        self.assertEqual(
            scientific["model"]["ddi_internal_normalization_impl"],
            "released_batchnorm1d_norm1_and_norm2_when_alpha_gt_0",
        )
        self.assertEqual(
            scientific["model"]["ddi_hidden_rule"],
            "max(32,2**ceil(log2(feature_count)))_when_alpha_gt_0",
        )
        self.assertEqual(
            scientific["model"]["selector_mode"],
            "horizon_shared_dense_emphasis",
        )

        for invalid in ("0.000000001", "nan", "inf"):
            with self.subTest(weight_decay=invalid):
                with self.assertRaises(ValueError):
                    runner.prepare_args(
                        runner.parse_args(["--weight_decay", invalid])
                    )

    def test_atomic_serializers_leave_complete_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.atomic_write_json(root / "value.json", {"value": 3})
            self.assertEqual(
                json.loads((root / "value.json").read_text(encoding="utf-8")),
                {"value": 3},
            )
            runner.atomic_torch_save(root / "value.pt", {"value": torch.tensor([3])})
            saved = torch.load(root / "value.pt", map_location="cpu")
            self.assertTrue(torch.equal(saved["value"], torch.tensor([3])))
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_atomic_torch_save_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "last.pt"
            runner.atomic_torch_save(destination, {"generation": 1})
            original_bytes = destination.read_bytes()

            with self.subTest(failure="torch.save"):
                with mock.patch.object(
                    runner.torch, "save", side_effect=RuntimeError("simulated save failure")
                ):
                    with self.assertRaisesRegex(RuntimeError, "simulated save failure"):
                        runner.atomic_torch_save(destination, {"generation": 2})
                self.assertEqual(destination.read_bytes(), original_bytes)
                self.assertEqual(torch.load(destination, map_location="cpu")["generation"], 1)
                self.assertEqual(list(root.glob(".*.tmp")), [])

            with self.subTest(failure="os.replace"):
                with mock.patch.object(
                    runner.os, "replace", side_effect=OSError("simulated replace failure")
                ):
                    with self.assertRaisesRegex(OSError, "simulated replace failure"):
                        runner.atomic_torch_save(destination, {"generation": 3})
                self.assertEqual(destination.read_bytes(), original_bytes)
                self.assertEqual(torch.load(destination, map_location="cpu")["generation"], 1)
                self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_run_lock_rejects_second_owner_for_same_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            first = runner.RunLock(run_dir)
            second = runner.RunLock(run_dir)
            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    second.acquire()
            finally:
                first.release()

            # Releasing the owner must make the directory lockable again.
            second.acquire()
            second.release()

    def test_load_resume_checkpoint_keeps_rng_states_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "fixture-run"
            config_hash, data_sha256 = _write_resume_fixture(run_dir)

            _, checkpoint, _ = runner._load_resume_checkpoint(
                run_dir, config_hash, data_sha256, train_epochs=2
            )

            generator_state = checkpoint["train_generator_state"]
            cpu_rng_state = checkpoint["rng_state"]["torch_cpu"]
            for state in (generator_state, cpu_rng_state):
                self.assertEqual(state.device.type, "cpu")
                self.assertEqual(state.dtype, torch.uint8)
                self.assertEqual(state.ndim, 1)
            for state in checkpoint["rng_state"].get("torch_cuda") or []:
                self.assertEqual(state.device.type, "cpu")
                self.assertEqual(state.dtype, torch.uint8)
                self.assertEqual(state.ndim, 1)

            # These APIs reject CUDA ByteTensors, so successful restoration is a
            # stronger contract check than inspecting the device string alone.
            restored_generator = torch.Generator()
            restored_generator.set_state(generator_state)
            previous_cpu_rng = torch.get_rng_state()
            try:
                torch.set_rng_state(cpu_rng_state)
            finally:
                torch.set_rng_state(previous_cpu_rng)

    def test_old_variant_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run-old-variant"
            config_hash, data_sha256 = _write_resume_fixture(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["implementation_variant"] = "AMD-mdm-u-to-ddi-v1"
            runner.atomic_write_json(manifest_path, manifest)

            with self.assertRaisesRegex(RuntimeError, "resume variant"):
                runner._load_resume_checkpoint(
                    run_dir,
                    config_hash=config_hash,
                    data_sha256=data_sha256,
                    train_epochs=2,
                )

    def test_nonexistent_resume_path_is_rejected_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "does-not-exist"
            args = runner.prepare_args(runner.parse_args(["--resume", str(missing)]))

            with self.assertRaisesRegex(FileNotFoundError, "resume path does not exist"):
                runner._run_directory(args)
            self.assertFalse(missing.exists())

    def test_tampered_resolved_or_checkpoint_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with self.subTest(location="config.resolved.json"):
                run_dir = root / "tampered-json"
                config_hash, data_sha256 = _write_resume_fixture(run_dir)
                config_path = run_dir / "config.resolved.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config["scientific_config"]["execution"]["seed"] = 999
                self.assertEqual(config["config_hash"], config_hash)
                runner.atomic_write_json(config_path, config)

                with self.assertRaisesRegex(RuntimeError, "scientific configuration was modified"):
                    runner._load_resume_checkpoint(
                        run_dir, config_hash, data_sha256, train_epochs=2
                    )

            with self.subTest(location="last.pt"):
                run_dir = root / "tampered-checkpoint"
                config_hash, data_sha256 = _write_resume_fixture(run_dir)
                last_path = run_dir / "last.pt"
                checkpoint = torch.load(last_path, map_location="cpu")
                checkpoint["resolved_config"]["scientific_config"]["execution"]["seed"] = 999
                self.assertEqual(checkpoint["config_hash"], config_hash)
                runner.atomic_torch_save(last_path, checkpoint)

                with self.assertRaisesRegex(
                    RuntimeError, "checkpoint scientific configuration mismatch"
                ):
                    runner._load_resume_checkpoint(
                        run_dir, config_hash, data_sha256, train_epochs=2
                    )


class RunnerResumeIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_dataset(path):
        rows = 80
        frame = pd.DataFrame({
            "date": [f"2024-01-{index + 1:02d}" for index in range(rows)],
            "a": [float((index * 3) % 17) for index in range(rows)],
            "b": [float((index * 5 + 1) % 19) for index in range(rows)],
        })
        frame.to_csv(path, index=False)

    @staticmethod
    def _args(data_path, artifact_root, resume=None, train_epochs=2):
        values = [
            "--data", str(data_path),
            "--dataset_id", "toy",
            "--artifact_root", str(artifact_root),
            "--device", "cpu",
            "--progress", "false",
            "--seed", "123",
            "--seq_len", "4",
            "--pred_len", "2",
            "--n_block", "1",
            "--alpha", "0",
            "--mix_layer_num", "0",
            "--mix_layer_scale", "2",
            "--patch", "4",
            "--norm", "false",
            "--layernorm", "false",
            "--dropout", "0.1",
            "--train_epochs", str(train_epochs),
            "--batch_size", "8",
            "--learning_rate", "0.001",
        ]
        if resume is not None:
            values.extend(["--resume", str(resume)])
        return runner.parse_args(values)

    @staticmethod
    def _single_run_dir(artifact_root):
        run_dirs = list(
            (Path(artifact_root) / runner.IMPLEMENTATION_VARIANT / "toy"
             / "sl4_pl2" / "seed123").iterdir()
        )
        if len(run_dirs) != 1:
            raise AssertionError(f"expected exactly one run directory, got {run_dirs}")
        return run_dirs[0]

    @staticmethod
    def _assert_state_dict_equal(test_case, left, right):
        test_case.assertEqual(left.keys(), right.keys())
        for key in left:
            test_case.assertTrue(
                torch.equal(left[key], right[key]),
                msg=f"model tensor differs after resume: {key}",
            )

    def test_interrupted_resume_matches_uninterrupted_and_completed_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            full_root = root / "full"
            resumed_root = root / "resumed"
            self._write_dataset(data_path)

            stable_git = {"commit": "test", "dirty": False, "status": []}
            patches = (
                mock.patch.object(runner, "environment_metadata", return_value=_runtime_metadata()),
                mock.patch.object(runner, "git_metadata", return_value=stable_git),
            )
            with patches[0], patches[1]:
                full_metrics = runner.main(self._args(data_path, full_root))
                full_dir = self._single_run_dir(full_root)

                original_train_one_epoch = runner.train_one_epoch

                def interrupt_before_second_epoch(*args, **kwargs):
                    epoch = args[5] if len(args) > 5 else kwargs["epoch"]
                    if epoch == 2:
                        raise RuntimeError("simulated interruption")
                    return original_train_one_epoch(*args, **kwargs)

                with mock.patch.object(
                    runner, "train_one_epoch", side_effect=interrupt_before_second_epoch
                ):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        runner.main(self._args(data_path, resumed_root))

                resumed_dir = self._single_run_dir(resumed_root)
                failed_manifest = json.loads(
                    (resumed_dir / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(failed_manifest["status"], "failed")
                self.assertEqual(failed_manifest["completed_epoch"], 1)

                resumed_metrics = runner.main(
                    self._args(data_path, resumed_root, resume=resumed_dir)
                )

            full_last = torch.load(full_dir / "last.pt", map_location="cpu")
            resumed_last = torch.load(resumed_dir / "last.pt", map_location="cpu")
            self.assertEqual(
                full_last["optimizer_state"]["param_groups"][0]["weight_decay"],
                1e-7,
            )
            self._assert_state_dict_equal(
                self, full_last["model_state"], resumed_last["model_state"]
            )
            self._assert_state_dict_equal(
                self,
                full_last["best_model_state"],
                resumed_last["best_model_state"],
            )
            _assert_nested_equal(
                self,
                full_last["optimizer_state"],
                resumed_last["optimizer_state"],
                "optimizer_state",
            )
            _assert_nested_equal(
                self,
                full_last["rng_state"],
                resumed_last["rng_state"],
                "rng_state",
            )
            self.assertTrue(torch.equal(
                full_last["train_generator_state"],
                resumed_last["train_generator_state"],
            ))
            def scientific_history(history):
                return [
                    {key: value for key, value in record.items()
                     if key not in {"duration_seconds", "finished_at"}}
                    for record in history
                ]

            self.assertEqual(
                scientific_history(full_last["history"]),
                scientific_history(resumed_last["history"]),
            )
            self.assertEqual(full_metrics["test"], resumed_metrics["test"])
            self.assertEqual(full_metrics["best_epoch"], resumed_metrics["best_epoch"])

            for required in (
                "manifest.json", "config.resolved.json", "best.pt", "last.pt",
                "history.jsonl", "metrics.json",
            ):
                self.assertTrue((resumed_dir / required).is_file(), required)
            manifest = json.loads(
                (resumed_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["implementation_variant"], runner.IMPLEMENTATION_VARIANT)

            manifest_before = (resumed_dir / "manifest.json").read_bytes()
            config_before = (resumed_dir / "config.resolved.json").read_bytes()
            with mock.patch.object(
                runner, "environment_metadata", return_value=_runtime_metadata()
            ), mock.patch.object(runner, "git_metadata", return_value=stable_git):
                with self.assertRaisesRegex(RuntimeError, "immutable"):
                    runner.main(self._args(data_path, resumed_root, resume=resumed_dir))
            self.assertEqual(manifest_before, (resumed_dir / "manifest.json").read_bytes())
            self.assertEqual(config_before, (resumed_dir / "config.resolved.json").read_bytes())

    def test_resume_below_completed_epoch_does_not_mutate_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            artifact_root = root / "artifacts"
            self._write_dataset(data_path)

            stable_git = {"commit": "test", "dirty": False, "status": []}
            original_train_one_epoch = runner.train_one_epoch

            def interrupt_before_third_epoch(*args, **kwargs):
                epoch = args[5] if len(args) > 5 else kwargs["epoch"]
                if epoch == 3:
                    raise RuntimeError("simulated interruption after two epochs")
                return original_train_one_epoch(*args, **kwargs)

            with mock.patch.object(
                runner, "environment_metadata", return_value=_runtime_metadata()
            ), mock.patch.object(
                runner, "git_metadata", return_value=stable_git
            ), mock.patch.object(
                runner, "train_one_epoch", side_effect=interrupt_before_third_epoch
            ):
                with self.assertRaisesRegex(RuntimeError, "after two epochs"):
                    runner.main(
                        self._args(data_path, artifact_root, train_epochs=3)
                    )

            run_dir = self._single_run_dir(artifact_root)
            manifest_path = run_dir / "manifest.json"
            config_path = run_dir / "config.resolved.json"
            manifest_before = manifest_path.read_bytes()
            config_before = config_path.read_bytes()

            with mock.patch.object(
                runner, "environment_metadata", return_value=_runtime_metadata()
            ), mock.patch.object(
                runner, "git_metadata", return_value=stable_git
            ):
                with self.assertRaisesRegex(RuntimeError, "below completed epoch 2"):
                    runner.main(
                        self._args(
                            data_path,
                            artifact_root,
                            resume=run_dir,
                            train_epochs=1,
                        )
                    )

            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            self.assertEqual(config_path.read_bytes(), config_before)

    @staticmethod
    def _fake_train_epoch(model, *args, **kwargs):
        with torch.no_grad():
            model.weight.add_(1.0)
        value = float(model.weight.item())
        return {
            "mse": value,
            "mae": value,
            "num_elements": 1,
            "num_batches": 1,
            "objective_mean_batches": value,
            "auxiliary_mean_batches": 0.0,
        }

    @staticmethod
    def _fake_evaluate(model, *args, **kwargs):
        description = kwargs.get("description", "")
        value = float(model.weight.item())
        if description.startswith("Val"):
            # Epoch one is the unique validation best; epoch two is worse.
            mse = 0.1 if value == 1.0 else 0.2
            mae = mse / 2
        else:
            # Expose which checkpoint was loaded for final testing.
            mse = value
            mae = value
        return {
            "mse": mse,
            "mae": mae,
            "num_elements": 1,
            "num_batches": 1,
        }

    def test_final_test_reloads_strict_validation_best_not_last_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            artifact_root = root / "artifacts"
            self._write_dataset(data_path)
            stable_git = {"commit": "test", "dirty": False, "status": []}

            with mock.patch.object(
                runner, "environment_metadata", return_value=_runtime_metadata()
            ), mock.patch.object(
                runner, "git_metadata", return_value=stable_git
            ), mock.patch.object(
                runner, "AMD", _TinyCheckpointModel
            ), mock.patch.object(
                runner, "train_one_epoch", side_effect=self._fake_train_epoch
            ), mock.patch.object(
                runner, "evaluate", side_effect=self._fake_evaluate
            ):
                metrics = runner.main(self._args(data_path, artifact_root))

            run_dir = self._single_run_dir(artifact_root)
            last = torch.load(run_dir / "last.pt", map_location="cpu")
            best = torch.load(run_dir / "best.pt", map_location="cpu")
            self.assertEqual(metrics["best_epoch"], 1)
            self.assertEqual(metrics["test"]["mse"], 1.0)
            self.assertEqual(last["model_state"]["weight"].item(), 2.0)
            self.assertEqual(last["best_model_state"]["weight"].item(), 1.0)
            self.assertEqual(best["model_state"]["weight"].item(), 1.0)

    def test_resume_reconciles_derivative_files_from_last_commit_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            artifact_root = root / "artifacts"
            self._write_dataset(data_path)
            stable_git = {"commit": "test", "dirty": False, "status": []}
            original_atomic_torch_save = runner.atomic_torch_save
            injected = {"done": False}

            def interrupt_first_best_write(path, value):
                if Path(path).name == "best.pt" and not injected["done"]:
                    injected["done"] = True
                    raise RuntimeError("interrupted after last commit")
                return original_atomic_torch_save(path, value)

            common_patches = (
                mock.patch.object(
                    runner, "environment_metadata", return_value=_runtime_metadata()
                ),
                mock.patch.object(
                    runner, "git_metadata", return_value=stable_git
                ),
                mock.patch.object(runner, "AMD", _TinyCheckpointModel),
                mock.patch.object(
                    runner, "train_one_epoch", side_effect=self._fake_train_epoch
                ),
                mock.patch.object(
                    runner, "evaluate", side_effect=self._fake_evaluate
                ),
            )
            with common_patches[0], common_patches[1], common_patches[2], \
                    common_patches[3], common_patches[4], mock.patch.object(
                        runner,
                        "atomic_torch_save",
                        side_effect=interrupt_first_best_write,
                    ):
                with self.assertRaisesRegex(RuntimeError, "after last commit"):
                    runner.main(
                        self._args(data_path, artifact_root, train_epochs=1)
                    )

            run_dir = self._single_run_dir(artifact_root)
            self.assertTrue((run_dir / "last.pt").is_file())
            self.assertFalse((run_dir / "best.pt").exists())
            failed = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed["status"], "failed")

            resumed_patches = (
                mock.patch.object(
                    runner, "environment_metadata", return_value=_runtime_metadata()
                ),
                mock.patch.object(
                    runner, "git_metadata", return_value=stable_git
                ),
                mock.patch.object(runner, "AMD", _TinyCheckpointModel),
                mock.patch.object(
                    runner, "train_one_epoch", side_effect=self._fake_train_epoch
                ),
                mock.patch.object(
                    runner, "evaluate", side_effect=self._fake_evaluate
                ),
            )
            with resumed_patches[0], resumed_patches[1], resumed_patches[2], \
                    resumed_patches[3], resumed_patches[4]:
                metrics = runner.main(
                    self._args(
                        data_path,
                        artifact_root,
                        resume=run_dir,
                        train_epochs=1,
                    )
                )

            self.assertEqual(metrics["status"], "completed")
            self.assertEqual(metrics["test"]["mse"], 1.0)
            self.assertTrue((run_dir / "best.pt").is_file())
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")

    def test_post_commit_print_failure_does_not_downgrade_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            artifact_root = root / "artifacts"
            self._write_dataset(data_path)
            stable_git = {"commit": "test", "dirty": False, "status": []}

            def fail_only_completed_print(*values, **kwargs):
                message = " ".join(str(value) for value in values)
                if message.startswith("completed run="):
                    raise BrokenPipeError("simulated closed stdout")

            with mock.patch.object(
                runner, "environment_metadata", return_value=_runtime_metadata()
            ), mock.patch.object(
                runner, "git_metadata", return_value=stable_git
            ), mock.patch.object(
                runner, "AMD", _TinyCheckpointModel
            ), mock.patch.object(
                runner, "train_one_epoch", side_effect=self._fake_train_epoch
            ), mock.patch.object(
                runner, "evaluate", side_effect=self._fake_evaluate
            ), mock.patch(
                "builtins.print", side_effect=fail_only_completed_print
            ):
                with self.assertRaisesRegex(BrokenPipeError, "closed stdout"):
                    runner.main(
                        self._args(data_path, artifact_root, train_epochs=1)
                    )

            run_dir = self._single_run_dir(artifact_root)
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (run_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(metrics["status"], "completed")
            self.assertNotIn("failure", manifest)


class EnhancedRunnerM3Tests(unittest.TestCase):
    @staticmethod
    def _write_dataset(path):
        rows = 80
        pd.DataFrame({
            "date": [f"2024-01-{index + 1:02d}" for index in range(rows)],
            "a": [float((index * 3) % 17) for index in range(rows)],
            "b": [float((index * 5 + 1) % 19) for index in range(rows)],
        }).to_csv(path, index=False)

    @staticmethod
    def _args(
        data_path,
        artifact_root,
        *,
        use_teb=True,
        ablation_id="U2",
        train_epochs=1,
        resume=None,
        teb_dropout=0.0,
    ):
        values = [
            "--implementation_variant", runner.ENHANCED_IMPLEMENTATION_VARIANT,
            "--data", str(data_path),
            "--dataset_id", "toy",
            "--artifact_root", str(artifact_root),
            "--device", "cpu",
            "--num_threads", "1",
            "--progress", "false",
            "--seed", "123",
            "--seq_len", "4",
            "--pred_len", "2",
            "--n_block", "1",
            "--alpha", "0",
            "--mix_layer_num", "0",
            "--mix_layer_scale", "2",
            "--patch", "4",
            "--norm", "true",
            "--layernorm", "false",
            "--dropout", "0",
            "--train_epochs", str(train_epochs),
            "--batch_size", "8",
            "--learning_rate", "0.001",
            "--feature_type", "MS",
            "--target", "b",
            "--task_mode", "target_exogenous",
            "--target_idx", "1",
            "--aux_idx", "0",
            "--feature_names", "a", "b",
            "--target_feature_name", "b",
            "--aux_feature_names", "a",
            "--schema_fingerprint", "synthetic-v1",
            "--fold", "official",
            "--horizon", "2",
            "--ablation_id", ablation_id,
            "--use_pmcr", "false",
            "--use_teb", str(use_teb).lower(),
            "--teb_context_dim", "4",
            "--teb_heads", "2",
            "--teb_dropout", str(teb_dropout),
            "--teb_gamma_init", "0.001",
        ]
        if resume is not None:
            values.extend(["--resume", str(resume)])
        return runner.parse_args(values)

    @staticmethod
    def _urbanev_args(data_root, artifact_root):
        return runner.parse_args(
            [
                "--implementation_variant", runner.ENHANCED_IMPLEMENTATION_VARIANT,
                "--data", str(data_root),
                "--dataset_id", "UrbanEV",
                "--artifact_root", str(artifact_root),
                "--device", "cpu",
                "--num_threads", "1",
                "--progress", "false",
                "--seed", "123",
                "--seq_len", "12",
                "--pred_len", "1",
                "--n_block", "1",
                "--alpha", "0",
                "--mix_layer_num", "0",
                "--mix_layer_scale", "2",
                "--patch", "12",
                "--norm", "true",
                "--layernorm", "false",
                "--dropout", "0",
                "--train_epochs", "1",
                "--batch_size", "4",
                "--learning_rate", "0.001",
                "--feature_type", "MS",
                "--target", "volume",
                "--task_mode", "target_exogenous",
                "--feature_preset", "F1",
                "--fold", "1",
                "--label_horizon", "3",
                "--ablation_id", "U2",
                "--use_pmcr", "false",
                "--use_teb", "true",
                "--teb_context_dim", "4",
                "--teb_heads", "2",
                "--teb_dropout", "0",
                "--teb_gamma_init", "0.001",
            ]
        )

    @staticmethod
    def _run_parent(artifact_root):
        return (
            Path(artifact_root)
            / runner.ENHANCED_IMPLEMENTATION_VARIANT
            / "toy"
            / "target_exogenous"
            / "b"
            / "horizon_2"
            / "fold_official"
            / "seed_123"
        )

    @classmethod
    def _single_run_dir(cls, artifact_root):
        runs = list(cls._run_parent(artifact_root).iterdir())
        if len(runs) != 1:
            raise AssertionError(f"expected one enhanced run, got {runs}")
        return runs[0]

    @staticmethod
    def _fake_train(model, *args, **kwargs):
        with torch.no_grad():
            model.weight.add_(1.0)
        value = float(model.weight.item())
        return {
            "mse": value,
            "mae": value,
            "num_elements": 1,
            "num_batches": 1,
            "objective_mean_batches": value,
            "auxiliary_mean_batches": 0.0,
        }

    @staticmethod
    def _fake_evaluate(model, *args, **kwargs):
        value = float(model.weight.item())
        return {
            "mse": value,
            "mae": value,
            "num_elements": 1,
            "num_batches": 1,
        }

    def test_u1_u2_share_data_and_optimization_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            self._write_dataset(data_path)
            u1 = runner.prepare_args(self._args(
                data_path, root / "u1", use_teb=False, ablation_id="U1"
            ))
            u2 = runner.prepare_args(self._args(
                data_path, root / "u2", use_teb=True, ablation_id="U2"
            ))

            for field in (
                "task_mode", "target_idx", "aux_idx", "feature_names",
                "target_feature_name", "aux_feature_names", "schema_fingerprint",
                "fold", "horizon", "batch_size", "learning_rate", "seed",
            ):
                self.assertEqual(getattr(u1, field), getattr(u2, field), field)
            self.assertEqual(u1.display_name, "AMD-Concat")
            self.assertEqual(u2.display_name, "AMD-Concat + TEB")

            preprocessing = {
                "columns": ["a", "b"],
                "target_indices": [1],
            }
            scientific = []
            for args in (u1, u2):
                scientific.append(runner._scientific_config(
                    args,
                    "data-hash",
                    "source-hash",
                    preprocessing,
                    torch.device("cpu"),
                    _runtime_metadata(),
                ))
            left, right = (
                json.loads(json.dumps(value)) for value in scientific
            )
            left["model"]["use_teb"] = right["model"]["use_teb"]
            left["experiment"]["ablation_id"] = right["experiment"]["ablation_id"]
            left["experiment"]["display_name"] = right["experiment"]["display_name"]
            self.assertEqual(left, right)

    def test_contradictory_enhanced_contracts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            self._write_dataset(data_path)

            f0_teb = self._args(data_path, root / "a")
            f0_teb.feature_names = ["b"]
            f0_teb.target_idx = 0
            f0_teb.aux_idx = []
            f0_teb.aux_feature_names = []
            with self.assertRaisesRegex(ValueError, "contradicts|requires"):
                runner.prepare_args(f0_teb)

            wrong_ablation = self._args(
                data_path, root / "b", use_teb=True, ablation_id="U1"
            )
            with self.assertRaisesRegex(ValueError, "contradicts"):
                runner.prepare_args(wrong_ablation)

            parallel_c1 = self._args(data_path, root / "c")
            parallel_c1.task_mode = "parallel_multivariate"
            parallel_c1.feature_type = "M"
            parallel_c1.target = "all"
            parallel_c1.target_feature_name = "b"
            parallel_c1.target_idx = 0
            parallel_c1.feature_names = ["b"]
            parallel_c1.aux_idx = []
            parallel_c1.aux_feature_names = []
            parallel_c1.ablation_id = "M2"
            with self.assertRaisesRegex(ValueError, "at least two variables"):
                runner.prepare_args(parallel_c1)

    def test_enhanced_synthetic_smoke_and_complete_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            artifact_root = root / "artifacts"
            self._write_dataset(data_path)
            stable_git = {"commit": "test", "dirty": False, "status": []}
            with mock.patch.object(
                runner, "environment_metadata", return_value=_runtime_metadata()
            ), mock.patch.object(
                runner, "git_metadata", return_value=stable_git
            ), mock.patch.object(
                runner, "AMDEnhanced", _TinyCheckpointModel
            ), mock.patch.object(
                runner, "train_one_epoch", side_effect=self._fake_train
            ), mock.patch.object(
                runner, "evaluate", side_effect=self._fake_evaluate
            ):
                metrics = runner.main(self._args(data_path, artifact_root))

            run_dir = self._single_run_dir(artifact_root)
            self.assertEqual(metrics["status"], "completed")
            self.assertEqual(
                run_dir.relative_to(artifact_root).parts[:-1],
                (
                    runner.ENHANCED_IMPLEMENTATION_VARIANT,
                    "toy",
                    "target_exogenous",
                    "b",
                    "horizon_2",
                    "fold_official",
                    "seed_123",
                ),
            )
            required = set(runner.ENHANCED_CHECKSUM_FILES) | {"checksums.sha256"}
            self.assertTrue(required.issubset({path.name for path in run_dir.iterdir()}))
            self.assertEqual(set(runner.verify_checksums(run_dir)), set(runner.ENHANCED_CHECKSUM_FILES))
            checked = subprocess.run(
                ["sha256sum", "-c", "checksums.sha256"],
                cwd=run_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)

            config = json.loads((run_dir / "config.resolved.json").read_text())
            scientific = config["scientific_config"]
            self.assertEqual(config["implementation_variant"], runner.ENHANCED_IMPLEMENTATION_VARIANT)
            self.assertEqual(scientific["dataset"]["feature_names"], ["a", "b"])
            self.assertEqual(scientific["dataset"]["aux_idx"], [0])
            self.assertEqual(scientific["model"]["target_slice"], None)
            self.assertTrue(scientific["model"]["use_teb"])
            self.assertEqual(scientific["model"]["teb"]["context_dim"], 4)
            self.assertEqual(scientific["experiment"]["ablation_id"], "U2")

            source = json.loads((run_dir / "source_fingerprint.json").read_text())
            source_paths = [entry["path"] for entry in source["files"]]
            self.assertIn("models/modules/target_exogenous_bridge.py", source_paths)
            self.assertEqual(source_paths, sorted(source_paths))
            argv = json.loads((run_dir / "sys.argv.json").read_text())
            self.assertEqual(len(argv["invocations"]), 1)
            self.assertTrue((run_dir / "command.txt").read_text().strip())
            self.assertIn("completed run=", (run_dir / "stdout.log").read_text())
            self.assertTrue((run_dir / "train.log").read_text())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["checksum_contract"]["required_files"],
                list(runner.ENHANCED_CHECKSUM_FILES),
            )

    def test_checksum_failure_never_publishes_hidden_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            artifact_root = root / "artifacts"
            self._write_dataset(data_path)
            with mock.patch.object(
                runner, "environment_metadata", return_value=_runtime_metadata()
            ), mock.patch.object(
                runner, "git_metadata", return_value={"commit": "test", "dirty": False, "status": []}
            ), mock.patch.object(
                runner, "AMDEnhanced", _TinyCheckpointModel
            ), mock.patch.object(
                runner, "train_one_epoch", side_effect=self._fake_train
            ), mock.patch.object(
                runner, "evaluate", side_effect=self._fake_evaluate
            ), mock.patch.object(
                runner, "verify_checksums", side_effect=RuntimeError("checksum gate")
            ):
                with self.assertRaisesRegex(RuntimeError, "checksum gate"):
                    runner.main(self._args(data_path, artifact_root))
            run_dir = self._single_run_dir(artifact_root)
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertTrue(run_dir.name.startswith("."))
            self.assertTrue(run_dir.name.endswith(".staging"))
            run_id = runner._enhanced_run_id_from_staging(run_dir)
            self.assertFalse((run_dir.parent / run_id).exists())
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue((run_dir / ".run.lock").exists())
            self.assertEqual(summary.load_completed_runs(
                artifact_root, implementation_variant=runner.ENHANCED_IMPLEMENTATION_VARIANT
            ), [])

    def test_resume_structure_mismatch_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            artifact_root = root / "artifacts"
            self._write_dataset(data_path)
            calls = {"count": 0}

            def interrupt_second_epoch(model, *args, **kwargs):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise RuntimeError("synthetic interruption")
                return self._fake_train(model, *args, **kwargs)

            common_patches = (
                mock.patch.object(runner, "environment_metadata", return_value=_runtime_metadata()),
                mock.patch.object(runner, "git_metadata", return_value={"commit": "test", "dirty": False, "status": []}),
                mock.patch.object(runner, "AMDEnhanced", _TinyCheckpointModel),
                mock.patch.object(runner, "train_one_epoch", side_effect=interrupt_second_epoch),
                mock.patch.object(runner, "evaluate", side_effect=self._fake_evaluate),
            )
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4]:
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    runner.main(self._args(
                        data_path, artifact_root, train_epochs=2
                    ))
            run_dir = self._single_run_dir(artifact_root)
            watched = {
                name: (run_dir / name).read_bytes()
                for name in (
                    "manifest.json", "config.resolved.json", "sys.argv.json",
                    "command.txt", "stdout.log", "stderr.log", "train.log",
                )
            }

            with mock.patch.object(
                runner, "environment_metadata", return_value=_runtime_metadata()
            ), mock.patch.object(
                runner, "git_metadata", return_value={"commit": "test", "dirty": False, "status": []}
            ):
                with self.assertRaisesRegex(RuntimeError, "resume manifest configuration hash mismatch"):
                    runner.main(self._args(
                        data_path,
                        artifact_root,
                        train_epochs=2,
                        resume=run_dir,
                        teb_dropout=0.2,
                    ))
            for name, content in watched.items():
                self.assertEqual((run_dir / name).read_bytes(), content, name)
            mismatch_args = []

            schema_changed = self._args(
                data_path,
                artifact_root,
                train_epochs=2,
                resume=run_dir,
            )
            schema_changed.schema_fingerprint = "synthetic-v2"
            mismatch_args.append(("schema", schema_changed))

            horizon_changed = self._args(
                data_path,
                artifact_root,
                train_epochs=2,
                resume=run_dir,
            )
            horizon_changed.horizon = 3
            horizon_changed.label_horizon = 3
            mismatch_args.append(("horizon", horizon_changed))

            fold_changed = self._args(
                data_path,
                artifact_root,
                train_epochs=2,
                resume=run_dir,
            )
            fold_changed.fold = "alternate"
            mismatch_args.append(("fold", fold_changed))

            teb_changed = self._args(
                data_path,
                artifact_root,
                use_teb=False,
                ablation_id="U1",
                train_epochs=2,
                resume=run_dir,
            )
            mismatch_args.append(("teb", teb_changed))

            pmcr_changed = self._args(
                data_path,
                artifact_root,
                use_teb=True,
                ablation_id="U4",
                train_epochs=2,
                resume=run_dir,
            )
            pmcr_changed.use_pmcr = True
            pmcr_changed.pmcr_hidden_dim = 4
            pmcr_changed.pmcr_kernel_small = 1
            pmcr_changed.pmcr_kernel_large = 3
            mismatch_args.append(("pmcr", pmcr_changed))

            for name, mismatch in mismatch_args:
                with self.subTest(resume_mismatch=name):
                    with mock.patch.object(
                        runner,
                        "environment_metadata",
                        return_value=_runtime_metadata(),
                    ), mock.patch.object(
                        runner,
                        "git_metadata",
                        return_value={
                            "commit": "test",
                            "dirty": False,
                            "status": [],
                        },
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "resume manifest configuration hash mismatch",
                        ):
                            runner.main(mismatch)
                    for filename, content in watched.items():
                        self.assertEqual(
                            (run_dir / filename).read_bytes(),
                            content,
                            filename,
                        )

    def test_urbanev_f1_production_bundle_single_batch_and_horizon_identity(self):
        data_root = runner.ROOT / "data" / "UrbanEV" / "data"
        self.assertTrue(data_root.is_dir())
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "artifacts"
            prepared = runner.prepare_args(
                self._urbanev_args(data_root, artifact_root)
            )
            self.assertEqual(prepared.label_horizon, 3)
            self.assertEqual(prepared.model_pred_len, 1)
            self.assertEqual(prepared.artifact_horizon, 3)
            self.assertEqual(prepared.fold, 1)
            self.assertEqual(prepared.feature_names, (
                "volume",
                "hour_sin",
                "hour_cos",
                "weekday_sin",
                "weekday_cos",
                "is_weekend",
            ))
            self.assertEqual(prepared.target_idx, 0)
            self.assertEqual(prepared.aux_idx, (1, 2, 3, 4, 5))

            generator = torch.Generator().manual_seed(prepared.seed)
            runtime = runner._build_urbanev_runtime_data(prepared, generator)
            for loader in (
                runtime.train_data,
                runtime.val_data,
                runtime.test_data,
            ):
                self.assertIs(loader.dataset.bundle, runtime.backend)
            self.assertEqual(
                runtime.preprocessing["loader_kind"],
                "urbanev_m1_temporal_region",
            )
            x, y = next(iter(runtime.val_data))
            model = runner._build_model(prepared, runtime).eval()
            with torch.no_grad():
                prediction, moe_loss, state_source = model(
                    x,
                    return_state_source=True,
                )
            self.assertEqual(x.shape, (4, 12, 6))
            self.assertEqual(y.shape, (4, 1))
            self.assertEqual(prediction.shape, (4, 1, 1))
            self.assertEqual(state_source.shape, (4, 28))
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertTrue(torch.isfinite(moe_loss))
            self.assertTrue(torch.isfinite(state_source).all())
            adapted = runner._prediction_for_loss(
                prediction,
                y,
                task_mode=runner.TARGET_EXOGENOUS,
            )
            self.assertEqual(adapted.shape, y.shape)

            observed = {}

            def one_batch(model, data):
                model.eval()
                batch_x, batch_y = next(iter(data))
                with torch.no_grad():
                    batch_prediction, batch_moe, batch_state = model(
                        batch_x,
                        return_state_source=True,
                    )
                loss_prediction = runner._prediction_for_loss(
                    batch_prediction,
                    batch_y,
                    task_mode=runner.TARGET_EXOGENOUS,
                )
                observed.update({
                    "x": tuple(batch_x.shape),
                    "y": tuple(batch_y.shape),
                    "prediction": tuple(batch_prediction.shape),
                    "state_source": tuple(batch_state.shape),
                    "finite": bool(
                        torch.isfinite(batch_prediction).all()
                        and torch.isfinite(batch_state).all()
                    ),
                })
                mse = float(torch.mean((loss_prediction - batch_y) ** 2))
                mae = float(torch.mean(torch.abs(loss_prediction - batch_y)))
                return {
                    "mse": mse,
                    "mae": mae,
                    "num_elements": int(batch_y.numel()),
                    "num_batches": 1,
                }

            def engineering_train(model, data, *args, **kwargs):
                metrics = one_batch(model, data)
                return {
                    **metrics,
                    "objective_mean_batches": metrics["mse"],
                    "auxiliary_mean_batches": 0.0,
                }

            def engineering_evaluate(model, data, *args, **kwargs):
                return one_batch(model, data)

            stable_git = {"commit": "test", "dirty": False, "status": []}
            with mock.patch.object(
                runner, "_build_runtime_data", return_value=runtime
            ), mock.patch.object(
                runner, "environment_metadata", return_value=_runtime_metadata()
            ), mock.patch.object(
                runner, "git_metadata", return_value=stable_git
            ), mock.patch.object(
                runner, "train_one_epoch", side_effect=engineering_train
            ), mock.patch.object(
                runner, "evaluate", side_effect=engineering_evaluate
            ):
                metrics = runner.main(
                    self._urbanev_args(data_root, artifact_root)
                )

            parent = (
                artifact_root
                / runner.ENHANCED_IMPLEMENTATION_VARIANT
                / "UrbanEV"
                / "target_exogenous"
                / "volume"
                / "horizon_3"
                / "fold_1"
                / "seed_123"
            )
            final_runs = [
                path for path in parent.iterdir()
                if not path.name.startswith(".")
            ]
            staging_runs = [
                path for path in parent.iterdir()
                if path.name.startswith(".") and path.name.endswith(".staging")
            ]
            self.assertEqual(len(final_runs), 1)
            self.assertEqual(staging_runs, [])
            run_dir = final_runs[0]
            self.assertEqual(metrics["artifact_horizon"], 3)
            self.assertEqual(metrics["model_pred_len"], 1)
            self.assertEqual(metrics["label_horizon"], 3)
            self.assertEqual(observed, {
                "x": (4, 12, 6),
                "y": (4, 1),
                "prediction": (4, 1, 1),
                "state_source": (4, 28),
                "finite": True,
            })
            self.assertFalse((run_dir / ".run.lock").exists())
            runner.verify_checksums(run_dir)
            self.assertIn("OK", runner.verify_checksums_with_sha256sum(run_dir))

            config = json.loads(
                (run_dir / "config.resolved.json").read_text(encoding="utf-8")
            )
            dataset = config["scientific_config"]["dataset"]
            experiment = config["scientific_config"]["experiment"]
            self.assertEqual(dataset["label_horizon"], 3)
            self.assertEqual(dataset["model_pred_len"], 1)
            self.assertEqual(dataset["artifact_horizon"], 3)
            self.assertEqual(experiment["artifact_horizon"], 3)
            checkpoint = torch.load(run_dir / "last.pt", map_location="cpu")
            self.assertEqual(
                checkpoint["resolved_config"]["scientific_config"]["dataset"]["fold"],
                1,
            )
            fingerprint = json.loads(
                (run_dir / "data_fingerprint.json").read_text(encoding="utf-8")
            )
            for field in (
                "data_fingerprint",
                "preprocessing_state_fingerprint",
                "schema_fingerprint",
                "timestamp_order_sha256",
                "node_order_sha256",
                "split_identity",
            ):
                self.assertIn(field, fingerprint)
            rows = summary.load_completed_runs(
                artifact_root,
                implementation_variant=runner.ENHANCED_IMPLEMENTATION_VARIANT,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label_horizon"], 3)
            self.assertEqual(rows[0]["fold"], "1")

    def test_urbanev_protocol_is_rejected_before_artifact_creation(self):
        data_root = runner.ROOT / "data" / "UrbanEV" / "data"
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "artifacts"
            cases = (
                ("seq_len", 11, "fixed at 12"),
                ("pred_len", 2, "fixed at 1"),
                ("label_horizon", 5, "one of"),
                ("fold", "7", "fold must be in"),
                ("target", "occupancy", "target is fixed"),
            )
            for field, value, message in cases:
                with self.subTest(field=field):
                    args = self._urbanev_args(data_root, artifact_root)
                    setattr(args, field, value)
                    if field == "label_horizon":
                        args.horizon = None
                    with self.assertRaisesRegex(ValueError, message):
                        runner.prepare_args(args)
                    self.assertFalse(artifact_root.exists())

            f0 = self._urbanev_args(data_root, artifact_root)
            f0.feature_preset = "F0"
            with self.assertRaisesRegex(ValueError, "requires use_teb=False"):
                runner.prepare_args(f0)
            self.assertFalse(artifact_root.exists())

            for gamma in (0.0, 1e-2, -1e-3, float("nan"), float("inf")):
                with self.subTest(teb_gamma_init=gamma):
                    args = self._urbanev_args(data_root, artifact_root)
                    args.teb_gamma_init = gamma
                    with self.assertRaisesRegex(ValueError, "fixed at 1e-3"):
                        runner.prepare_args(args)
                    self.assertFalse(artifact_root.exists())

    def test_prediction_loss_adapter_rejects_all_broadcasting(self):
        target_prediction = torch.zeros(2, 1, 1)
        target = torch.zeros(2, 1)
        adapted = runner._prediction_for_loss(
            target_prediction,
            target,
            task_mode=runner.TARGET_EXOGENOUS,
        )
        self.assertEqual(adapted.shape, target.shape)
        with self.assertRaisesRegex(RuntimeError, "must be \[B,H,1\]"):
            runner._prediction_for_loss(
                target_prediction,
                target.unsqueeze(-1),
                task_mode=runner.TARGET_EXOGENOUS,
            )

        parallel_prediction = torch.zeros(2, 1, 3)
        parallel_target = torch.zeros(2, 1, 3)
        self.assertEqual(
            runner._prediction_for_loss(
                parallel_prediction,
                parallel_target,
                task_mode=runner.PARALLEL_MULTIVARIATE,
            ).shape,
            parallel_target.shape,
        )
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            runner._prediction_for_loss(
                parallel_prediction,
                torch.zeros(2, 1, 1),
                task_mode=runner.PARALLEL_MULTIVARIATE,
            )

    def test_atomic_publication_fault_windows_and_success(self):
        class ClosedTranscript:
            def __init__(self):
                self.finalized = False

            def finalize(self):
                self.finalized = True
                return self

        def make_staging(case_root, run_id):
            artifact_root = case_root / "artifacts"
            parent = (
                artifact_root
                / runner.ENHANCED_IMPLEMENTATION_VARIANT
                / "toy"
                / "target_exogenous"
                / "b"
                / "horizon_2"
                / "fold_official"
                / "seed_123"
            )
            parent.mkdir(parents=True)
            staging = parent / f".{run_id}.staging"
            final = parent / run_id
            staging.mkdir()
            for name in runner.ENHANCED_CHECKSUM_FILES:
                (staging / name).write_bytes(f"{name}\n".encode("utf-8"))
            runner.atomic_write_json(
                staging / "manifest.json",
                {"status": "running", "run_id": run_id},
            )
            lock = runner.RunLock(staging).acquire()
            completed = {
                "status": "completed",
                "run_id": run_id,
                "artifact_dir": str(final),
            }
            return artifact_root, staging, final, lock, completed

        fault_stages = (
            "after_completed_manifest",
            "after_checksums",
            "before_atomic_rename",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in fault_stages:
                with self.subTest(stage=stage):
                    artifact_root, staging, final, lock, completed = make_staging(
                        root / stage,
                        stage,
                    )

                    def inject(observed_stage):
                        if observed_stage == stage:
                            raise RuntimeError(f"fault:{stage}")

                    try:
                        with mock.patch.object(
                            runner,
                            "_artifact_fault_point",
                            side_effect=inject,
                        ):
                            with self.assertRaisesRegex(RuntimeError, f"fault:{stage}"):
                                runner._publish_enhanced_artifact(
                                    staging_dir=staging,
                                    final_dir=final,
                                    transcript=ClosedTranscript(),
                                    run_lock=lock,
                                    manifest_path=staging / "manifest.json",
                                    completed_manifest=completed,
                                )
                    finally:
                        lock.release()
                    self.assertFalse(final.exists())
                    self.assertTrue(staging.is_dir())
                    staged_manifest = json.loads(
                        (staging / "manifest.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(staged_manifest["status"], "completed")
                    if stage == "after_completed_manifest":
                        self.assertFalse((staging / "checksums.sha256").exists())
                    else:
                        self.assertTrue((staging / "checksums.sha256").is_file())
                    self.assertEqual(summary.load_completed_runs(
                        artifact_root,
                        implementation_variant=runner.ENHANCED_IMPLEMENTATION_VARIANT,
                    ), [])

            verifier_cases = (
                ("python_verify", "verify_checksums"),
                ("sha256sum_verify", "verify_checksums_with_sha256sum"),
            )
            for case, function_name in verifier_cases:
                with self.subTest(case=case):
                    artifact_root, staging, final, lock, completed = make_staging(
                        root / case,
                        case,
                    )
                    try:
                        with mock.patch.object(
                            runner,
                            function_name,
                            side_effect=RuntimeError(f"fault:{case}"),
                        ):
                            with self.assertRaisesRegex(RuntimeError, f"fault:{case}"):
                                runner._publish_enhanced_artifact(
                                    staging_dir=staging,
                                    final_dir=final,
                                    transcript=ClosedTranscript(),
                                    run_lock=lock,
                                    manifest_path=staging / "manifest.json",
                                    completed_manifest=completed,
                                )
                    finally:
                        lock.release()
                    self.assertFalse(final.exists())
                    self.assertTrue(staging.is_dir())
                    self.assertEqual(summary.load_completed_runs(
                        artifact_root,
                        implementation_variant=runner.ENHANCED_IMPLEMENTATION_VARIANT,
                    ), [])

            artifact_root, staging, final, lock, completed = make_staging(
                root / "success",
                "success",
            )
            transcript = ClosedTranscript()
            published = runner._publish_enhanced_artifact(
                staging_dir=staging,
                final_dir=final,
                transcript=transcript,
                run_lock=lock,
                manifest_path=staging / "manifest.json",
                completed_manifest=completed,
            )
            self.assertTrue(transcript.finalized)
            self.assertEqual(published, final)
            self.assertFalse(staging.exists())
            self.assertTrue(final.is_dir())
            self.assertFalse((final / ".run.lock").exists())
            before = {
                name: runner.sha256_file(final / name)
                for name in runner.ENHANCED_CHECKSUM_FILES
            }
            runner.verify_checksums(final)
            runner.verify_checksums_with_sha256sum(final)
            after = {
                name: runner.sha256_file(final / name)
                for name in runner.ENHANCED_CHECKSUM_FILES
            }
            self.assertEqual(before, after)

    def test_subprocess_cli_records_replayable_command_and_real_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "toy.csv"
            artifact_root = root / "artifacts"
            self._write_dataset(data_path)
            cli = [
                "--implementation_variant", runner.ENHANCED_IMPLEMENTATION_VARIANT,
                "--data", str(data_path),
                "--dataset_id", "toy_cli",
                "--artifact_root", str(artifact_root),
                "--device", "cpu",
                "--num_threads", "1",
                "--progress", "false",
                "--seed", "321",
                "--seq_len", "4",
                "--pred_len", "2",
                "--n_block", "1",
                "--alpha", "0",
                "--mix_layer_num", "0",
                "--mix_layer_scale", "2",
                "--patch", "4",
                "--norm", "true",
                "--layernorm", "false",
                "--dropout", "0",
                "--train_epochs", "1",
                "--batch_size", "8",
                "--learning_rate", "0.001",
                "--feature_type", "M",
                "--target", "all",
                "--task_mode", "parallel_multivariate",
                "--target_idx", "0",
                "--aux_idx",
                "--feature_names", "a", "b",
                "--target_feature_name", "a",
                "--aux_feature_names",
                "--schema_fingerprint", "synthetic-parallel-v1",
                "--fold", "official",
                "--horizon", "2",
                "--ablation_id", "M2",
                "--use_pmcr", "false",
                "--use_teb", "true",
                "--teb_context_dim", "4",
                "--teb_heads", "2",
                "--teb_dropout", "0",
                "--teb_gamma_init", "0.001",
            ]
            executable = runner.sys.executable
            main_path = runner.ROOT / "main.py"
            command = [executable, "-B", str(main_path), *cli]
            environment = dict(runner.os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                command,
                cwd=runner.ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            parent = (
                artifact_root
                / runner.ENHANCED_IMPLEMENTATION_VARIANT
                / "toy_cli"
                / "parallel_multivariate"
                / "all"
                / "horizon_2"
                / "fold_official"
                / "seed_321"
            )
            runs = [
                path for path in parent.iterdir()
                if not path.name.startswith(".")
            ]
            self.assertEqual(len(runs), 1)
            run_dir = runs[0]
            invocation = json.loads(
                (run_dir / "sys.argv.json").read_text(encoding="utf-8")
            )["invocations"][-1]["argv"]
            self.assertEqual(invocation, [str(main_path), *cli])
            expected_command = runner.shlex.join(
                [executable, str(main_path), *cli]
            )
            command_lines = [
                line for line in (run_dir / "command.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line and not line.startswith("#")
            ]
            self.assertEqual(command_lines, [expected_command])
            stdout_text = (run_dir / "stdout.log").read_text(encoding="utf-8")
            stderr_text = (run_dir / "stderr.log").read_text(encoding="utf-8")
            train_text = (run_dir / "train.log").read_text(encoding="utf-8")
            self.assertIn("completed run=", stdout_text)
            self.assertIn("completed run=", completed.stdout)
            self.assertTrue(stderr_text)
            self.assertTrue(train_text)
            self.assertIn("completed run=", train_text)
            runner.verify_checksums(run_dir)
            self.assertIn("OK", runner.verify_checksums_with_sha256sum(run_dir))



if __name__ == "__main__":
    unittest.main()
