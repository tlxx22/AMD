import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

import main as runner


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


if __name__ == "__main__":
    unittest.main()
