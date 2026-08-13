import shutil
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


EXPECTED = {
    "ECL.sh": {
        "data": "electricity.csv",
        "model_id": "electricity",
        "data_name": "electricity",
        "seq_len": "512",
        "alpha": "0.0",
        "patch": "16",
        "norm": "True",
        "layernorm": "False",
        "epochs": "20",
        "batch_size": "128",
        "learning_rate": "0.0003",
    },
    "ETTh1.sh": {
        "data": "ETTh1.csv",
        "model_id": "ETTh1",
        "data_name": "ETTh1",
        "seq_len": "512",
        "alpha": "0.0",
        "patch": "16",
        "norm": "True",
        "layernorm": "True",
        "epochs": "10",
        "batch_size": "128",
        "learning_rate": "0.00005",
    },
    "ETTh2.sh": {
        "data": "ETTh2.csv",
        "model_id": "ETTh2",
        "data_name": "ETTh2",
        "seq_len": "512",
        "alpha": "1.0",
        "patch": "4",
        "norm": "False",
        "layernorm": "False",
        "epochs": "10",
        "batch_size": "128",
        "learning_rate": "0.00005",
    },
    "ETTm1.sh": {
        "data": "ETTm1.csv",
        "model_id": "ETTm1",
        "data_name": "ETTm1",
        "seq_len": "512",
        "alpha": "0.0",
        "patch": "16",
        "norm": "True",
        "layernorm": "True",
        "epochs": "10",
        "batch_size": "128",
        "learning_rate": "0.00003",
    },
    "ETTm2.sh": {
        "data": "ETTm2.csv",
        "model_id": "ETTm2",
        "data_name": "ETTm2",
        "seq_len": "512",
        "alpha": "0.0",
        "patch": "8",
        "norm": "True",
        "layernorm": "True",
        "epochs": "10",
        "batch_size": "128",
        "learning_rate": "0.00001",
    },
    "Exchange.sh": {
        "data": "exchange_rate.csv",
        "model_id": "exchange_rate",
        "data_name": "exchange_rate",
        "seq_len": "96",
        "alpha": "0.0",
        "patch": "4",
        "norm": "True",
        "layernorm": "True",
        "epochs": "10",
        "batch_size": "512",
        "learning_rate": "0.0003",
    },
    "Solar_AL.sh": {
        "data": "solar_AL.txt",
        "model_id": "solar_AL",
        "data_name": "solar_AL",
        "seq_len": "512",
        "alpha": "1.0",
        "patch": "8",
        "norm": "True",
        "layernorm": "True",
        "epochs": "10",
        "batch_size": "128",
        "learning_rate": "0.00002",
    },
    "Traffic.sh": {
        "data": "traffic.csv",
        "model_id": "traffic",
        "data_name": "traffic",
        "seq_len": "512",
        "alpha": "0.0",
        "patch": "16",
        "norm": "True",
        "layernorm": "False",
        "epochs": "20",
        "batch_size": "32",
        "learning_rate": "0.00008",
    },
    "Weather.sh": {
        "data": "weather.csv",
        "model_id": "weather",
        "data_name": "weather",
        "seq_len": "512",
        "alpha": "0.0",
        "patch": "16",
        "norm": "True",
        "layernorm": "True",
        "epochs": "10",
        "batch_size": "128",
        "learning_rate": "0.00005",
    },
}


def _script_bytes(name):
    return (SCRIPTS_DIR / name).read_bytes()


def _script_text(name):
    return _script_bytes(name).decode("utf-8")


def _find_working_bash():
    candidates = []
    path_bash = shutil.which("bash")
    if path_bash:
        candidates.append(Path(path_bash))

    git = shutil.which("git")
    if git:
        git_root = Path(git).resolve().parent.parent
        candidates.extend((git_root / "bin" / "bash.exe", git_root / "usr" / "bin" / "bash.exe"))

    for candidate in candidates:
        if not candidate.is_file():
            continue
        probe = subprocess.run(
            [str(candidate), "--version"],
            text=True,
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0 and "GNU bash" in probe.stdout:
            return str(candidate)
    return None


class ScriptTests(unittest.TestCase):
    def test_expected_experiment_scripts_are_present(self):
        actual = {path.name for path in SCRIPTS_DIR.glob("*.sh")}
        self.assertEqual(actual, set(EXPECTED))

    def test_scripts_are_strict_path_independent_and_lf_only(self):
        for name in EXPECTED:
            with self.subTest(script=name):
                raw = _script_bytes(name)
                text = raw.decode("utf-8")

                self.assertNotIn(b"\r\n", raw)
                self.assertTrue(text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
                self.assertIn(
                    'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
                    text,
                )
                self.assertIn('PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"', text)
                self.assertIn('cd "${PROJECT_ROOT}"', text)
                self.assertIn('artifact_root="${PROJECT_ROOT}/artifacts"', text)
                self.assertIn('--data "${PROJECT_ROOT}/data/${data_path_name}"', text)
                self.assertIn('--artifact_root "${artifact_root}"', text)
                self.assertIn('--device "cuda:0"', text)
                self.assertIn(
                    '--implementation_variant "AMD-paper-norm-wd-ddi-v1"', text
                )
                self.assertIn('--weight_decay 0.0000001', text)

                self.assertNotIn("--checkpoint_dir", text)
                self.assertNotIn("../checkpoints", text)
                self.assertNotIn("--result_path", text)
                self.assertNotIn("result.csv", text)

    def test_scripts_loop_over_configurable_seeds_and_all_horizons(self):
        for name in EXPECTED:
            with self.subTest(script=name):
                text = _script_text(name)

                self.assertIn('python_bin="${PYTHON_BIN:-python}"', text)
                self.assertIn('read -r -a seeds <<< "${SEEDS:-2024}"', text)
                self.assertIn('for seed in "${seeds[@]}"; do', text)
                self.assertIn("for pred_len in 96 192 336 720; do", text)
                self.assertIn('"${python_bin}" -u "${PROJECT_ROOT}/main.py"', text)
                self.assertIn('--seed "${seed}"', text)
                self.assertIn('--pred_len "${pred_len}"', text)
                self.assertEqual(text.count("\ndone\n"), 1)
                self.assertEqual(text.count("  done\n"), 1)

    def test_scripts_preserve_dataset_hyperparameters_and_paper_weight_decay(self):
        for name, expected in EXPECTED.items():
            with self.subTest(script=name):
                text = _script_text(name)
                expected_fragments = (
                    f'data_path_name="{expected["data"]}"',
                    f'model_id_name="{expected["model_id"]}"',
                    f'data_name="{expected["data_name"]}"',
                    f'seq_len={expected["seq_len"]}',
                    '--dataset_id "${data_name}"',
                    '--name "${model_id_name}"',
                    '--seq_len "${seq_len}"',
                    "--n_block 1",
                    f'--alpha {expected["alpha"]}',
                    "--mix_layer_num 3",
                    "--mix_layer_scale 2",
                    f'--patch {expected["patch"]}',
                    f'--norm {expected["norm"]}',
                    f'--layernorm {expected["layernorm"]}',
                    "--dropout 0.1",
                    f'--train_epochs {expected["epochs"]}',
                    f'--batch_size {expected["batch_size"]}',
                    f'--learning_rate {expected["learning_rate"]}',
                )
                for fragment in expected_fragments:
                    self.assertIn(fragment, text, f"{name} is missing or changed: {fragment}")

    def test_all_scripts_are_valid_bash_when_bash_is_available(self):
        bash = _find_working_bash()
        if bash is None:
            self.skipTest("bash is not available on PATH")

        for name in EXPECTED:
            with self.subTest(script=name):
                completed = subprocess.run(
                    [bash, "-n"],
                    input=_script_text(name),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, f"{name}: {completed.stderr}")


if __name__ == "__main__":
    unittest.main()
