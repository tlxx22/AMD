# coding=utf-8
"""Train the reproducible AMD paper-close interpretation variant."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm

from models.tsAMD import AMD
from utils.dataloader import CustomDataLoader
from utils.general import capture_rng_state, restore_rng_state, set_seed


IMPLEMENTATION_VARIANT = "AMD-paper-norm-wd-ddi-v1"
SCHEMA_VERSION = 1
PAPER_WEIGHT_DECAY = 1e-7
METRIC_SPACE = "train-standardized"


def str2bool(value):
    """Parse explicit boolean strings without Python's ``bool('False')`` trap."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value (true/false, yes/no, on/off, 1/0), got {value!r}"
    )


def infer_extension(dataset_name):
    name = dataset_name.lower()
    if name.startswith("solar"):
        return "txt"
    if name.startswith("pems"):
        return "npz"
    return "csv"


def parse_args(argv=None):
    dataset = "ETTh1"
    parser = argparse.ArgumentParser(
        description=(
            "Train AMD-paper-norm-wd-ddi-v1: MDM(U)->DDI wiring, true "
            "last-dimension LayerNorm at flag-controlled entries, paper DDI "
            "hidden width and weight decay, with reproducible bookkeeping."
        )
    )
    parser.add_argument("--implementation_variant", default=IMPLEMENTATION_VARIANT,
                        choices=[IMPLEMENTATION_VARIANT])
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--device", default="cuda:0",
                        help="explicit torch device; formal runs should use cuda:0")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="number of PyTorch CPU threads")
    parser.add_argument("--progress", type=str2bool, default=True,
                        help="show tqdm progress bars")

    parser.add_argument(
        "--data", type=str,
        default=str(ROOT / "data" / f"{dataset}.{infer_extension(dataset)}"),
        help="dataset path",
    )
    parser.add_argument("--dataset_id", default=None,
                        help="stable dataset identifier; defaults to the data filename stem")
    parser.add_argument("--feature_type", default="M", choices=["S", "M", "MS"])
    parser.add_argument("--target", default="OT", help="target column for S/MS tasks")
    parser.add_argument("--name", default=dataset,
                        help="human-readable experiment name (metadata only)")

    parser.add_argument("--seq_len", type=int, default=720)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--mix_layer_num", type=int, default=3)
    parser.add_argument("--mix_layer_scale", type=int, default=2)
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--norm", type=str2bool, default=True, help="enable RevIN")
    parser.add_argument(
        "--layernorm", type=str2bool, default=True,
        help=(
            "enable torch.nn.LayerNorm over the final look-back dimension at "
            "the MDM and DDI entries controlled by the released switch"
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--train_epochs", type=int, default=10,
                        help="target total epoch count when resuming")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--weight_decay", type=float, default=PAPER_WEIGHT_DECAY)

    parser.add_argument(
        "--artifact_root", type=str, default=None,
        help="root for isolated run artifacts (default: PROJECT_ROOT/artifacts)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="resume an interrupted run from its directory or last.pt",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="deprecated alias for --artifact_root",
    )
    parser.add_argument(
        "--result_path", type=str, default=None,
        help="deprecated; each run now writes an isolated metrics.json",
    )
    return parser.parse_args(argv)


def _absolute_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _safe_component(value, field_name):
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not sanitized:
        raise ValueError(f"{field_name} has no filesystem-safe characters: {value!r}")
    return sanitized


def prepare_args(args):
    """Normalize paths and reject configurations outside this baseline contract."""

    if args.implementation_variant != IMPLEMENTATION_VARIANT:
        raise ValueError(f"unsupported implementation variant: {args.implementation_variant}")
    if args.checkpoint_dir:
        if args.artifact_root:
            raise ValueError("use only one of --artifact_root and --checkpoint_dir")
        warnings.warn("--checkpoint_dir is deprecated; use --artifact_root", stacklevel=2)
        args.artifact_root = args.checkpoint_dir
    if args.artifact_root is None:
        args.artifact_root = str(ROOT / "artifacts")
    if args.result_path:
        warnings.warn(
            "--result_path is deprecated and ignored; this run writes metrics.json in its artifact directory",
            stacklevel=2,
        )

    args.data = str(_absolute_path(args.data))
    args.artifact_root = str(_absolute_path(args.artifact_root))
    args.resume = str(_absolute_path(args.resume)) if args.resume else None
    args.dataset_id = _safe_component(args.dataset_id or Path(args.data).stem, "dataset_id")
    args.name = str(args.name).strip() or args.dataset_id

    positive_ints = {
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "n_block": args.n_block,
        "patch": args.patch,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "num_threads": args.num_threads,
    }
    for name, value in positive_ints.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if args.batch_size < 2 and args.n_block > 0 and args.seq_len > args.patch:
        raise ValueError(
            "batch_size must be at least 2 because DDI's retained internal "
            "BatchNorm1d is active when seq_len > patch"
        )
    if (
        isinstance(args.mix_layer_num, bool)
        or not isinstance(args.mix_layer_num, int)
        or args.mix_layer_num < 0
    ):
        raise ValueError("mix_layer_num must be a non-negative integer")
    if (
        isinstance(args.mix_layer_scale, bool)
        or not isinstance(args.mix_layer_scale, int)
        or args.mix_layer_scale < 1
    ):
        raise ValueError("mix_layer_scale must be a positive integer")
    if not math.isfinite(args.alpha) or args.alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if (
        not math.isfinite(args.weight_decay)
        or args.weight_decay != PAPER_WEIGHT_DECAY
    ):
        raise ValueError(
            f"{IMPLEMENTATION_VARIANT} fixes weight_decay at {PAPER_WEIGHT_DECAY:g}; "
            f"got {args.weight_decay:g}"
        )
    return args


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_run_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(root=ROOT):
    """Hash the executable Python source, independent of Git staging state."""

    root = Path(root)
    files = [root / "main.py"]
    files.extend((root / "models").rglob("*.py"))
    files.extend((root / "utils").glob("*.py"))
    files = sorted(files, key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _run_readonly(command):
    try:
        result = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_metadata():
    status = _run_readonly(["git", "status", "--porcelain=v1", "--untracked-files=normal"])
    return {
        "commit": _run_readonly(["git", "rev-parse", "HEAD"]),
        "dirty": None if status is None else bool(status),
        "status": [] if not status else status.splitlines(),
    }


def environment_metadata(device):
    def package_version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    driver = _run_readonly(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scipy": package_version("scipy"),
        "scikit_learn": package_version("scikit-learn"),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "torchaudio": package_version("torchaudio"),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_driver": driver.splitlines()[0] if driver else None,
        "device": str(device),
        "device_name": None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)
    return result


def stable_hash(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path, value):
    serialized = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    _atomic_write_text(path, serialized + "\n")


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_history(path, history):
    lines = [
        json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False)
        for item in history
    ]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


class RunLock:
    """Cross-platform, process-scoped exclusive lock for one run directory.

    The lock is owned by the open file handle, so the operating system releases
    it automatically if the process exits or crashes.  The small lock file may
    remain on disk; its presence alone never means a run is locked.
    """

    def __init__(self, run_dir):
        self.path = Path(run_dir) / ".run.lock"
        self._handle = None

    def acquire(self):
        if self._handle is not None:
            raise RuntimeError("run lock is already acquired")
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                f"run is already locked by another process: {self.path.parent}"
            ) from exc
        self._handle = handle
        return self

    def release(self):
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback_value):
        self.release()
        return False


def _resolve_device(specification):
    try:
        device = torch.device(specification)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid device {specification!r}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {specification!r} was requested, but CUDA is unavailable; "
                "pass --device cpu explicitly for a CPU diagnostic run"
            )
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; found {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    return device


def _assert_prediction_batch(outputs, targets):
    if outputs.shape != targets.shape:
        raise RuntimeError(
            f"prediction/target shape mismatch: {tuple(outputs.shape)} != {tuple(targets.shape)}"
        )
    if outputs.numel() == 0:
        raise RuntimeError("model produced an empty prediction batch")
    if not torch.isfinite(outputs).all():
        raise FloatingPointError("model predictions contain NaN or Inf")
    if not torch.isfinite(targets).all():
        raise FloatingPointError("targets contain NaN or Inf")


def _accumulate_errors(outputs, targets, accumulator):
    difference = (outputs - targets).double()
    accumulator["sse"] += difference.square().sum().item()
    accumulator["sae"] += difference.abs().sum().item()
    accumulator["num_elements"] += targets.numel()
    accumulator["num_batches"] += 1


def _finalize_errors(accumulator):
    count = accumulator["num_elements"]
    if count <= 0:
        raise RuntimeError("cannot compute metrics from an empty data loader")
    metrics = {
        "mse": accumulator["sse"] / count,
        "mae": accumulator["sae"] / count,
        "num_elements": count,
        "num_batches": accumulator["num_batches"],
    }
    if not math.isfinite(metrics["mse"]) or not math.isfinite(metrics["mae"]):
        raise FloatingPointError(f"non-finite metrics: {metrics}")
    return metrics


def evaluate(model, data_loader, device, description="Eval", show_progress=True):
    """Evaluate global element-wise MSE/MAE, including a final partial batch."""

    model.eval()
    accumulator = {"sse": 0.0, "sae": 0.0, "num_elements": 0, "num_batches": 0}
    iterator = tqdm(data_loader, desc=description, disable=not show_progress, leave=False)
    with torch.no_grad():
        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs, _ = model(batch_x)
            _assert_prediction_batch(outputs, batch_y)
            _accumulate_errors(outputs, batch_y, accumulator)
    return _finalize_errors(accumulator)


def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch,
                    total_epochs, show_progress=True):
    model.train()
    accumulator = {"sse": 0.0, "sae": 0.0, "num_elements": 0, "num_batches": 0}
    objective_sum = 0.0
    auxiliary_sum = 0.0
    iterator = tqdm(
        data_loader, desc=f"Train {epoch}/{total_epochs}",
        disable=not show_progress, leave=False,
    )
    for batch_x, batch_y in iterator:
        if batch_x.shape[0] < 2 and getattr(model, "_uses_batch_norm", False):
            raise RuntimeError(
                "DDI's retained internal BatchNorm1d cannot train on a batch of size 1; "
                "increase the dataset/batch size or keep training drop_last=True"
            )
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs, auxiliary_loss = model(batch_x)
        _assert_prediction_batch(outputs, batch_y)
        if not torch.is_tensor(auxiliary_loss) or auxiliary_loss.numel() != 1:
            raise RuntimeError("AMD auxiliary loss must be a scalar tensor")
        prediction_loss = criterion(outputs, batch_y)
        loss = prediction_loss + auxiliary_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("training objective contains NaN or Inf")
        loss.backward()
        optimizer.step()

        objective_sum += loss.detach().item()
        auxiliary_sum += auxiliary_loss.detach().item()
        _accumulate_errors(outputs.detach(), batch_y, accumulator)
        iterator.set_postfix(loss=f"{objective_sum / accumulator['num_batches']:.6g}")

    metrics = _finalize_errors(accumulator)
    metrics["objective_mean_batches"] = objective_sum / metrics["num_batches"]
    metrics["auxiliary_mean_batches"] = auxiliary_sum / metrics["num_batches"]
    return metrics


def should_update_best(candidate_mse, best_mse):
    return math.isfinite(candidate_mse) and candidate_mse < best_mse


def _scientific_config(args, data_sha256, source_sha256, preprocessing, device,
                       environment):
    """Return fields that must match exactly when resuming a run."""

    return {
        "implementation_variant": IMPLEMENTATION_VARIANT,
        "source_sha256": source_sha256,
        "dataset": {
            "id": args.dataset_id,
            "sha256": data_sha256,
            "feature_type": args.feature_type,
            "target": args.target,
            "preprocessing": preprocessing,
        },
        "model": {
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "n_block": args.n_block,
            "alpha": args.alpha,
            "mix_layer_num": args.mix_layer_num,
            "mix_layer_scale": args.mix_layer_scale,
            "patch": args.patch,
            "norm": args.norm,
            "layernorm_flag": args.layernorm,
            "entry_normalization_impl": "torch_layernorm_last_dim_sequence",
            "entry_normalization_scope": "mdm_and_ddi_entries_controlled_by_layernorm_flag",
            "ddi_internal_normalization_impl": (
                "released_batchnorm1d_norm1_and_norm2_when_alpha_gt_0"
            ),
            "ddi_hidden_rule": "max(32,2**ceil(log2(feature_count)))_when_alpha_gt_0",
            "module_connection": "X->MDM(U)->DDI; AMS_selector<-U",
            "selector_mode": "horizon_shared_dense_emphasis",
            "dropout": args.dropout,
        },
        "optimization": {
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "train_drop_last": True,
            "validation_drop_last": False,
        },
        "execution": {
            "seed": args.seed,
            "device": str(device),
            "num_threads": args.num_threads,
            "metric_space": METRIC_SPACE,
        },
        "runtime_contract": {
            key: environment.get(key)
            for key in (
                "python", "numpy", "pandas", "scipy", "scikit_learn",
                "torch", "torch_cuda", "cudnn", "device_name",
                "cudnn_benchmark", "cudnn_deterministic",
                "deterministic_algorithms", "cuda_matmul_allow_tf32",
                "cudnn_allow_tf32",
            )
        },
    }


def _resolved_config(args, scientific, config_hash, run_dir, source, environment):
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_variant": IMPLEMENTATION_VARIANT,
        "config_hash": config_hash,
        "scientific_config": scientific,
        "run": {
            "name": args.name,
            "train_epochs": args.train_epochs,
            "data_path": args.data,
            "artifact_root": args.artifact_root,
            "run_dir": str(run_dir),
            "resume": args.resume,
        },
        "source": source,
        "environment": environment,
    }


def _checkpoint_common(resolved_config, config_hash, data_sha256, preprocessing):
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_variant": IMPLEMENTATION_VARIANT,
        "config_hash": config_hash,
        "data_sha256": data_sha256,
        "resolved_config": resolved_config,
        "preprocessing": preprocessing,
    }


def _cpu_state_dict(state_dict):
    """Snapshot a module state without retaining live parameter storage."""

    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def _load_resume_checkpoint(run_dir, config_hash, data_sha256, train_epochs):
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.resolved.json"
    last_path = run_dir / "last.pt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"resume manifest not found: {manifest_path}")
    if not last_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"resume resolved config not found: {config_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"resume metadata is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(previous_config, dict):
        raise RuntimeError("resume manifest and resolved config must be JSON objects")
    if manifest.get("status") == "completed":
        raise RuntimeError("completed runs are immutable and cannot be resumed")
    if manifest.get("status") not in {"running", "failed"}:
        raise RuntimeError(f"unsupported resume manifest status: {manifest.get('status')!r}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("resume manifest schema version mismatch")
    if manifest.get("implementation_variant") != IMPLEMENTATION_VARIANT:
        raise RuntimeError(f"resume variant does not match {IMPLEMENTATION_VARIANT}")
    if manifest.get("run_id") != run_dir.name:
        raise RuntimeError("resume manifest run_id/path mismatch")
    manifest_artifact_dir = manifest.get("artifact_dir")
    if (
        not isinstance(manifest_artifact_dir, str)
        or Path(manifest_artifact_dir).resolve() != run_dir.resolve()
    ):
        raise RuntimeError("resume manifest artifact path mismatch")
    if manifest.get("config_hash") != config_hash:
        raise RuntimeError("resume manifest configuration hash mismatch")
    if manifest.get("data_sha256") != data_sha256:
        raise RuntimeError("resume manifest data fingerprint mismatch")
    if previous_config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("resume resolved config schema version mismatch")
    if previous_config.get("implementation_variant") != IMPLEMENTATION_VARIANT:
        raise RuntimeError("resume resolved config variant mismatch")
    if previous_config.get("config_hash") != config_hash:
        raise RuntimeError("resume resolved config hash mismatch")
    previous_scientific = previous_config.get("scientific_config")
    if not isinstance(previous_scientific, dict):
        raise RuntimeError("resume resolved config has no scientific_config object")
    if stable_hash(previous_scientific) != config_hash:
        raise RuntimeError("resume resolved scientific configuration was modified")
    previous_run = previous_config.get("run")
    if not isinstance(previous_run, dict):
        raise RuntimeError("resume resolved config has no run object")
    previous_run_dir = previous_run.get("run_dir")
    if (
        not isinstance(previous_run_dir, str)
        or Path(previous_run_dir).resolve() != run_dir.resolve()
    ):
        raise RuntimeError("resume resolved config run path mismatch")

    # Always deserialize checkpoint tensors onto CPU.  Mapping the whole object
    # to CUDA also maps CPU RNG/DataLoader generator ByteTensors, which makes
    # torch.set_rng_state and Generator.set_state fail on resume.
    checkpoint = torch.load(last_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("resume checkpoint must contain a dictionary")
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema version mismatch")
    if checkpoint.get("implementation_variant") != IMPLEMENTATION_VARIANT:
        raise RuntimeError("checkpoint implementation variant mismatch")
    if checkpoint.get("config_hash") != config_hash:
        raise RuntimeError(
            "resume configuration mismatch; start a new run instead of combining experiments"
        )
    if checkpoint.get("data_sha256") != data_sha256:
        raise RuntimeError("resume data fingerprint mismatch")
    checkpoint_config = checkpoint.get("resolved_config")
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError("resume checkpoint has no resolved configuration")
    checkpoint_scientific = checkpoint_config.get("scientific_config")
    if (
        not isinstance(checkpoint_scientific, dict)
        or stable_hash(checkpoint_scientific) != config_hash
    ):
        raise RuntimeError("resume checkpoint scientific configuration mismatch")
    completed_epoch = int(checkpoint.get("completed_epoch", -1))
    if completed_epoch < 1:
        raise RuntimeError("resume checkpoint has no completed epoch")
    manifest_epoch = int(manifest.get("completed_epoch", 0))
    if manifest_epoch > completed_epoch:
        raise RuntimeError("manifest is ahead of the committed last checkpoint")
    if train_epochs < completed_epoch:
        raise RuntimeError(
            f"train_epochs={train_epochs} is below completed epoch {completed_epoch}"
        )
    history = checkpoint.get("history")
    if not isinstance(history, list) or len(history) != completed_epoch:
        raise RuntimeError("resume checkpoint history does not match completed_epoch")
    if [record.get("epoch") for record in history if isinstance(record, dict)] != list(
        range(1, completed_epoch + 1)
    ):
        raise RuntimeError("resume checkpoint history has invalid epoch numbering")
    if not isinstance(checkpoint.get("model_state"), dict):
        raise RuntimeError("resume checkpoint has no model state")
    if not isinstance(checkpoint.get("optimizer_state"), dict):
        raise RuntimeError("resume checkpoint has no optimizer state")
    if not isinstance(checkpoint.get("best_model_state"), dict):
        raise RuntimeError("resume checkpoint has no committed best-model state")
    best_epoch = checkpoint.get("best_epoch")
    best_mse = checkpoint.get("best_mse")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= completed_epoch
        or isinstance(best_mse, bool)
        or not isinstance(best_mse, (int, float))
        or not math.isfinite(float(best_mse))
    ):
        raise RuntimeError("resume checkpoint has invalid best-validation metadata")
    best_val_metrics = checkpoint.get("best_val_metrics")
    if (
        not isinstance(best_val_metrics, dict)
        or isinstance(best_val_metrics.get("mse"), bool)
        or not isinstance(best_val_metrics.get("mse"), (int, float))
        or not math.isfinite(float(best_val_metrics["mse"]))
        or float(best_val_metrics["mse"]) != float(best_mse)
    ):
        raise RuntimeError("resume checkpoint best-validation metrics are inconsistent")
    generator_state = checkpoint.get("train_generator_state")
    if (
        not torch.is_tensor(generator_state)
        or generator_state.device.type != "cpu"
        or generator_state.dtype != torch.uint8
        or generator_state.ndim != 1
    ):
        raise RuntimeError("resume checkpoint has an invalid train generator state")
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict):
        raise RuntimeError("resume checkpoint has no RNG state")
    torch_cpu_state = rng_state.get("torch_cpu")
    if (
        not torch.is_tensor(torch_cpu_state)
        or torch_cpu_state.device.type != "cpu"
        or torch_cpu_state.dtype != torch.uint8
        or torch_cpu_state.ndim != 1
    ):
        raise RuntimeError("resume checkpoint has an invalid CPU RNG state")
    cuda_states = rng_state.get("torch_cuda")
    if cuda_states is not None and (
        not isinstance(cuda_states, (list, tuple))
        or any(
            not torch.is_tensor(state)
            or state.device.type != "cpu"
            or state.dtype != torch.uint8
            or state.ndim != 1
            for state in cuda_states
        )
    ):
        raise RuntimeError("resume checkpoint has invalid CUDA RNG state")
    return manifest, checkpoint, previous_config


def _run_directory(args):
    if args.resume:
        resume = Path(args.resume)
        if not resume.exists():
            raise FileNotFoundError(f"resume path does not exist: {resume}")
        if resume.is_file():
            if resume.name != "last.pt":
                raise ValueError("a resume file must be the run's last.pt")
            return resume.parent
        if not resume.is_dir():
            raise ValueError(f"resume path must be a run directory or last.pt: {resume}")
        return resume
    parent = (
        Path(args.artifact_root)
        / IMPLEMENTATION_VARIANT
        / args.dataset_id
        / f"sl{args.seq_len}_pl{args.pred_len}"
        / f"seed{args.seed}"
    )
    run_dir = parent / _new_run_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(args):
    args = prepare_args(args)
    device = _resolve_device(args.device)
    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"dataset not found: {data_path}")
    data_sha256 = sha256_file(data_path)
    source_sha256 = source_fingerprint()

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    data_loader = CustomDataLoader(
        args.data, args.batch_size, args.seq_len, args.pred_len,
        args.feature_type, args.target, dataset_id=args.dataset_id,
    )
    train_data = data_loader.get_train(generator=train_generator)
    val_data = data_loader.get_val()
    test_data = data_loader.get_test()
    if len(train_data) == 0:
        raise ValueError(
            "training loader has zero full batches; batch_size exceeds the available training windows"
        )

    preprocessing = data_loader.metadata()
    environment = environment_metadata(device)
    scientific = _scientific_config(
        args, data_sha256, source_sha256, preprocessing, device, environment
    )
    config_hash = stable_hash(scientific)
    run_dir = _run_directory(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.resolved.json"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.jsonl"
    metrics_path = run_dir / "metrics.json"

    source = {"sha256": source_sha256, "git": git_metadata()}
    resolved_config = _resolved_config(
        args, scientific, config_hash, run_dir, source, environment
    )
    run_started = time.time()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "implementation_variant": IMPLEMENTATION_VARIANT,
        "run_id": run_dir.name,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "config_hash": config_hash,
        "data_sha256": data_sha256,
        "artifact_dir": str(run_dir),
    }
    resume_checkpoint = None
    previous_config = None
    manifest_is_mutable = False
    run_committed = False
    elapsed_before_resume = 0.0
    run_lock = RunLock(run_dir).acquire()

    # Validate a resume before writing anything into the existing run.  A typo,
    # a config mismatch, or an attempt to resume an immutable completed run must
    # leave the original manifest and config byte-for-byte untouched.
    try:
        if args.resume:
            manifest, resume_checkpoint, previous_config = _load_resume_checkpoint(
                run_dir, config_hash, data_sha256, args.train_epochs
            )
            manifest["status"] = "running"
            manifest["updated_at"] = _utc_now()
            manifest.setdefault("resumed_at", []).append(_utc_now())
            previous_failure = manifest.pop("failure", None)
            if previous_failure is not None:
                manifest.setdefault("failure_history", []).append(previous_failure)
            manifest_is_mutable = True
            elapsed_before_resume = float(
                resume_checkpoint.get(
                    "active_duration_seconds",
                    sum(
                        float(item.get("duration_seconds", 0.0))
                        for item in resume_checkpoint.get("history", [])
                    ),
                )
            )
            # Preserve the original invocation and append resume provenance while
            # updating only the permitted target total epoch count.
            resolved_config = deepcopy(previous_config)
            resolved_config.setdefault("resume_invocations", []).append({
                "invoked_at": _utc_now(),
                "target_train_epochs": args.train_epochs,
                "resume_argument": args.resume,
            })
            resolved_config["run"]["train_epochs"] = args.train_epochs
            resolved_config["run"]["resume"] = args.resume
        else:
            manifest_is_mutable = True
    except BaseException:
        run_lock.release()
        raise

    try:
        atomic_write_json(config_path, resolved_config)
        atomic_write_json(manifest_path, manifest)
        model = AMD(
            input_shape=(args.seq_len, data_loader.n_feature),
            pred_len=args.pred_len,
            dropout=args.dropout,
            n_block=args.n_block,
            patch=args.patch,
            k=args.mix_layer_num,
            c=args.mix_layer_scale,
            alpha=args.alpha,
            target_slice=data_loader.target_slice,
            norm=args.norm,
            layernorm=args.layernorm,
        ).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        start_epoch = 0
        best_mse = float("inf")
        best_epoch = None
        best_val_metrics = None
        best_model_state = None
        history = []

        if resume_checkpoint is not None:
            model.load_state_dict(resume_checkpoint["model_state"])
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            start_epoch = int(resume_checkpoint["completed_epoch"])
            best_mse = float(resume_checkpoint["best_mse"])
            best_epoch = resume_checkpoint["best_epoch"]
            best_val_metrics = resume_checkpoint.get("best_val_metrics")
            best_model_state = resume_checkpoint["best_model_state"]
            history = resume_checkpoint.get("history", [])
            train_generator.set_state(
                resume_checkpoint["train_generator_state"].detach().cpu()
            )
            restore_rng_state(resume_checkpoint["rng_state"])
        common_checkpoint = _checkpoint_common(
            resolved_config, config_hash, data_sha256, preprocessing
        )

        if resume_checkpoint is not None:
            # last.pt is the epoch commit point.  Rebuild derivative files from
            # it before doing more work, so an interruption between their
            # writes cannot leave a logically mixed run.
            best_checkpoint = {
                **common_checkpoint,
                "model_state": best_model_state,
                "best_epoch": best_epoch,
                "best_mse": best_mse,
                "best_val_metrics": best_val_metrics,
            }
            atomic_torch_save(best_path, best_checkpoint)
            write_history(history_path, history)
            manifest["completed_epoch"] = start_epoch
            manifest["best_epoch"] = best_epoch
            manifest["best_validation_mse"] = best_mse
            manifest["updated_at"] = _utc_now()
            atomic_write_json(manifest_path, manifest)

        for epoch_index in range(start_epoch, args.train_epochs):
            epoch_started = time.time()
            epoch_number = epoch_index + 1
            train_metrics = train_one_epoch(
                model, train_data, optimizer, criterion, device,
                epoch_number, args.train_epochs, args.progress,
            )
            val_metrics = evaluate(
                model, val_data, device,
                description=f"Val {epoch_number}/{args.train_epochs}",
                show_progress=args.progress,
            )
            improved = should_update_best(val_metrics["mse"], best_mse)
            if improved:
                best_mse = val_metrics["mse"]
                best_epoch = epoch_number
                best_val_metrics = val_metrics
                best_model_state = _cpu_state_dict(model.state_dict())

            epoch_record = {
                "epoch": epoch_number,
                "train": train_metrics,
                "validation": val_metrics,
                "is_best": improved,
                "duration_seconds": time.time() - epoch_started,
                "finished_at": _utc_now(),
            }
            history.append(epoch_record)
            if best_model_state is None:
                raise RuntimeError("epoch completed without a finite validation best")
            last_checkpoint = {
                **common_checkpoint,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "completed_epoch": epoch_number,
                "best_epoch": best_epoch,
                "best_mse": best_mse,
                "best_val_metrics": best_val_metrics,
                "best_model_state": best_model_state,
                "rng_state": capture_rng_state(),
                "train_generator_state": train_generator.get_state(),
                "history": history,
                "active_duration_seconds": (
                    elapsed_before_resume + time.time() - run_started
                ),
            }
            # This is the sole epoch commit point.  Every artifact written
            # below is derivable from last.pt and is reconciled on resume.
            atomic_torch_save(last_path, last_checkpoint)
            best_checkpoint = {
                **common_checkpoint,
                "model_state": best_model_state,
                "best_epoch": best_epoch,
                "best_mse": best_mse,
                "best_val_metrics": best_val_metrics,
            }
            atomic_torch_save(best_path, best_checkpoint)
            write_history(history_path, history)
            manifest["completed_epoch"] = epoch_number
            manifest["best_epoch"] = best_epoch
            manifest["best_validation_mse"] = best_mse
            manifest["updated_at"] = _utc_now()
            atomic_write_json(manifest_path, manifest)
            print(
                f"epoch={epoch_number} train_objective={train_metrics['objective_mean_batches']:.8g} "
                f"val_mse={val_metrics['mse']:.8g} val_mae={val_metrics['mae']:.8g} "
                f"best_epoch={best_epoch}"
            )

        if best_epoch is None or not best_path.is_file():
            raise RuntimeError("training completed without a valid best checkpoint")

        # The test set is evaluated only after validation-based selection, using
        # the on-disk best checkpoint rather than an in-memory last-epoch copy.
        best_checkpoint = torch.load(best_path, map_location="cpu")
        if not isinstance(best_checkpoint, dict):
            raise RuntimeError("best checkpoint must contain a dictionary")
        if best_checkpoint.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("best checkpoint schema version mismatch")
        if best_checkpoint.get("implementation_variant") != IMPLEMENTATION_VARIANT:
            raise RuntimeError("best checkpoint implementation variant mismatch")
        if best_checkpoint.get("config_hash") != config_hash:
            raise RuntimeError("best checkpoint config hash mismatch")
        if best_checkpoint.get("data_sha256") != data_sha256:
            raise RuntimeError("best checkpoint data fingerprint mismatch")
        if best_checkpoint.get("best_epoch") != best_epoch:
            raise RuntimeError("best checkpoint epoch metadata mismatch")
        if not isinstance(best_checkpoint.get("model_state"), dict):
            raise RuntimeError("best checkpoint has no model state")
        # A resumed training budget may not improve the validation best.  Refresh
        # checkpoint provenance without changing its selected model state.
        best_checkpoint["resolved_config"] = resolved_config
        atomic_torch_save(best_path, best_checkpoint)
        model.load_state_dict(best_checkpoint["model_state"])
        test_metrics = evaluate(
            model, test_data, device, description="Final Test",
            show_progress=args.progress,
        )
        elapsed = elapsed_before_resume + time.time() - run_started
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "implementation_variant": IMPLEMENTATION_VARIANT,
            "run_id": run_dir.name,
            "status": "completed",
            "dataset_id": args.dataset_id,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "seed": args.seed,
            "metric_space": METRIC_SPACE,
            "best_epoch": best_epoch,
            "best_validation": best_val_metrics,
            "test": test_metrics,
            "parameter_count": parameter_count,
            "train_epochs": args.train_epochs,
            "duration_seconds": elapsed,
            "config_hash": config_hash,
            "data_sha256": data_sha256,
            "artifact_dir": str(run_dir),
            "completed_at": _utc_now(),
        }
        atomic_write_json(metrics_path, metrics)
        manifest.update({
            "status": "completed",
            "updated_at": _utc_now(),
            "completed_at": _utc_now(),
            "metrics_file": metrics_path.name,
            "best_epoch": best_epoch,
            "best_validation_mse": best_mse,
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
        })
        atomic_write_json(manifest_path, manifest)
        run_committed = True
        print(
            f"completed run={run_dir}\n"
            f"best_epoch={best_epoch} val_mse={best_mse:.8g} "
            f"test_mse={test_metrics['mse']:.8g} test_mae={test_metrics['mae']:.8g}"
        )
        return metrics
    except BaseException as exc:
        if manifest_is_mutable and not run_committed:
            manifest.update({
                "status": "failed",
                "updated_at": _utc_now(),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            })
            try:
                atomic_write_json(manifest_path, manifest)
            except Exception as status_error:
                warnings.warn(
                    "could not persist failed run status after "
                    f"{type(exc).__name__}: {status_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        raise
    finally:
        run_lock.release()


if __name__ == "__main__":
    main(parse_args())
