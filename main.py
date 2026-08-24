# coding=utf-8
"""Train the reproducible AMD paper-close interpretation variant."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.modules.target_exogenous_bridge import (
    PARALLEL_MULTIVARIATE,
    TARGET_EXOGENOUS,
)
from models.tsAMD import AMD
from models.tsAMD_enhanced import AMDEnhanced
from utils.dataloader import CustomDataLoader
from utils.dataloader_urbanev import (
    ALLOWED_HORIZONS,
    SPLIT_NAMES,
    UrbanEVFoldPreprocessor,
    UrbanEVRawData,
)
from utils.feature_schema import TARGET_NAME, get_feature_schema
from utils.temporal_region_dataset import TemporalRegionDataset
from utils.general import capture_rng_state, restore_rng_state, set_seed


BASELINE_IMPLEMENTATION_VARIANT = "AMD-paper-norm-wd-ddi-v1"
ENHANCED_IMPLEMENTATION_VARIANT = "el-amd-pmcr-teb-v1"
IMPLEMENTATION_VARIANT = BASELINE_IMPLEMENTATION_VARIANT
SUPPORTED_IMPLEMENTATION_VARIANTS = (
    BASELINE_IMPLEMENTATION_VARIANT,
    ENHANCED_IMPLEMENTATION_VARIANT,
)
SCHEMA_VERSION = 1
ENHANCED_ARTIFACT_SCHEMA_VERSION = 2
PAPER_WEIGHT_DECAY = 1e-7
METRIC_SPACE = "train-standardized"
ENHANCED_CHECKSUM_FILES = (
    "best.pt",
    "last.pt",
    "config.resolved.json",
    "history.jsonl",
    "metrics.json",
    "manifest.json",
    "sys.argv.json",
    "command.txt",
    "stdout.log",
    "stderr.log",
    "train.log",
    "source_fingerprint.json",
    "data_fingerprint.json",
)


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
    parser.add_argument(
        "--implementation_variant",
        default=IMPLEMENTATION_VARIANT,
        choices=SUPPORTED_IMPLEMENTATION_VARIANTS,
    )
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
    parser.add_argument(
        "--task_mode",
        default=None,
        choices=[TARGET_EXOGENOUS, PARALLEL_MULTIVARIATE],
    )
    parser.add_argument("--target_idx", type=int, default=None)
    parser.add_argument(
        "--aux_idx",
        type=int,
        nargs="*",
        default=None,
        help="ordered auxiliary indices; pass the flag with no values for an empty tuple",
    )
    parser.add_argument("--feature_names", nargs="+", default=None)
    parser.add_argument("--target_feature_name", default=None)
    parser.add_argument("--aux_feature_names", nargs="*", default=None)
    parser.add_argument("--schema_fingerprint", default=None)
    parser.add_argument(
        "--feature_preset",
        default=None,
        choices=["F0", "F1", "F2", "F3", "F4"],
        help="M1 UrbanEV feature preset; required by the production UrbanEV path",
    )
    parser.add_argument("--fold", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--label_horizon",
        type=int,
        default=None,
        help="single-point label offset; distinct from model_pred_len",
    )
    parser.add_argument(
        "--ablation_id",
        default=None,
        choices=[
            "U0", "U1", "U2", "U3", "U4", "target_only_pmcr",
            "M0", "M1", "M2", "M3",
        ],
    )

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

    parser.add_argument("--use_pmcr", type=str2bool, default=False)
    parser.add_argument("--pmcr_hidden_dim", type=int, default=None)
    parser.add_argument("--pmcr_kernel_small", type=int, default=None)
    parser.add_argument("--pmcr_kernel_large", type=int, default=None)
    parser.add_argument("--pmcr_dropout", type=float, default=0.1)
    parser.add_argument("--pmcr_gamma_init", type=float, default=1e-3)
    parser.add_argument("--pmcr_deploy", type=str2bool, default=False)

    parser.add_argument("--use_teb", type=str2bool, default=False)
    parser.add_argument("--teb_context_dim", type=int, default=32)
    parser.add_argument("--teb_heads", type=int, default=4)
    parser.add_argument("--teb_dropout", type=float, default=0.1)
    parser.add_argument("--teb_gamma_init", type=float, default=1e-3)
    parser.add_argument(
        "--teb_query_policy",
        default="linear_then_layernorm",
        choices=["linear_then_layernorm"],
    )
    parser.add_argument(
        "--teb_projector_policy",
        default="shared_linear_then_layernorm",
        choices=["shared_linear_then_layernorm"],
    )
    parser.add_argument(
        "--parallel_aux_policy",
        default="all_other_variables",
        choices=["all_other_variables"],
    )
    parser.add_argument(
        "--parallel_self_mask",
        default="diagonal_exclusion",
        choices=["diagonal_exclusion"],
    )
    parser.add_argument(
        "--empty_aux_policy",
        default="reject_when_teb_enabled",
        choices=["reject_when_teb_enabled"],
    )
    parser.add_argument(
        "--parallel_c1_policy",
        default="reject_when_teb_enabled",
        choices=["reject_when_teb_enabled"],
    )
    parser.add_argument(
        "--target_selection_policy",
        default="full_denorm_then_task_select",
        choices=["full_denorm_then_task_select"],
    )

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


def _ordered_unique_names(value, field_name, *, allow_empty=False):
    if value is None:
        raise ValueError(f"{field_name} must be provided explicitly")
    result = tuple(str(item).strip() for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{field_name} must not be empty")
    if any(not item for item in result):
        raise ValueError(f"{field_name} must contain non-empty names")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _is_urbanev_production(args):
    return (
        args.implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT
        and str(args.dataset_id).casefold() == "urbanev"
        and args.task_mode == TARGET_EXOGENOUS
    )


def _resolve_horizon_contract(args):
    """Separate label offset, model output length, and artifact identity."""

    if args.task_mode == PARALLEL_MULTIVARIATE:
        if args.label_horizon is not None:
            raise ValueError(
                "parallel_multivariate requires label_horizon=None; pred_len is "
                "the model and artifact horizon"
            )
        if args.horizon is not None and args.horizon != args.pred_len:
            raise ValueError(
                "parallel_multivariate horizon and pred_len must agree when both "
                f"are supplied, got {args.horizon} and {args.pred_len}"
            )
        if str(args.fold) != "official":
            raise ValueError("parallel_multivariate requires fold='official'")
        args.label_horizon = None
        args.model_pred_len = args.pred_len
        args.artifact_horizon = args.pred_len
        args.horizon = args.artifact_horizon
        return

    supplied = [
        value for value in (args.label_horizon, args.horizon)
        if value is not None
    ]
    if not supplied:
        raise ValueError(
            "target_exogenous requires an explicit label_horizon (or legacy --horizon alias)"
        )
    if len(supplied) == 2 and supplied[0] != supplied[1]:
        raise ValueError(
            "label_horizon and horizon disagree: "
            f"{args.label_horizon} != {args.horizon}"
        )
    label_horizon = supplied[0]
    if (
        isinstance(label_horizon, bool)
        or not isinstance(label_horizon, int)
        or label_horizon <= 0
    ):
        raise ValueError(
            f"label_horizon must be a positive integer, got {label_horizon!r}"
        )
    args.label_horizon = label_horizon
    args.model_pred_len = args.pred_len
    args.artifact_horizon = label_horizon
    args.horizon = args.artifact_horizon


def _bind_urbanev_schema_contract(args):
    """Derive UrbanEV schema identities from the frozen M1 FeatureSchema API."""

    if args.feature_preset is None:
        raise ValueError("UrbanEV target_exogenous requires --feature_preset F0--F4")
    schema = get_feature_schema(args.feature_preset)
    actual = {
        "feature_names": tuple(schema.feature_names),
        "target_idx": schema.target_idx,
        "target_feature_name": schema.target_name,
        "schema_fingerprint": schema.fingerprint,
    }
    actual["aux_idx"] = tuple(
        index for index in range(len(schema.feature_names))
        if index != schema.target_idx
    )
    actual["aux_feature_names"] = tuple(
        schema.feature_names[index] for index in actual["aux_idx"]
    )

    expected_values = {
        "feature_names": (
            None if args.feature_names is None
            else tuple(str(value) for value in args.feature_names)
        ),
        "target_idx": args.target_idx,
        "target_feature_name": args.target_feature_name,
        "schema_fingerprint": args.schema_fingerprint,
        "aux_idx": None if args.aux_idx is None else tuple(args.aux_idx),
        "aux_feature_names": (
            None if args.aux_feature_names is None
            else tuple(str(value) for value in args.aux_feature_names)
        ),
    }
    for name, expected in expected_values.items():
        if expected is not None and expected != actual[name]:
            raise ValueError(
                f"UrbanEV CLI expected {name} does not match M1 schema: "
                f"expected {expected!r}, observed {actual[name]!r}"
            )
        setattr(args, name, actual[name])

    args.feature_preset = schema.preset
    return schema


def _validate_urbanev_protocol(args):
    if not Path(args.data).is_dir():
        raise ValueError("UrbanEV production data path must be a data_root directory")
    if args.seq_len != 12:
        raise ValueError("UrbanEV production history_len/seq_len is fixed at 12")
    if args.label_horizon not in ALLOWED_HORIZONS:
        raise ValueError(
            f"UrbanEV label_horizon must be one of {ALLOWED_HORIZONS}, "
            f"got {args.label_horizon}"
        )
    if args.pred_len != 1 or args.model_pred_len != 1:
        raise ValueError("UrbanEV production model_pred_len/pred_len is fixed at 1")
    try:
        fold = int(args.fold)
    except (TypeError, ValueError) as exc:
        raise ValueError("UrbanEV fold must be an integer in {1,2,3,4,5,6}") from exc
    if fold not in range(1, 7):
        raise ValueError("UrbanEV fold must be in {1,2,3,4,5,6}")
    args.fold = fold
    if args.target != TARGET_NAME:
        raise ValueError("UrbanEV v1 target is fixed at 'volume'")
    if args.feature_type != "MS":
        raise ValueError("UrbanEV target_exogenous requires feature_type='MS'")
    _bind_urbanev_schema_contract(args)
    if args.feature_preset == "F0" and args.use_teb:
        raise ValueError("UrbanEV F0 has no auxiliary variables and requires use_teb=False")

def _prepare_enhanced_contract(args):
    if args.task_mode is None:
        raise ValueError("enhanced runner requires explicit task_mode")
    if args.fold is None or args.ablation_id is None:
        raise ValueError("enhanced runner requires explicit fold and ablation_id")

    _resolve_horizon_contract(args)
    urbanev_production = _is_urbanev_production(args)
    if urbanev_production:
        _validate_urbanev_protocol(args)
    else:
        required = {
            "target_idx": args.target_idx,
            "aux_idx": args.aux_idx,
            "feature_names": args.feature_names,
            "target_feature_name": args.target_feature_name,
            "aux_feature_names": args.aux_feature_names,
            "schema_fingerprint": args.schema_fingerprint,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "enhanced runner requires explicit " + ", ".join(missing)
            )
        args.fold = _safe_component(args.fold, "fold")

    args.feature_names = _ordered_unique_names(
        args.feature_names, "feature_names"
    )
    args.aux_feature_names = _ordered_unique_names(
        args.aux_feature_names,
        "aux_feature_names",
        allow_empty=True,
    )
    args.target_feature_name = str(args.target_feature_name).strip()
    if not args.target_feature_name:
        raise ValueError("target_feature_name must not be empty")
    args.schema_fingerprint = str(args.schema_fingerprint).strip()
    if not args.schema_fingerprint:
        raise ValueError("schema_fingerprint must not be empty")

    if (
        isinstance(args.target_idx, bool)
        or not isinstance(args.target_idx, int)
        or not 0 <= args.target_idx < len(args.feature_names)
    ):
        raise ValueError(
            f"target_idx must index feature_names, got {args.target_idx!r}"
        )
    if args.feature_names[args.target_idx] != args.target_feature_name:
        raise ValueError(
            "target_idx and target_feature_name disagree: "
            f"feature_names[{args.target_idx}]={args.feature_names[args.target_idx]!r}, "
            f"target_feature_name={args.target_feature_name!r}"
        )

    args.aux_idx = tuple(args.aux_idx)
    for index in args.aux_idx:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("aux_idx must contain only non-bool integers")
    if len(set(args.aux_idx)) != len(args.aux_idx):
        raise ValueError("aux_idx must not contain duplicate indices")
    if any(not 0 <= index < len(args.feature_names) for index in args.aux_idx):
        raise ValueError("aux_idx contains an out-of-range feature index")
    if args.target_idx in args.aux_idx:
        raise ValueError("aux_idx must exclude target_idx")
    expected_aux_names = tuple(args.feature_names[index] for index in args.aux_idx)
    if args.aux_feature_names != expected_aux_names:
        raise ValueError(
            "aux_feature_names must exactly match ordered aux_idx: "
            f"expected {expected_aux_names}, got {args.aux_feature_names}"
        )

    if (
        isinstance(args.teb_context_dim, bool)
        or not isinstance(args.teb_context_dim, int)
        or args.teb_context_dim <= 0
    ):
        raise ValueError("teb_context_dim must be a positive integer")
    if (
        isinstance(args.teb_heads, bool)
        or not isinstance(args.teb_heads, int)
        or args.teb_heads <= 0
    ):
        raise ValueError("teb_heads must be a positive integer")
    if args.teb_context_dim % args.teb_heads != 0:
        raise ValueError("teb_context_dim must be divisible by teb_heads")
    if not math.isfinite(args.teb_dropout) or not 0 <= args.teb_dropout < 1:
        raise ValueError("teb_dropout must satisfy 0 <= dropout < 1")
    if not math.isfinite(args.teb_gamma_init) or args.teb_gamma_init != 1e-3:
        raise ValueError("teb_gamma_init is fixed at 1e-3")
    if args.pmcr_deploy:
        raise ValueError("formal training runner requires pmcr_deploy=False")
    if not math.isfinite(args.pmcr_dropout) or not 0 <= args.pmcr_dropout < 1:
        raise ValueError("pmcr_dropout must satisfy 0 <= dropout < 1")
    if not math.isfinite(args.pmcr_gamma_init) or args.pmcr_gamma_init == 0:
        raise ValueError("pmcr_gamma_init must be finite and non-zero")
    if args.use_pmcr:
        pmcr_required = {
            "pmcr_hidden_dim": args.pmcr_hidden_dim,
            "pmcr_kernel_small": args.pmcr_kernel_small,
            "pmcr_kernel_large": args.pmcr_kernel_large,
        }
        missing_pmcr = [
            name for name, value in pmcr_required.items() if value is None
        ]
        if missing_pmcr:
            raise ValueError(
                "use_pmcr=True requires explicit " + ", ".join(missing_pmcr)
            )
        if args.pmcr_kernel_large > args.seq_len:
            raise ValueError("pmcr_kernel_large must not exceed seq_len")
    elif any(
        value is not None
        for value in (
            args.pmcr_hidden_dim,
            args.pmcr_kernel_small,
            args.pmcr_kernel_large,
        )
    ):
        raise ValueError(
            "PMCR dimensions/kernels must be omitted when use_pmcr=False"
        )

    aux_nonempty = bool(args.aux_idx)
    if args.task_mode == TARGET_EXOGENOUS:
        if args.feature_type != "MS":
            raise ValueError("target_exogenous requires feature_type='MS'")
        if args.target != args.target_feature_name:
            raise ValueError("target must equal target_feature_name")
        expected = {
            "U0": (False, False, False, "AMD-TargetOnly"),
            "U1": (False, False, True, "AMD-Concat"),
            "U2": (False, True, True, "AMD-Concat + TEB"),
            "U3": (True, False, True, "AMD-Concat + PMCR"),
            "U4": (True, True, True, "EL-AMD"),
            "target_only_pmcr": (
                True,
                False,
                False,
                "AMD-TargetOnly + PMCR",
            ),
        }
        if args.ablation_id not in expected:
            raise ValueError(
                "target_exogenous ablation_id must be U0--U4 or target_only_pmcr"
            )
        expected_pmcr, expected_teb, expected_aux, display = expected[
            args.ablation_id
        ]
        if (
            args.use_pmcr != expected_pmcr
            or args.use_teb != expected_teb
            or aux_nonempty != expected_aux
        ):
            raise ValueError(
                f"ablation_id={args.ablation_id} contradicts PMCR/TEB/aux contract"
            )
        if args.use_teb and not aux_nonempty:
            raise ValueError("TEB requires at least one auxiliary variable.")
    else:
        if args.feature_type != "M":
            raise ValueError("parallel_multivariate requires feature_type='M'")
        if args.target != "all":
            raise ValueError("parallel_multivariate requires target='all'")
        if args.aux_idx or args.aux_feature_names:
            raise ValueError(
                "parallel_multivariate uses all other variables; aux_idx must be empty"
            )
        expected = {
            "M0": (False, False),
            "M1": (True, False),
            "M2": (False, True),
            "M3": (True, True),
        }
        if args.ablation_id not in expected:
            raise ValueError("parallel_multivariate ablation_id must be M0--M3")
        if (args.use_pmcr, args.use_teb) != expected[args.ablation_id]:
            raise ValueError(
                f"ablation_id={args.ablation_id} contradicts PMCR/TEB contract"
            )
        if args.use_teb and len(args.feature_names) < 2:
            raise ValueError("Parallel TEB requires at least two variables.")
        display = {
            "M0": "AMD",
            "M1": "AMD + PMCR",
            "M2": "AMD + TEB",
            "M3": "EL-AMD",
        }[args.ablation_id]

    args.display_name = display
    return args


def prepare_args(args):
    """Normalize paths and validate baseline or enhanced scientific contracts."""

    if args.implementation_variant not in SUPPORTED_IMPLEMENTATION_VARIANTS:
        raise ValueError(
            f"unsupported implementation variant: {args.implementation_variant}"
        )
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
            f"{args.implementation_variant} fixes weight_decay at {PAPER_WEIGHT_DECAY:g}; "
            f"got {args.weight_decay:g}"
        )
    if args.implementation_variant == BASELINE_IMPLEMENTATION_VARIANT:
        baseline_only_values = {
            "task_mode": args.task_mode,
            "target_idx": args.target_idx,
            "aux_idx": args.aux_idx,
            "feature_names": args.feature_names,
            "target_feature_name": args.target_feature_name,
            "aux_feature_names": args.aux_feature_names,
            "schema_fingerprint": args.schema_fingerprint,
            "feature_preset": args.feature_preset,
            "fold": args.fold,
            "horizon": args.horizon,
            "label_horizon": args.label_horizon,
            "ablation_id": args.ablation_id,
        }
        configured = [
            name for name, value in baseline_only_values.items()
            if value is not None
        ]
        if configured or args.use_pmcr or args.use_teb:
            raise ValueError(
                "baseline variant does not accept enhanced configuration: "
                + ", ".join(configured)
            )
        args.display_name = "AMD"
        return args
    return _prepare_enhanced_contract(args)


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


def _source_files(root=ROOT):
    root = Path(root)
    files = [root / "main.py"]
    files.extend((root / "models").rglob("*.py"))
    files.extend((root / "utils").glob("*.py"))
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def source_fingerprint(root=ROOT):
    """Hash the executable Python source, independent of Git staging state."""

    root = Path(root)
    files = _source_files(root)
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def source_fingerprint_metadata(root=ROOT):
    """Return the deterministic source fingerprint and its exact file scope."""

    root = Path(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "sha256_length_prefixed_relative_path_and_content_v1",
        "sha256": source_fingerprint(root),
        "files": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            for path in _source_files(root)
        ],
    }


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


class _TranscriptStream:
    """File-like proxy used by :class:`RunTranscript`."""

    def __init__(self, transcript, channel):
        self._transcript = transcript
        self._channel = channel

    def write(self, value):
        return self._transcript.write(self._channel, value)

    def flush(self):
        return self._transcript.flush(self._channel)

    def __getattr__(self, name):
        return getattr(self._transcript.original_stream(self._channel), name)


class RunTranscript:
    """Tee stdout/stderr into three atomically finalized run logs.

    The proxy is installed before data loading. Output is buffered in memory
    until a new run has a directory, or until resume validation succeeds. This
    keeps rejected resume attempts byte-for-byte immutable.
    """

    _LOG_NAMES = {
        "stdout": "stdout.log",
        "stderr": "stderr.log",
        "train": "train.log",
    }

    def __init__(self):
        self._original = {"stdout": sys.stdout, "stderr": sys.stderr}
        self._proxies = {
            channel: _TranscriptStream(self, channel)
            for channel in ("stdout", "stderr")
        }
        self._events = []
        self._handles = {}
        self._temporary_paths = {}
        self._final_paths = {}
        self._installed = False
        self._bound = False
        self._finalized = False

    def original_stream(self, channel):
        return self._original[channel]

    def install(self):
        if self._installed:
            return self
        sys.stdout = self._proxies["stdout"]
        sys.stderr = self._proxies["stderr"]
        self._installed = True
        return self

    def _open_atomic_log(self, final_path):
        temporary = final_path.with_name(
            f".{final_path.name}.{uuid.uuid4().hex}.tmp"
        )
        handle = temporary.open("w", encoding="utf-8", newline="\n")
        if final_path.is_file():
            handle.write(final_path.read_text(encoding="utf-8", errors="replace"))
        return temporary, handle

    def bind(self, run_dir):
        if self._bound:
            if Path(run_dir) != next(iter(self._final_paths.values())).parent:
                raise RuntimeError("run transcript is already bound to another directory")
            return self
        if self._finalized:
            raise RuntimeError("cannot bind a finalized run transcript")

        run_dir = Path(run_dir)
        try:
            for channel, name in self._LOG_NAMES.items():
                final_path = run_dir / name
                temporary, handle = self._open_atomic_log(final_path)
                self._final_paths[channel] = final_path
                self._temporary_paths[channel] = temporary
                self._handles[channel] = handle
            marker = f"\n=== invocation {_utc_now()} ===\n"
            for handle in self._handles.values():
                handle.write(marker)
            for channel, value in self._events:
                self._handles[channel].write(value)
                self._handles["train"].write(value)
            self._events.clear()
            self._bound = True
            return self
        except BaseException:
            for handle in self._handles.values():
                handle.close()
            for path in self._temporary_paths.values():
                if path.exists():
                    path.unlink()
            self._handles.clear()
            self._temporary_paths.clear()
            self._final_paths.clear()
            raise

    def write(self, channel, value):
        value = str(value)
        written = self._original[channel].write(value)
        if self._bound and not self._finalized:
            self._handles[channel].write(value)
            self._handles["train"].write(value)
        elif not self._finalized:
            self._events.append((channel, value))
        return len(value) if written is None else written

    def flush(self, channel):
        result = self._original[channel].flush()
        if self._bound and not self._finalized:
            self._handles[channel].flush()
            self._handles["train"].flush()
        return result

    def finalize(self):
        if self._finalized:
            return self
        if self._bound:
            for handle in self._handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            for channel in self._LOG_NAMES:
                os.replace(
                    self._temporary_paths[channel],
                    self._final_paths[channel],
                )
        self._finalized = True
        return self

    def restore(self):
        if self._installed:
            if sys.stdout is self._proxies["stdout"]:
                sys.stdout = self._original["stdout"]
            if sys.stderr is self._proxies["stderr"]:
                sys.stderr = self._original["stderr"]
            self._installed = False


def _write_enhanced_provenance(
    args,
    run_dir,
    source_sha256,
    data_sha256,
    data_fingerprint_document,
):
    """Atomically record invocation, source, and data identities."""

    run_dir = Path(run_dir)
    argv_path = run_dir / "sys.argv.json"
    command_path = run_dir / "command.txt"
    recorded_at = _utc_now()
    invocation = {
        "recorded_at": recorded_at,
        "argv": list(sys.argv),
    }
    if args.resume and argv_path.is_file():
        try:
            argv_document = json.loads(argv_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("existing sys.argv.json is invalid") from exc
        if not isinstance(argv_document, dict) or not isinstance(
            argv_document.get("invocations"), list
        ):
            raise RuntimeError("existing sys.argv.json has an invalid contract")
    else:
        argv_document = {
            "schema_version": SCHEMA_VERSION,
            "invocations": [],
        }
    argv_document["invocations"].append(invocation)
    atomic_write_json(argv_path, argv_document)

    command_record = (
        f"# invocation {recorded_at}\n"
        f"{shlex.join([sys.executable, *sys.argv])}\n"
    )
    previous_commands = ""
    if args.resume and command_path.is_file():
        previous_commands = command_path.read_text(encoding="utf-8")
        if previous_commands and not previous_commands.endswith("\n"):
            previous_commands += "\n"
    _atomic_write_text(command_path, previous_commands + command_record)

    source_document = source_fingerprint_metadata()
    if source_document["sha256"] != source_sha256:
        raise RuntimeError("source fingerprint changed while preparing provenance")
    atomic_write_json(run_dir / "source_fingerprint.json", source_document)
    data_document = deepcopy(data_fingerprint_document)
    if data_document.get("sha256") != data_sha256:
        raise RuntimeError("data fingerprint changed while preparing provenance")
    atomic_write_json(run_dir / "data_fingerprint.json", data_document)


def write_checksums(run_dir, filenames=ENHANCED_CHECKSUM_FILES):
    """Write a sha256sum-compatible manifest for the controlled run files."""

    run_dir = Path(run_dir)
    names = tuple(filenames)
    if len(set(names)) != len(names):
        raise ValueError("checksum filenames must be unique")
    lines = []
    for name in names:
        path = Path(name)
        if path.is_absolute() or len(path.parts) != 1 or path.name != name:
            raise ValueError(f"checksum filename must be a direct relative name: {name!r}")
        artifact = run_dir / name
        if not artifact.is_file():
            raise FileNotFoundError(f"required artifact is missing: {artifact}")
        lines.append(f"{sha256_file(artifact)}  {name}")
    checksum_path = run_dir / "checksums.sha256"
    _atomic_write_text(checksum_path, "\n".join(lines) + "\n")
    return checksum_path


def verify_checksums(run_dir, filenames=ENHANCED_CHECKSUM_FILES):
    """Verify the exact controlled checksum set without trusting shell parsing."""

    run_dir = Path(run_dir)
    checksum_path = run_dir / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"checksum manifest is missing: {checksum_path}")
    observed = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"invalid checksum line: {line!r}")
        path = Path(name)
        if path.is_absolute() or len(path.parts) != 1 or path.name != name:
            raise RuntimeError(f"unsafe checksum path: {name!r}")
        if name in observed:
            raise RuntimeError(f"duplicate checksum entry: {name}")
        observed[name] = digest
    expected = set(filenames)
    if set(observed) != expected:
        raise RuntimeError(
            "checksum file set mismatch: "
            f"missing={sorted(expected - set(observed))}, "
            f"unexpected={sorted(set(observed) - expected)}"
        )
    for name, expected_digest in observed.items():
        actual = sha256_file(run_dir / name)
        if actual != expected_digest:
            raise RuntimeError(
                f"checksum mismatch for {name}: {actual} != {expected_digest}"
            )
    return observed


def verify_checksums_with_sha256sum(run_dir):
    """Run the independent system verifier without writing into the artifact."""

    run_dir = Path(run_dir)
    try:
        result = subprocess.run(
            ["sha256sum", "-c", "checksums.sha256"],
            cwd=run_dir,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise RuntimeError(f"could not execute sha256sum verifier: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "sha256sum -c verification failed: "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
    return result.stdout


def _artifact_fault_point(stage):
    """No-op seam used by permanent fault-injection tests."""

    return None


def _publish_enhanced_artifact(
    *,
    staging_dir,
    final_dir,
    transcript,
    run_lock,
    manifest_path,
    completed_manifest,
):
    """Seal, independently verify, and atomically publish one schema-v2 run."""

    staging_dir = Path(staging_dir)
    final_dir = Path(final_dir)
    if staging_dir.parent != final_dir.parent:
        raise RuntimeError("staging and final artifact directories must share a parent")
    if final_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable completed artifact: {final_dir}"
        )

    transcript.finalize()
    _artifact_fault_point("after_writers_closed")
    atomic_write_json(manifest_path, completed_manifest)
    _artifact_fault_point("after_completed_manifest")
    write_checksums(staging_dir)
    _artifact_fault_point("after_checksums")
    verify_checksums(staging_dir)
    _artifact_fault_point("after_python_verify")
    verify_checksums_with_sha256sum(staging_dir)
    _artifact_fault_point("after_sha256sum_verify")

    run_lock.release()
    if run_lock.path.exists():
        run_lock.path.unlink()
    _artifact_fault_point("before_atomic_rename")
    if final_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable completed artifact: {final_dir}"
        )
    os.replace(staging_dir, final_dir)
    return final_dir



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


def _prediction_for_loss(outputs, targets, task_mode=None):
    if task_mode == TARGET_EXOGENOUS:
        if outputs.ndim != targets.ndim + 1 or outputs.shape[-1] != 1:
            raise RuntimeError(
                "target_exogenous prediction must be [B,H,1] while target is [B,H], "
                f"got {tuple(outputs.shape)} and {tuple(targets.shape)}"
            )
        prediction = outputs.squeeze(-1)
    else:
        prediction = outputs
    if prediction.shape != targets.shape:
        raise RuntimeError(
            "prediction/target shape mismatch after the explicit task adapter: "
            f"{tuple(prediction.shape)} != {tuple(targets.shape)}"
        )
    if prediction.numel() == 0:
        raise RuntimeError("model produced an empty prediction batch")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("model predictions contain NaN or Inf")
    if not torch.isfinite(targets).all():
        raise FloatingPointError("targets contain NaN or Inf")
    return prediction


def _assert_prediction_batch(outputs, targets, task_mode=None):
    return _prediction_for_loss(outputs, targets, task_mode=task_mode)


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


def evaluate(
    model,
    data_loader,
    device,
    description="Eval",
    show_progress=True,
    task_mode=None,
):
    """Evaluate global element-wise MSE/MAE, including a final partial batch."""

    model.eval()
    accumulator = {"sse": 0.0, "sae": 0.0, "num_elements": 0, "num_batches": 0}
    iterator = tqdm(data_loader, desc=description, disable=not show_progress, leave=False)
    with torch.no_grad():
        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs, _ = model(batch_x)
            prediction = _assert_prediction_batch(
                outputs, batch_y, task_mode=task_mode
            )
            _accumulate_errors(prediction, batch_y, accumulator)
    return _finalize_errors(accumulator)


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
    show_progress=True,
    task_mode=None,
):
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
        prediction = _assert_prediction_batch(
            outputs, batch_y, task_mode=task_mode
        )
        if not torch.is_tensor(auxiliary_loss) or auxiliary_loss.numel() != 1:
            raise RuntimeError("AMD auxiliary loss must be a scalar tensor")
        prediction_loss = criterion(prediction, batch_y)
        loss = prediction_loss + auxiliary_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("training objective contains NaN or Inf")
        loss.backward()
        optimizer.step()

        objective_sum += loss.detach().item()
        auxiliary_sum += auxiliary_loss.detach().item()
        _accumulate_errors(prediction.detach(), batch_y, accumulator)
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

    dataset_config = {
        "id": args.dataset_id,
        "sha256": data_sha256,
        "feature_type": args.feature_type,
        "target": args.target,
        "preprocessing": preprocessing,
    }
    model_config = {
        "model_class": (
            "AMDEnhanced"
            if args.implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT
            else "AMD"
        ),
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "model_pred_len": (
            args.model_pred_len
            if args.implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT
            else args.pred_len
        ),
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
    }
    experiment = None
    if args.implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT:
        dataset_config.update({
            "task_mode": args.task_mode,
            "feature_names": list(args.feature_names),
            "target_feature_name": args.target_feature_name,
            "target_idx": args.target_idx,
            "aux_feature_names": list(args.aux_feature_names),
            "aux_idx": list(args.aux_idx),
            "schema_fingerprint": args.schema_fingerprint,
            "feature_preset": args.feature_preset,
            "fold": args.fold,
            "label_horizon": args.label_horizon,
            "model_pred_len": args.model_pred_len,
            "artifact_horizon": args.artifact_horizon,
        })
        model_config.update({
            "target_idx": args.target_idx,
            "target_slice": None,
            "target_selection_policy": args.target_selection_policy,
            "use_pmcr": args.use_pmcr,
            "pmcr": {
                "hidden_dim": args.pmcr_hidden_dim,
                "kernel_small": args.pmcr_kernel_small,
                "kernel_large": args.pmcr_kernel_large,
                "dropout": args.pmcr_dropout,
                "gamma_init": args.pmcr_gamma_init,
                "deploy": args.pmcr_deploy,
                "norm": "feature_wise_layernorm",
                "ffn_ratio": 2,
            },
            "use_teb": args.use_teb,
            "teb": {
                "context_dim": args.teb_context_dim,
                "heads": args.teb_heads,
                "dropout": args.teb_dropout,
                "gamma_init": args.teb_gamma_init,
                "query_policy": args.teb_query_policy,
                "projector_policy": args.teb_projector_policy,
                "variable_identity_embedding": False,
                "output_dropout": False,
                "query_residual": False,
                "post_attention_ffn": False,
            },
            "parallel_aux_policy": args.parallel_aux_policy,
            "parallel_self_mask": args.parallel_self_mask,
            "empty_aux_policy": args.empty_aux_policy,
            "parallel_c1_policy": args.parallel_c1_policy,
        })
        experiment = {
            "ablation_id": args.ablation_id,
            "display_name": args.display_name,
            "task_mode": args.task_mode,
            "target": args.target,
            "label_horizon": args.label_horizon,
            "model_pred_len": args.model_pred_len,
            "artifact_horizon": args.artifact_horizon,
            "fold": args.fold,
        }

    result = {
        "implementation_variant": args.implementation_variant,
        "source_sha256": source_sha256,
        "dataset": dataset_config,
        "model": model_config,
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
    if experiment is not None:
        result["experiment"] = experiment
    return result

def _resolved_config(args, scientific, config_hash, run_dir, source, environment):
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_schema_version": (
            ENHANCED_ARTIFACT_SCHEMA_VERSION
            if args.implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT
            else None
        ),
        "implementation_variant": args.implementation_variant,
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
        "artifact_schema_version": resolved_config.get("artifact_schema_version"),
        "implementation_variant": resolved_config["implementation_variant"],
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


def _load_resume_checkpoint(
    run_dir,
    config_hash,
    data_sha256,
    train_epochs,
    implementation_variant=IMPLEMENTATION_VARIANT,
    *,
    run_id=None,
    artifact_dir=None,
):
    run_dir = Path(run_dir)
    run_id = run_dir.name if run_id is None else str(run_id)
    artifact_dir = run_dir if artifact_dir is None else Path(artifact_dir)
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
    if (
        implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT
        and manifest.get("artifact_schema_version")
        != ENHANCED_ARTIFACT_SCHEMA_VERSION
    ):
        raise RuntimeError("resume artifact schema version mismatch")
    if manifest.get("implementation_variant") != implementation_variant:
        raise RuntimeError(
            f"resume variant does not match {implementation_variant}"
        )
    if manifest.get("run_id") != run_id:
        raise RuntimeError("resume manifest run_id/path mismatch")
    manifest_artifact_dir = manifest.get("artifact_dir")
    if (
        not isinstance(manifest_artifact_dir, str)
        or Path(manifest_artifact_dir).resolve() != artifact_dir.resolve()
    ):
        raise RuntimeError("resume manifest artifact path mismatch")
    if manifest.get("config_hash") != config_hash:
        raise RuntimeError("resume manifest configuration hash mismatch")
    if manifest.get("data_sha256") != data_sha256:
        raise RuntimeError("resume manifest data fingerprint mismatch")
    if previous_config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("resume resolved config schema version mismatch")
    if (
        implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT
        and previous_config.get("artifact_schema_version")
        != ENHANCED_ARTIFACT_SCHEMA_VERSION
    ):
        raise RuntimeError("resume resolved artifact schema version mismatch")
    if previous_config.get("implementation_variant") != implementation_variant:
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
        or Path(previous_run_dir).resolve() != artifact_dir.resolve()
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
    if (
        implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT
        and checkpoint.get("artifact_schema_version")
        != ENHANCED_ARTIFACT_SCHEMA_VERSION
    ):
        raise RuntimeError("resume checkpoint artifact schema version mismatch")
    if checkpoint.get("implementation_variant") != implementation_variant:
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


@dataclass(frozen=True)
class ArtifactPaths:
    work_dir: Path
    final_dir: Path
    run_id: str
    is_enhanced: bool


def _enhanced_artifact_parent(args):
    return (
        Path(args.artifact_root)
        / args.implementation_variant
        / args.dataset_id
        / args.task_mode
        / _safe_component(args.target, "target")
        / f"horizon_{args.artifact_horizon}"
        / f"fold_{args.fold}"
        / f"seed_{args.seed}"
    )


def _enhanced_run_id_from_staging(staging_dir):
    name = Path(staging_dir).name
    if not name.startswith(".") or not name.endswith(".staging"):
        raise ValueError(
            "enhanced resume path must be the hidden staging directory "
            "'.<run_id>.staging' or its last.pt"
        )
    run_id = name[1:-len(".staging")]
    if not run_id:
        raise ValueError("enhanced staging path contains an empty run_id")
    return run_id


def _artifact_paths(args):
    if args.resume:
        resume = Path(args.resume)
        if not resume.exists():
            raise FileNotFoundError(f"resume path does not exist: {resume}")
        if resume.is_file():
            if resume.name != "last.pt":
                raise ValueError("a resume file must be the run's last.pt")
            work_dir = resume.parent
        elif resume.is_dir():
            work_dir = resume
        else:
            raise ValueError(
                f"resume path must be a run directory or last.pt: {resume}"
            )
        if args.implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT:
            run_id = _enhanced_run_id_from_staging(work_dir)
            final_dir = work_dir.parent / run_id
            if final_dir.exists():
                raise FileExistsError(
                    f"immutable completed artifact already exists: {final_dir}"
                )
            return ArtifactPaths(work_dir, final_dir, run_id, True)
        return ArtifactPaths(work_dir, work_dir, work_dir.name, False)

    if args.implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT:
        parent = _enhanced_artifact_parent(args)
        parent.mkdir(parents=True, exist_ok=True)
        run_id = _new_run_id()
        final_dir = parent / run_id
        work_dir = parent / f".{run_id}.staging"
        if final_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite completed artifact: {final_dir}"
            )
        work_dir.mkdir(parents=False, exist_ok=False)
        return ArtifactPaths(work_dir, final_dir, run_id, True)

    parent = (
        Path(args.artifact_root)
        / args.implementation_variant
        / args.dataset_id
        / f"sl{args.seq_len}_pl{args.pred_len}"
        / f"seed{args.seed}"
    )
    run_dir = parent / _new_run_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    return ArtifactPaths(run_dir, run_dir, run_dir.name, False)


def _run_directory(args):
    """Compatibility helper returning the mutable work directory."""

    return _artifact_paths(args).work_dir


@dataclass(frozen=True)
class RuntimeData:
    """Resolved in-memory data contract consumed by one runner invocation."""

    n_feature: int
    target_slice: object
    train_data: object
    val_data: object
    test_data: object
    preprocessing: dict
    data_fingerprint: str
    data_fingerprint_document: dict
    backend: object


def _urbanev_split_identity(bundle, dataset, split):
    split_slice = bundle.split_slice(split)
    timestamps = bundle.split_timestamps(split)
    return {
        "name": split,
        "start_idx": int(split_slice.start),
        "end_idx": int(split_slice.stop),
        "end_idx_semantics": "exclusive",
        "timestamp_count": len(timestamps),
        "start_timestamp": timestamps[0].isoformat(),
        "end_timestamp": timestamps[-1].isoformat(),
        "window_count": int(dataset.window_count),
        "temporal_region_sample_count": len(dataset),
        "history_context_policy": "split_local_no_borrowing",
    }


def _build_urbanev_runtime_data(args, train_generator):
    raw = UrbanEVRawData.load(args.data)
    bundle = UrbanEVFoldPreprocessor(raw).fit_transform(
        fold=args.fold,
        preset=args.feature_preset,
    )
    observed = {
        "feature_names": tuple(bundle.feature_names),
        "target_idx": bundle.target_idx,
        "target_feature_name": bundle.schema.target_name,
        "schema_fingerprint": bundle.feature_schema_fingerprint,
    }
    observed["aux_idx"] = tuple(
        index for index in range(len(bundle.feature_names))
        if index != bundle.target_idx
    )
    observed["aux_feature_names"] = tuple(
        bundle.feature_names[index] for index in observed["aux_idx"]
    )
    for name, value in observed.items():
        if getattr(args, name) != value:
            raise RuntimeError(
                f"M1 FoldBundle {name} changed after argument validation: "
                f"{getattr(args, name)!r} != {value!r}"
            )

    datasets = {
        split: TemporalRegionDataset(
            bundle,
            split=split,
            label_horizon=args.label_horizon,
            history_len=args.seq_len,
        )
        for split in SPLIT_NAMES
    }
    if len(datasets["train"]) < args.batch_size:
        raise ValueError(
            "UrbanEV training split has fewer temporal-region samples than "
            f"batch_size: {len(datasets['train'])} < {args.batch_size}"
        )
    train_data = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
    )
    val_data = DataLoader(
        datasets["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_data = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    split_identity = {
        split: _urbanev_split_identity(bundle, datasets[split], split)
        for split in SPLIT_NAMES
    }
    preprocessing = {
        "loader_kind": "urbanev_m1_temporal_region",
        "data_root": str(raw.data_root),
        "data_fingerprint": raw.data_fingerprint,
        "source_file_sha256": raw.file_sha256,
        "timestamp_order_sha256": raw.timestamp_order_sha256,
        "node_order_sha256": raw.node_order_sha256,
        "fold": bundle.fold_definition.to_dict(),
        "preset": bundle.schema.preset,
        "feature_schema": bundle.schema.to_dict(),
        "feature_schema_fingerprint": bundle.feature_schema_fingerprint,
        "preprocessing_state": bundle.preprocessing_state.to_dict(),
        "preprocessing_state_fingerprint": bundle.preprocessing_state_fingerprint,
        "columns": list(bundle.feature_names),
        "target": bundle.schema.target_name,
        "target_indices": [bundle.target_idx],
        "aux_idx": list(args.aux_idx),
        "aux_feature_names": list(args.aux_feature_names),
        "history_len": args.seq_len,
        "label_horizon": args.label_horizon,
        "model_pred_len": args.model_pred_len,
        "split_identity": split_identity,
        "node_count": len(bundle.node_ids),
        "timestamp_semantics": raw.timestamp_semantics,
        "timezone": raw.timezone,
    }
    fingerprint_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_schema_version": ENHANCED_ARTIFACT_SCHEMA_VERSION,
        "algorithm": "M1_deterministic_fingerprint_contract",
        "dataset_id": args.dataset_id,
        "path": str(raw.data_root),
        "sha256": raw.data_fingerprint,
        "data_fingerprint": raw.data_fingerprint,
        "preprocessing_state_fingerprint": bundle.preprocessing_state_fingerprint,
        "schema_fingerprint": bundle.feature_schema_fingerprint,
        "timestamp_order_sha256": raw.timestamp_order_sha256,
        "node_order_sha256": raw.node_order_sha256,
        "fold": args.fold,
        "preset": bundle.schema.preset,
        "label_horizon": args.label_horizon,
        "model_pred_len": args.model_pred_len,
        "artifact_horizon": args.artifact_horizon,
        "target_name": bundle.schema.target_name,
        "feature_names": list(bundle.feature_names),
        "target_idx": bundle.target_idx,
        "aux_idx": list(args.aux_idx),
        "aux_feature_names": list(args.aux_feature_names),
        "split_identity": split_identity,
    }
    return RuntimeData(
        n_feature=len(bundle.feature_names),
        target_slice=None,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        preprocessing=preprocessing,
        data_fingerprint=raw.data_fingerprint,
        data_fingerprint_document=fingerprint_document,
        backend=bundle,
    )


def _build_generic_runtime_data(args, train_generator):
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"dataset not found: {data_path}")
    data_fingerprint = sha256_file(data_path)
    loader = CustomDataLoader(
        args.data,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target,
        dataset_id=args.dataset_id,
    )
    train_data = loader.get_train(generator=train_generator)
    preprocessing = loader.metadata()
    fingerprint_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_schema_version": (
            ENHANCED_ARTIFACT_SCHEMA_VERSION
            if args.implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT
            else None
        ),
        "algorithm": "sha256_file_bytes",
        "dataset_id": args.dataset_id,
        "path": args.data,
        "sha256": data_fingerprint,
        "schema_fingerprint": getattr(args, "schema_fingerprint", None),
        "loader_kind": "generic_custom_data_loader",
    }
    return RuntimeData(
        n_feature=loader.n_feature,
        target_slice=loader.target_slice,
        train_data=train_data,
        val_data=loader.get_val(),
        test_data=loader.get_test(),
        preprocessing=preprocessing,
        data_fingerprint=data_fingerprint,
        data_fingerprint_document=fingerprint_document,
        backend=loader,
    )


def _build_runtime_data(args, train_generator):
    if _is_urbanev_production(args):
        return _build_urbanev_runtime_data(args, train_generator)
    return _build_generic_runtime_data(args, train_generator)

def _validate_loader_contract(args, data_loader, preprocessing):
    if args.implementation_variant != ENHANCED_IMPLEMENTATION_VARIANT:
        return
    observed_names = tuple(str(name) for name in preprocessing["columns"])
    if observed_names != args.feature_names:
        raise ValueError(
            "feature_names do not match loader column order: "
            f"expected {args.feature_names}, observed {observed_names}"
        )
    if data_loader.n_feature != len(args.feature_names):
        raise ValueError("loader feature width does not match feature_names")
    if args.task_mode == TARGET_EXOGENOUS:
        target_indices = tuple(preprocessing["target_indices"])
        if target_indices != (args.target_idx,):
            raise ValueError(
                "target_idx does not match the loader's resolved target: "
                f"{args.target_idx} != {target_indices}"
            )


def _build_model(args, data_loader):
    common = {
        "input_shape": (args.seq_len, data_loader.n_feature),
        "pred_len": args.pred_len,
        "dropout": args.dropout,
        "n_block": args.n_block,
        "patch": args.patch,
        "k": args.mix_layer_num,
        "c": args.mix_layer_scale,
        "alpha": args.alpha,
        "norm": args.norm,
        "layernorm": args.layernorm,
    }
    if args.implementation_variant == BASELINE_IMPLEMENTATION_VARIANT:
        return AMD(
            **common,
            target_slice=data_loader.target_slice,
        )
    return AMDEnhanced(
        **common,
        target_slice=None,
        target_idx=args.target_idx,
        teb_context_dim=args.teb_context_dim,
        task_mode=args.task_mode,
        aux_idx=args.aux_idx,
        use_pmcr=args.use_pmcr,
        pmcr_hidden_dim=args.pmcr_hidden_dim,
        pmcr_kernel_small=args.pmcr_kernel_small,
        pmcr_kernel_large=args.pmcr_kernel_large,
        pmcr_dropout=args.pmcr_dropout,
        pmcr_gamma_init=args.pmcr_gamma_init,
        use_teb=args.use_teb,
        teb_heads=args.teb_heads,
        teb_dropout=args.teb_dropout,
        teb_gamma_init=args.teb_gamma_init,
    )


def _main_impl(args, transcript=None):
    args = prepare_args(args)
    device = _resolve_device(args.device)
    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)

    source_sha256 = source_fingerprint()
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    data_loader = _build_runtime_data(args, train_generator)
    train_data = data_loader.train_data
    val_data = data_loader.val_data
    test_data = data_loader.test_data
    if len(train_data) == 0:
        raise ValueError(
            "training loader has zero full batches; batch_size exceeds the available training windows"
        )

    data_sha256 = data_loader.data_fingerprint
    preprocessing = data_loader.preprocessing
    _validate_loader_contract(args, data_loader, preprocessing)
    environment = environment_metadata(device)
    scientific = _scientific_config(
        args, data_sha256, source_sha256, preprocessing, device, environment
    )
    config_hash = stable_hash(scientific)
    artifact_paths = _artifact_paths(args)
    run_dir = artifact_paths.work_dir
    final_run_dir = artifact_paths.final_dir
    run_id = artifact_paths.run_id
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.resolved.json"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.jsonl"
    metrics_path = run_dir / "metrics.json"

    source = {"sha256": source_sha256, "git": git_metadata()}
    resolved_config = _resolved_config(
        args, scientific, config_hash, final_run_dir, source, environment
    )
    run_started = time.time()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_schema_version": (
            ENHANCED_ARTIFACT_SCHEMA_VERSION
            if artifact_paths.is_enhanced
            else None
        ),
        "implementation_variant": args.implementation_variant,
        "run_id": run_id,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "config_hash": config_hash,
        "data_sha256": data_sha256,
        "artifact_dir": str(final_run_dir),
        "publication": (
            {
                "mode": "same_filesystem_staging_atomic_directory_rename",
                "staging_dir": str(run_dir),
            }
            if artifact_paths.is_enhanced
            else {"mode": "legacy_direct"}
        ),
    }
    if artifact_paths.is_enhanced:
        manifest.update({
            "task_mode": args.task_mode,
            "target": args.target,
            "label_horizon": args.label_horizon,
            "model_pred_len": args.model_pred_len,
            "artifact_horizon": args.artifact_horizon,
            "fold": args.fold,
            "seed": args.seed,
        })
    resume_checkpoint = None
    previous_config = None
    manifest_is_mutable = False
    artifact_sealed = False
    run_committed = False
    elapsed_before_resume = 0.0
    run_lock = RunLock(run_dir).acquire()

    # Validate a resume before writing anything into the existing run.  A typo,
    # a config mismatch, or an attempt to resume an immutable completed run must
    # leave the original manifest and config byte-for-byte untouched.
    try:
        if args.resume:
            manifest, resume_checkpoint, previous_config = _load_resume_checkpoint(
                run_dir,
                config_hash,
                data_sha256,
                args.train_epochs,
                implementation_variant=args.implementation_variant,
                run_id=run_id,
                artifact_dir=final_run_dir,
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

        if args.implementation_variant == ENHANCED_IMPLEMENTATION_VARIANT:
            if transcript is None:
                raise RuntimeError("enhanced runner requires an active run transcript")
            transcript.bind(run_dir)
            _write_enhanced_provenance(
                args,
                run_dir,
                source_sha256,
                data_sha256,
                data_loader.data_fingerprint_document,
            )
    except BaseException:
        run_lock.release()
        raise

    try:
        atomic_write_json(config_path, resolved_config)
        atomic_write_json(manifest_path, manifest)
        model = _build_model(args, data_loader).to(device)
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
                task_mode=args.task_mode,
            )
            val_metrics = evaluate(
                model, val_data, device,
                description=f"Val {epoch_number}/{args.train_epochs}",
                show_progress=args.progress,
                task_mode=args.task_mode,
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
        if best_checkpoint.get("implementation_variant") != args.implementation_variant:
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
            task_mode=args.task_mode,
        )
        elapsed = elapsed_before_resume + time.time() - run_started
        completed_at = _utc_now()
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "artifact_schema_version": (
                ENHANCED_ARTIFACT_SCHEMA_VERSION
                if artifact_paths.is_enhanced
                else None
            ),
            "implementation_variant": args.implementation_variant,
            "run_id": run_id,
            "status": "completed",
            "dataset_id": args.dataset_id,
            "task_mode": args.task_mode,
            "target": args.target,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "label_horizon": getattr(args, "label_horizon", None),
            "model_pred_len": getattr(args, "model_pred_len", args.pred_len),
            "artifact_horizon": getattr(args, "artifact_horizon", args.pred_len),
            "fold": args.fold,
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
            "artifact_dir": str(final_run_dir),
            "completed_at": completed_at,
        }
        atomic_write_json(metrics_path, metrics)
        completed_manifest = deepcopy(manifest)
        completed_manifest.update({
            "status": "completed",
            "updated_at": completed_at,
            "completed_at": completed_at,
            "metrics_file": metrics_path.name,
            "task_mode": args.task_mode,
            "target": args.target,
            "label_horizon": getattr(args, "label_horizon", None),
            "model_pred_len": getattr(args, "model_pred_len", args.pred_len),
            "artifact_horizon": getattr(args, "artifact_horizon", args.pred_len),
            "fold": args.fold,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation_mse": best_mse,
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
        })
        completion_message = (
            f"completed run={final_run_dir}\n"
            f"best_epoch={best_epoch} val_mse={best_mse:.8g} "
            f"test_mse={test_metrics['mse']:.8g} test_mae={test_metrics['mae']:.8g}"
        )
        if artifact_paths.is_enhanced:
            completed_manifest["checksum_contract"] = {
                "algorithm": "sha256",
                "file": "checksums.sha256",
                "required_files": list(ENHANCED_CHECKSUM_FILES),
                "validation": [
                    "python_exact_file_set_and_digest",
                    "system_sha256sum_-c",
                ],
            }
            print(completion_message)
            artifact_sealed = True
            _publish_enhanced_artifact(
                staging_dir=run_dir,
                final_dir=final_run_dir,
                transcript=transcript,
                run_lock=run_lock,
                manifest_path=manifest_path,
                completed_manifest=completed_manifest,
            )
            run_committed = True
        else:
            atomic_write_json(manifest_path, completed_manifest)
            run_committed = True
            print(completion_message)
        return metrics
    except BaseException as exc:
        if manifest_is_mutable and not run_committed and not artifact_sealed:
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


def main(args):
    """Run one experiment, adding native transcripts for the enhanced variant."""

    is_enhanced = (
        getattr(args, "implementation_variant", IMPLEMENTATION_VARIANT)
        == ENHANCED_IMPLEMENTATION_VARIANT
    )
    transcript = RunTranscript().install() if is_enhanced else None
    try:
        return _main_impl(args, transcript=transcript)
    finally:
        if transcript is not None:
            try:
                transcript.finalize()
            finally:
                transcript.restore()


if __name__ == "__main__":
    main(parse_args())
