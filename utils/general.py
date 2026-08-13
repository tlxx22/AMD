"""Reproducibility helpers shared by training and checkpoint resume."""

import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed=0):
    """Seed the RNGs used by the released training code.

    The public implementation already requested deterministic cuDNN kernels.  We
    preserve that behavior here instead of enabling PyTorch's stricter global
    deterministic mode, which can change kernel availability and performance.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    """Return all RNG state required to resume at an epoch boundary."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    """Restore a state produced by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state, but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)
