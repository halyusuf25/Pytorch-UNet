"""Reproducibility and device helpers used by command-line entry points."""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except (AttributeError, TypeError):
        pass


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def capture_rng_state(
    *,
    train_generator: torch.Generator | None = None,
    validation_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Capture epoch-boundary RNG state for exact structured resume."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader_generator": (
            None if train_generator is None else train_generator.get_state()
        ),
        "validation_loader_generator": (
            None if validation_generator is None else validation_generator.get_state()
        ),
    }


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    train_generator: torch.Generator | None = None,
    validation_generator: torch.Generator | None = None,
) -> None:
    """Restore every state emitted by :func:`capture_rng_state`."""

    required = {
        "python",
        "numpy",
        "torch",
        "cuda",
        "train_loader_generator",
        "validation_loader_generator",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Resume checkpoint RNG state is incomplete; missing {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    torch.set_rng_state(torch_state.detach().cpu() if torch.is_tensor(torch_state) else torch_state)
    cuda_state = state["cuda"]
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [value.detach().cpu() if torch.is_tensor(value) else value for value in cuda_state]
        )
    train_state = state["train_loader_generator"]
    validation_state = state["validation_loader_generator"]
    if train_generator is not None:
        if train_state is None:
            raise ValueError("Resume checkpoint lacks the training DataLoader generator state")
        train_generator.set_state(
            train_state.detach().cpu() if torch.is_tensor(train_state) else train_state
        )
    if validation_generator is not None:
        if validation_state is None:
            raise ValueError("Resume checkpoint lacks the validation DataLoader generator state")
        validation_generator.set_state(
            validation_state.detach().cpu()
            if torch.is_tensor(validation_state)
            else validation_state
        )


def resolve_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_num_workers() -> int:
    return min(8, int(os.cpu_count() or 1))


def jsonable_arguments(namespace: Any) -> dict[str, Any]:
    return {key: value for key, value in vars(namespace).items()}
