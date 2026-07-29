"""Shared, strict checkpoint loading and structured checkpoint saving."""

from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


MODEL_STATE_KEYS = ("model_state_dict", "state_dict", "model")
CHECKPOINT_METADATA_FIELDS = (
    "dataset",
    "n_channels",
    "n_classes",
    "bilinear",
    "img_size",
    "class_names",
    "normalization",
    "arguments",
)
_VALID_LOAD_MODES = frozenset(("test", "resume", "init"))


@dataclass
class CheckpointLoadResult:
    """Detailed report returned by :func:`load_checkpoint`."""

    path: Path
    mode: str
    structured: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    epoch: Optional[int] = None
    best_mean_dice: Optional[float] = None
    global_step: Optional[int] = None
    mask_values: Any = None
    loaded_keys: Tuple[str, ...] = ()
    missing_keys: Tuple[str, ...] = ()
    unexpected_keys: Tuple[str, ...] = ()
    shape_mismatches: Dict[str, Dict[str, Tuple[int, ...]]] = field(
        default_factory=dict
    )
    metadata_mismatches: Tuple[str, ...] = ()
    optimizer_restored: bool = False
    scheduler_restored: bool = False
    scaler_restored: bool = False
    rng_state: Any = None

    @property
    def partial(self) -> bool:
        return bool(
            self.missing_keys
            or self.unexpected_keys
            or self.shape_mismatches
            or self.metadata_mismatches
        )

    @property
    def skipped_keys(self) -> Tuple[str, ...]:
        """Return every checkpoint tensor omitted by partial initialization."""

        return tuple(
            dict.fromkeys(self.unexpected_keys + tuple(self.shape_mismatches.keys()))
        )

    @property
    def next_epoch(self) -> int:
        """Return the conventional first epoch after the saved epoch."""

        return 0 if self.epoch is None else int(self.epoch) + 1

    @property
    def iter_num(self) -> Optional[int]:
        """TransUNet-compatible alias for the restored optimization step."""

        return self.global_step

    def summary(self) -> str:
        details = [
            "checkpoint={}".format(self.path),
            "mode={}".format(self.mode),
            "loaded_keys={}".format(len(self.loaded_keys)),
        ]
        if self.missing_keys:
            details.append("missing_keys={}".format(list(self.missing_keys)))
        if self.unexpected_keys:
            details.append("unexpected_keys={}".format(list(self.unexpected_keys)))
        if self.shape_mismatches:
            details.append("shape_mismatches={}".format(self.shape_mismatches))
        if self.metadata_mismatches:
            details.append("metadata_mismatches={}".format(list(self.metadata_mismatches)))
        if self.global_step is not None:
            details.append("global_step={}".format(self.global_step))
        return "; ".join(details)


def _unwrap_model(model):
    data_parallel_types = (torch.nn.DataParallel,)
    distributed_type = getattr(torch.nn.parallel, "DistributedDataParallel", None)
    if distributed_type is not None:
        data_parallel_types = data_parallel_types + (distributed_type,)
    if isinstance(model, data_parallel_types):
        return model.module
    return model


def _torch_load(path: Path, map_location):
    """Load old and new PyTorch checkpoints across ``weights_only`` defaults."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch releases predating the weights_only argument.
        return torch.load(path, map_location=map_location)


def load_checkpoint_file(checkpoint_path, map_location="cpu"):
    """Read a checkpoint after validating that its explicit path exists."""

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(path))
    if not path.is_file():
        raise FileNotFoundError("Checkpoint path is not a file: {}".format(path))
    checkpoint = _torch_load(path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint {} must contain a mapping, got {}".format(
                path, type(checkpoint).__name__
            )
        )
    return checkpoint


def _looks_like_raw_state_dict(checkpoint: Mapping[str, Any]) -> bool:
    tensor_items = [
        value
        for key, value in checkpoint.items()
        if key != "mask_values" and torch.is_tensor(value)
    ]
    non_tensor_keys = [
        key
        for key, value in checkpoint.items()
        if key != "mask_values" and not torch.is_tensor(value)
    ]
    return bool(tensor_items) and not non_tensor_keys


def extract_model_state_dict(
    checkpoint: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], bool, Any]:
    """Return ``(state_dict, is_structured, mask_values)``.

    Besides the canonical ``model_state_dict`` field, ``state_dict`` and
    ``model`` are accepted for interoperability.  Legacy raw state dictionaries
    may contain a non-parameter ``mask_values`` entry.
    """

    state_dict = None
    structured = False
    for key in MODEL_STATE_KEYS:
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state_dict = candidate
            structured = True
            break

    if state_dict is None:
        if not _looks_like_raw_state_dict(checkpoint):
            raise ValueError(
                "Checkpoint contains neither a model_state_dict nor a recognizable raw state_dict"
            )
        state_dict = checkpoint

    mask_values = checkpoint.get("mask_values") if structured else None
    if "mask_values" in state_dict:
        if mask_values is None:
            mask_values = state_dict["mask_values"]
        filtered_state_dict = OrderedDict(
            (key, value) for key, value in state_dict.items() if key != "mask_values"
        )
        if hasattr(state_dict, "_metadata"):
            filtered_state_dict._metadata = state_dict._metadata
        state_dict = filtered_state_dict
    return state_dict, structured, mask_values


def normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> "OrderedDict[str, Any]":
    """Strip one or more leading ``module.`` prefixes without collisions."""

    normalized = OrderedDict()
    for original_key, value in state_dict.items():
        if not isinstance(original_key, str):
            raise TypeError(
                "State-dict keys must be strings, got {}".format(
                    type(original_key).__name__
                )
            )
        key = original_key
        while key.startswith("module."):
            key = key[len("module.") :]
        if key in normalized:
            raise ValueError(
                "State-dict key collision after normalizing module. prefixes: {}".format(
                    key
                )
            )
        normalized[key] = value
    source_metadata = getattr(state_dict, "_metadata", None)
    if isinstance(source_metadata, Mapping):
        normalized_metadata = OrderedDict()
        for original_key, value in source_metadata.items():
            key = original_key
            while key == "module" or key.startswith("module."):
                key = "" if key == "module" else key[len("module.") :]
            normalized_metadata[key] = value
        normalized._metadata = normalized_metadata
    return normalized


def checkpoint_metadata(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Collect supported metadata from nested and canonical top-level fields."""

    metadata = {}
    nested_metadata = checkpoint.get("metadata")
    if isinstance(nested_metadata, Mapping):
        metadata.update(dict(nested_metadata))
    for key in CHECKPOINT_METADATA_FIELDS:
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    return metadata


def _canonical_dataset_name(value):
    if value is None:
        return None
    text = str(value).strip()
    known = {
        "synapse": "Synapse",
        "acdc": "ACDC",
        "cataract1k": "Cataract1k",
        "catrakt1k": "Cataract1k",
        "carvana": "Carvana",
    }
    return known.get(text.lower(), text)


def _coerce_bool(value, field_name: str) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError("Invalid boolean metadata for {}: {!r}".format(field_name, value))
    return bool(value)


def checkpoint_metadata_mismatches(
    metadata: Mapping[str, Any],
    *,
    dataset=None,
    n_channels=None,
    n_classes=None,
    bilinear=None,
) -> Tuple[str, ...]:
    """Return every selected-model/checkpoint metadata disagreement."""

    expected = {
        "dataset": dataset,
        "n_channels": n_channels,
        "n_classes": n_classes,
        "bilinear": bilinear,
    }
    mismatches = []
    for key, expected_value in expected.items():
        if expected_value is None or key not in metadata or metadata[key] is None:
            continue
        actual_value = metadata[key]
        try:
            if key == "dataset":
                actual_comparable = _canonical_dataset_name(actual_value)
                expected_comparable = _canonical_dataset_name(expected_value)
            elif key in ("n_channels", "n_classes"):
                actual_comparable = int(actual_value)
                expected_comparable = int(expected_value)
            elif key == "bilinear":
                actual_comparable = _coerce_bool(actual_value, key)
                expected_comparable = _coerce_bool(expected_value, key)
            else:  # pragma: no cover - the expected set above is fixed.
                actual_comparable = actual_value
                expected_comparable = expected_value
        except (TypeError, ValueError) as error:
            mismatches.append(
                "{} metadata is invalid ({!r}): {}".format(key, actual_value, error)
            )
            continue
        if actual_comparable != expected_comparable:
            mismatches.append(
                "{}: checkpoint={!r}, selected={!r}".format(
                    key, actual_value, expected_value
                )
            )
    return tuple(mismatches)


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    dataset=None,
    n_channels=None,
    n_classes=None,
    bilinear=None,
) -> None:
    """Raise if metadata present in a checkpoint disagrees with the model."""

    mismatches = checkpoint_metadata_mismatches(
        metadata,
        dataset=dataset,
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
    )
    if mismatches:
        raise ValueError(
            "Checkpoint metadata is incompatible: {}".format("; ".join(mismatches))
        )


def _state_shape(value) -> Tuple[int, ...]:
    shape = getattr(value, "shape", ())
    return tuple(int(dimension) for dimension in shape)


def _load_partial_state_dict(model, state_dict):
    model_state = model.state_dict()
    compatible = OrderedDict()
    unexpected_keys = []
    shape_mismatches = {}

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        checkpoint_shape = _state_shape(value)
        model_shape = _state_shape(model_state[key])
        if checkpoint_shape != model_shape:
            shape_mismatches[key] = {
                "checkpoint": checkpoint_shape,
                "model": model_shape,
            }
            continue
        compatible[key] = value

    missing_keys = [key for key in model_state if key not in compatible]
    model.load_state_dict(compatible, strict=False)
    return (
        tuple(compatible.keys()),
        tuple(missing_keys),
        tuple(unexpected_keys),
        shape_mismatches,
    )


def _state_value(checkpoint: Mapping[str, Any], *keys):
    for key in keys:
        value = checkpoint.get(key)
        if value is not None:
            return value
    return None


def load_checkpoint(
    model,
    checkpoint_path,
    *,
    mode: str = "test",
    map_location="cpu",
    optimizer=None,
    scheduler=None,
    scaler=None,
    dataset=None,
    n_channels=None,
    n_classes=None,
    bilinear=None,
    allow_partial_init: bool = False,
) -> CheckpointLoadResult:
    """Load a checkpoint with strict test/resume/init semantics.

    ``mode`` must be ``"test"``, ``"resume"``, or ``"init"``.  All three
    modes load model tensors strictly by default.  Shape-compatible partial
    loading is available only when ``mode="init"`` and
    ``allow_partial_init=True``; every omitted tensor is returned in the load
    report.  Optimizer, scheduler, and scaler states are restored only in
    resume mode and only when present in the structured checkpoint.
    """

    mode = str(mode).lower()
    if mode not in _VALID_LOAD_MODES:
        raise ValueError(
            "mode must be one of {}, got {!r}".format(
                sorted(_VALID_LOAD_MODES), mode
            )
        )
    if allow_partial_init and mode != "init":
        raise ValueError("allow_partial_init is valid only for initialization checkpoints")
    if mode != "resume" and any(
        component is not None for component in (optimizer, scheduler, scaler)
    ):
        raise ValueError(
            "optimizer, scheduler, and scaler may be restored only in resume mode"
        )

    path = Path(checkpoint_path).expanduser()
    checkpoint = load_checkpoint_file(path, map_location=map_location)
    raw_state_dict, structured, mask_values = extract_model_state_dict(checkpoint)
    state_dict = normalize_state_dict_keys(raw_state_dict)
    metadata = checkpoint_metadata(checkpoint) if structured else {}

    target_model = _unwrap_model(model)
    if n_channels is None:
        n_channels = getattr(target_model, "n_channels", None)
    if n_classes is None:
        n_classes = getattr(target_model, "n_classes", None)
    if bilinear is None:
        bilinear = getattr(target_model, "bilinear", None)
    metadata_mismatches = checkpoint_metadata_mismatches(
        metadata,
        dataset=dataset,
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
    )
    if metadata_mismatches and not (mode == "init" and allow_partial_init):
        raise ValueError(
            "Checkpoint metadata is incompatible: {}".format(
                "; ".join(metadata_mismatches)
            )
        )

    if mode == "init" and allow_partial_init:
        loaded_keys, missing_keys, unexpected_keys, shape_mismatches = (
            _load_partial_state_dict(target_model, state_dict)
        )
        logging.info(
            "Partial checkpoint initialization loaded %d shape-compatible keys",
            len(loaded_keys),
        )
        logging.warning(
            "Partial checkpoint initialization missing model keys: %s",
            list(missing_keys),
        )
        logging.warning(
            "Partial checkpoint initialization unexpected checkpoint keys: %s",
            list(unexpected_keys),
        )
        logging.warning(
            "Partial checkpoint initialization shape mismatches: %s",
            shape_mismatches,
        )
        if metadata_mismatches:
            logging.warning(
                "Partial checkpoint initialization metadata mismatches: %s",
                list(metadata_mismatches),
            )
    else:
        try:
            incompatible = target_model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            suffix = (
                " Use allow_partial_init=True only for an explicitly requested "
                "partial initialization."
                if mode == "init"
                else ""
            )
            raise RuntimeError(
                "Strict {} checkpoint load failed for {}: {}{}".format(
                    mode, path, error, suffix
                )
            ) from error
        loaded_keys = tuple(state_dict.keys())
        missing_keys = tuple(incompatible.missing_keys)
        unexpected_keys = tuple(incompatible.unexpected_keys)
        shape_mismatches = {}

    optimizer_restored = False
    scheduler_restored = False
    scaler_restored = False
    if mode == "resume":
        optimizer_state = _state_value(checkpoint, "optimizer_state_dict", "optimizer")
        scheduler_state = _state_value(checkpoint, "scheduler_state_dict", "scheduler")
        scaler_state = _state_value(
            checkpoint, "scaler_state_dict", "grad_scaler_state_dict", "scaler"
        )
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            optimizer_restored = True
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            scheduler_restored = True
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
            scaler_restored = True

    epoch = checkpoint.get("epoch") if structured else None
    if epoch is not None:
        epoch = int(epoch)
    global_step = checkpoint.get("global_step") if structured else None
    iter_num = checkpoint.get("iter_num") if structured else None
    if global_step is not None and iter_num is not None:
        if int(global_step) != int(iter_num):
            raise ValueError(
                "Checkpoint global_step={} disagrees with iter_num={}".format(
                    global_step,
                    iter_num,
                )
            )
    if global_step is None:
        global_step = iter_num
    if global_step is not None:
        global_step = int(global_step)
        if global_step < 0:
            raise ValueError(
                "Checkpoint global_step/iter_num must be non-negative, got {}".format(
                    global_step
                )
            )
    best_mean_dice = checkpoint.get("best_mean_dice") if structured else None
    if best_mean_dice is not None:
        best_mean_dice = float(best_mean_dice)

    return CheckpointLoadResult(
        path=path,
        mode=mode,
        structured=structured,
        metadata=metadata,
        epoch=epoch,
        global_step=global_step,
        best_mean_dice=best_mean_dice,
        mask_values=mask_values,
        loaded_keys=loaded_keys,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
        metadata_mismatches=metadata_mismatches,
        optimizer_restored=optimizer_restored,
        scheduler_restored=scheduler_restored,
        scaler_restored=scaler_restored,
        rng_state=checkpoint.get("rng_state") if structured else None,
    )


def _arguments_dict(arguments) -> Dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, argparse.Namespace):
        return dict(vars(arguments))
    if isinstance(arguments, Mapping):
        return dict(arguments)
    try:
        return dict(vars(arguments))
    except TypeError as error:
        raise TypeError("arguments must be a mapping or argparse namespace") from error


def build_checkpoint(
    model,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    best_mean_dice: Optional[float] = None,
    dataset: Optional[str] = None,
    n_channels: Optional[int] = None,
    n_classes: Optional[int] = None,
    bilinear: Optional[bool] = None,
    img_size: Optional[int] = None,
    class_names=None,
    normalization=None,
    arguments=None,
    mask_values=None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical structured, resume-capable checkpoint mapping."""

    target_model = _unwrap_model(model)
    if n_channels is None:
        n_channels = getattr(target_model, "n_channels", None)
    if n_classes is None:
        n_classes = getattr(target_model, "n_classes", None)
    if bilinear is None:
        bilinear = getattr(target_model, "bilinear", None)
    if global_step is not None and int(global_step) < 0:
        raise ValueError(
            "global_step must be non-negative, got {}".format(global_step)
        )

    checkpoint = {
        "model_state_dict": target_model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "epoch": None if epoch is None else int(epoch),
        "best_mean_dice": (
            None if best_mean_dice is None else float(best_mean_dice)
        ),
        "dataset": None if dataset is None else _canonical_dataset_name(dataset),
        "n_channels": None if n_channels is None else int(n_channels),
        "n_classes": None if n_classes is None else int(n_classes),
        "bilinear": None if bilinear is None else bool(bilinear),
        "img_size": None if img_size is None else int(img_size),
        "class_names": None if class_names is None else list(class_names),
        "normalization": normalization,
        "arguments": _arguments_dict(arguments),
    }
    if global_step is not None:
        checkpoint["global_step"] = int(global_step)
        checkpoint["iter_num"] = int(global_step)
    if mask_values is not None:
        checkpoint["mask_values"] = mask_values
    if extra is not None:
        collisions = sorted(set(extra).intersection(checkpoint))
        if collisions:
            raise ValueError(
                "extra checkpoint fields cannot replace canonical fields: {}".format(
                    collisions
                )
            )
        checkpoint.update(dict(extra))
    return checkpoint


def save_checkpoint(
    checkpoint_path,
    model,
    **checkpoint_fields,
) -> Path:
    """Build and save a structured checkpoint, creating its parent directory."""

    path = Path(checkpoint_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = build_checkpoint(model, **checkpoint_fields)
    torch.save(checkpoint, path)
    return path


def load_initialization_checkpoint(
    model,
    checkpoint_path,
    *,
    map_location="cpu",
    dataset=None,
    allow_partial: bool = False,
) -> CheckpointLoadResult:
    """Convenience wrapper for exact or explicitly partial initialization."""

    return load_checkpoint(
        model,
        checkpoint_path,
        mode="init",
        map_location=map_location,
        dataset=dataset,
        allow_partial_init=allow_partial,
    )


__all__ = [
    "CHECKPOINT_METADATA_FIELDS",
    "CheckpointLoadResult",
    "MODEL_STATE_KEYS",
    "build_checkpoint",
    "checkpoint_metadata",
    "checkpoint_metadata_mismatches",
    "extract_model_state_dict",
    "load_checkpoint",
    "load_checkpoint_file",
    "load_initialization_checkpoint",
    "normalize_state_dict_keys",
    "save_checkpoint",
    "validate_checkpoint_metadata",
]
