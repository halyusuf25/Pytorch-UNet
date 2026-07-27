"""Reproducible real-sample inference benchmarks for the repository U-Net.

The timing paths in this module consume batches which have already been loaded,
resized, and normalized by :func:`collate_for_benchmark`.  Consequently the
reported latency measures model inference rather than dataset I/O or CPU image
preprocessing.  FLOPs follow the project convention ``FLOPs = 2 * MACs``.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from functools import partial
from pathlib import Path, PurePath
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset


DEFAULT_BENCHMARK_BATCH_SIZE = 36
DEFAULT_WARMUP_STEPS = 20
DEFAULT_MEASURE_BATCHES = 50
DEFAULT_SINGLE_IMAGE_LATENCY_SAMPLES = 1000
DEFAULT_SINGLE_IMAGE_WARMUP_STEPS = 50
DEFAULT_REPEATED_RUNS = 1

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================= JSON-safe results =============================

def _make_json_safe(value: Any) -> Any:
    """Recursively convert scientific-Python values to strict JSON values.

    Undefined floating-point metrics remain undefined in memory, but become
    ``None`` in serialized output.  This is important for metrics such as HD95:
    replacing an undefined result with zero would incorrectly imply perfection.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _make_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _make_json_safe(value.tolist())
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return _make_json_safe(tensor.item() if tensor.ndim == 0 else tensor.tolist())
    if isinstance(value, (torch.device, torch.dtype, PurePath)):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if is_dataclass(value) and not isinstance(value, type):
        return _make_json_safe({
            item.name: getattr(value, item.name) for item in dataclass_fields(value)
        })
    if isinstance(value, Mapping):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_make_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _make_json_safe(vars(value))
    return str(value)


def _make_console_safe(value: Any) -> Any:
    """Prepare nested values for display while retaining NaN/inf semantics."""

    if isinstance(value, np.generic):
        return _make_console_safe(value.item())
    if isinstance(value, np.ndarray):
        return _make_console_safe(value.tolist())
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return _make_console_safe(tensor.item() if tensor.ndim == 0 else tensor.tolist())
    if isinstance(value, (torch.device, torch.dtype, PurePath)):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if is_dataclass(value) and not isinstance(value, type):
        return _make_console_safe({
            item.name: getattr(value, item.name) for item in dataclass_fields(value)
        })
    if isinstance(value, Mapping):
        return {str(key): _make_console_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_make_console_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _make_console_safe(vars(value))
    return value


@dataclass
class BenchmarkResults:
    """Accuracy/system metrics and reproducibility metadata.

    Keep this two-field constructor stable: callers combine their accuracy
    diagnostics with the system metrics before printing and saving one result.
    """

    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return the stable ``{metrics, notes}`` strict-JSON representation."""

        return {
            "metrics": _make_json_safe(self.metrics),
            "notes": _make_json_safe(self.notes),
        }

    def pretty(self) -> str:
        """Render scalar values compactly and nested diagnostics readably."""

        lines = ["== Segmentation Inference Benchmark =="]
        for key, raw_value in self.metrics.items():
            value = _make_console_safe(raw_value)
            label = f"{key:28s}: "
            if isinstance(value, (dict, list)):
                rendered = json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=True,
                )
                rendered_lines = rendered.splitlines()
                lines.append(label + rendered_lines[0])
                padding = " " * len(label)
                lines.extend(padding + line for line in rendered_lines[1:])
            elif value is None:
                lines.append(label + "null")
            elif isinstance(value, float) and "gflops" in key:
                lines.append(label + f"{value:.3f}")
            elif isinstance(value, float) and "img_s" in key:
                lines.append(label + f"{value:.2f}")
            elif isinstance(value, float) and (
                "_ms" in key or key.endswith(("_mib", "_mb"))
            ):
                lines.append(label + f"{value:.2f}")
            elif isinstance(value, float) and (
                key == "params" or key.endswith("_params_m")
            ):
                lines.append(label + f"{value:.3f}")
            elif isinstance(value, int) and key.endswith("_params"):
                lines.append(label + f"{value:,}")
            else:
                lines.append(label + str(value))
        return "\n".join(lines)

    def save_json(self, path: str | Path) -> Path:
        """Save strict JSON, creating the destination directory if necessary."""

        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        return output_path


# ============================= Real-sample collate ============================

def _extract_img_only(sample: Any) -> Any:
    """Extract the image while intentionally discarding labels and metadata."""

    if isinstance(sample, Mapping):
        for key in ("image", "img", "images", "inputs", "x"):
            if key in sample:
                return sample[key]
    if isinstance(sample, (list, tuple)) and sample:
        return sample[0]
    return sample


def _declared_model_channels(model: Optional[nn.Module]) -> Optional[int]:
    if model is None:
        return None
    candidates = (model, getattr(model, "module", None))
    for candidate in candidates:
        channels = getattr(candidate, "n_channels", None)
        if channels is not None:
            return int(channels)
    # This keeps the helper useful for small test models while U-Net callers
    # continue to use the authoritative ``n_channels`` attribute.
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return int(module.in_channels)
    return None


def _resolve_expected_channels(
    *,
    input_channels: Optional[int],
    expected_channels: Optional[int],
    model: Optional[nn.Module],
) -> Optional[int]:
    declared = _declared_model_channels(model)
    supplied = [
        int(value)
        for value in (input_channels, expected_channels, declared)
        if value is not None
    ]
    if supplied and any(value != supplied[0] for value in supplied[1:]):
        raise ValueError(
            "Conflicting benchmark channel counts were supplied: "
            + ", ".join(str(value) for value in supplied)
        )
    resolved = supplied[0] if supplied else None
    if resolved is not None and resolved <= 0:
        raise ValueError(f"input_channels must be positive, got {resolved}")
    return resolved


def _as_channel_aware_chw(
    image: Any,
    *,
    expected_channels: Optional[int],
    volume_input: bool,
) -> torch.Tensor:
    """Convert a 2D slice, DHW volume, HWC RGB, or CHW image to float CHW."""

    source_was_numpy = isinstance(image, np.ndarray)
    if source_was_numpy:
        # ``torch.from_numpy`` rejects the negative strides produced by flips.
        image = torch.from_numpy(np.ascontiguousarray(image))
    elif not torch.is_tensor(image):
        image = torch.as_tensor(image)

    tensor = image.detach().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and volume_input:
        # Dataset context removes the otherwise unavoidable ambiguity between
        # a depth-three medical volume and a 3,H,W RGB tensor.
        tensor = tensor[int(tensor.shape[0]) // 2].unsqueeze(0)
    elif tensor.ndim == 3:
        first = int(tensor.shape[0])
        last = int(tensor.shape[-1])

        first_matches = first in (1, 3)
        last_matches = last in (1, 3)
        if expected_channels is not None:
            first_matches = first == expected_channels
            last_matches = last == expected_channels

        if last_matches and not first_matches:
            tensor = tensor.permute(2, 0, 1)
        elif first_matches and not last_matches:
            pass
        elif first_matches and last_matches:
            # In the rare ambiguous shape, raw NumPy dataset frames are HWC
            # while tensors produced by PyTorch transforms are normally CHW.
            if source_was_numpy:
                tensor = tensor.permute(2, 0, 1)
        elif expected_channels == 1 or expected_channels is None:
            tensor = tensor[int(tensor.shape[0]) // 2].unsqueeze(0)
        else:
            raise ValueError(
                "Cannot interpret benchmark image as HWC or CHW with "
                f"{expected_channels} channels; received {tuple(tensor.shape)}"
            )
    else:
        raise ValueError(
            "Benchmark images must be 2D slices, 3D DHW volumes, or HWC/CHW "
            f"images; received shape {tuple(tensor.shape)}"
        )

    channels = int(tensor.shape[0])
    if expected_channels is not None and channels != expected_channels:
        raise ValueError(
            f"Benchmark preprocessing produced {channels} channels, but the model "
            f"expects {expected_channels}; grayscale inputs are not expanded to RGB"
        )
    if channels not in (1, 3):
        raise ValueError(
            f"Benchmark input must resolve to one grayscale or three RGB channels, got {channels}"
        )
    return tensor.contiguous()


def collate_for_benchmark(
    batch: Iterable[Any],
    img_size: int,
    *,
    input_channels: Optional[int] = None,
    expected_channels: Optional[int] = None,
    model: Optional[nn.Module] = None,
    volume_input: bool = False,
    normalize: bool = False,
    image_mean: Sequence[float] = IMAGENET_MEAN,
    image_std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Prepare real samples as contiguous ``B,C,img_size,img_size`` tensors.

    Synapse/ACDC volume loaders should set ``volume_input=True`` so a
    deterministic center slice is selected.  Cataract1k should set
    ``normalize=True`` to resize raw HWC RGB data and then apply the ImageNet
    preprocessing used during training.  All work here occurs before timing.
    """

    size = int(img_size)
    if size <= 0:
        raise ValueError(f"img_size must be positive, got {img_size!r}")
    channels = _resolve_expected_channels(
        input_channels=input_channels,
        expected_channels=expected_channels,
        model=model,
    )

    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError("image_mean and image_std must each contain three values")
    if any(float(value) <= 0 for value in image_std):
        raise ValueError("image_std values must be positive")
    if normalize and channels not in (None, 3):
        raise ValueError("ImageNet normalization is only valid for three-channel RGB input")

    images: List[torch.Tensor] = []
    for sample in batch:
        image = _as_channel_aware_chw(
            _extract_img_only(sample),
            expected_channels=channels,
            volume_input=bool(volume_input),
        )
        if tuple(image.shape[-2:]) != (size, size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if normalize:
            if int(image.shape[0]) != 3:
                raise ValueError(
                    "ImageNet normalization requires RGB input, got "
                    f"{int(image.shape[0])} channels"
                )
            mean = image.new_tensor(image_mean).view(3, 1, 1)
            std = image.new_tensor(image_std).view(3, 1, 1)
            image = (image / 255.0 - mean) / std
        images.append(image.contiguous())

    if not images:
        raise ValueError("Cannot collate an empty benchmark batch")
    try:
        result = torch.stack(images, dim=0).contiguous()
    except RuntimeError as error:
        shapes = [tuple(image.shape) for image in images]
        raise ValueError(f"Benchmark samples resolved to incompatible shapes: {shapes}") from error

    if result.ndim != 4 or tuple(result.shape[-2:]) != (size, size):
        raise RuntimeError(f"Unexpected benchmark batch shape {tuple(result.shape)}")
    if channels is not None and int(result.shape[1]) != channels:
        raise ValueError(
            f"Benchmark batch has {int(result.shape[1])} channels; expected {channels}"
        )
    return result


def _arg(args: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(args, Mapping) and name in args:
            return args[name]
        if hasattr(args, name):
            return getattr(args, name)
    return default


class _BenchmarkInputSpec(NamedTuple):
    name: str
    input_channels: int
    protocol: str
    imagenet_normalization: bool


def _benchmark_input_spec(dataset_name: str) -> _BenchmarkInputSpec:
    """Resolve preprocessing metadata without importing dataset dependencies.

    Registry construction remains authoritative whenever this module creates a
    dataset.  This small read-only projection lets callers benchmark an already
    constructed dataset in lightweight environments which may not have h5py or
    OpenCV imported yet.
    """

    key = "".join(
        character
        for character in str(dataset_name).strip().casefold()
        if character.isalnum()
    )
    if key == "synapse":
        return _BenchmarkInputSpec("Synapse", 1, "volume", False)
    if key == "acdc":
        return _BenchmarkInputSpec("ACDC", 1, "volume", False)
    if key in {"cataract1k", "catrakt1k"}:
        return _BenchmarkInputSpec("Cataract1k", 3, "present_class_frame", True)
    raise ValueError(
        f"Unknown dataset {dataset_name!r}. Supported datasets: Synapse, ACDC, Cataract1k"
    )


def build_benchmark_loader(
    args: Any = None,
    batch_size: int = DEFAULT_BENCHMARK_BATCH_SIZE,
    num_workers: int = 0,
    shuffle: bool = False,
    *,
    model: Optional[nn.Module] = None,
    dataset: Optional[Dataset] = None,
    dataset_name: Optional[str] = None,
    img_size: Optional[int] = None,
    input_channels: Optional[int] = None,
    normalize: Optional[bool] = None,
) -> DataLoader:
    """Build the canonical real test-split loader used for benchmarking.

    ``args`` may be an argparse-style namespace (in which case the test split
    is built through the registry), or it may be an already-created dataset
    when ``dataset_name`` and ``img_size`` are supplied explicitly.  The latter
    form lets accuracy evaluation and benchmarking reuse one real dataset.
    """

    if dataset is None and isinstance(args, Dataset):
        dataset = args
        args = None

    selected_dataset_name = dataset_name or _arg(args, "dataset")
    if not selected_dataset_name:
        raise ValueError("args.dataset is required to build a benchmark loader")
    selected_img_size = img_size
    if selected_img_size is None:
        selected_img_size = _arg(args, "img_size", default=224)
    size = int(selected_img_size)
    if size <= 0:
        raise ValueError(f"img_size must be positive, got {size}")

    if dataset is None:
        from utils.dataset_registry import build_test_dataset, get_dataset_spec

        registry_spec = get_dataset_spec(str(selected_dataset_name))
        spec = _BenchmarkInputSpec(
            registry_spec.name,
            registry_spec.input_channels,
            registry_spec.protocol,
            registry_spec.imagenet_normalization,
        )
        common_root = _arg(args, "data_root", "root_path", "root")
        volume_root = _arg(args, "volume_path", "volume_root", "test_root")
        list_dir = _arg(args, "list_dir")
        fold_id = _arg(args, "fold_id")
        dataset = build_test_dataset(
            registry_spec,
            size,
            data_root=common_root,
            volume_root=volume_root,
            list_dir=list_dir,
            fold_id=fold_id,
        )
    else:
        spec = _benchmark_input_spec(str(selected_dataset_name))

    requested_batch = int(batch_size)
    if requested_batch <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")
    workers = int(num_workers)
    if workers < 0:
        raise ValueError(f"num_workers cannot be negative, got {num_workers!r}")

    try:
        dataset_size = len(dataset)
    except TypeError as error:
        raise TypeError("Benchmark dataset must implement __len__") from error
    if dataset_size <= 0:
        raise ValueError(f"The {spec.name} test dataset is empty")
    if dataset_size < requested_batch:
        copies = (requested_batch + dataset_size - 1) // dataset_size
        dataset = ConcatDataset([dataset] * copies)

    resolved_channels = _resolve_expected_channels(
        input_channels=(spec.input_channels if input_channels is None else input_channels),
        expected_channels=None,
        model=model,
    )
    use_normalization = spec.imagenet_normalization if normalize is None else bool(normalize)
    collate = partial(
        collate_for_benchmark,
        img_size=size,
        input_channels=resolved_channels,
        volume_input=spec.protocol == "volume",
        normalize=use_normalization,
    )
    return DataLoader(
        dataset,
        batch_size=requested_batch,
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate,
    )


# ============================== Parameters and FLOPs =========================

class ParameterCounts(NamedTuple):
    trainable: int
    total: int


def count_parameters(model: nn.Module) -> ParameterCounts:
    """Return trainable and total stored parameter counts."""

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return ParameterCounts(trainable=int(trainable), total=int(total))


def _model_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _fallback_hook_macs_unet(model: nn.Module, example: torch.Tensor) -> int:
    """Count Conv2d, ConvTranspose2d, and Linear MACs with forward hooks."""

    macs_total = 0
    handles: List[Any] = []

    def _conv(
        module: nn.Conv2d,
        inputs: Tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal macs_total
        image = inputs[0]
        batch = int(image.shape[0])
        channels_in = int(image.shape[1])
        channels_out = int(output.shape[1])
        height_out, width_out = int(output.shape[-2]), int(output.shape[-1])
        kernel_height, kernel_width = module.kernel_size
        macs_total += (
            batch
            * (channels_in // int(module.groups))
            * channels_out
            * int(kernel_height)
            * int(kernel_width)
            * height_out
            * width_out
        )

    def _conv_transpose(
        module: nn.ConvTranspose2d,
        inputs: Tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal macs_total
        image = inputs[0]
        batch = int(image.shape[0])
        channels_in = int(image.shape[1])
        channels_out = int(output.shape[1])
        height_out, width_out = int(output.shape[-2]), int(output.shape[-1])
        kernel_height, kernel_width = module.kernel_size
        macs_total += (
            batch
            * channels_in
            * (channels_out // int(module.groups))
            * int(kernel_height)
            * int(kernel_width)
            * height_out
            * width_out
        )

    def _linear(
        module: nn.Linear,
        inputs: Tuple[Any, ...],
        output: Any,
    ) -> None:
        nonlocal macs_total
        input_tensor = inputs[0]
        instances = int(input_tensor.numel()) // int(module.in_features)
        macs_total += instances * int(module.in_features) * int(module.out_features)

    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(_conv))
            elif isinstance(module, nn.ConvTranspose2d):
                handles.append(module.register_forward_hook(_conv_transpose))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(_linear))

        with torch.inference_mode():
            model(example.to(_model_device(model)))
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training_states:
            module.training = was_training
    return int(macs_total)


def count_flops_gflops(
    model: nn.Module,
    example: torch.Tensor,
    quantized_model: bool = False,
) -> Tuple[float, ParameterCounts]:
    """Return per-image GFLOPs and parameters using ``FLOPs = 2 * MACs``."""

    if quantized_model:
        raise ValueError("Quantized/AWQ counting is not applicable to this plain U-Net")
    if not torch.is_tensor(example) or example.ndim != 4 or int(example.shape[0]) != 1:
        shape = tuple(example.shape) if torch.is_tensor(example) else type(example).__name__
        raise ValueError(f"FLOPs require one BCHW image; received {shape}")
    expected = _declared_model_channels(model)
    _validate_benchmark_images(example, expected)
    macs = _fallback_hook_macs_unet(model, example)
    return (2.0 * float(macs)) / 1e9, count_parameters(model)


def _parameter_metrics(counts: ParameterCounts) -> Dict[str, Any]:
    return {
        "params": round(float(counts.total) / 1e6, 3),
        "trainable_params": int(counts.trainable),
        "total_params": int(counts.total),
        "trainable_params_m": round(float(counts.trainable) / 1e6, 3),
        "total_params_m": round(float(counts.total) / 1e6, 3),
    }


# ============================== Model/runtime memory =========================

def model_size_mb_benchmark(
    model: nn.Module,
    *,
    include_buffers: bool = True,
    use_state_dict: bool = True,
    binary_mebibytes: bool = True,
    deduplicate_shared_tensors: bool = True,
    exclude_prefixes: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Measure persistent model storage from tensor element sizes."""

    tensors: List[torch.Tensor] = []
    if use_state_dict:
        prefixes = (
            (exclude_prefixes,)
            if isinstance(exclude_prefixes, str)
            else tuple(exclude_prefixes or ())
        )
        for name, value in model.state_dict().items():
            if prefixes and name.startswith(prefixes):
                continue
            if torch.is_tensor(value):
                tensors.append(value)
        counted_via = "state_dict"
    else:
        tensors.extend(model.parameters())
        if include_buffers:
            tensors.extend(model.buffers())
        counted_via = "parameters(+buffers)" if include_buffers else "parameters_only"

    total_bytes = 0
    dtype_bytes: Dict[str, int] = {}
    seen: set[Tuple[int, int, int]] = set()
    for tensor in tensors:
        storage_key = (
            int(tensor.data_ptr()),
            int(tensor.numel()),
            int(tensor.element_size()),
        )
        if deduplicate_shared_tensors and storage_key in seen:
            continue
        seen.add(storage_key)
        byte_count = int(tensor.numel()) * int(tensor.element_size())
        total_bytes += byte_count
        dtype_name = str(tensor.dtype)
        dtype_bytes[dtype_name] = dtype_bytes.get(dtype_name, 0) + byte_count

    denominator = (1024 ** 2) if binary_mebibytes else (10 ** 6)
    return {
        "total_bytes": int(total_bytes),
        "total_mb": float(total_bytes / denominator),
        "total_mib": float(total_bytes / (1024 ** 2)),
        "unit": "MiB" if binary_mebibytes else "MB",
        "num_tensors_counted": len(tensors),
        "dtype_bytes": dtype_bytes,
        "dtype_mb": {
            dtype_name: byte_count / denominator
            for dtype_name, byte_count in dtype_bytes.items()
        },
        "counted_via": counted_via,
    }


def _extract_images(batch: Any) -> torch.Tensor:
    images = _extract_img_only(batch)
    if not torch.is_tensor(images):
        raise TypeError("Extracted benchmark images are not a torch.Tensor")
    return images


def _validate_benchmark_images(
    images: torch.Tensor,
    expected_channels: Optional[int],
) -> None:
    if not torch.is_tensor(images):
        raise TypeError(f"Benchmark images must be tensors, got {type(images).__name__}")
    if images.ndim != 4:
        raise ValueError(f"Benchmark loader must return BCHW images, got {tuple(images.shape)}")
    if int(images.shape[0]) <= 0:
        raise ValueError("Benchmark loader returned an empty batch")
    if expected_channels is not None and int(images.shape[1]) != expected_channels:
        raise ValueError(
            f"Benchmark batch has {int(images.shape[1])} channels, but model.n_channels "
            f"is {expected_channels}"
        )


def _unavailable_runtime_memory(
    *,
    device: torch.device,
    reason: str,
    binary_mebibytes: bool,
    batch_size: int,
    input_size: Sequence[int],
    warmup: int,
    iterations: int,
    autocast: bool,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "current_allocated_bytes": None,
        "current_reserved_bytes": None,
        "peak_allocated_mb": None,
        "peak_reserved_mb": None,
        "current_allocated_mb": None,
        "current_reserved_mb": None,
        "unit": "MiB" if binary_mebibytes else "MB",
        "device": str(device),
        "batch_size": int(batch_size),
        "input_size": tuple(int(value) for value in input_size),
        "warmup": int(warmup),
        "iterations": int(iterations),
        "amp_autocast": bool(autocast),
        "amp_dtype": str(amp_dtype),
    }


def runtime_memory_mb_benchmark(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 50,
    test_loader: Optional[Iterable[Any]] = None,
    device: Optional[Any] = None,
    autocast: bool = False,
    amp_dtype: Optional[torch.dtype] = torch.float16,
    binary_mebibytes: bool = True,
) -> Dict[str, Any]:
    """Measure peak CUDA allocator memory, or return explicit unavailability."""

    device_object = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    warmups = max(0, int(warmup))
    measured_iterations = max(1, int(iterations))
    selected_batch_size = max(1, int(batch_size))
    selected_input_size = tuple(int(value) for value in input_size)

    if device_object.type != "cuda":
        return _unavailable_runtime_memory(
            device=device_object,
            reason="CUDA allocator statistics are unavailable on CPU",
            binary_mebibytes=binary_mebibytes,
            batch_size=selected_batch_size,
            input_size=selected_input_size,
            warmup=warmups,
            iterations=measured_iterations,
            autocast=autocast,
            amp_dtype=amp_dtype,
        )
    if not torch.cuda.is_available():
        return _unavailable_runtime_memory(
            device=device_object,
            reason="CUDA was requested but is not available in this PyTorch environment",
            binary_mebibytes=binary_mebibytes,
            batch_size=selected_batch_size,
            input_size=selected_input_size,
            warmup=warmups,
            iterations=measured_iterations,
            autocast=autocast,
            amp_dtype=amp_dtype,
        )

    model.to(device_object).eval()
    if test_loader is not None:
        try:
            batch = next(iter(test_loader))
        except StopIteration as error:
            raise ValueError("test_loader is empty; runtime memory cannot be measured") from error
        images = _extract_images(batch)
        _validate_benchmark_images(images, _declared_model_channels(model))
        input_tensor = images.to(device_object, non_blocking=True).contiguous()
        selected_batch_size = int(input_tensor.shape[0])
        selected_input_size = tuple(int(value) for value in input_tensor.shape[1:4])
    else:
        if len(selected_input_size) != 3 or any(value <= 0 for value in selected_input_size):
            raise ValueError(f"input_size must be three positive C,H,W values, got {input_size}")
        expected = _declared_model_channels(model)
        if expected is not None and selected_input_size[0] != expected:
            raise ValueError(
                f"input_size specifies {selected_input_size[0]} channels, but model.n_channels "
                f"is {expected}"
            )
        input_tensor = torch.randn(
            (selected_batch_size, *selected_input_size), device=device_object
        )

    def _amp_context():
        if autocast:
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16 if amp_dtype is None else amp_dtype,
            )
        return contextlib.nullcontext()

    with torch.inference_mode():
        with _amp_context():
            for _ in range(warmups):
                model(input_tensor)
        torch.cuda.synchronize(device_object)
        torch.cuda.reset_peak_memory_stats(device_object)
        with _amp_context():
            for _ in range(measured_iterations):
                model(input_tensor)
        torch.cuda.synchronize(device_object)

    peak_allocated = int(torch.cuda.max_memory_allocated(device_object))
    peak_reserved = int(torch.cuda.max_memory_reserved(device_object))
    current_allocated = int(torch.cuda.memory_allocated(device_object))
    current_reserved = int(torch.cuda.memory_reserved(device_object))
    denominator = (1024 ** 2) if binary_mebibytes else (10 ** 6)
    return {
        "available": True,
        "reason": None,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "current_allocated_bytes": current_allocated,
        "current_reserved_bytes": current_reserved,
        "peak_allocated_mb": peak_allocated / denominator,
        "peak_reserved_mb": peak_reserved / denominator,
        "current_allocated_mb": current_allocated / denominator,
        "current_reserved_mb": current_reserved / denominator,
        "unit": "MiB" if binary_mebibytes else "MB",
        "device": str(device_object),
        "batch_size": selected_batch_size,
        "input_size": selected_input_size,
        "warmup": warmups,
        "iterations": measured_iterations,
        "amp_autocast": bool(autocast),
        "amp_dtype": str(amp_dtype),
    }


# ============================== Timing/public API ============================

@torch.inference_mode()
def _infer_one_batch(model: nn.Module, images: torch.Tensor) -> Any:
    return model(images)


def _set_cudnn_benchmark(enable: bool) -> None:
    try:
        torch.backends.cudnn.benchmark = bool(enable)
    except Exception:
        pass


def _timed_forward_gpu(model: nn.Module, images: torch.Tensor) -> float:
    with torch.cuda.device(images.device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _infer_one_batch(model, images)
        end.record()
        torch.cuda.synchronize(images.device)
        return float(start.elapsed_time(end))


def _timed_forward_cpu(model: nn.Module, images: torch.Tensor) -> float:
    start = time.perf_counter()
    _infer_one_batch(model, images)
    return float((time.perf_counter() - start) * 1000.0)


def _percentiles(
    values: List[float],
    quantiles: Tuple[int, ...] = (50, 90, 95, 99),
) -> Dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"p{quantile}": float(np.percentile(array, quantile))
        for quantile in quantiles
    }


def _run_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"runs": [], "mean": float("nan"), "std": float("nan")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "runs": [float(value) for value in values],
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _runtime_memory_metrics(memory: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "runtime_memory_peak_allocated_bytes": memory["peak_allocated_bytes"],
        "runtime_memory_peak_reserved_bytes": memory["peak_reserved_bytes"],
        "runtime_memory_current_allocated_bytes": memory["current_allocated_bytes"],
        "runtime_memory_current_reserved_bytes": memory["current_reserved_bytes"],
        "runtime_memory_peak_allocated_mib": memory["peak_allocated_mb"],
        "runtime_memory_peak_reserved_mib": memory["peak_reserved_mb"],
        "runtime_memory_current_allocated_mib": memory["current_allocated_mb"],
        "runtime_memory_current_reserved_mib": memory["current_reserved_mb"],
    }


def benchmark_segmentation_model(
    model: nn.Module,
    test_loader: Iterable[Any],
    device: Any = "cuda",
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    measure_batches: int = DEFAULT_MEASURE_BATCHES,
    single_image_latency_samples: int = DEFAULT_SINGLE_IMAGE_LATENCY_SAMPLES,
    enable_cudnn_benchmark: bool = True,
    autocast: bool = False,
    amp_dtype: Optional[torch.dtype] = torch.float16,
    single_image_warmup_steps: int = DEFAULT_SINGLE_IMAGE_WARMUP_STEPS,
    quantized_model: bool = False,
    args: Any = None,
    repeated_runs: Optional[int] = None,
    measure_runtime_memory: bool = True,
    fp32_reference_size_bytes: Optional[int] = None,
    deployed_model_size_bytes: Optional[int] = None,
) -> BenchmarkResults:
    """Benchmark fixed-size, preprocessed real test images.

    Defaults match the supplied protocol: batch size is selected by the loader,
    followed by 20 warmups, 50 measured batches, 50 single-image warmups, 1000
    latency samples, cuDNN benchmark mode enabled, no AMP, and one run.
    """

    warmups = int(warmup_steps)
    measured_batches = int(measure_batches)
    latency_samples = int(single_image_latency_samples)
    single_warmups = int(single_image_warmup_steps)
    if warmups < 0:
        raise ValueError("warmup_steps cannot be negative")
    if measured_batches <= 0:
        raise ValueError("measure_batches must be positive")
    if latency_samples < 0:
        raise ValueError("single_image_latency_samples cannot be negative")
    if single_warmups < 0:
        raise ValueError("single_image_warmup_steps cannot be negative")
    if repeated_runs is None:
        repeated_runs = int(_arg(args, "repeated_runs", default=DEFAULT_REPEATED_RUNS))
    repeats = int(repeated_runs)
    if not 1 <= repeats <= 10:
        raise ValueError("repeated_runs must be between 1 and 10")
    if quantized_model:
        raise ValueError("AWQ is not applicable to this plain U-Net benchmark")

    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmarking was requested but CUDA is unavailable")
    model.to(device_object).eval()
    _set_cudnn_benchmark(enable_cudnn_benchmark)
    expected_channels = _declared_model_channels(model)

    try:
        first_batch = next(iter(test_loader))
    except StopIteration as error:
        raise ValueError("test_loader is empty; cannot benchmark the model") from error
    first_images = _extract_images(first_batch)
    _validate_benchmark_images(first_images, expected_channels)
    batch, channels, height, width = [int(value) for value in first_images.shape]

    example = first_images[:1].contiguous()
    gflops, parameter_counts = count_flops_gflops(model, example)

    def _amp_context():
        if device_object.type == "cuda" and autocast:
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16 if amp_dtype is None else amp_dtype,
            )
        return contextlib.nullcontext()

    def _timed(images: torch.Tensor) -> float:
        if device_object.type == "cuda":
            return _timed_forward_gpu(model, images)
        return _timed_forward_cpu(model, images)

    def _next(iterator: Any) -> Tuple[Any, Any]:
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(test_loader)
            try:
                return next(iterator), iterator
            except StopIteration as error:
                raise ValueError("test_loader became empty during benchmarking") from error

    iterator = iter(test_loader)
    with torch.inference_mode(), _amp_context():
        for _ in range(warmups):
            batch_value, iterator = _next(iterator)
            images = _extract_images(batch_value)
            _validate_benchmark_images(images, expected_channels)
            images = images.to(
                device_object,
                non_blocking=device_object.type == "cuda",
            ).contiguous()
            _timed(images)

    throughput_runs: List[float] = []
    mean_latency_runs: List[float] = []
    for _ in range(repeats):
        total_images = 0
        total_ms = 0.0
        iterator = iter(test_loader)
        with torch.inference_mode(), _amp_context():
            for _ in range(measured_batches):
                batch_value, iterator = _next(iterator)
                images = _extract_images(batch_value)
                _validate_benchmark_images(images, expected_channels)
                current_batch_size = int(images.shape[0])
                images = images.to(
                    device_object,
                    non_blocking=device_object.type == "cuda",
                ).contiguous()
                total_ms += _timed(images)
                total_images += current_batch_size
        throughput_runs.append(
            total_images / (total_ms / 1000.0) if total_ms > 0 else float("nan")
        )
        mean_latency_runs.append(total_ms / max(1, total_images))

    if latency_samples > 0 and single_warmups > 0:
        iterator = iter(test_loader)
        warmed = 0
        with torch.inference_mode(), _amp_context():
            while warmed < single_warmups:
                batch_value, iterator = _next(iterator)
                images = _extract_images(batch_value)
                _validate_benchmark_images(images, expected_channels)
                for index in range(int(images.shape[0])):
                    single = images[index:index + 1].to(
                        device_object,
                        non_blocking=device_object.type == "cuda",
                    ).contiguous()
                    _timed(single)
                    warmed += 1
                    if warmed >= single_warmups:
                        break

    percentile_runs: List[Dict[str, float]] = []
    for _ in range(repeats):
        latencies: List[float] = []
        iterator = iter(test_loader)
        with torch.inference_mode(), _amp_context():
            while len(latencies) < latency_samples:
                batch_value, iterator = _next(iterator)
                images = _extract_images(batch_value)
                _validate_benchmark_images(images, expected_channels)
                for index in range(int(images.shape[0])):
                    single = images[index:index + 1].to(
                        device_object,
                        non_blocking=device_object.type == "cuda",
                    ).contiguous()
                    latencies.append(_timed(single))
                    if len(latencies) >= latency_samples:
                        break
        percentile_runs.append(_percentiles(latencies))

    throughput_summary = _run_summary(throughput_runs)
    mean_latency_summary = _run_summary(mean_latency_runs)
    percentile_summaries = {
        quantile: _run_summary([
            run.get(f"p{quantile}", float("nan")) for run in percentile_runs
        ])
        for quantile in (50, 90, 95, 99)
    }

    model_size = model_size_mb_benchmark(model)
    if deployed_model_size_bytes is None:
        deployed_model_size_bytes = int(model_size["total_bytes"])
    elif int(deployed_model_size_bytes) < 0:
        raise ValueError("deployed_model_size_bytes cannot be negative")
    if fp32_reference_size_bytes is not None and int(fp32_reference_size_bytes) <= 0:
        raise ValueError("fp32_reference_size_bytes must be positive when supplied")
    mebibyte = 1024 ** 2

    if measure_runtime_memory:
        runtime_memory = runtime_memory_mb_benchmark(
            model,
            input_size=(channels, height, width),
            batch_size=batch,
            test_loader=test_loader,
            device=device_object,
            warmup=warmups,
            iterations=measured_batches,
            autocast=autocast,
            amp_dtype=amp_dtype,
        )
    else:
        runtime_memory = _unavailable_runtime_memory(
            device=device_object,
            reason="Runtime memory measurement was disabled",
            binary_mebibytes=True,
            batch_size=batch,
            input_size=(channels, height, width),
            warmup=warmups,
            iterations=measured_batches,
            autocast=autocast,
            amp_dtype=amp_dtype,
        )

    metrics: Dict[str, Any] = {
        "throughput_img_s": throughput_summary["mean"],
        "latency_ms_mean": mean_latency_summary["mean"],
        "latency_ms_p50": percentile_summaries[50]["mean"],
        "latency_ms_p90": percentile_summaries[90]["mean"],
        "latency_ms_p95": percentile_summaries[95]["mean"],
        "latency_ms_p99": percentile_summaries[99]["mean"],
        **_parameter_metrics(parameter_counts),
        "flops_gflops_per_image": float(gflops),
        "input_shape": [batch, channels, height, width],
        "input_height": height,
        "input_width": width,
        "image_shape_HxW": height * width,
        "channels": channels,
        "batch_size_first": batch,
        "model_size": model_size,
        "model_size_bytes": int(deployed_model_size_bytes),
        "model_size_mib": float(int(deployed_model_size_bytes) / mebibyte),
        "fp32_reference_size_bytes": fp32_reference_size_bytes,
        "fp32_reference_size_mib": (
            int(fp32_reference_size_bytes) / mebibyte
            if fp32_reference_size_bytes is not None
            else None
        ),
        "compression_ratio": (
            int(fp32_reference_size_bytes) / int(deployed_model_size_bytes)
            if fp32_reference_size_bytes is not None and int(deployed_model_size_bytes) > 0
            else None
        ),
        "model_runtime_memory": runtime_memory,
        **_runtime_memory_metrics(runtime_memory),
    }

    if repeats > 1:
        metrics.update({
            "repeated_runs": repeats,
            "throughput_img_s_runs": throughput_summary["runs"],
            "throughput_img_s_mean": throughput_summary["mean"],
            "throughput_img_s_std": throughput_summary["std"],
            "latency_ms_mean_runs": mean_latency_summary["runs"],
            "latency_ms_mean_mean": mean_latency_summary["mean"],
            "latency_ms_mean_std": mean_latency_summary["std"],
        })
        for quantile, summary in percentile_summaries.items():
            prefix = f"latency_ms_p{quantile}"
            metrics[f"{prefix}_runs"] = summary["runs"]
            metrics[f"{prefix}_mean"] = summary["mean"]
            metrics[f"{prefix}_std"] = summary["std"]

    checkpoint = _arg(args, "checkpoint", "ckpt", default="N/A")
    notes = {
        "checkpoint": checkpoint,
        "device": str(device_object),
        "warmup_steps": warmups,
        "measure_batches": measured_batches,
        "single_image_warmup_steps": single_warmups,
        "single_image_latency_samples": latency_samples,
        "repeated_runs": repeats,
        "cudnn_benchmark": bool(enable_cudnn_benchmark),
        "amp_autocast": bool(autocast),
        "amp_dtype": str(amp_dtype),
        "runtime_memory_measured": bool(runtime_memory["available"]),
        "runtime_memory_unavailable_reason": runtime_memory["reason"],
        "flops_convention": "FLOPs = 2 * MACs",
        "preprocessing_outside_timing": True,
    }
    return BenchmarkResults(metrics=metrics, notes=notes)


__all__ = [
    "BenchmarkResults",
    "DEFAULT_BENCHMARK_BATCH_SIZE",
    "DEFAULT_MEASURE_BATCHES",
    "DEFAULT_REPEATED_RUNS",
    "DEFAULT_SINGLE_IMAGE_LATENCY_SAMPLES",
    "DEFAULT_SINGLE_IMAGE_WARMUP_STEPS",
    "DEFAULT_WARMUP_STEPS",
    "ParameterCounts",
    "benchmark_segmentation_model",
    "build_benchmark_loader",
    "collate_for_benchmark",
    "count_flops_gflops",
    "count_parameters",
    "model_size_mb_benchmark",
    "runtime_memory_mb_benchmark",
]
