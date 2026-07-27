"""Dataset-correct medical segmentation inference and aggregation."""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.ndimage import zoom

from utils.medical_metrics import (
    _case_group_sort_key,
    _mean_metric_array,
    _present_class_metric,
    _safe_nanmean,
    calculate_metric_percase,
    extract_acdc_voxelspacing_zyx,
    fallback_acdc_voxelspacing_zyx,
)
from utils.model_output import _extract_logits, validate_labels, validate_model_input


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _preserve_model_mode(model_argument_index: int):
    """Restore train/eval mode even when an inference helper raises."""

    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            model = kwargs.get("model")
            if model is None:
                model = args[model_argument_index]
            was_training = model.training
            try:
                return function(*args, **kwargs)
            finally:
                model.train(was_training)

        return wrapped

    return decorator


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _case_name(batch: Mapping[str, Any], index: int = 0) -> str:
    value = batch.get("case_name", f"case_{index:04d}")
    if isinstance(value, (list, tuple)):
        value = value[index]
    elif torch.is_tensor(value) and value.ndim:
        value = value[index].item()
    return str(value)


def _case_group_name(case_name: str) -> str:
    return str(case_name).split("_", 1)[0]


def _resize_prediction(prediction: np.ndarray, output_shape: Sequence[int]) -> np.ndarray:
    output_shape = tuple(int(v) for v in output_shape)
    if tuple(prediction.shape) == output_shape:
        return prediction
    factors = tuple(out / current for out, current in zip(output_shape, prediction.shape))
    return zoom(prediction, factors, order=0)


@torch.inference_mode()
@_preserve_model_mode(1)
def infer_volume_slicewise(
    image: Any,
    model: torch.nn.Module,
    *,
    img_size: int,
    device: torch.device,
    expected_channels: int = 1,
    amp: bool = False,
) -> np.ndarray:
    """Reconstruct a D,H,W prediction one resized 2-D slice at a time."""

    volume = _as_numpy(image).squeeze(0)
    if volume.ndim != 3:
        raise ValueError(f"Volume inference expects D,H,W after removing batch, got {volume.shape}")
    if expected_channels != 1:
        raise ValueError("Slice-wise Synapse/ACDC inference requires a one-channel U-Net")

    prediction = np.zeros(volume.shape, dtype=np.int64)
    model_was_training = model.training
    model.eval()
    autocast_device = device.type if device.type != "mps" else "cpu"
    for index, image_slice in enumerate(volume):
        height, width = image_slice.shape
        if (height, width) != (img_size, img_size):
            resized = zoom(image_slice, (img_size / height, img_size / width), order=3)
        else:
            resized = image_slice
        inputs = torch.from_numpy(np.ascontiguousarray(resized)).unsqueeze(0).unsqueeze(0)
        inputs = inputs.to(device=device, dtype=torch.float32)
        validate_model_input(inputs, expected_channels)
        with torch.autocast(autocast_device, enabled=amp):
            logits = _extract_logits(model(inputs))
        predicted = torch.softmax(logits, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
        prediction[index] = _resize_prediction(predicted, (height, width)).astype(np.int64)
    if model_was_training:
        model.train()
    return prediction


def _volume_metrics(
    prediction: np.ndarray,
    label: np.ndarray,
    num_classes: int,
    *,
    voxelspacing: Sequence[float] | None,
) -> np.ndarray:
    return np.asarray(
        [
            calculate_metric_percase(
                prediction == class_id,
                label == class_id,
                voxelspacing=voxelspacing,
            )
            for class_id in range(1, num_classes)
        ],
        dtype=np.float32,
    )


def _save_volume_triplet(
    destination: Path,
    case_name: str,
    image: np.ndarray,
    prediction: np.ndarray,
    label: np.ndarray,
    spacing_zyx: Sequence[float] | None,
) -> None:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("SimpleITK is required by --save-predictions") from exc
    destination.mkdir(parents=True, exist_ok=True)
    spacing_xyz = tuple(reversed(spacing_zyx)) if spacing_zyx is not None else (1.0, 1.0, 1.0)
    for suffix, array in (("img", image), ("pred", prediction), ("gt", label)):
        itk_image = sitk.GetImageFromArray(array.astype(np.float32))
        itk_image.SetSpacing(tuple(float(v) for v in spacing_xyz))
        sitk.WriteImage(itk_image, str(destination / f"{case_name}_{suffix}.nii.gz"))


def evaluate_volume_loader(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    dataset_name: str,
    num_classes: int,
    class_names: Sequence[str],
    img_size: int,
    device: torch.device,
    input_channels: int = 1,
    acdc_zspacing: float = 5.0,
    amp: bool = False,
    save_predictions: str | Path | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    if dataset_name not in {"Synapse", "ACDC"}:
        raise ValueError(f"Volume protocol does not support {dataset_name}")
    try:
        loader_length = len(loader)  # type: ignore[arg-type]
    except TypeError:
        loader_length = None
    if loader_length == 0:
        raise RuntimeError("Validation/test loader is empty")

    all_metrics: list[np.ndarray] = []
    spacing_sources: dict[str, str] = {}
    save_path = Path(save_predictions) if save_predictions else None
    for case_index, batch in enumerate(loader):
        if max_cases is not None and case_index >= max_cases:
            break
        if "image" not in batch or "label" not in batch:
            raise KeyError("Medical volume batches must contain image and label")
        name = _case_name(batch)
        label = _as_numpy(batch["label"]).squeeze(0)
        validate_labels(label, num_classes, context=f"case {name}")
        spacing = None
        if dataset_name == "ACDC":
            spacing = extract_acdc_voxelspacing_zyx(batch, name)
            if spacing is None:
                spacing = fallback_acdc_voxelspacing_zyx(acdc_zspacing, name)
                spacing_sources[name] = "fallback"
            else:
                spacing_sources[name] = "dataset_metadata"
            logging.info(
                "ACDC case %s HD95 voxelspacing_zyx=%s source=%s",
                name,
                spacing,
                spacing_sources[name],
            )
        prediction = infer_volume_slicewise(
            batch["image"],
            model,
            img_size=img_size,
            device=device,
            expected_channels=input_channels,
            amp=amp,
        )
        metrics = _volume_metrics(prediction, label, num_classes, voxelspacing=spacing)
        all_metrics.append(metrics)
        logging.info(
            "%s case %s mean_dice %.6f mean_hd95 %.6f mean_iou %.6f",
            dataset_name,
            name,
            _safe_nanmean(metrics[:, 0]),
            _safe_nanmean(metrics[:, 1]),
            _safe_nanmean(metrics[:, 2]),
        )
        if save_path is not None:
            image = _as_numpy(batch["image"]).squeeze(0)
            _save_volume_triplet(save_path, name, image, prediction, label, spacing)

    if not all_metrics:
        raise RuntimeError("No volume metrics were collected")
    metric_stack = np.stack(all_metrics, axis=0)  # case,class,metric
    per_class_mean = _mean_metric_array(metric_stack)
    per_class = {
        class_names[class_id]: {
            "dice": float(per_class_mean[class_id - 1, 0]),
            "hd95": float(per_class_mean[class_id - 1, 1]),
            "iou": float(per_class_mean[class_id - 1, 2]),
        }
        for class_id in range(1, num_classes)
    }
    result = {
        "mean_dice": _safe_nanmean(per_class_mean[:, 0]),
        "mean_hd95": _safe_nanmean(per_class_mean[:, 1]),
        "mean_iou": _safe_nanmean(per_class_mean[:, 2]),
        "per_class": per_class,
        "protocol": f"{dataset_name}_volume_foreground_case_then_class_nanmean",
        "num_cases": len(all_metrics),
    }
    if dataset_name == "ACDC":
        result["spacing_source_per_case"] = spacing_sources
    return result


def present_class_frame_result(
    prediction: np.ndarray,
    label: np.ndarray,
    *,
    num_classes: int,
    case_name: str,
) -> dict[str, Any]:
    """Compute one Cataract1k frame using only GT-present foreground classes."""

    prediction = np.asarray(prediction)
    label = np.asarray(label)
    if prediction.shape != label.shape or label.ndim != 2:
        raise ValueError(
            f"Cataract1k prediction and label must be matching 2-D arrays, got "
            f"{prediction.shape} and {label.shape}"
        )
    validate_labels(label, num_classes, context=f"frame {case_name}")
    present = [class_id for class_id in range(1, num_classes) if np.any(label == class_id)]
    false_positive_absent = [
        class_id
        for class_id in range(1, num_classes)
        if not np.any(label == class_id) and np.any(prediction == class_id)
    ]
    per_class = np.full((num_classes - 1, 3), np.nan, dtype=np.float32)
    for class_id in present:
        dice, hd95, iou, ignored = _present_class_metric(
            prediction,
            label,
            class_id,
            voxelspacing=(1.0, 1.0),
        )
        if not ignored:
            per_class[class_id - 1] = (dice, hd95, iou)
    frame_metrics = np.asarray(
        [
            _safe_nanmean(per_class[:, 0]),
            _safe_nanmean(per_class[:, 1]),
            _safe_nanmean(per_class[:, 2]),
        ],
        dtype=np.float32,
    )
    return {
        "case": str(case_name),
        "per_class_metrics": per_class,
        "frame_metrics": frame_metrics,
        "present_class_ids": present,
        "false_positive_absent_class_ids": false_positive_absent,
        "discounted_frame": not present,
    }


@torch.inference_mode()
def infer_cataract_frame(
    image: Any,
    model: torch.nn.Module,
    *,
    img_size: int,
    device: torch.device,
    normalize: bool,
    input_channels: int = 3,
    amp: bool = False,
) -> np.ndarray:
    """Infer one original-resolution raw HWC RGB Cataract1k frame."""

    array = _as_numpy(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Raw Cataract1k inference expects H,W,3 RGB, got {array.shape}")
    height, width, _ = array.shape
    channels = [zoom(array[..., ch], (img_size / height, img_size / width), order=3) for ch in range(3)]
    prepared = np.stack(channels, axis=0).astype(np.float32)
    if normalize:
        prepared /= 255.0
        prepared = (
            prepared - np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        ) / np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    inputs = torch.from_numpy(np.ascontiguousarray(prepared)).unsqueeze(0).to(device)
    validate_model_input(inputs, input_channels)
    autocast_device = device.type if device.type != "mps" else "cpu"
    with torch.autocast(autocast_device, enabled=amp):
        logits = _extract_logits(model(inputs))
    predicted = torch.softmax(logits, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
    return _resize_prediction(predicted, (height, width)).astype(np.int64)


def _save_cataract_mask(destination: Path, case_name: str, prediction: np.ndarray) -> None:
    from PIL import Image

    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.astype(np.uint8), mode="L").save(destination / f"{case_name}_pred.png")


def aggregate_present_class_frames(
    frame_results: Sequence[Mapping[str, Any]],
    *,
    class_names: Sequence[str],
    num_classes: int,
) -> dict[str, Any]:
    if not frame_results:
        raise RuntimeError("No Cataract1k frame metrics were collected")
    frame_stack = np.stack([result["frame_metrics"] for result in frame_results])
    class_stack = np.stack([result["per_class_metrics"] for result in frame_results])
    class_mean = _mean_metric_array(class_stack)
    per_class_diagnostic = {
        class_names[class_id]: {
            "dice": float(class_mean[class_id - 1, 0]),
            "hd95": float(class_mean[class_id - 1, 1]),
            "iou": float(class_mean[class_id - 1, 2]),
        }
        for class_id in range(1, num_classes)
    }

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for result in frame_results:
        groups.setdefault(_case_group_name(str(result["case"])), []).append(result)
    per_case_group: dict[str, Any] = {}
    for group_name in sorted(groups, key=_case_group_sort_key):
        group_frame_stack = np.stack([result["frame_metrics"] for result in groups[group_name]])
        group_class_mean = _mean_metric_array(
            np.stack([result["per_class_metrics"] for result in groups[group_name]])
        )
        per_case_group[group_name] = {
            "mean_dice": _safe_nanmean(group_frame_stack[:, 0]),
            "mean_hd95": _safe_nanmean(group_frame_stack[:, 1]),
            "mean_iou": _safe_nanmean(group_frame_stack[:, 2]),
            "num_frames": len(groups[group_name]),
            "per_class_diagnostic": {
                class_names[class_id]: {
                    "dice": float(group_class_mean[class_id - 1, 0]),
                    "hd95": float(group_class_mean[class_id - 1, 1]),
                    "iou": float(group_class_mean[class_id - 1, 2]),
                }
                for class_id in range(1, num_classes)
            },
        }

    presence_counts = {
        class_names[class_id]: sum(
            class_id in result["present_class_ids"] for result in frame_results
        )
        for class_id in range(1, num_classes)
    }
    false_positive_counts = {
        class_names[class_id]: sum(
            class_id in result["false_positive_absent_class_ids"] for result in frame_results
        )
        for class_id in range(1, num_classes)
    }
    discounted = sum(bool(result["discounted_frame"]) for result in frame_results)
    return {
        "mean_dice": _safe_nanmean(frame_stack[:, 0]),
        "mean_hd95": _safe_nanmean(frame_stack[:, 1]),
        "mean_iou": _safe_nanmean(frame_stack[:, 2]),
        "protocol": "Cataract1k_frame_present_background_excluded",
        "per_class_diagnostic": per_class_diagnostic,
        "per_case_group": per_case_group,
        "class_presence_counts": presence_counts,
        "false_positive_absent_class_counts": false_positive_counts,
        "discounted_frames": int(discounted),
        "evaluated_frames": int(len(frame_results) - discounted),
        "num_frames": int(len(frame_results)),
    }


@torch.inference_mode()
@_preserve_model_mode(0)
def evaluate_cataract_loader(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    num_classes: int,
    class_names: Sequence[str],
    img_size: int,
    device: torch.device,
    normalize_raw: bool,
    input_channels: int = 3,
    amp: bool = False,
    save_predictions: str | Path | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    try:
        if len(loader) == 0:  # type: ignore[arg-type]
            raise RuntimeError("Validation/test loader is empty")
    except TypeError:
        pass
    save_path = Path(save_predictions) if save_predictions else None
    results: list[dict[str, Any]] = []
    model_was_training = model.training
    model.eval()
    processed = 0
    autocast_device = device.type if device.type != "mps" else "cpu"
    for batch_index, batch in enumerate(loader):
        images = batch["image"]
        labels = batch["label"]
        if not torch.is_tensor(images):
            images = torch.as_tensor(images)
        batch_size = int(images.shape[0]) if images.ndim == 4 else 1
        for item_index in range(batch_size):
            if max_cases is not None and processed >= max_cases:
                break
            name_value = batch.get("case_name", f"frame_{batch_index:04d}_{item_index:02d}")
            name = str(name_value[item_index] if isinstance(name_value, (list, tuple)) else name_value)
            label = _as_numpy(labels[item_index] if batch_size > 1 else labels).squeeze()
            item_image = images[item_index] if batch_size > 1 else images.squeeze(0)
            # Raw accuracy-test frames are HWC.  Training validation samples are
            # already normalized CHW and can be evaluated directly.
            if item_image.ndim == 3 and int(item_image.shape[-1]) == 3:
                prediction = infer_cataract_frame(
                    item_image,
                    model,
                    img_size=img_size,
                    device=device,
                    normalize=normalize_raw,
                    input_channels=input_channels,
                    amp=amp,
                )
            elif item_image.ndim == 3 and int(item_image.shape[0]) == 3:
                inputs = item_image.unsqueeze(0).to(device=device, dtype=torch.float32)
                validate_model_input(inputs, input_channels)
                with torch.autocast(autocast_device, enabled=amp):
                    logits = _extract_logits(model(inputs))
                prediction = logits.softmax(dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
                prediction = _resize_prediction(prediction, label.shape).astype(np.int64)
            else:
                raise ValueError(f"Cataract1k image must be HWC or CHW RGB, got {tuple(item_image.shape)}")
            result = present_class_frame_result(
                prediction,
                label,
                num_classes=num_classes,
                case_name=name,
            )
            results.append(result)
            if save_path is not None:
                _save_cataract_mask(save_path, name, prediction)
            processed += 1
        if max_cases is not None and processed >= max_cases:
            break
    if model_was_training:
        model.train()
    return aggregate_present_class_frames(
        results,
        class_names=class_names,
        num_classes=num_classes,
    )


def foreground_validation_dice(result: Mapping[str, Any]) -> float:
    value = float(result["mean_dice"])
    if not np.isfinite(value):
        raise RuntimeError("Validation produced no finite foreground Dice values")
    return value
