"""Medical-segmentation metrics and nan-aware aggregation helpers.

The metric definitions in this module intentionally mirror the TransUNet
``multiconfig`` evaluation protocol.  In particular, HD95 is never replaced
with zero when it is undefined, and volume metrics are reduced over cases
before they are reduced over foreground classes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from medpy import metric

try:  # Keep spacing helpers usable in lightweight metric-only environments.
    import torch
except ImportError:  # pragma: no cover - torch is a project dependency.
    torch = None


METRIC_NAMES = ("dice", "hd95", "iou")

ACDC_SPACING_KEYS = (
    "voxelspacing_zyx",
    "spacing_zyx",
    "voxelspacing",
    "spacing",
    "spacing_mm",
    "pixdim",
    "zooms",
)
ACDC_SPACING_ZYX_KEYS = frozenset(("voxelspacing_zyx", "spacing_zyx"))


def _hd95_empty_mask_penalty(shape: Sequence[int], voxelspacing=None) -> float:
    """Return the physical image diagonal used for one-empty-mask HD95.

    The distance is ``norm((shape - 1) * voxelspacing)``.  Unit spacing is
    used when ``voxelspacing`` is omitted.
    """

    shape_arr = np.asarray(shape, dtype=np.float64)
    if shape_arr.ndim != 1:
        shape_arr = shape_arr.reshape(-1)

    if voxelspacing is None:
        spacing_arr = np.ones_like(shape_arr, dtype=np.float64)
    else:
        spacing_arr = np.asarray(voxelspacing, dtype=np.float64).reshape(-1)
        if spacing_arr.size != shape_arr.size:
            raise ValueError(
                "voxelspacing length {} does not match mask ndim {}; "
                "shape={}, voxelspacing={}".format(
                    spacing_arr.size,
                    shape_arr.size,
                    tuple(shape_arr),
                    voxelspacing,
                )
            )

    side_lengths = np.maximum(shape_arr - 1.0, 0.0) * spacing_arr
    return float(np.linalg.norm(side_lengths))


def calculate_metric_percase(pred, gt, voxelspacing=None) -> Tuple[float, float, float]:
    """Return Dice, HD95, and IoU with Synapse/ACDC absent reward.

    A class absent from both masks receives perfect Dice/IoU and undefined
    (NaN) HD95.  Exactly one empty mask receives zero overlap and the image
    diagonal as its HD95 penalty.
    """

    pred = np.asarray(pred)
    gt = np.asarray(gt)
    pred_present = pred.sum() > 0
    gt_present = gt.sum() > 0

    if pred_present and gt_present:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)
        iou = metric.binary.jc(pred, gt)
        return float(dice), float(hd95), float(iou)

    if pred_present and not gt_present:
        return 0.0, _hd95_empty_mask_penalty(pred.shape, voxelspacing), 0.0

    if not pred_present and gt_present:
        return 0.0, _hd95_empty_mask_penalty(pred.shape, voxelspacing), 0.0

    return 1.0, np.nan, 1.0


def calculate_metric_percase_without_absent_reward(
    pred,
    gt,
    voxelspacing=None,
) -> Tuple[float, float, float]:
    """Return Dice, HD95, and IoU without rewarding mutual absence."""

    pred = np.asarray(pred)
    gt = np.asarray(gt)
    pred_present = pred.sum() > 0
    gt_present = gt.sum() > 0

    if pred_present and gt_present:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)
        iou = metric.binary.jc(pred, gt)
        return float(dice), float(hd95), float(iou)

    if pred_present and not gt_present:
        return 0.0, _hd95_empty_mask_penalty(pred.shape, voxelspacing), 0.0

    if not pred_present and gt_present:
        return 0.0, _hd95_empty_mask_penalty(pred.shape, voxelspacing), 0.0

    return np.nan, np.nan, np.nan


def _safe_nanmean(values) -> float:
    """Return a float nanmean without warnings for empty/all-NaN input."""

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(values))


def _mean_metric_array(metric_stack) -> np.ndarray:
    """Nan-average the first axis while preserving all remaining axes.

    Computing each element through :func:`_safe_nanmean` preserves the
    reference behavior for all-NaN classes without emitting warnings.
    """

    metric_stack = np.asarray(metric_stack, dtype=np.float32)
    if metric_stack.ndim < 1:
        raise ValueError("metric_stack must have at least one dimension")
    mean_metrics = np.full(metric_stack.shape[1:], np.nan, dtype=np.float32)
    for index in np.ndindex(mean_metrics.shape):
        mean_metrics[index] = _safe_nanmean(metric_stack[(slice(None),) + index])
    return mean_metrics


def _present_class_metric(
    prediction,
    label,
    class_id: int,
    voxelspacing=None,
) -> Tuple[float, float, float, bool]:
    """Measure one class for the frame-present protocol.

    The final boolean indicates that the class is ignored because it is absent
    from ground truth.  A prediction of an absent class is deliberately still
    ignored here; it is counted separately as a false-positive diagnostic.
    """

    prediction = np.asarray(prediction)
    label = np.asarray(label)
    pred_mask = prediction == class_id
    gt_mask = label == class_id
    pred_present = pred_mask.sum() > 0
    gt_present = gt_mask.sum() > 0

    if not gt_present:
        return np.nan, np.nan, np.nan, True

    if not pred_present:
        return 0.0, _hd95_empty_mask_penalty(gt_mask.shape, voxelspacing), 0.0, False

    dice = metric.binary.dc(pred_mask, gt_mask)
    hd95 = metric.binary.hd95(pred_mask, gt_mask, voxelspacing=voxelspacing)
    iou = metric.binary.jc(pred_mask, gt_mask)
    return float(dice), float(hd95), float(iou), False


def present_class_frame_metrics(
    prediction,
    label,
    num_classes: int,
    *,
    case_name: Optional[str] = None,
    voxelspacing=(1, 1),
) -> Dict[str, Any]:
    """Compute one Cataract1k frame's metrics and diagnostics.

    Background is excluded.  Only foreground classes present in ground truth
    contribute to the official per-frame mean.
    """

    prediction = np.asarray(prediction)
    label = np.asarray(label)
    if prediction.shape != label.shape:
        raise ValueError(
            "prediction and label must have identical shapes, got {} and {}".format(
                prediction.shape, label.shape
            )
        )
    if int(num_classes) < 2:
        raise ValueError("num_classes must include background and at least one foreground class")

    num_classes = int(num_classes)
    per_class_metrics = np.full((num_classes - 1, 3), np.nan, dtype=np.float32)
    present_class_ids = [
        class_id
        for class_id in range(1, num_classes)
        if np.any(label == class_id)
    ]
    false_positive_absent_class_ids = [
        class_id
        for class_id in range(1, num_classes)
        if not np.any(label == class_id) and np.any(prediction == class_id)
    ]
    discounted_frame = len(present_class_ids) == 0

    if not discounted_frame:
        for class_id in present_class_ids:
            dice, hd95, iou, ignored = _present_class_metric(
                prediction,
                label,
                class_id,
                voxelspacing=voxelspacing,
            )
            if not ignored:
                per_class_metrics[class_id - 1] = (dice, hd95, iou)

    frame_metrics = np.asarray(
        [
            _safe_nanmean(per_class_metrics[:, 0]),
            _safe_nanmean(per_class_metrics[:, 1]),
            _safe_nanmean(per_class_metrics[:, 2]),
        ],
        dtype=np.float32,
    )
    return {
        "per_class_metrics": per_class_metrics,
        "frame_metrics": frame_metrics,
        "present_class_ids": present_class_ids,
        "discounted_frame": discounted_frame,
        "false_positive_absent_class_ids": false_positive_absent_class_ids,
        "case": case_name,
    }


def _case_group_name(case_name: Any) -> str:
    """Group frame names by the prefix before their first underscore."""

    return str(case_name).split("_", 1)[0]


def _case_group_sort_key(case_group_name: str):
    """Sort ``seq<number>`` groups numerically, then other names lexically."""

    prefix = "seq"
    if case_group_name.startswith(prefix) and case_group_name[len(prefix) :].isdigit():
        return 0, int(case_group_name[len(prefix) :])
    return 1, case_group_name


def _class_label(class_names: Optional[Sequence[str]], class_id: int) -> str:
    if class_names is not None and class_id < len(class_names):
        return str(class_names[class_id])
    return "class_{}".format(class_id)


def _named_per_class_metrics(
    mean_metrics: np.ndarray,
    num_classes: int,
    class_names: Optional[Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    named = {}
    for class_id in range(1, num_classes):
        values = mean_metrics[class_id - 1]
        named[_class_label(class_names, class_id)] = {
            "dice": float(values[0]),
            "hd95": float(values[1]),
            "iou": float(values[2]),
        }
    return named


def aggregate_present_class_metrics(
    frame_results: Sequence[Mapping[str, Any]],
    num_classes: int,
    *,
    class_names: Optional[Sequence[str]] = None,
    dataset_name: str = "Cataract1k",
) -> Dict[str, Any]:
    """Aggregate frame-present results using the official frame-first order."""

    if not frame_results:
        raise RuntimeError("No metrics were collected during inference.")
    num_classes = int(num_classes)
    if num_classes < 2:
        raise ValueError("num_classes must include background and foreground classes")

    frame_metrics_all: List[np.ndarray] = []
    per_class_metrics_all: List[np.ndarray] = []
    case_group_frame_metrics: Dict[str, List[np.ndarray]] = {}
    case_group_per_class_metrics: Dict[str, List[np.ndarray]] = {}
    class_presence_counts = np.zeros(num_classes, dtype=np.int64)
    false_positive_absent_class_counts = np.zeros(num_classes, dtype=np.int64)
    discounted_frames = 0

    for result in frame_results:
        frame_metrics = np.asarray(result["frame_metrics"], dtype=np.float32)
        per_class_metrics = np.asarray(result["per_class_metrics"], dtype=np.float32)
        expected_class_shape = (num_classes - 1, len(METRIC_NAMES))
        if frame_metrics.shape != (len(METRIC_NAMES),):
            raise ValueError(
                "frame_metrics must have shape {}, got {}".format(
                    (len(METRIC_NAMES),), frame_metrics.shape
                )
            )
        if per_class_metrics.shape != expected_class_shape:
            raise ValueError(
                "per_class_metrics must have shape {}, got {}".format(
                    expected_class_shape, per_class_metrics.shape
                )
            )

        frame_metrics_all.append(frame_metrics)
        per_class_metrics_all.append(per_class_metrics)
        if bool(result.get("discounted_frame", False)):
            discounted_frames += 1
        for class_id in result.get("present_class_ids", ()):
            class_presence_counts[int(class_id)] += 1
        for class_id in result.get("false_positive_absent_class_ids", ()):
            false_positive_absent_class_counts[int(class_id)] += 1

        group_name = _case_group_name(result.get("case"))
        case_group_frame_metrics.setdefault(group_name, []).append(frame_metrics)
        case_group_per_class_metrics.setdefault(group_name, []).append(per_class_metrics)

    official_frame_stack = np.stack(frame_metrics_all, axis=0)
    mean_dice = _safe_nanmean(official_frame_stack[:, 0])
    mean_hd95 = _safe_nanmean(official_frame_stack[:, 1])
    mean_iou = _safe_nanmean(official_frame_stack[:, 2])

    per_class_mean = _mean_metric_array(np.stack(per_class_metrics_all, axis=0))
    per_class_diagnostic = _named_per_class_metrics(
        per_class_mean, num_classes, class_names
    )

    per_case_group: Dict[str, Dict[str, Any]] = {}
    for group_name in sorted(case_group_frame_metrics, key=_case_group_sort_key):
        group_frame_stack = np.stack(case_group_frame_metrics[group_name], axis=0)
        group_class_mean = _mean_metric_array(
            np.stack(case_group_per_class_metrics[group_name], axis=0)
        )
        per_case_group[group_name] = {
            "mean_dice": float(_safe_nanmean(group_frame_stack[:, 0])),
            "mean_hd95": float(_safe_nanmean(group_frame_stack[:, 1])),
            "mean_iou": float(_safe_nanmean(group_frame_stack[:, 2])),
            "num_frames": len(case_group_frame_metrics[group_name]),
            "per_class_diagnostic": _named_per_class_metrics(
                group_class_mean, num_classes, class_names
            ),
        }

    class_presence_counts_dict = {
        _class_label(class_names, class_id): int(class_presence_counts[class_id])
        for class_id in range(1, num_classes)
    }
    false_positive_counts_dict = {
        _class_label(class_names, class_id): int(
            false_positive_absent_class_counts[class_id]
        )
        for class_id in range(1, num_classes)
    }
    num_frames = len(frame_metrics_all)
    evaluated_frames = num_frames - discounted_frames

    return {
        "mean_dice": float(mean_dice),
        "mean_hd95": float(mean_hd95),
        "mean_iou": float(mean_iou),
        "protocol": "{}_frame_present_background_excluded".format(dataset_name),
        "per_class_diagnostic": per_class_diagnostic,
        "per_case_group": per_case_group,
        "class_presence_counts": class_presence_counts_dict,
        "false_positive_absent_class_counts": false_positive_counts_dict,
        "discounted_frames": int(discounted_frames),
        "evaluated_frames": int(evaluated_frames),
        "num_frames": int(num_frames),
    }


def aggregate_volume_metrics(
    case_metrics,
    *,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Aggregate ``[case, foreground_class, metric]`` values case-first."""

    metrics_stack = np.asarray(case_metrics, dtype=np.float32)
    if metrics_stack.ndim != 3 or metrics_stack.shape[2] != len(METRIC_NAMES):
        raise ValueError(
            "case_metrics must have shape [case, foreground_class, 3], got {}".format(
                metrics_stack.shape
            )
        )
    if metrics_stack.shape[0] == 0 or metrics_stack.shape[1] == 0:
        raise RuntimeError("No metrics were collected during inference.")

    per_class_mean = _mean_metric_array(metrics_stack)
    num_classes = per_class_mean.shape[0] + 1
    return {
        "mean_dice": float(_safe_nanmean(per_class_mean[:, 0])),
        "mean_hd95": float(_safe_nanmean(per_class_mean[:, 1])),
        "mean_iou": float(_safe_nanmean(per_class_mean[:, 2])),
        "per_class": _named_per_class_metrics(
            per_class_mean, num_classes, class_names
        ),
        "per_class_array": per_class_mean,
    }


def _spacing_value_to_vector(value) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy().reshape(-1)
    if isinstance(value, np.ndarray):
        return value.reshape(-1)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _spacing_value_to_vector(value[0])
        if len(value) == 0:
            return np.asarray(value).reshape(-1)
        return np.concatenate([_spacing_value_to_vector(item) for item in value])
    return np.asarray(value).reshape(-1)


def _normalize_acdc_spacing_zyx(value, key: str, case_name: Any) -> Tuple[float, float, float]:
    spacing = _spacing_value_to_vector(value).astype(np.float32)
    if key == "pixdim" and spacing.size >= 4:
        spacing = spacing[1:4]
    else:
        spacing = spacing[:3]
    if spacing.size != 3:
        raise ValueError(
            "ACDC case {} spacing key {} must provide 3 spatial values, got {}".format(
                case_name, key, spacing.tolist()
            )
        )
    if key not in ACDC_SPACING_ZYX_KEYS:
        spacing = spacing[::-1]
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(
            "ACDC case {} spacing key {} must be positive finite values, got {}".format(
                case_name, key, spacing.tolist()
            )
        )
    return tuple(float(value) for value in spacing)


def extract_acdc_voxelspacing_zyx(
    sampled_batch: Mapping[str, Any],
    case_name: Any,
) -> Optional[Tuple[float, float, float]]:
    """Extract and normalize the first supported ACDC spacing field."""

    for key in ACDC_SPACING_KEYS:
        if key in sampled_batch:
            return _normalize_acdc_spacing_zyx(sampled_batch[key], key, case_name)
    return None


def _extract_acdc_voxelspacing_zyx(sampled_batch, case_name):
    """Compatibility alias for the TransUNet helper name."""

    return extract_acdc_voxelspacing_zyx(sampled_batch, case_name)


def validate_acdc_voxelspacing_zyx(
    voxelspacing,
    case_name: Any,
) -> Tuple[float, float, float]:
    """Require exactly three positive, finite z-y-x spacing values."""

    spacing = np.asarray(voxelspacing, dtype=np.float64).reshape(-1)
    if spacing.size != 3:
        raise ValueError(
            "ACDC HD95 voxelspacing for case {} must have 3 values, got {}".format(
                case_name, tuple(float(value) for value in spacing)
            )
        )
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(
            "ACDC HD95 voxelspacing for case {} must be positive finite values, got {}".format(
                case_name, tuple(float(value) for value in spacing)
            )
        )
    return tuple(float(value) for value in spacing)


def fallback_acdc_voxelspacing_zyx(
    z_spacing: float,
    case_name: Any,
) -> Tuple[float, float, float]:
    """Build the validated ``(z, 1, 1)`` ACDC fallback spacing."""

    z_spacing = float(z_spacing)
    if not np.isfinite(z_spacing) or z_spacing <= 0:
        raise ValueError(
            "ACDC case {} fallback --acdc_zspacing must be a positive finite value, got {}".format(
                case_name, z_spacing
            )
        )
    return z_spacing, 1.0, 1.0


def _fallback_acdc_voxelspacing_zyx(args, case_name):
    """Compatibility wrapper accepting the reference argparse namespace."""

    return fallback_acdc_voxelspacing_zyx(
        getattr(args, "acdc_zspacing", 5.0), case_name
    )


def resolve_acdc_voxelspacing_zyx(
    sampled_batch: Mapping[str, Any],
    case_name: Any,
    fallback_z_spacing: float = 5.0,
) -> Tuple[Tuple[float, float, float], str]:
    """Resolve dataset metadata first and a z-spacing fallback second.

    Returns ``(spacing_zyx, source)`` so callers can log the source per case.
    """

    for key in ACDC_SPACING_KEYS:
        if key in sampled_batch:
            return (
                _normalize_acdc_spacing_zyx(sampled_batch[key], key, case_name),
                "dataset metadata ({})".format(key),
            )
    return (
        fallback_acdc_voxelspacing_zyx(fallback_z_spacing, case_name),
        "fallback --acdc_zspacing",
    )


__all__ = [
    "ACDC_SPACING_KEYS",
    "ACDC_SPACING_ZYX_KEYS",
    "METRIC_NAMES",
    "_case_group_name",
    "_case_group_sort_key",
    "_class_label",
    "_extract_acdc_voxelspacing_zyx",
    "_fallback_acdc_voxelspacing_zyx",
    "_hd95_empty_mask_penalty",
    "_mean_metric_array",
    "_present_class_metric",
    "_safe_nanmean",
    "aggregate_present_class_metrics",
    "aggregate_volume_metrics",
    "calculate_metric_percase",
    "calculate_metric_percase_without_absent_reward",
    "extract_acdc_voxelspacing_zyx",
    "fallback_acdc_voxelspacing_zyx",
    "present_class_frame_metrics",
    "resolve_acdc_voxelspacing_zyx",
    "validate_acdc_voxelspacing_zyx",
]
