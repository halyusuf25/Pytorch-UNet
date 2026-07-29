#!/usr/bin/env python3
"""Re-evaluate a saved tiny-subset checkpoint on its exact training samples."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.tiny_common import (
    OVERFIT_TARGET,
    ResizeOnlyTransform,
    TinySubsetDataset,
    build_raw_training_dataset,
    format_metrics,
    read_json,
    records_from_document,
    save_visualization_triplet,
)
from unet import UNet
from utils.checkpointing import (
    checkpoint_metadata,
    load_checkpoint,
    load_checkpoint_file,
)
from utils.dataset_registry import (
    available_dataset_names,
    get_dataset_spec,
    resolve_dataset_paths,
)
from utils.model_output import _extract_logits
from utils.runtime import resolve_device, seed_everything


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly load a tiny-overfit best_model.pth and independently compute "
            "classwise hard Dice on the exact training slices/frames recorded in "
            "tiny_subset.json. This does not use the volume benchmark protocol."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=available_dataset_names(),
        help="Dataset identity; must match tiny_subset.json and the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the best_model.pth written by training_tiny.py.",
    )
    parser.add_argument(
        "--subset-json",
        required=True,
        help="Path to the corresponding tiny_subset.json manifest.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help="Inference batch size; defaults to all recorded samples.",
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help=(
            "Override the training root saved in tiny_subset.json, for example after "
            "moving the dataset."
        ),
    )
    parser.add_argument(
        "--list-dir",
        default=None,
        help="Override the saved Synapse split-list directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cpu, cuda, or cuda:1.",
    )
    parser.add_argument(
        "--num-workers",
        type=_nonnegative_int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--save-predictions",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "Save image, ground-truth, and prediction PNGs. With no DIR, use a "
            "'tiny_predictions' directory beside the checkpoint."
        ),
    )
    return parser


def _require_int(document: dict[str, Any], key: str) -> int:
    try:
        return int(document[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Subset JSON requires an integer {key!r}") from error


def _stored_path(document: dict[str, Any], key: str) -> str | None:
    paths = document.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Subset JSON requires a 'paths' object")
    value = paths.get(key)
    return None if value is None else str(value)


def _infer_bilinear_and_validate_metadata(
    checkpoint_path: Path,
    *,
    spec,
    img_size: int,
) -> bool:
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    metadata = checkpoint_metadata(checkpoint)
    required = ("dataset", "n_channels", "n_classes", "bilinear", "img_size")
    missing = [key for key in required if metadata.get(key) is None]
    if missing:
        raise ValueError(
            "Tiny diagnostic checkpoint is missing required metadata: "
            + ", ".join(missing)
        )
    if str(metadata["dataset"]) != spec.name:
        raise ValueError(
            f"Checkpoint dataset is {metadata['dataset']!r}, expected {spec.name!r}"
        )
    if int(metadata["n_channels"]) != spec.input_channels:
        raise ValueError(
            f"Checkpoint has {metadata['n_channels']} input channels, "
            f"but {spec.name} requires {spec.input_channels}"
        )
    if int(metadata["n_classes"]) != spec.num_classes:
        raise ValueError(
            f"Checkpoint has {metadata['n_classes']} classes, "
            f"but {spec.name} requires {spec.num_classes}"
        )
    if int(metadata["img_size"]) != img_size:
        raise ValueError(
            f"Checkpoint image size is {metadata['img_size']}, "
            f"but tiny_subset.json records {img_size}"
        )
    class_names = metadata.get("class_names")
    if class_names is not None and list(class_names) != list(spec.class_names):
        raise ValueError("Checkpoint class_names do not match the dataset registry")
    normalization = metadata.get("normalization")
    expected_normalization = "ImageNet" if spec.imagenet_normalization else None
    if normalization != expected_normalization:
        raise ValueError(
            f"Checkpoint normalization is {normalization!r}, "
            f"expected {expected_normalization!r}"
        )
    value = metadata["bilinear"]
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered not in {"true", "false"}:
            raise ValueError(f"Checkpoint bilinear metadata is invalid: {value!r}")
        return lowered == "true"
    return bool(value)


def _independent_hard_dice(
    intersections: np.ndarray,
    predicted_counts: np.ndarray,
    ground_truth_counts: np.ndarray,
) -> tuple[list[float], float]:
    """Compute test metrics from integer counts, without training loss helpers."""

    values: list[float] = []
    for class_id in range(len(intersections)):
        denominator = predicted_counts[class_id] + ground_truth_counts[class_id]
        if class_id > 0 and ground_truth_counts[class_id] == 0:
            values.append(float("nan"))
        elif denominator == 0:
            values.append(1.0)
        else:
            values.append(float(2.0 * intersections[class_id] / denominator))
    foreground_present_values = [
        value
        for class_id, value in enumerate(values)
        if class_id > 0 and not math.isnan(value)
    ]
    if not foreground_present_values:
        raise RuntimeError("The reconstructed tiny subset contains no foreground classes")
    return values, float(np.mean(foreground_present_values))


def run(args: argparse.Namespace) -> dict[str, Any]:
    subset_document = read_json(args.subset_json)
    spec = get_dataset_spec(args.dataset)
    if subset_document.get("dataset") != spec.name:
        raise ValueError(
            f"--dataset {spec.name} does not match subset JSON dataset "
            f"{subset_document.get('dataset')!r}"
        )

    img_size = _require_int(subset_document, "img_size")
    seed = _require_int(subset_document, "seed")
    if img_size < 16:
        raise ValueError(f"Subset JSON img_size must be at least 16, got {img_size}")
    stored_class_names = subset_document.get("class_names")
    if stored_class_names != list(spec.class_names):
        raise ValueError("Subset JSON class_names do not match the dataset registry")
    checkpoint_path = Path(args.checkpoint).expanduser()
    bilinear = _infer_bilinear_and_validate_metadata(
        checkpoint_path,
        spec=spec,
        img_size=img_size,
    )

    saved_root = _stored_path(subset_document, "train_root")
    saved_list_dir = _stored_path(subset_document, "list_dir")
    paths = resolve_dataset_paths(
        spec,
        root=args.root_path if args.root_path is not None else saved_root,
        list_dir=args.list_dir if args.list_dir is not None else saved_list_dir,
    )
    records = records_from_document(
        subset_document,
        expected_dataset=spec.name,
        num_classes=spec.num_classes,
    )
    if _require_int(subset_document, "num_samples") != len(records):
        raise ValueError("Subset JSON num_samples disagrees with its samples list")

    seed_everything(seed, deterministic=True)
    raw_dataset = build_raw_training_dataset(spec, paths)
    if any(record.index >= len(raw_dataset) for record in records):
        raise ValueError(
            f"Subset JSON contains an index outside the current training split "
            f"(length {len(raw_dataset)})"
        )
    transform = ResizeOnlyTransform(
        img_size,
        rgb_imagenet=spec.imagenet_normalization,
    )
    subset = TinySubsetDataset(raw_dataset, records, transform, spec.num_classes)
    batch_size = len(subset) if args.batch_size is None else args.batch_size
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    device = resolve_device(args.device)
    model = UNet(
        n_channels=spec.input_channels,
        n_classes=spec.num_classes,
        bilinear=bilinear,
    ).to(device)
    report = load_checkpoint(
        model,
        checkpoint_path,
        mode="test",
        map_location=device,
        dataset=spec.name,
        n_channels=spec.input_channels,
        n_classes=spec.num_classes,
        bilinear=bilinear,
    )
    if report.partial:
        raise RuntimeError(
            "Tiny checkpoint load was unexpectedly partial: " + report.summary()
        )
    model.eval()

    intersections = np.zeros(spec.num_classes, dtype=np.int64)
    predicted_counts = np.zeros(spec.num_classes, dtype=np.int64)
    ground_truth_counts = np.zeros(spec.num_classes, dtype=np.int64)
    sample_count = 0
    foreground_sample_count = 0
    prediction_dir = None
    if args.save_predictions is not None:
        prediction_dir = (
            checkpoint_path.parent / "tiny_predictions"
            if args.save_predictions == ""
            else Path(args.save_predictions).expanduser()
        )

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            target = batch["label"].to(device=device, dtype=torch.long)
            if image.ndim != 4 or image.shape[1] != spec.input_channels:
                raise ValueError(
                    f"Expected BCHW input with {spec.input_channels} channels, "
                    f"got {tuple(image.shape)}"
                )
            logits = _extract_logits(model(image))
            expected_shape = (image.shape[0], spec.num_classes, *target.shape[-2:])
            if logits.shape != expected_shape:
                raise ValueError(
                    f"U-Net returned logits {tuple(logits.shape)}, expected {expected_shape}"
                )
            prediction = logits.argmax(dim=1)
            prediction_cpu = prediction.cpu()
            target_cpu = target.cpu()
            batch_count = int(target.shape[0])
            sample_count += batch_count
            foreground_sample_count += int(
                (target_cpu.reshape(batch_count, -1) > 0).any(dim=1).sum().item()
            )

            for class_id in range(spec.num_classes):
                predicted_mask = prediction_cpu == class_id
                target_mask = target_cpu == class_id
                intersections[class_id] += int((predicted_mask & target_mask).sum())
                predicted_counts[class_id] += int(predicted_mask.sum())
                ground_truth_counts[class_id] += int(target_mask.sum())

            if prediction_dir is not None:
                for position in range(batch_count):
                    save_visualization_triplet(
                        prediction_dir,
                        image=batch["image"][position],
                        target=batch["label"][position],
                        prediction=prediction_cpu[position],
                        case_name=str(batch["case_name"][position]),
                        index=int(batch["sample_index"][position]),
                        imagenet_normalized=spec.imagenet_normalization,
                    )

    per_class, mean_dice = _independent_hard_dice(
        intersections,
        predicted_counts,
        ground_truth_counts,
    )
    metrics = {
        "foreground_mean_dice": mean_dice,
        "per_class_dice": per_class,
        "predicted_counts": predicted_counts.tolist(),
        "ground_truth_counts": ground_truth_counts.tolist(),
        "sample_count": sample_count,
        "foreground_sample_count": foreground_sample_count,
    }
    threshold = float(subset_document.get("success_threshold", OVERFIT_TARGET))
    if not 0 < threshold <= 1:
        raise ValueError(
            f"Subset JSON success_threshold must be in (0, 1], got {threshold}"
        )
    reached = mean_dice >= threshold

    print(f"Dataset: {spec.name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Subset manifest: {Path(args.subset_json).expanduser()}")
    print(f"Number of samples: {sample_count}")
    print(f"Foreground-containing samples: {foreground_sample_count}")
    print(format_metrics(metrics, spec.class_names))
    print(
        f"Overfitting target reached: {'YES' if reached else 'NO'} "
        f"({mean_dice:.6f} {'>=' if reached else '<'} {threshold:.2f})"
    )
    if prediction_dir is not None:
        print(f"Saved image/ground-truth/prediction visualizations to: {prediction_dir}")
    return {**metrics, "success_threshold": threshold, "target_reached": reached}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
