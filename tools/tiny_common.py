"""Shared data and metric helpers for the tiny-subset diagnostics.

These helpers intentionally operate on the training *slice/frame* datasets.
They do not call the volume-level Synapse or ACDC evaluation protocol.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from datasets import ACDC_Dataset, Cataract1kDataset, Synapse_dataset
from datasets.dataset_cataract import IMAGENET_MEAN, IMAGENET_STD
from utils.dataset_registry import (
    DatasetPaths,
    DatasetSpec,
    validate_dataset_paths,
)


MIN_TINY_SAMPLES = 8
MAX_TINY_SAMPLES = 16
OVERFIT_TARGET = 0.90


def validate_num_samples(value: int) -> int:
    value = int(value)
    if not MIN_TINY_SAMPLES <= value <= MAX_TINY_SAMPLES:
        raise ValueError(
            f"num_samples must be between {MIN_TINY_SAMPLES} and "
            f"{MAX_TINY_SAMPLES}, got {value}"
        )
    return value


class ResizeOnlyTransform:
    """Resize one training slice/frame without any stochastic augmentation."""

    def __init__(self, img_size: int, *, rgb_imagenet: bool) -> None:
        self.img_size = int(img_size)
        if self.img_size < 16:
            raise ValueError(
                f"img_size must be at least 16 for the four-level U-Net, got {img_size}"
            )
        self.output_size = (self.img_size, self.img_size)
        self.rgb_imagenet = bool(rgb_imagenet)
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def resize_label(self, label: Any) -> torch.Tensor:
        """Resize only a mask, avoiding unnecessary image interpolation while scanning."""

        label = np.asarray(label)
        if label.ndim != 2:
            raise ValueError(f"Tiny-subset labels must be 2-D, got shape {label.shape}")
        height, width = label.shape
        if (height, width) != self.output_size:
            label = zoom(
                label,
                (self.img_size / height, self.img_size / width),
                order=0,
            )
        if label.shape != self.output_size:
            raise RuntimeError(
                f"Resize produced label shape {label.shape}, expected {self.output_size}"
            )
        return torch.from_numpy(
            np.ascontiguousarray(label, dtype=np.int64)
        ).contiguous()

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        image = np.asarray(sample["image"])
        label = np.asarray(sample["label"])
        if label.ndim != 2:
            raise ValueError(f"Tiny-subset labels must be 2-D, got shape {label.shape}")

        if self.rgb_imagenet:
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    "Cataract1k tiny-subset images must be HWC RGB arrays, "
                    f"got shape {image.shape}"
                )
            if image.shape[:2] != label.shape:
                raise ValueError(
                    f"Image and label shapes differ: {image.shape[:2]} versus {label.shape}"
                )
        else:
            if image.ndim != 2:
                raise ValueError(
                    f"Synapse/ACDC tiny-subset images must be 2-D, got {image.shape}"
                )
            if image.shape != label.shape:
                raise ValueError(
                    f"Image and label shapes differ: {image.shape} versus {label.shape}"
                )

        height, width = label.shape
        if (height, width) != self.output_size:
            spatial_factors = (self.img_size / height, self.img_size / width)
            if self.rgb_imagenet:
                image = zoom(image, (*spatial_factors, 1), order=3)
            else:
                image = zoom(image, spatial_factors, order=3)

        label_tensor = self.resize_label(label)

        image = np.asarray(image, dtype=np.float32)
        if self.rgb_imagenet:
            if image.shape != (*self.output_size, 3):
                raise RuntimeError(
                    f"Resize produced image shape {image.shape}, "
                    f"expected {(*self.output_size, 3)}"
                )
            image = (image / 255.0 - self.mean) / self.std
            image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        else:
            if image.shape != self.output_size:
                raise RuntimeError(
                    f"Resize produced image shape {image.shape}, expected {self.output_size}"
                )
            image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)

        return {
            "image": image_tensor.contiguous(),
            "label": label_tensor.contiguous(),
        }


def build_raw_training_dataset(spec: DatasetSpec, paths: DatasetPaths) -> Dataset:
    """Construct the registry's explicit training split with no transform."""

    validate_dataset_paths(spec, paths, "train")
    if spec.name == "Synapse":
        if paths.list_dir is None:
            raise ValueError("Synapse requires --list-dir")
        return Synapse_dataset(
            base_dir=paths.train_root,
            list_dir=paths.list_dir,
            split="train",
            transform=None,
        )
    if spec.name == "ACDC":
        return ACDC_Dataset(
            base_dir=paths.train_root,
            split="train",
            transform=None,
        )
    if spec.name == "Cataract1k":
        return Cataract1kDataset(
            base_dir=paths.train_root,
            split="train",
            transform=None,
        )
    raise ValueError(f"Tiny-subset diagnostics do not support dataset {spec.name!r}")


def _validate_resized_label(
    label: torch.Tensor,
    *,
    num_classes: int,
    index: int,
    case_name: str,
) -> tuple[int, ...]:
    if label.ndim != 2:
        raise ValueError(
            f"Sample {index} ({case_name}) produced label shape {tuple(label.shape)}; "
            "expected a 2-D label"
        )
    if label.numel() == 0:
        raise ValueError(f"Sample {index} ({case_name}) has an empty label")
    minimum = int(label.min().item())
    maximum = int(label.max().item())
    if minimum < 0 or maximum >= num_classes:
        raise ValueError(
            f"Sample {index} ({case_name}) has labels in [{minimum}, {maximum}], "
            f"but {num_classes} classes require [0, {num_classes - 1}]"
        )
    return tuple(
        int(value)
        for value in torch.unique(label).tolist()
        if 0 < int(value) < num_classes
    )


@dataclass(frozen=True)
class TinySampleRecord:
    index: int
    case_name: str
    foreground_classes: tuple[int, ...]

    def as_dict(self, class_names: Sequence[str]) -> dict[str, Any]:
        return {
            "index": self.index,
            "case_name": self.case_name,
            "foreground_classes": list(self.foreground_classes),
            "foreground_class_names": [
                class_names[class_id] for class_id in self.foreground_classes
            ],
        }


def inspect_sample(
    raw_dataset: Dataset,
    transform: ResizeOnlyTransform,
    index: int,
    num_classes: int,
) -> TinySampleRecord:
    sample = raw_dataset[int(index)]
    if not isinstance(sample, Mapping) or "image" not in sample or "label" not in sample:
        raise TypeError(
            f"Training dataset sample {index} must contain image and label mappings"
        )
    case_name = str(sample.get("case_name", index))
    resized_label = transform.resize_label(sample["label"])
    classes = _validate_resized_label(
        resized_label,
        num_classes=num_classes,
        index=int(index),
        case_name=case_name,
    )
    return TinySampleRecord(int(index), case_name, classes)


def select_tiny_records(
    raw_dataset: Dataset,
    transform: ResizeOnlyTransform,
    *,
    num_samples: int,
    num_classes: int,
    seed: int,
) -> list[TinySampleRecord]:
    """Greedily maximize class coverage, with seeded deterministic tie-breaking."""

    num_samples = validate_num_samples(num_samples)
    if len(raw_dataset) < num_samples:
        raise ValueError(
            f"Training split has only {len(raw_dataset)} samples; "
            f"cannot select {num_samples}"
        )

    order = np.random.default_rng(int(seed)).permutation(len(raw_dataset))
    foreground_records: list[TinySampleRecord] = []
    represented_classes: set[int] = set()
    desired_classes = set(range(1, int(num_classes)))
    for index in order:
        record = inspect_sample(
            raw_dataset,
            transform,
            int(index),
            num_classes,
        )
        if record.foreground_classes:
            foreground_records.append(record)
            represented_classes.update(record.foreground_classes)
            # Once every registered foreground class is represented, no later
            # sample can improve class coverage. Stop after enough candidates
            # to fill the requested subset.
            if (
                len(foreground_records) >= num_samples
                and represented_classes == desired_classes
            ):
                break

    if len(foreground_records) < num_samples:
        raise RuntimeError(
            f"Only {len(foreground_records)} of {len(raw_dataset)} resized training "
            f"samples contain foreground; need {num_samples}"
        )

    available_classes = {
        class_id
        for record in foreground_records
        for class_id in record.foreground_classes
    }
    selected: list[TinySampleRecord] = []
    remaining = list(foreground_records)
    covered: set[int] = set()

    while remaining and covered != available_classes and len(selected) < num_samples:
        best_position = max(
            range(len(remaining)),
            key=lambda position: len(
                set(remaining[position].foreground_classes).difference(covered)
            ),
        )
        chosen = remaining.pop(best_position)
        selected.append(chosen)
        covered.update(chosen.foreground_classes)

    if len(selected) < num_samples:
        selected.extend(remaining[: num_samples - len(selected)])

    if any(not record.foreground_classes for record in selected):
        raise AssertionError("Tiny-subset selection admitted a background-only sample")
    return selected


class TinySubsetDataset(Dataset):
    """Apply the deterministic resize to an explicitly indexed raw subset."""

    def __init__(
        self,
        raw_dataset: Dataset,
        records: Sequence[TinySampleRecord],
        transform: ResizeOnlyTransform,
        num_classes: int,
    ) -> None:
        if not records:
            raise ValueError("Tiny subset must contain at least one record")
        self.raw_dataset = raw_dataset
        self.records = tuple(records)
        self.transform = transform
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        raw = self.raw_dataset[record.index]
        case_name = str(raw.get("case_name", record.index))
        if case_name != record.case_name:
            raise RuntimeError(
                f"Subset case mismatch at dataset index {record.index}: "
                f"JSON selected {record.case_name!r}, dataset now returns {case_name!r}"
            )
        transformed = self.transform(raw)
        classes = _validate_resized_label(
            transformed["label"],
            num_classes=self.num_classes,
            index=record.index,
            case_name=case_name,
        )
        if classes != record.foreground_classes:
            raise RuntimeError(
                f"Subset mask changed for {case_name!r}: JSON records "
                f"{list(record.foreground_classes)}, current resized mask has {list(classes)}"
            )
        return {
            "image": transformed["image"],
            "label": transformed["label"],
            "case_name": case_name,
            "sample_index": record.index,
        }


def records_from_document(
    document: Mapping[str, Any],
    *,
    expected_dataset: str,
    num_classes: int,
) -> list[TinySampleRecord]:
    if document.get("dataset") != expected_dataset:
        raise ValueError(
            f"Subset JSON dataset is {document.get('dataset')!r}, "
            f"expected {expected_dataset!r}"
        )
    raw_records = document.get("samples")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Subset JSON must contain a non-empty 'samples' list")

    records = []
    for position, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise ValueError(f"Subset JSON sample {position} is not an object")
        try:
            index = int(item["index"])
            case_name = str(item["case_name"])
            classes = tuple(int(value) for value in item["foreground_classes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Subset JSON sample {position} is invalid: {error}") from error
        if index < 0:
            raise ValueError(f"Subset JSON sample {position} has negative index {index}")
        if not case_name:
            raise ValueError(f"Subset JSON sample {position} has an empty case_name")
        if not classes:
            raise ValueError(
                f"Subset JSON sample {position} ({case_name}) contains no foreground"
            )
        if tuple(sorted(set(classes))) != classes:
            raise ValueError(
                f"Subset JSON sample {position} foreground classes must be sorted and unique"
            )
        if classes[0] <= 0 or classes[-1] >= num_classes:
            raise ValueError(
                f"Subset JSON sample {position} has foreground class outside "
                f"[1, {num_classes - 1}]"
            )
        records.append(TinySampleRecord(index, case_name, classes))

    stored_indices = document.get("selected_indices")
    if stored_indices is not None and [record.index for record in records] != [
        int(value) for value in stored_indices
    ]:
        raise ValueError("Subset JSON selected_indices disagrees with samples")
    if len({record.index for record in records}) != len(records):
        raise ValueError("Subset JSON contains duplicate sample indices")
    return records


def build_subset_document(
    *,
    spec: DatasetSpec,
    paths: DatasetPaths,
    records: Sequence[TinySampleRecord],
    img_size: int,
    seed: int,
    arguments: Mapping[str, Any],
    success_threshold: float,
) -> dict[str, Any]:
    represented = sorted(
        {
            class_id
            for record in records
            for class_id in record.foreground_classes
        }
    )
    return {
        "format_version": 1,
        "dataset": spec.name,
        "img_size": int(img_size),
        "seed": int(seed),
        "num_samples": len(records),
        "success_threshold": float(success_threshold),
        "normalization": "ImageNet" if spec.imagenet_normalization else None,
        "class_names": list(spec.class_names),
        "foreground_classes": represented,
        "foreground_class_names": [spec.class_names[value] for value in represented],
        "selected_indices": [record.index for record in records],
        "case_names": [record.case_name for record in records],
        "samples": [record.as_dict(spec.class_names) for record in records],
        "paths": paths.as_dict(),
        "training_arguments": _json_safe(dict(arguments)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: str | Path, document: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(document), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Subset JSON does not exist: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Subset JSON is invalid ({source}): {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Subset JSON root must be an object: {source}")
    return document


@dataclass
class HardDiceAccumulator:
    intersections: torch.Tensor
    predicted_counts: torch.Tensor
    ground_truth_counts: torch.Tensor
    sample_count: int = 0
    foreground_sample_count: int = 0

    @classmethod
    def create(cls, num_classes: int) -> "HardDiceAccumulator":
        zeros = lambda: torch.zeros(int(num_classes), dtype=torch.float64)
        return cls(zeros(), zeros(), zeros())

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().to("cpu")
        target = target.detach().to("cpu")
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {prediction.shape} versus {target.shape}"
            )
        if prediction.ndim != 3:
            raise ValueError(
                f"Hard Dice expects BHW prediction/target tensors, got {prediction.shape}"
            )
        self.sample_count += int(target.shape[0])
        self.foreground_sample_count += int(
            (target.reshape(target.shape[0], -1) > 0).any(dim=1).sum().item()
        )
        for class_id in range(len(self.intersections)):
            predicted_mask = prediction == class_id
            target_mask = target == class_id
            self.intersections[class_id] += (
                predicted_mask & target_mask
            ).sum(dtype=torch.float64)
            self.predicted_counts[class_id] += predicted_mask.sum(dtype=torch.float64)
            self.ground_truth_counts[class_id] += target_mask.sum(dtype=torch.float64)

    def result(self) -> dict[str, Any]:
        dice: list[float] = []
        for class_id in range(len(self.intersections)):
            gt_count = float(self.ground_truth_counts[class_id])
            denominator = float(
                self.predicted_counts[class_id] + self.ground_truth_counts[class_id]
            )
            if class_id > 0 and gt_count == 0:
                dice.append(float("nan"))
            elif denominator == 0:
                dice.append(1.0)
            else:
                dice.append(2.0 * float(self.intersections[class_id]) / denominator)
        present_foreground = [
            value
            for class_id, value in enumerate(dice)
            if class_id > 0 and not math.isnan(value)
        ]
        if not present_foreground:
            raise RuntimeError("The evaluated tiny subset contains no foreground classes")
        return {
            "foreground_mean_dice": float(np.mean(present_foreground)),
            "per_class_dice": dice,
            "predicted_counts": [
                int(value) for value in self.predicted_counts.tolist()
            ],
            "ground_truth_counts": [
                int(value) for value in self.ground_truth_counts.tolist()
            ],
            "sample_count": self.sample_count,
            "foreground_sample_count": self.foreground_sample_count,
        }


def format_metrics(metrics: Mapping[str, Any], class_names: Sequence[str]) -> str:
    lines = [
        f"foreground mean hard Dice: {metrics['foreground_mean_dice']:.6f}",
        "class metrics:",
    ]
    dice = metrics["per_class_dice"]
    predicted = metrics["predicted_counts"]
    target = metrics["ground_truth_counts"]
    for class_id, name in enumerate(class_names):
        value = dice[class_id]
        dice_text = "N/A (absent in GT)" if math.isnan(value) else f"{value:.6f}"
        lines.append(
            f"  [{class_id}] {name}: Dice={dice_text}, "
            f"predicted_pixels={predicted[class_id]}, "
            f"ground_truth_pixels={target[class_id]}"
        )
    return "\n".join(lines)


def safe_case_stem(case_name: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_name)).strip("._")
    return f"{index:05d}_{cleaned or 'sample'}"


def save_visualization_triplet(
    output_dir: str | Path,
    *,
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    case_name: str,
    index: int,
    imagenet_normalized: bool,
) -> None:
    from PIL import Image

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    stem = safe_case_stem(case_name, index)

    image_array = image.detach().cpu().numpy()
    if imagenet_normalized:
        image_array = np.moveaxis(image_array, 0, -1)
        image_array = image_array * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
        image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    else:
        image_array = image_array[0]
        finite = image_array[np.isfinite(image_array)]
        if finite.size == 0:
            image_array = np.zeros_like(image_array, dtype=np.uint8)
        else:
            low, high = np.percentile(finite, (1, 99))
            if high <= low:
                low, high = float(finite.min()), float(finite.max())
            scale = max(float(high - low), 1e-12)
            image_array = np.clip((image_array - low) / scale * 255.0, 0, 255).astype(
                np.uint8
            )

    palette = np.asarray(
        [
            (0, 0, 0),
            (230, 25, 75),
            (60, 180, 75),
            (255, 225, 25),
            (0, 130, 200),
            (245, 130, 48),
            (145, 30, 180),
            (70, 240, 240),
            (240, 50, 230),
        ],
        dtype=np.uint8,
    )
    target_array = target.detach().cpu().numpy().astype(np.int64)
    prediction_array = prediction.detach().cpu().numpy().astype(np.int64)
    if target_array.max(initial=0) >= len(palette):
        raise ValueError("Visualization palette does not cover target class IDs")

    Image.fromarray(image_array).save(destination / f"{stem}_image.png")
    Image.fromarray(palette[target_array], mode="RGB").save(
        destination / f"{stem}_ground_truth.png"
    )
    Image.fromarray(palette[prediction_array], mode="RGB").save(
        destination / f"{stem}_prediction.png"
    )
