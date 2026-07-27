"""Cataract1k RGB frame segmentation dataset."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CATARACT_CLASS_MAP = {
    "Pupil": 1,
    "pupil1": 1,
    "Cornea": 2,
    "cornea1": 2,
    "Lens": 3,
    "Instruments": 4,
}

CATARACT_INSTRUMENTS = (
    "Slit Knife",
    "Gauge",
    "Capsulorhexis Cystotome",
    "Spatula",
    "Phacoemulsification Tip",
    "Irrigation-Aspiration",
    "Lens Injector",
    "Incision Knife",
    "Katena Forceps",
    "Capsulorhexis Forceps",
)


def random_rot_flip(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator4Cataract:
    """Resize/augment RGB frames and optionally apply ImageNet normalization."""

    def __init__(
        self,
        output_size: Sequence[int],
        augment: bool = True,
        normalize: bool = True,
        image_mean: Sequence[float] = IMAGENET_MEAN,
        image_std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        if len(output_size) != 2 or any(int(value) <= 0 for value in output_size):
            raise ValueError(f"output_size must contain two positive values, got {output_size!r}")
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("Cataract1k image_mean and image_std must each contain 3 values")
        if any(float(value) <= 0 for value in image_std):
            raise ValueError("Cataract1k image_std values must be positive")

        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.augment = bool(augment)
        self.normalize = bool(normalize)
        self.image_mean = np.asarray(image_mean, dtype=np.float32).reshape(1, 1, 3)
        self.image_std = np.asarray(image_std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        image = np.asarray(sample["image"])
        label = np.asarray(sample["label"])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Cataract1k images must be HWC RGB arrays, got {image.shape}")
        if label.ndim != 2 or image.shape[:2] != label.shape:
            raise ValueError(
                "Cataract1k image/label spatial shapes must match; "
                f"got image {image.shape} and label {label.shape}"
            )

        # Keep the supplied transform's independent augmentation draws.
        if self.augment and random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif self.augment and random.random() > 0.5:
            image, label = random_rotate(image, label)

        height, width = image.shape[:2]
        if (height, width) != self.output_size:
            image = zoom(
                image,
                (self.output_size[0] / height, self.output_size[1] / width, 1),
                order=3,
            )
            label = zoom(
                label,
                (self.output_size[0] / height, self.output_size[1] / width),
                order=0,
            )

        image = np.asarray(image, dtype=np.float32)
        if self.normalize:
            image = image / 255.0
            image = (image - self.image_mean) / self.image_std

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous()
        label_tensor = torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))
        return {"image": image_tensor, "label": label_tensor}


def load_split(
    csv_path: str | Path,
    base_dir: str | Path,
) -> tuple[list[str], list[str]]:
    """Resolve the image/annotation pairs listed by a Cataract1k split CSV."""

    csv_file = Path(csv_path).expanduser()
    data_root = Path(base_dir).expanduser()
    if not csv_file.is_file():
        raise FileNotFoundError(f"Cataract1k split CSV is missing: {csv_file}")

    with csv_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "imgs" not in reader.fieldnames:
            raise ValueError(
                f"Cataract1k split CSV {csv_file} must contain an 'imgs' column; "
                f"found {reader.fieldnames or []}"
            )
        filenames = [Path(row["imgs"]).name for row in reader if row.get("imgs", "").strip()]

    image_files = [str(data_root / "img" / filename) for filename in filenames]
    # The supplied annotations retain the image extension, e.g. frame.png.json.
    annotation_files = [str(data_root / "ann" / f"{filename}.json") for filename in filenames]
    return image_files, annotation_files


class Cataract1kDataset(Dataset):
    """Load original RGB Cataract1k frames and rasterize JSON polygons."""

    class_map_template = CATARACT_CLASS_MAP
    instrument_titles = CATARACT_INSTRUMENTS

    def __init__(
        self,
        base_dir: str | Path,
        split: str,
        transform: Any | None = None,
        train_csv: str = "train.csv",
        test_csv: str = "test.csv",
    ) -> None:
        self.transform = transform
        self.split = str(split).lower()
        if self.split not in {"train", "test"}:
            raise ValueError(
                f"Unsupported Cataract1k split {split!r}; expected 'train' or 'test'"
            )

        self.data_dir = Path(base_dir).expanduser()
        self.image_dir = self.data_dir / "img"
        self.annotation_dir = self.data_dir / "ann"
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Cataract1k root does not exist or is not a directory: {self.data_dir}"
            )
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Cataract1k image directory is missing: {self.image_dir}")
        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Cataract1k annotation directory is missing: {self.annotation_dir}"
            )

        csv_name = train_csv if self.split == "train" else test_csv
        csv_path = self.data_dir / csv_name
        self.image_files, self.annotation_files = load_split(csv_path, self.data_dir)
        if len(self.image_files) != len(self.annotation_files):
            raise RuntimeError(
                "Cataract1k split produced different image and annotation counts: "
                f"{len(self.image_files)} versus {len(self.annotation_files)}"
            )
        if not self.image_files:
            raise RuntimeError(f"Cataract1k split CSV contains no frames: {csv_path}")

        # Instance attributes preserve the public shape of the supplied class.
        self.class_map = dict(CATARACT_CLASS_MAP)
        self.instruments = list(CATARACT_INSTRUMENTS)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path = Path(self.image_files[idx])
        annotation_path = Path(self.annotation_files[idx])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read Cataract1k image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not annotation_path.is_file():
            raise FileNotFoundError(f"Cataract1k annotation is missing: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        label = self.process_annotation(annotation, image.shape[:2])

        sample: dict[str, Any] = {"image": image, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample["case_name"] = image_path.stem
        return sample

    def process_annotation(
        self,
        annotation: Mapping[str, Any],
        image_shape: Sequence[int],
    ) -> np.ndarray:
        """Rasterize supplied exterior polygons into the five-class mask."""

        if len(image_shape) != 2:
            raise ValueError(f"image_shape must be (height, width), got {image_shape!r}")
        height, width = int(image_shape[0]), int(image_shape[1])
        mask = np.zeros((height, width), dtype=np.uint8)

        for obj in annotation.get("objects", []):
            class_title = obj.get("classTitle")
            if class_title in self.class_map:
                class_id = self.class_map[class_title]
            elif class_title in self.instruments:
                class_id = 4
            else:
                continue

            try:
                exterior = obj["points"]["exterior"]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"Cataract1k object {class_title!r} is missing points.exterior"
                ) from exc
            exterior_points = np.asarray(exterior, dtype=np.int32)
            if exterior_points.ndim != 2 or exterior_points.shape[1:] != (2,):
                raise ValueError(
                    f"Cataract1k object {class_title!r} has invalid exterior points shape "
                    f"{exterior_points.shape}"
                )
            if len(exterior_points) < 3:
                continue
            cv2.fillPoly(mask, [exterior_points], int(class_id))

        return mask


__all__ = [
    "CATARACT_CLASS_MAP",
    "CATARACT_INSTRUMENTS",
    "Cataract1kDataset",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "RandomGenerator4Cataract",
    "load_split",
    "random_rot_flip",
    "random_rotate",
]
