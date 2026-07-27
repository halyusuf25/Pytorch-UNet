"""Synapse slice-training and volume-evaluation dataset.

This module follows the dataset implementation used by the TransUNet
``multiconfig`` branch.  Training samples are individual ``.npz`` slices;
validation and test samples are complete HDF5 volumes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def random_rot_flip(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a shared random 90-degree rotation and flip."""

    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the reference small-angle image/label rotation."""

    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator:
    """Reference Synapse training augmentation and resize transform."""

    def __init__(self, output_size: Sequence[int]):
        if len(output_size) != 2 or any(int(value) <= 0 for value in output_size):
            raise ValueError(f"output_size must contain two positive values, got {output_size!r}")
        self.output_size = (int(output_size[0]), int(output_size[1]))

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        image = np.asarray(sample["image"])
        label = np.asarray(sample["label"])

        # Keep the two independent draws used by the supplied implementation.
        if random.random() > 0.6:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.35:
            image, label = random_rotate(image, label)

        if image.ndim != 2 or label.ndim != 2:
            raise ValueError(
                "Synapse training samples must be 2D image/label pairs; "
                f"got image {image.shape} and label {label.shape}"
            )
        if image.shape != label.shape:
            raise ValueError(
                f"Synapse image and label shapes differ: {image.shape} versus {label.shape}"
            )

        height, width = image.shape
        if (height, width) != self.output_size:
            factors = (self.output_size[0] / height, self.output_size[1] / width)
            image = zoom(image, factors, order=3)
            label = zoom(label, factors, order=0)

        return {
            "image": torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0),
            "label": torch.from_numpy(np.asarray(label, dtype=np.int64)),
        }


class Synapse_dataset(Dataset):
    """Load Synapse training slices or complete test volumes.

    ``split="train"`` expects ``<base_dir>/<case>.npz``.  Other splits,
    notably ``test_vol``, expect ``<base_dir>/<case>.npy.h5``.
    """

    def __init__(
        self,
        base_dir: str | Path,
        list_dir: str | Path,
        split: str,
        transform: Any | None = None,
    ) -> None:
        if not split or not str(split).strip():
            raise ValueError("Synapse split must be a non-empty string")

        self.transform = transform
        self.split = str(split).strip()
        self.data_dir = Path(base_dir).expanduser()
        self.list_dir = Path(list_dir).expanduser()

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Synapse data directory does not exist or is not a directory: {self.data_dir}"
            )
        if not self.list_dir.is_dir():
            raise FileNotFoundError(
                f"Synapse list directory does not exist or is not a directory: {self.list_dir}"
            )

        split_file = self.list_dir / f"{self.split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"Synapse split file is missing: {split_file}. "
                f"Expected {self.split}.txt in the configured list directory."
            )
        self.sample_list = [
            line.strip() for line in split_file.read_text().splitlines() if line.strip()
        ]
        if not self.sample_list:
            raise RuntimeError(f"Synapse split file contains no samples: {split_file}")

    def __len__(self) -> int:
        return len(self.sample_list)

    def _sample_path(self, case_name: str) -> Path:
        suffix = ".npz" if self.split == "train" else ".npy.h5"
        return self.data_dir / f"{case_name}{suffix}"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case_name = self.sample_list[idx]
        data_path = self._sample_path(case_name)
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Synapse sample '{case_name}' from split '{self.split}' is missing: {data_path}"
            )

        if self.split == "train":
            with np.load(data_path) as data:
                image = data["image"]
                label = data["label"]
        else:
            with h5py.File(data_path, "r") as data:
                image = data["image"][:]
                label = data["label"][:]

        sample: dict[str, Any] = {"image": image, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample["case_name"] = case_name
        return sample


# Conventional spelling for new callers without breaking the supplied API.
SynapseDataset = Synapse_dataset


__all__ = [
    "RandomGenerator",
    "SynapseDataset",
    "Synapse_dataset",
    "random_rot_flip",
    "random_rotate",
]
