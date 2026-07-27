"""ACDC slice-training and volume-evaluation dataset."""

from __future__ import annotations

import itertools
import random
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


SPACING_KEYS = (
    "voxelspacing_zyx",
    "spacing_zyx",
    "voxelspacing",
    "spacing",
    "spacing_mm",
    "pixdim",
    "zooms",
)
SPACING_ZYX_KEYS = frozenset({"voxelspacing_zyx", "spacing_zyx"})


class ACDC_Dataset(Dataset):
    """Load ACDC HDF5 slices or complete volumes.

    The active supplied split is intentionally preserved: patients 021--100
    train, while patients 001--020 are used for both validation and testing.
    ``fold_id`` remains accepted for compatibility but does not alter that
    split; this is not a five-fold cross-validation implementation.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        split: str = "train",
        list_dir: str | Path | None = None,
        transform: Any | None = None,
        fold_id: int = 0,
    ) -> None:
        del list_dir  # retained only for compatibility with older callers
        if base_dir is None:
            raise ValueError("ACDC base_dir is required")

        self._base_dir = Path(base_dir).expanduser()
        self.split = str(split)
        self.transform = transform
        self.fold_id = int(fold_id)
        if self.fold_id != 0:
            warnings.warn(
                "ACDC fold_id is accepted for API compatibility but ignored; "
                "the supplied active split always uses 001-020 for validation/test "
                "and 021-100 for training.",
                UserWarning,
                stacklevel=2,
            )

        split_lower = self.split.lower()
        if "train" in split_lower:
            self._split_kind = "train"
            split_dir = self._base_dir / "ACDC_training_slices"
            selected_ids = self._get_ids()[0]
        elif "val" in split_lower:
            self._split_kind = "val"
            split_dir = self._base_dir / "ACDC_training_volumes"
            selected_ids = self._get_ids()[1]
        elif "test" in split_lower:
            self._split_kind = "test"
            split_dir = self._base_dir / "ACDC_training_volumes"
            selected_ids = self._get_ids()[2]
        else:
            raise ValueError(
                f"Unsupported ACDC split {split!r}; expected a train, val, or test split"
            )

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"ACDC {self._split_kind} directory does not exist or is not a directory: "
                f"{split_dir}"
            )
        self._sample_dir = split_dir
        all_entries = sorted(path.name for path in split_dir.iterdir() if path.is_file())
        self.sample_list = [
            filename
            for patient_id in selected_ids
            for filename in all_entries
            if filename.startswith(patient_id)
        ]
        if not self.sample_list:
            patient_range = "021-100" if self._split_kind == "train" else "001-020"
            raise RuntimeError(
                f"No ACDC {self._split_kind} samples for patients {patient_range} were found "
                f"in {split_dir}"
            )

    @staticmethod
    def _get_ids() -> list[list[str]]:
        """Return the supplied fixed train/validation/test patient IDs."""

        all_cases = [f"patient{i:03d}" for i in range(1, 101)]
        testing = [f"patient{i:03d}" for i in range(1, 21)]
        validation = [f"patient{i:03d}" for i in range(1, 21)]
        training = [case for case in all_cases if case not in testing + validation]
        return [training, validation, testing]

    @staticmethod
    def _normalize_spacing_zyx(
        spacing: Any,
        key: str,
        case: str,
    ) -> np.ndarray:
        """Normalize supported source metadata to positive ``(z, y, x)`` spacing."""

        normalized = np.asarray(spacing, dtype=np.float32).reshape(-1)
        if key == "pixdim" and normalized.size >= 4:
            normalized = normalized[1:4]
        else:
            normalized = normalized[:3]
        if normalized.size != 3:
            raise ValueError(
                f"ACDC case {case} spacing key {key} must contain 3 spatial values"
            )
        if key not in SPACING_ZYX_KEYS:
            normalized = normalized[::-1]
        if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0):
            raise ValueError(
                f"ACDC case {case} spacing key {key} has invalid values "
                f"{normalized.tolist()}"
            )
        return normalized.astype(np.float32)

    def _spacing_from_h5(self, h5f: h5py.File, case: str) -> np.ndarray | None:
        for key in SPACING_KEYS:
            if key in h5f.attrs:
                return self._normalize_spacing_zyx(h5f.attrs[key], key, case)
            if key in h5f:
                return self._normalize_spacing_zyx(h5f[key][()], key, case)
            for dataset_key in ("image", "label"):
                if dataset_key in h5f and key in h5f[dataset_key].attrs:
                    return self._normalize_spacing_zyx(
                        h5f[dataset_key].attrs[key], key, case
                    )
        return None

    def _nifti_spacing_zyx(self, path: Path, case: str) -> np.ndarray | None:
        try:
            import nibabel as nib
        except ImportError:
            nib = None
        if nib is not None:
            spacing = nib.load(str(path)).header.get_zooms()[:3]
            return self._normalize_spacing_zyx(spacing, "zooms", case)

        try:
            import SimpleITK as sitk
        except ImportError:
            return None
        spacing = sitk.ReadImage(str(path)).GetSpacing()[:3]
        return self._normalize_spacing_zyx(spacing, "spacing", case)

    def _original_nifti_candidates(self, case: str) -> list[Path]:
        stem = case[:-3] if case.endswith(".h5") else case
        patient_id = stem.split("_")[0]
        directories = [
            self._base_dir,
            self._base_dir / "ACDC_training_volumes",
            self._base_dir / patient_id,
            self._base_dir / "training" / patient_id,
            self._base_dir / "database" / "training" / patient_id,
            self._base_dir / "ACDC_training" / patient_id,
        ]
        return [
            directory / f"{stem}{extension}"
            for directory in directories
            for extension in (".nii.gz", ".nii")
        ]

    def _volume_spacing_zyx(self, h5f: h5py.File, case: str) -> np.ndarray | None:
        spacing = self._spacing_from_h5(h5f, case)
        if spacing is not None:
            return spacing
        for path in self._original_nifti_candidates(case):
            if path.is_file():
                spacing = self._nifti_spacing_zyx(path, case)
                if spacing is not None:
                    return spacing
        return None

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case = self.sample_list[idx]
        data_path = self._sample_dir / case
        if not data_path.is_file():
            raise FileNotFoundError(f"ACDC sample is missing: {data_path}")

        with h5py.File(data_path, "r") as h5f:
            image = h5f["image"][:]
            label = h5f["label"][:]
            spacing = (
                None
                if self._split_kind == "train"
                else self._volume_spacing_zyx(h5f, case)
            )

        sample: dict[str, Any] = {"image": image, "label": label}
        if self._split_kind == "train" and self.transform is not None:
            sample = self.transform(sample)
        if spacing is not None:
            sample["voxelspacing_zyx"] = spacing
        sample["idx"] = idx
        sample["case_name"] = case[:-3] if case.endswith(".h5") else case
        return sample


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
    # Preserve the supplied ACDC interpolation choices (nearest for both).
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator4ACDC:
    """Reference ACDC training augmentation and nearest-neighbor resize."""

    def __init__(self, output_size: Sequence[int]):
        if len(output_size) != 2 or any(int(value) <= 0 for value in output_size):
            raise ValueError(f"output_size must contain two positive values, got {output_size!r}")
        self.output_size = (int(output_size[0]), int(output_size[1]))

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        image = np.asarray(sample["image"])
        label = np.asarray(sample["label"])
        if image.ndim != 2 or label.ndim != 2:
            raise ValueError(
                "ACDC training samples must be 2D image/label pairs; "
                f"got image {image.shape} and label {label.shape}"
            )
        if image.shape != label.shape:
            raise ValueError(
                f"ACDC image and label shapes differ: {image.shape} versus {label.shape}"
            )

        augment = random.random()
        if augment > 0.6:
            image, label = random_rot_flip(image, label)
        elif augment > 0.35:
            image, label = random_rotate(image, label)

        height, width = image.shape
        if (height, width) != self.output_size:
            factors = (self.output_size[0] / height, self.output_size[1] / width)
            image = zoom(image, factors, order=0)
            label = zoom(label, factors, order=0)

        if image.shape != self.output_size or label.shape != self.output_size:
            raise RuntimeError(
                f"ACDC transform produced image {image.shape} and label {label.shape}; "
                f"expected {self.output_size}"
            )
        return {
            "image": torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0),
            "label": torch.from_numpy(np.asarray(label, dtype=np.uint8)),
        }


def iterate_once(iterable: Any) -> np.ndarray:
    return np.random.permutation(iterable)


def iterate_eternally(indices: Any):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable: Any, n: int):
    """Collect data into fixed-size groups, as in the supplied implementation."""

    args = [iter(iterable)] * n
    return zip(*args)


# Conventional spelling for new callers without breaking the supplied API.
ACDCDataset = ACDC_Dataset


__all__ = [
    "ACDCDataset",
    "ACDC_Dataset",
    "RandomGenerator4ACDC",
    "SPACING_KEYS",
    "SPACING_ZYX_KEYS",
    "grouper",
    "iterate_eternally",
    "iterate_once",
    "random_rot_flip",
    "random_rotate",
]
