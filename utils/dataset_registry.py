"""Central medical-dataset metadata, path resolution, and construction.

Registry lookup is deliberately side-effect free: default paths are not touched
until a builder (or :func:`validate_dataset_paths`) is called.  This keeps CLI
help and configuration inspection usable on machines without the datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, Union

from torch.utils.data import Dataset

from datasets import (
    ACDC_Dataset,
    Cataract1kDataset,
    RandomGenerator,
    RandomGenerator4ACDC,
    RandomGenerator4Cataract,
    Synapse_dataset,
)


Purpose: TypeAlias = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable model/data contract for one supported medical dataset."""

    name: str
    aliases: tuple[str, ...]
    train_root: str
    volume_root: str
    list_dir: str | None
    dataset_class: type[Dataset]
    training_transform_class: type
    num_classes: int
    input_channels: int
    class_names: tuple[str, ...]
    protocol: str
    imagenet_normalization: bool

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError(f"{self.name}: num_classes must be greater than one")
        if self.input_channels <= 0:
            raise ValueError(f"{self.name}: input_channels must be positive")
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"{self.name}: got {len(self.class_names)} class names for "
                f"{self.num_classes} classes"
            )
        if self.protocol not in {"volume", "present_class_frame"}:
            raise ValueError(f"{self.name}: unsupported test protocol {self.protocol!r}")

    @property
    def canonical_name(self) -> str:
        return self.name

    @property
    def validation_root(self) -> str:
        return self.volume_root

    @property
    def test_root(self) -> str:
        return self.volume_root

    @property
    def train_transform_class(self) -> type:
        return self.training_transform_class

    @property
    def validation_protocol(self) -> str:
        return self.protocol

    @property
    def test_protocol(self) -> str:
        return self.protocol

    @property
    def uses_imagenet_normalization(self) -> bool:
        return self.imagenet_normalization

    @property
    def normalization(self) -> str | None:
        return "ImageNet" if self.imagenet_normalization else None


DatasetNameOrSpec: TypeAlias = Union[str, DatasetSpec]


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved paths suitable for logging and passing to dataset builders."""

    canonical_name: str
    train_root: Path
    volume_root: Path
    list_dir: Path | None

    @property
    def data_root(self) -> Path:
        return self.train_root

    @property
    def validation_root(self) -> Path:
        return self.volume_root

    @property
    def test_root(self) -> Path:
        return self.volume_root

    def as_dict(self) -> dict[str, str | None]:
        return {
            "dataset": self.canonical_name,
            "train_root": str(self.train_root),
            "volume_root": str(self.volume_root),
            "validation_root": str(self.volume_root),
            "test_root": str(self.volume_root),
            "list_dir": None if self.list_dir is None else str(self.list_dir),
        }


SYNAPSE_SPEC = DatasetSpec(
    name="Synapse",
    aliases=("synapse",),
    train_root="/data/halyusuf/data/Synapse/train_npz/",
    volume_root="/data/halyusuf/data/Synapse/test_vol_h5",
    list_dir="./lists/lists_Synapse",
    dataset_class=Synapse_dataset,
    training_transform_class=RandomGenerator,
    num_classes=9,
    input_channels=1,
    class_names=(
        "Background",
        "Aorta",
        "Gallbladder",
        "Kidney(L)",
        "Kidney(R)",
        "Liver",
        "Pancreas",
        "Spleen",
        "Stomach",
    ),
    protocol="volume",
    imagenet_normalization=False,
)

ACDC_SPEC = DatasetSpec(
    name="ACDC",
    aliases=("acdc",),
    train_root="/data/halyusuf/data/ACDC",
    volume_root="/data/halyusuf/data/ACDC",
    list_dir=None,
    dataset_class=ACDC_Dataset,
    training_transform_class=RandomGenerator4ACDC,
    num_classes=4,
    input_channels=1,
    class_names=(
        "Background",
        "Right Ventricle",
        "Myocardium",
        "Left Ventricle",
    ),
    protocol="volume",
    imagenet_normalization=False,
)

CATARACT1K_SPEC = DatasetSpec(
    name="Cataract1k",
    aliases=("cataract1k", "Catrakt1k", "catrakt1k"),
    train_root="/data/halyusuf/data/CataractData/",
    volume_root="/data/halyusuf/data/CataractData/",
    list_dir=None,
    dataset_class=Cataract1kDataset,
    training_transform_class=RandomGenerator4Cataract,
    num_classes=5,
    input_channels=3,
    class_names=(
        "Background",
        "Pupil",
        "Cornea",
        "Lens",
        "Instruments",
    ),
    protocol="present_class_frame",
    imagenet_normalization=True,
)


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    SYNAPSE_SPEC,
    ACDC_SPEC,
    CATARACT1K_SPEC,
)
DATASET_REGISTRY: dict[str, DatasetSpec] = {spec.name: spec for spec in DATASET_SPECS}


def _name_key(value: str) -> str:
    return "".join(character for character in value.strip().casefold() if character.isalnum())


_ALIASES: dict[str, str] = {}
for _spec in DATASET_SPECS:
    for _candidate in (_spec.name, *_spec.aliases):
        _key = _name_key(_candidate)
        _existing = _ALIASES.get(_key)
        if _existing is not None and _existing != _spec.name:
            raise RuntimeError(
                f"Dataset alias {_candidate!r} is shared by {_existing!r} and {_spec.name!r}"
            )
        _ALIASES[_key] = _spec.name

# Public read-only-by-convention copy useful for CLI diagnostics.
DATASET_ALIASES: dict[str, str] = dict(_ALIASES)


def canonicalize_dataset_name(name: str) -> str:
    """Return a canonical dataset name, including the ``Catrakt1k`` alias."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Dataset name must be a non-empty string")
    key = _name_key(name)
    try:
        return _ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(DATASET_REGISTRY)
        raise ValueError(f"Unknown dataset {name!r}. Supported datasets: {choices}") from exc


# Readable alias used by some entry points.
canonical_dataset_name = canonicalize_dataset_name


def get_dataset_spec(dataset: DatasetNameOrSpec) -> DatasetSpec:
    """Look up metadata without checking any dataset paths."""

    if isinstance(dataset, DatasetSpec):
        canonical = canonicalize_dataset_name(dataset.name)
        registered = DATASET_REGISTRY[canonical]
        if dataset != registered:
            raise ValueError(
                f"DatasetSpec named {dataset.name!r} differs from the registered specification"
            )
        return registered
    return DATASET_REGISTRY[canonicalize_dataset_name(dataset)]


def available_dataset_names(*, include_aliases: bool = False) -> tuple[str, ...]:
    """Return stable names for CLI help; no paths are accessed."""

    canonical = tuple(DATASET_REGISTRY)
    if not include_aliases:
        return canonical
    aliases = tuple(
        alias
        for spec in DATASET_SPECS
        for alias in spec.aliases
        if alias.casefold() != spec.name.casefold()
    )
    return canonical + aliases


dataset_choices = available_dataset_names


def is_medical_dataset(name: str) -> bool:
    try:
        canonicalize_dataset_name(name)
    except (TypeError, ValueError):
        return False
    return True


def _resolved_path(value: str | Path, *, option: str) -> Path:
    if not str(value).strip():
        raise ValueError(f"{option} must not be empty")
    return Path(value).expanduser().resolve(strict=False)


def resolve_dataset_paths(
    dataset: DatasetNameOrSpec,
    *,
    data_root: str | Path | None = None,
    root: str | Path | None = None,
    train_root: str | Path | None = None,
    volume_root: str | Path | None = None,
    validation_root: str | Path | None = None,
    test_root: str | Path | None = None,
    list_dir: str | Path | None = None,
) -> DatasetPaths:
    """Apply CLI-style path overrides without touching the filesystem.

    ``train_root`` has priority over ``data_root``/``root``.  A common-root
    override is also applied to validation/test only for datasets whose default
    train and volume roots are the same (ACDC and Cataract1k).  Synapse keeps its
    distinct volume root unless a volume/validation/test override is supplied.
    """

    spec = get_dataset_spec(dataset)
    common_root = data_root if data_root is not None else root
    selected_train = train_root
    if selected_train is None:
        selected_train = common_root if common_root is not None else spec.train_root

    selected_volume = next(
        (
            candidate
            for candidate in (test_root, validation_root, volume_root)
            if candidate is not None
        ),
        None,
    )
    if selected_volume is None:
        if common_root is not None and Path(spec.train_root) == Path(spec.volume_root):
            selected_volume = common_root
        else:
            selected_volume = spec.volume_root

    selected_list = spec.list_dir if list_dir is None else list_dir
    return DatasetPaths(
        canonical_name=spec.name,
        train_root=_resolved_path(selected_train, option="train_root"),
        volume_root=_resolved_path(selected_volume, option="volume_root"),
        list_dir=(
            None
            if selected_list is None
            else _resolved_path(selected_list, option="list_dir")
        ),
    )


def _require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist or is not a directory: {path}")


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")


def validate_dataset_paths(
    dataset: DatasetNameOrSpec,
    paths: DatasetPaths,
    purpose: Purpose,
) -> None:
    """Fail before DataLoader startup when required roots/split files are absent."""

    spec = get_dataset_spec(dataset)
    if paths.canonical_name != spec.name:
        raise ValueError(
            f"Resolved paths are for {paths.canonical_name}, not requested dataset {spec.name}"
        )
    if purpose not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported dataset purpose {purpose!r}")

    if spec.name == "Synapse":
        if paths.list_dir is None:
            raise ValueError("Synapse requires a list_dir")
        _require_directory(paths.list_dir, "Synapse list directory")
        split_name = "train" if purpose == "train" else "test_vol"
        _require_file(
            paths.list_dir / f"{split_name}.txt",
            f"Synapse {split_name} split file",
        )
        selected_root = paths.train_root if purpose == "train" else paths.volume_root
        _require_directory(selected_root, f"Synapse {purpose} data root")
        return

    if spec.name == "ACDC":
        selected_root = paths.train_root if purpose == "train" else paths.volume_root
        subdirectory = (
            "ACDC_training_slices" if purpose == "train" else "ACDC_training_volumes"
        )
        _require_directory(selected_root, f"ACDC {purpose} root")
        _require_directory(selected_root / subdirectory, f"ACDC {purpose} data directory")
        return

    selected_root = paths.train_root if purpose == "train" else paths.volume_root
    _require_directory(selected_root, f"Cataract1k {purpose} root")
    _require_directory(selected_root / "img", "Cataract1k image directory")
    _require_directory(selected_root / "ann", "Cataract1k annotation directory")
    csv_name = "train.csv" if purpose == "train" else "test.csv"
    _require_file(selected_root / csv_name, f"Cataract1k {purpose} split CSV")


def _validate_img_size(img_size: int) -> int:
    size = int(img_size)
    if size <= 0:
        raise ValueError(f"img_size must be positive, got {img_size!r}")
    return size


def _paths_for_builder(
    dataset: DatasetNameOrSpec,
    paths: DatasetPaths | None,
    path_overrides: dict[str, Any],
) -> tuple[DatasetSpec, DatasetPaths]:
    spec = get_dataset_spec(dataset)
    supplied_overrides = {key: value for key, value in path_overrides.items() if value is not None}
    if paths is not None and supplied_overrides:
        names = ", ".join(sorted(supplied_overrides))
        raise ValueError(f"Pass either paths or path overrides, not both (got: {names})")
    resolved = paths if paths is not None else resolve_dataset_paths(spec, **path_overrides)
    if resolved.canonical_name != spec.name:
        raise ValueError(f"DatasetPaths for {resolved.canonical_name} cannot build {spec.name}")
    return spec, resolved


def build_train_dataset(
    dataset: DatasetNameOrSpec,
    img_size: int,
    *,
    paths: DatasetPaths | None = None,
    data_root: str | Path | None = None,
    root: str | Path | None = None,
    train_root: str | Path | None = None,
    volume_root: str | Path | None = None,
    list_dir: str | Path | None = None,
    fold_id: int | None = None,
) -> Dataset:
    """Build the explicit medical training split (never a random split)."""

    size = _validate_img_size(img_size)
    spec, resolved = _paths_for_builder(
        dataset,
        paths,
        {
            "data_root": data_root,
            "root": root,
            "train_root": train_root,
            "volume_root": volume_root,
            "list_dir": list_dir,
        },
    )
    validate_dataset_paths(spec, resolved, "train")

    if spec.name == "Synapse":
        return Synapse_dataset(
            base_dir=resolved.train_root,
            list_dir=resolved.list_dir,
            split="train",
            transform=RandomGenerator([size, size]),
        )
    if spec.name == "ACDC":
        kwargs: dict[str, Any] = {}
        if fold_id is not None:
            kwargs["fold_id"] = fold_id
        return ACDC_Dataset(
            base_dir=resolved.train_root,
            split="train",
            transform=RandomGenerator4ACDC([size, size]),
            **kwargs,
        )
    if fold_id not in (None, 0):
        raise ValueError("fold_id is only accepted for ACDC and is ignored by its active split")
    return Cataract1kDataset(
        base_dir=resolved.train_root,
        split="train",
        transform=RandomGenerator4Cataract(
            [size, size],
            augment=True,
            normalize=True,
        ),
    )


def build_validation_dataset(
    dataset: DatasetNameOrSpec,
    img_size: int,
    *,
    paths: DatasetPaths | None = None,
    data_root: str | Path | None = None,
    root: str | Path | None = None,
    train_root: str | Path | None = None,
    volume_root: str | Path | None = None,
    validation_root: str | Path | None = None,
    list_dir: str | Path | None = None,
    fold_id: int | None = None,
) -> Dataset:
    """Build the supplied volume or deterministic frame validation split."""

    size = _validate_img_size(img_size)
    spec, resolved = _paths_for_builder(
        dataset,
        paths,
        {
            "data_root": data_root,
            "root": root,
            "train_root": train_root,
            "volume_root": volume_root,
            "validation_root": validation_root,
            "list_dir": list_dir,
        },
    )
    validate_dataset_paths(spec, resolved, "validation")

    if spec.name == "Synapse":
        return Synapse_dataset(
            base_dir=resolved.volume_root,
            list_dir=resolved.list_dir,
            split="test_vol",
        )
    if spec.name == "ACDC":
        kwargs: dict[str, Any] = {}
        if fold_id is not None:
            kwargs["fold_id"] = fold_id
        # The supplied active val/test patient IDs are identical.  Use the
        # prompt's consistent spelling for volume validation and testing.
        return ACDC_Dataset(base_dir=resolved.volume_root, split="test", **kwargs)
    if fold_id not in (None, 0):
        raise ValueError("fold_id is only accepted for ACDC and is ignored by its active split")
    return Cataract1kDataset(
        base_dir=resolved.volume_root,
        split="test",
        transform=RandomGenerator4Cataract(
            [size, size],
            augment=False,
            normalize=True,
        ),
    )


def build_test_dataset(
    dataset: DatasetNameOrSpec,
    img_size: int,
    *,
    paths: DatasetPaths | None = None,
    data_root: str | Path | None = None,
    root: str | Path | None = None,
    train_root: str | Path | None = None,
    volume_root: str | Path | None = None,
    test_root: str | Path | None = None,
    list_dir: str | Path | None = None,
    fold_id: int | None = None,
) -> Dataset:
    """Build the accuracy test split; Cataract1k remains raw HWC RGB."""

    _validate_img_size(img_size)  # validates the shared CLI contract
    spec, resolved = _paths_for_builder(
        dataset,
        paths,
        {
            "data_root": data_root,
            "root": root,
            "train_root": train_root,
            "volume_root": volume_root,
            "test_root": test_root,
            "list_dir": list_dir,
        },
    )
    validate_dataset_paths(spec, resolved, "test")

    if spec.name == "Synapse":
        return Synapse_dataset(
            base_dir=resolved.volume_root,
            list_dir=resolved.list_dir,
            split="test_vol",
        )
    if spec.name == "ACDC":
        kwargs: dict[str, Any] = {}
        if fold_id is not None:
            kwargs["fold_id"] = fold_id
        return ACDC_Dataset(base_dir=resolved.volume_root, split="test", **kwargs)
    if fold_id not in (None, 0):
        raise ValueError("fold_id is only accepted for ACDC and is ignored by its active split")
    return Cataract1kDataset(base_dir=resolved.volume_root, split="test", transform=None)


def build_dataset(
    dataset: DatasetNameOrSpec,
    split: str,
    img_size: int,
    **kwargs: Any,
) -> Dataset:
    """Generic dispatcher for callers that select a split dynamically."""

    normalized = split.strip().casefold()
    if normalized in {"train", "training"}:
        return build_train_dataset(dataset, img_size, **kwargs)
    if normalized in {"val", "valid", "validation"}:
        return build_validation_dataset(dataset, img_size, **kwargs)
    if normalized in {"test", "testing", "test_vol"}:
        return build_test_dataset(dataset, img_size, **kwargs)
    raise ValueError(f"Unsupported dataset split {split!r}; expected train, validation, or test")


__all__ = [
    "ACDC_SPEC",
    "CATARACT1K_SPEC",
    "DATASET_ALIASES",
    "DATASET_REGISTRY",
    "DATASET_SPECS",
    "DatasetPaths",
    "DatasetSpec",
    "SYNAPSE_SPEC",
    "available_dataset_names",
    "build_dataset",
    "build_test_dataset",
    "build_train_dataset",
    "build_validation_dataset",
    "canonical_dataset_name",
    "canonicalize_dataset_name",
    "dataset_choices",
    "get_dataset_spec",
    "is_medical_dataset",
    "resolve_dataset_paths",
    "validate_dataset_paths",
]
