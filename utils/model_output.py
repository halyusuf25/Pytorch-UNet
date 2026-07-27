"""Small model I/O validation helpers shared by training and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


_LOGITS_KEYS = ("logits", "out", "output", "prediction", "pred")


def _extract_logits(output: Any) -> torch.Tensor:
    """Return a logits tensor from the common model-output containers.

    ``UNet`` returns a tensor directly.  Supporting sequences and mappings keeps
    evaluation and benchmarking usable with wrappers such as DataParallel and
    lightweight inference adapters without changing the U-Net output contract.
    """

    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model returned an empty tuple/list; logits are unavailable")
        return _extract_logits(output[0])
    if isinstance(output, dict):
        for key in _LOGITS_KEYS:
            if key in output:
                return _extract_logits(output[key])
        raise KeyError(
            "Model output dictionary has no logits key; expected one of "
            + ", ".join(_LOGITS_KEYS)
        )
    raise TypeError(
        "Unsupported model output type {}; expected a tensor, tuple/list, or dict".format(
            type(output).__name__
        )
    )


def extract_target(batch: dict[str, Any]) -> Any:
    """Read a medical ``label`` or legacy Carvana ``mask`` from a batch."""

    if "label" in batch:
        return batch["label"]
    if "mask" in batch:
        return batch["mask"]
    raise KeyError("Batch must contain either a 'label' or 'mask' key")


def validate_model_input(images: torch.Tensor, expected_channels: int) -> None:
    if not torch.is_tensor(images):
        raise TypeError(f"Model input must be a tensor, got {type(images).__name__}")
    if images.ndim != 4:
        raise ValueError(f"Model input must have shape B,C,H,W, got {tuple(images.shape)}")
    if int(images.shape[1]) != int(expected_channels):
        raise ValueError(
            f"Model expects {expected_channels} input channels, got {int(images.shape[1])}; "
            "check the selected dataset and preprocessing"
        )


def validate_labels(labels: Any, num_classes: int, *, context: str = "batch") -> None:
    """Fail early when segmentation labels are fractional or out of range."""

    if torch.is_tensor(labels):
        if labels.numel() == 0:
            raise ValueError(f"{context} contains an empty label tensor")
        if labels.is_floating_point() and not torch.equal(labels, labels.round()):
            raise ValueError(f"{context} labels must be integer-valued")
        label_min = int(labels.min().item())
        label_max = int(labels.max().item())
    else:
        array = np.asarray(labels)
        if array.size == 0:
            raise ValueError(f"{context} contains an empty label array")
        if np.issubdtype(array.dtype, np.floating) and not np.array_equal(array, np.rint(array)):
            raise ValueError(f"{context} labels must be integer-valued")
        label_min = int(array.min())
        label_max = int(array.max())
    if label_min < 0 or label_max >= int(num_classes):
        raise ValueError(
            f"{context} labels must be in [0, {num_classes - 1}], "
            f"observed [{label_min}, {label_max}]"
        )
