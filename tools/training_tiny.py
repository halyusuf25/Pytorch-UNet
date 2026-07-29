#!/usr/bin/env python3
"""Overfit U-Net on 8--16 deterministic training slices/frames.

Run from the repository root, for example:

    python tools/training_tiny.py --dataset Synapse \
        --output-dir outputs/tiny/synapse
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.tiny_common import (
    OVERFIT_TARGET,
    HardDiceAccumulator,
    ResizeOnlyTransform,
    TinySubsetDataset,
    build_raw_training_dataset,
    build_subset_document,
    format_metrics,
    select_tiny_records,
    validate_num_samples,
    write_json,
)
from unet import UNet
from utils.checkpointing import save_checkpoint
from utils.dataset_registry import (
    available_dataset_names,
    get_dataset_spec,
    resolve_dataset_paths,
)
from utils.model_output import _extract_logits
from utils.runtime import (
    dataloader_generator,
    resolve_device,
    seed_everything,
    seed_worker,
)


def classwise_soft_dice_loss(
    logits,
    target,
    include_background=False,
    smooth=1e-5,
):
    """Classwise soft Dice; unlike flattened Dice, classes cannot dominate each other."""

    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(
        target.long(),
        num_classes=logits.shape[1],
    ).permute(0, 3, 1, 2).to(probabilities.dtype)

    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = (
        probabilities.square().sum(dims)
        + one_hot.square().sum(dims)
    )

    dice = (2 * intersection + smooth) / (denominator + smooth)

    if not include_background:
        dice = dice[1:]

    return 1.0 - dice.mean()


def _num_samples_argument(value: str) -> int:
    try:
        return validate_num_samples(int(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {value}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overfit the existing U-Net on a deterministic 8--16 sample "
            "foreground-containing training subset. This is a slice/frame diagnostic, "
            "not the official volume-level benchmark."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=available_dataset_names(),
        help="Medical training dataset to diagnose.",
    )
    parser.add_argument(
        "--num-samples",
        type=_num_samples_argument,
        default=12,
        help="Number of foreground-containing training slices/frames.",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=300,
        help="Number of complete tiny-subset training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help="Training/evaluation batch size; defaults to the entire tiny subset.",
    )
    parser.add_argument(
        "--img-size",
        type=_positive_int,
        default=224,
        help="Square resize used for both images and labels (minimum 16).",
    )
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=3e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (must be non-negative).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Reproducibility seed.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for best_model.pth, last_model.pth, and tiny_subset.json.",
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Override the registry's training-data root.",
    )
    parser.add_argument(
        "--list-dir",
        default=None,
        help="Override the Synapse split-list directory.",
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
        help="DataLoader workers; zero is simplest for a tiny deterministic subset.",
    )
    parser.add_argument(
        "--bilinear",
        action="store_true",
        help="Use bilinear U-Net upsampling instead of transposed convolution.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=OVERFIT_TARGET,
        help="Foreground mean hard-Dice target used for the pass/fail message.",
    )
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.img_size < 16:
        raise ValueError(
            f"--img-size must be at least 16 for the four-level U-Net, got {args.img_size}"
        )
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ValueError("--weight-decay must be a non-negative finite number")
    if not 0 < args.success_threshold <= 1:
        raise ValueError("--success-threshold must be in (0, 1]")
    if args.batch_size is None:
        args.batch_size = args.num_samples


def _validate_batch(
    image: torch.Tensor,
    target: torch.Tensor,
    *,
    channels: int,
    classes: int,
) -> None:
    if image.ndim != 4 or image.shape[1] != channels:
        raise ValueError(
            f"Expected BCHW input with {channels} channels, got {tuple(image.shape)}"
        )
    if target.ndim != 3 or target.shape != image.shape[:1] + image.shape[2:]:
        raise ValueError(
            f"Expected BHW labels matching images, got image {tuple(image.shape)} "
            f"and label {tuple(target.shape)}"
        )
    minimum = int(target.min().item())
    maximum = int(target.max().item())
    if minimum < 0 or maximum >= classes:
        raise ValueError(
            f"Batch label IDs [{minimum}, {maximum}] are outside [0, {classes - 1}]"
        )


def _evaluate(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    totals = {"total_loss": 0.0, "cross_entropy": 0.0, "dice_loss": 0.0}
    example_count = 0
    hard_metrics = HardDiceAccumulator.create(num_classes)

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            target = batch["label"].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            _validate_batch(
                image,
                target,
                channels=model.n_channels,
                classes=num_classes,
            )
            logits = _extract_logits(model(image))
            if logits.shape != (image.shape[0], num_classes, *target.shape[-2:]):
                raise ValueError(
                    f"U-Net returned logits {tuple(logits.shape)} for image "
                    f"{tuple(image.shape)} and {num_classes} classes"
                )
            cross_entropy = F.cross_entropy(logits, target)
            dice_loss = classwise_soft_dice_loss(logits, target)
            total_loss = 0.5 * cross_entropy + 0.5 * dice_loss
            count = int(image.shape[0])
            totals["total_loss"] += float(total_loss.item()) * count
            totals["cross_entropy"] += float(cross_entropy.item()) * count
            totals["dice_loss"] += float(dice_loss.item()) * count
            example_count += count
            hard_metrics.update(logits.argmax(dim=1), target)

    if example_count == 0:
        raise RuntimeError("Tiny-subset evaluation DataLoader returned no samples")
    return {
        **{key: value / example_count for key, value in totals.items()},
        **hard_metrics.result(),
    }


def _checkpoint_fields(
    *,
    args: argparse.Namespace,
    spec,
    optimizer,
    epoch: int,
    best_score: float,
    subset_json: Path,
    records,
) -> dict[str, Any]:
    return {
        "optimizer": optimizer,
        "epoch": epoch,
        "best_mean_dice": best_score,
        "dataset": spec.name,
        "n_channels": spec.input_channels,
        "n_classes": spec.num_classes,
        "bilinear": args.bilinear,
        "img_size": args.img_size,
        "class_names": spec.class_names,
        "normalization": "ImageNet" if spec.imagenet_normalization else None,
        "arguments": args,
        "extra": {
            "tiny_subset_file": str(subset_json),
            "tiny_subset_indices": [record.index for record in records],
            "diagnostic": "same_training_slice_or_frame_subset",
        },
    }


def run(args: argparse.Namespace) -> float:
    _validate_arguments(args)
    seed_everything(args.seed, deterministic=True)
    spec = get_dataset_spec(args.dataset)
    paths = resolve_dataset_paths(
        spec,
        root=args.root_path,
        list_dir=args.list_dir,
    )
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {spec.name}")
    print(f"Device: {device}")
    print(f"Resolved training root: {paths.train_root}")
    if paths.list_dir is not None:
        print(f"Resolved list directory: {paths.list_dir}")
    print(
        f"Scanning the training split for {args.num_samples} foreground-containing "
        "samples at the requested resize..."
    )

    raw_dataset = build_raw_training_dataset(spec, paths)
    transform = ResizeOnlyTransform(
        args.img_size,
        rgb_imagenet=spec.imagenet_normalization,
    )
    records = select_tiny_records(
        raw_dataset,
        transform,
        num_samples=args.num_samples,
        num_classes=spec.num_classes,
        seed=args.seed,
    )
    subset = TinySubsetDataset(raw_dataset, records, transform, spec.num_classes)

    subset_json = output_dir / "tiny_subset.json"
    document = build_subset_document(
        spec=spec,
        paths=paths,
        records=records,
        img_size=args.img_size,
        seed=args.seed,
        arguments=vars(args),
        success_threshold=args.success_threshold,
    )
    write_json(subset_json, document)
    print(f"Selected indices: {[record.index for record in records]}")
    for record in records:
        names = [spec.class_names[value] for value in record.foreground_classes]
        print(
            f"  index={record.index}, case={record.case_name}, "
            f"foreground={names}"
        )
    print(f"Wrote reproducible subset manifest: {subset_json}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker if args.num_workers else None,
    }
    train_loader = DataLoader(
        subset,
        shuffle=True,
        generator=dataloader_generator(args.seed),
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        subset,
        shuffle=False,
        generator=dataloader_generator(args.seed + 1),
        **loader_kwargs,
    )

    model = UNet(
        n_channels=spec.input_channels,
        n_classes=spec.num_classes,
        bilinear=args.bilinear,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_score = -math.inf
    reached_target = False
    best_path = output_dir / "best_model.pth"
    last_path = output_dir / "last_model.pth"

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            image = batch["image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            target = batch["label"].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            _validate_batch(
                image,
                target,
                channels=spec.input_channels,
                classes=spec.num_classes,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = _extract_logits(model(image))
            cross_entropy = F.cross_entropy(logits, target)
            dice_loss = classwise_soft_dice_loss(logits, target)
            loss = 0.5 * cross_entropy + 0.5 * dice_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch + 1}: {loss.item()}"
                )
            loss.backward()
            optimizer.step()

        metrics = _evaluate(
            model,
            validation_loader,
            device,
            num_classes=spec.num_classes,
        )
        score = metrics["foreground_mean_dice"]
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"\nEpoch {epoch + 1:04d}/{args.epochs:04d} | "
            f"total_loss={metrics['total_loss']:.6f} | "
            f"cross_entropy={metrics['cross_entropy']:.6f} | "
            f"classwise_dice_loss={metrics['dice_loss']:.6f} | "
            f"foreground_mean_hard_dice={score:.6f} | "
            f"lr={learning_rate:.8g}"
        )
        print(format_metrics(metrics, spec.class_names))

        if score > best_score:
            best_score = score
            save_checkpoint(
                best_path,
                model,
                **_checkpoint_fields(
                    args=args,
                    spec=spec,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_score=best_score,
                    subset_json=subset_json,
                    records=records,
                ),
            )
            print(f"Saved new best checkpoint: {best_path}")

        save_checkpoint(
            last_path,
            model,
            **_checkpoint_fields(
                args=args,
                spec=spec,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_score,
                subset_json=subset_json,
                records=records,
            ),
        )

        if score >= args.success_threshold and not reached_target:
            reached_target = True
            print(
                f"SUCCESS: tiny-subset training Dice reached {score:.4f}, "
                f"meeting the {args.success_threshold:.2f} overfitting target."
            )

    print(f"\nBest foreground mean hard Dice: {best_score:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    if best_score >= args.success_threshold:
        print(
            "SUCCESS: U-Net passed the tiny-subset overfitting test "
            f"({best_score:.4f} >= {args.success_threshold:.2f})."
        )
    else:
        print(
            "WARNING: U-Net failed the tiny-subset overfitting test "
            f"({best_score:.4f} < {args.success_threshold:.2f})."
        )
    return best_score


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
