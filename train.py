"""Train the repository U-Net on Carvana or the supported medical datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.checkpointing import load_checkpoint, save_checkpoint
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dataset_registry import (
    build_train_dataset,
    build_validation_dataset,
    canonicalize_dataset_name,
    get_dataset_spec,
    resolve_dataset_paths,
)
from utils.dice_score import dice_loss
from utils.medical_inference import (
    evaluate_cataract_loader,
    evaluate_volume_loader,
    foreground_validation_dice,
)
from utils.medical_metrics import fallback_acdc_voxelspacing_zyx
from utils.model_output import _extract_logits, extract_target, validate_labels, validate_model_input
from utils.runtime import (
    capture_rng_state,
    dataloader_generator,
    default_num_workers,
    resolve_device,
    restore_rng_state,
    seed_everything,
    seed_worker,
)


DEFAULT_IMAGE_DIR = Path("./data/imgs/")
DEFAULT_MASK_DIR = Path("./data/masks/")
MEDICAL_DATASETS = ("Synapse", "ACDC", "Cataract1k", "Catrakt1k")


class _NoOpExperiment:
    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None


def _start_wandb(args: argparse.Namespace) -> Any:
    if not args.wandb:
        return _NoOpExperiment()
    try:
        import wandb
    except Exception as exc:  # importing a broken optional install should be actionable
        raise RuntimeError("--wandb was requested, but Weights & Biases is unavailable") from exc
    return wandb.init(
        project=args.wandb_project,
        config=vars(args),
        notes=args.description,
    )


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _make_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    kwargs = dict(
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    try:
        return optim.RMSprop(model.parameters(), foreach=True, **kwargs)
    except TypeError:  # foreach is unavailable on older supported PyTorch builds
        return optim.RMSprop(model.parameters(), **kwargs)


def _loader_kwargs(
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    return {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": dataloader_generator(args.seed + seed_offset),
        # Recreate workers each epoch so restoring the loader generator also
        # restores worker seeds during strict resume.
        "persistent_workers": False,
    }


def _build_medical_data(args: argparse.Namespace, device: torch.device):
    canonical_name = canonicalize_dataset_name(args.dataset)
    spec = get_dataset_spec(canonical_name)
    if args.classes is not None and args.classes != spec.num_classes:
        raise ValueError(
            f"{canonical_name} has {spec.num_classes} registry classes; "
            f"--classes {args.classes} is incompatible"
        )
    paths = resolve_dataset_paths(
        spec,
        root=args.root_path,
        volume_root=args.volume_path,
        list_dir=args.list_dir,
    )
    train_set = build_train_dataset(
        spec,
        args.img_size,
        paths=paths,
        fold_id=args.fold_id if canonical_name == "ACDC" else None,
    )
    validation_set = build_validation_dataset(
        spec,
        args.img_size,
        paths=paths,
        fold_id=args.fold_id if canonical_name == "ACDC" else None,
    )
    if len(train_set) == 0:
        raise RuntimeError(f"{canonical_name} training split is empty")
    if len(validation_set) == 0:
        raise RuntimeError(f"{canonical_name} validation split is empty")

    train_loader_args = _loader_kwargs(args, device, seed_offset=0)
    validation_loader_args = _loader_kwargs(args, device, seed_offset=1)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **train_loader_args,
    )
    validation_batch_size = 1 if spec.protocol == "volume" else args.batch_size
    validation_loader = DataLoader(
        validation_set,
        batch_size=validation_batch_size,
        shuffle=False,
        drop_last=False,
        **validation_loader_args,
    )
    return spec, paths, train_set, validation_set, train_loader, validation_loader, None


def _build_carvana_data(args: argparse.Namespace, device: torch.device):
    image_dir = Path(args.images_dir)
    mask_dir = Path(args.masks_dir)
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Carvana image/mask directories are required: images={image_dir}, masks={mask_dir}"
        )
    try:
        dataset = CarvanaDataset(image_dir, mask_dir, args.scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(image_dir, mask_dir, args.scale)
    if len(dataset) < 2:
        raise RuntimeError("Carvana needs at least two samples for a train/validation split")
    n_validation = max(1, int(len(dataset) * args.validation / 100.0))
    n_validation = min(n_validation, len(dataset) - 1)
    n_train = len(dataset) - n_validation
    train_set, validation_set = random_split(
        dataset,
        (n_train, n_validation),
        generator=dataloader_generator(args.seed),
    )
    train_loader_args = _loader_kwargs(args, device, seed_offset=0)
    validation_loader_args = _loader_kwargs(args, device, seed_offset=1)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **train_loader_args,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **validation_loader_args,
    )
    num_classes = 2 if args.classes is None else args.classes
    if num_classes < 1:
        raise ValueError("--classes must be positive")
    class_names = tuple("class_{}".format(index) for index in range(num_classes))
    spec = SimpleNamespace(
        name="Carvana",
        num_classes=num_classes,
        input_channels=3,
        class_names=class_names,
        protocol="legacy_2d",
        imagenet_normalization=False,
    )
    paths = SimpleNamespace(
        train_root=image_dir,
        volume_root=image_dir,
        list_dir=None,
        as_dict=lambda: {
            "dataset": "Carvana",
            "images_dir": str(image_dir),
            "masks_dir": str(mask_dir),
        },
    )
    return (
        spec,
        paths,
        train_set,
        validation_set,
        train_loader,
        validation_loader,
        getattr(dataset, "mask_values", None),
    )


def _validate_medical(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    spec: Any,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, dict[str, Any]]:
    if spec.protocol == "volume":
        result = evaluate_volume_loader(
            model,
            validation_loader,
            dataset_name=spec.name,
            num_classes=spec.num_classes,
            class_names=spec.class_names,
            img_size=args.img_size,
            device=device,
            input_channels=spec.input_channels,
            acdc_zspacing=args.acdc_zspacing,
            amp=amp_enabled,
        )
    else:
        result = evaluate_cataract_loader(
            model,
            validation_loader,
            num_classes=spec.num_classes,
            class_names=spec.class_names,
            img_size=args.img_size,
            device=device,
            normalize_raw=False,
            input_channels=spec.input_channels,
            amp=amp_enabled,
        )
    return foreground_validation_dice(result), result


def _checkpoint_fields(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_mean_dice: float,
    spec: Any,
    args: argparse.Namespace,
    mask_values: Any,
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> dict[str, Any]:
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "epoch": epoch,
        "best_mean_dice": best_mean_dice,
        "dataset": spec.name,
        "n_channels": spec.input_channels,
        "n_classes": spec.num_classes,
        "bilinear": args.bilinear,
        "img_size": args.img_size,
        "class_names": spec.class_names,
        "normalization": "imagenet" if spec.imagenet_normalization else "none",
        "arguments": args,
        "mask_values": mask_values,
        "extra": {
            "rng_state": capture_rng_state(
                train_generator=train_loader.generator,
                validation_generator=validation_loader.generator,
            )
        },
    }


def run_training(args: argparse.Namespace) -> None:
    if args.resume and (args.init_checkpoint or args.load):
        raise ValueError("--resume cannot be combined with an initialization checkpoint")
    if args.init_checkpoint and args.load:
        raise ValueError("Use only one of --init-checkpoint and legacy --load")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive")
    if args.save_every < 0:
        raise ValueError("--save-every cannot be negative")
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("--batch-size and --epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.ce_weight < 0 or args.dice_weight < 0 or not (args.ce_weight or args.dice_weight):
        raise ValueError("Loss weights must be non-negative and at least one must be positive")
    if args.dataset == "Carvana" and not 0 < args.validation < 100:
        raise ValueError("Carvana --validation must be between 0 and 100 percent")
    for option, checkpoint_value in (
        ("--resume", args.resume),
        ("--init-checkpoint", args.init_checkpoint or args.load),
    ):
        if checkpoint_value is not None and not Path(checkpoint_value).expanduser().is_file():
            raise FileNotFoundError(
                f"{option} checkpoint does not exist or is not a file: {checkpoint_value}"
            )

    device = resolve_device(args.device)
    seed_everything(args.seed, args.deterministic)
    medical = args.dataset != "Carvana"
    if medical:
        args.dataset = canonicalize_dataset_name(args.dataset)
        if args.dataset == "ACDC":
            fallback_acdc_voxelspacing_zyx(args.acdc_zspacing, "CLI validation")
        data = _build_medical_data(args, device)
    else:
        data = _build_carvana_data(args, device)
    spec, paths, train_set, validation_set, train_loader, validation_loader, mask_values = data

    if spec.name == "ACDC":
        logging.warning(
            "ACDC fold_id=%d is retained for compatibility but the supplied active split ignores it: "
            "patients 021..100 train and 001..020 validation/test",
            args.fold_id,
        )
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise RuntimeError("A non-empty train and validation DataLoader is required")

    model = UNet(
        n_channels=spec.input_channels,
        n_classes=spec.num_classes,
        bilinear=args.bilinear,
    ).to(device)
    logging.info(
        "U-Net dataset=%s channels=%d classes=%d img_size=%d bilinear=%s device=%s",
        spec.name,
        spec.input_channels,
        spec.num_classes,
        args.img_size,
        args.bilinear,
        device,
    )
    logging.info("Dataset paths: %s", paths.as_dict())
    logging.info(
        "Split sizes: train=%d validation=%d; seed=%d deterministic=%s",
        len(train_set),
        len(validation_set),
        args.seed,
        args.deterministic,
    )

    init_path = args.init_checkpoint or args.load
    if init_path:
        init_result = load_checkpoint(
            model,
            init_path,
            mode="init",
            map_location=device,
            dataset=spec.name,
            n_channels=spec.input_channels,
            n_classes=spec.num_classes,
            bilinear=args.bilinear,
            allow_partial_init=args.allow_partial_init,
        )
        logging.info("Initialization loaded: %s", init_result.summary())

    optimizer = _make_optimizer(model, args)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.scheduler_patience
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.amp and not amp_enabled:
        logging.warning("AMP requested on %s; CUDA AMP is disabled", device.type)
    scaler = _make_grad_scaler(amp_enabled)
    criterion = nn.CrossEntropyLoss() if spec.num_classes > 1 else nn.BCEWithLogitsLoss()

    first_epoch = 1
    best_mean_dice = float("-inf")
    if args.resume:
        resume_result = load_checkpoint(
            model,
            args.resume,
            mode="resume",
            map_location=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            dataset=spec.name,
            n_channels=spec.input_channels,
            n_classes=spec.num_classes,
            bilinear=args.bilinear,
        )
        first_epoch = max(1, resume_result.next_epoch)
        if resume_result.best_mean_dice is not None:
            best_mean_dice = resume_result.best_mean_dice
        missing_continuation = []
        if not resume_result.structured:
            missing_continuation.append("structured checkpoint format")
        if resume_result.epoch is None:
            missing_continuation.append("epoch")
        if resume_result.best_mean_dice is None:
            missing_continuation.append("best validation Dice")
        if not resume_result.optimizer_restored:
            missing_continuation.append("optimizer state")
        if not resume_result.scheduler_restored:
            missing_continuation.append("scheduler state")
        if not resume_result.scaler_restored:
            missing_continuation.append("gradient-scaler state")
        if resume_result.rng_state is None:
            missing_continuation.append("RNG/DataLoader state")
        checkpoint_img_size = resume_result.metadata.get("img_size")
        if checkpoint_img_size is None:
            missing_continuation.append("image-size metadata")
        elif int(checkpoint_img_size) != args.img_size:
            raise ValueError(
                f"Resume checkpoint img_size={checkpoint_img_size} disagrees with "
                f"--img-size {args.img_size}"
            )
        checkpoint_normalization = resume_result.metadata.get("normalization")
        expected_normalization = "imagenet" if spec.imagenet_normalization else "none"
        if checkpoint_normalization is None:
            missing_continuation.append("normalization metadata")
        elif str(checkpoint_normalization).strip().lower() != expected_normalization:
            raise ValueError(
                "Resume checkpoint normalization={!r} disagrees with dataset normalization={!r}".format(
                    checkpoint_normalization, expected_normalization
                )
            )
        if missing_continuation:
            raise ValueError(
                "--resume requires an exact structured continuation checkpoint; missing "
                + ", ".join(missing_continuation)
                + ". Use --init-checkpoint to load weights without continuation state."
            )
        restore_rng_state(
            resume_result.rng_state,
            train_generator=train_loader.generator,
            validation_generator=validation_loader.generator,
        )
        logging.info("Resumed training: %s", resume_result.summary())
    if first_epoch > args.epochs:
        raise ValueError(
            f"Resume checkpoint starts at epoch {first_epoch}, beyond requested --epochs {args.epochs}"
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Optimizer=RMSprop lr=%g weight_decay=%g momentum=%g AMP=%s CE_weight=%g "
        "Dice_weight=%g gradient_clip=%g checkpoints=%s init=%s resume=%s",
        args.learning_rate,
        args.weight_decay,
        args.momentum,
        amp_enabled,
        args.ce_weight,
        args.dice_weight,
        args.gradient_clipping,
        checkpoint_dir,
        init_path,
        args.resume,
    )
    experiment = _start_wandb(args)
    global_step = 0

    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(
            total=len(train_set),
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="img",
        )
        for batch_index, batch in enumerate(train_loader):
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = extract_target(batch).to(device=device)
            validate_model_input(images, spec.input_channels)
            label_classes = 2 if spec.num_classes == 1 else spec.num_classes
            validate_labels(targets, label_classes, context=f"epoch {epoch} batch {batch_index}")
            targets = targets.long()

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp_enabled):
                logits = _extract_logits(model(images))
                if spec.num_classes == 1:
                    ce_component = criterion(logits.squeeze(1), targets.float())
                    dice_component = dice_loss(
                        torch.sigmoid(logits.squeeze(1)), targets.float(), multiclass=False
                    )
                else:
                    ce_component = criterion(logits, targets)
                    one_hot = F.one_hot(targets, spec.num_classes).permute(0, 3, 1, 2).float()
                    dice_component = dice_loss(
                        F.softmax(logits, dim=1).float(), one_hot, multiclass=True
                    )
                loss = args.ce_weight * ce_component + args.dice_weight * dice_component
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            scaler.step(optimizer)
            scaler.update()

            batch_count = int(images.shape[0])
            global_step += 1
            epoch_loss += float(loss.item())
            progress.update(batch_count)
            progress.set_postfix(loss=float(loss.item()))
            experiment.log(
                {
                    "train/loss": float(loss.item()),
                    "train/cross_entropy": float(ce_component.item()),
                    "train/dice_loss": float(dice_component.item()),
                    "epoch": epoch,
                    "step": global_step,
                }
            )
        progress.close()

        should_evaluate = epoch % args.eval_interval == 0 or epoch == args.epochs
        validation_result: dict[str, Any] | None = None
        if should_evaluate:
            if medical:
                validation_score, validation_result = _validate_medical(
                    model, validation_loader, spec, args, device, amp_enabled
                )
            else:
                score = evaluate(model, validation_loader, device, amp_enabled)
                validation_score = float(score.item() if torch.is_tensor(score) else score)
                validation_result = {"mean_dice": validation_score, "protocol": "legacy_2d_dice"}
            scheduler.step(validation_score)
            logging.info(
                "Epoch %d validation foreground mean Dice %.6f (%s)",
                epoch,
                validation_score,
                validation_result.get("protocol"),
            )
            experiment.log(
                {
                    "validation/mean_dice": validation_score,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "step": global_step,
                }
            )
            if validation_score > best_mean_dice:
                best_mean_dice = validation_score
                best_path = save_checkpoint(
                    checkpoint_dir
                    / (
                        f"{spec.name.replace(' ', '_')}_epoch{epoch}_score{validation_score:.6f}.pth"
                    ),
                    **_checkpoint_fields(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_mean_dice=best_mean_dice,
                        spec=spec,
                        args=args,
                        mask_values=mask_values,
                        train_loader=train_loader,
                        validation_loader=validation_loader,
                    ),
                )
                logging.info("New best checkpoint saved: %s", best_path)

        fields = _checkpoint_fields(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_mean_dice=best_mean_dice,
            spec=spec,
            args=args,
            mask_values=mask_values,
            train_loader=train_loader,
            validation_loader=validation_loader,
        )
        last_path = save_checkpoint(checkpoint_dir / "last_model.pth", **fields)
        if args.save_every and epoch % args.save_every == 0:
            periodic = save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch}.pth", **fields)
            logging.info("Periodic checkpoint saved: %s", periodic)
        logging.info(
            "Epoch %d mean training loss %.6f; last checkpoint=%s",
            epoch,
            epoch_loss / max(1, len(train_loader)),
            last_path,
        )



def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the U-Net on Synapse, ACDC, Cataract1k, or legacy Carvana"
    )
    parser.add_argument(
        "--dataset",
        choices=(*MEDICAL_DATASETS, "Carvana"),
        default="Carvana",
    )
    parser.add_argument("--root-path", "--root_path", dest="root_path")
    parser.add_argument("--volume-path", "--volume_path", dest="volume_path")
    parser.add_argument("--list-dir", "--list_dir", dest="list_dir")
    parser.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    parser.add_argument("--epochs", "--max_epochs", "-e", type=int, default=5)
    parser.add_argument("--batch-size", "--batch_size", "-b", type=int, default=1)
    parser.add_argument(
        "--learning-rate", "--base_lr", "-l", dest="learning_rate", type=float, default=1e-5
    )
    parser.add_argument(
        "--checkpoint-dir", "--output_dir", dest="checkpoint_dir", default="checkpoints"
    )
    parser.add_argument(
        "--init-checkpoint",
        "--pretrained-checkpoint",
        "--pretrained_checkpoint",
        dest="init_checkpoint",
    )
    parser.add_argument("--load", "-f", help="Legacy alias for --init-checkpoint")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument("--allow-partial-init", action="store_true")
    parser.add_argument("--bilinear", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers", type=int, default=default_num_workers()
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--eval-interval", "--eval_interval", dest="eval_interval", type=int, default=1
    )
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.999)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--fold-id", "--fold_id", dest="fold_id", type=int, default=0)
    parser.add_argument(
        "--acdc-zspacing", "--acdc_zspacing", dest="acdc_zspacing", type=float, default=5.0
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true", help="Opt in to Weights & Biases logging")
    parser.add_argument("--wandb-project", default="Pytorch-UNet-medical")
    parser.add_argument("--description", default="")

    # Legacy Carvana-only options remain available for predict/train compatibility.
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--masks-dir", default=str(DEFAULT_MASK_DIR))
    parser.add_argument("--scale", "-s", type=float, default=0.5)
    parser.add_argument("--validation", "-v", type=float, default=10.0)
    parser.add_argument("--classes", "-c", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_training(args)


if __name__ == "__main__":
    main()
