"""Medical accuracy evaluation followed by real-sample U-Net benchmarking."""

from __future__ import annotations

import argparse
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from unet import UNet
from utils.benchmark import (
    BenchmarkResults,
    benchmark_segmentation_model,
    build_benchmark_loader,
)
from utils.checkpointing import checkpoint_metadata, load_checkpoint, load_checkpoint_file
from utils.dataset_registry import (
    build_test_dataset,
    canonicalize_dataset_name,
    get_dataset_spec,
    resolve_dataset_paths,
)
from utils.medical_inference import evaluate_cataract_loader, evaluate_volume_loader
from utils.medical_metrics import fallback_acdc_voxelspacing_zyx
from utils.runtime import resolve_device, seed_everything, seed_worker


DATASET_CHOICES = ("Synapse", "ACDC", "Cataract1k", "Catrakt1k")


def _bounded_repeats(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 10:
        raise argparse.ArgumentTypeError("repeated runs must be in the range 1..10")
    return parsed


def _metadata_bool(value: Any, field: str) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Checkpoint {field} metadata is not boolean: {value!r}")
    return bool(value)


def _select_bilinear(args: argparse.Namespace, metadata: dict[str, Any]) -> bool:
    if metadata.get("bilinear") is None:
        return bool(args.bilinear)
    checkpoint_value = _metadata_bool(metadata["bilinear"], "bilinear")
    if args.bilinear is not None and bool(args.bilinear) != checkpoint_value:
        raise ValueError(
            f"--bilinear={args.bilinear} disagrees with checkpoint metadata "
            f"bilinear={checkpoint_value}"
        )
    return checkpoint_value


def _select_cataract_normalization(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    dataset_name: str,
) -> bool:
    enabled = bool(args.normalize_present_class_eval)
    if dataset_name != "Cataract1k":
        if enabled:
            raise ValueError("--normalize-present-class-eval is only valid for Cataract1k")
        return False
    normalization_value = metadata.get("normalization")
    normalization = (
        "" if normalization_value is None else str(normalization_value).strip().lower()
    )
    imagenet_values = {"imagenet", "image_net", "imagenet_mean_std", "true", "1"}
    none_values = {"none", "false", "unnormalized", "0"}
    if normalization and normalization not in imagenet_values | none_values:
        raise ValueError(f"Unsupported checkpoint normalization metadata: {normalization!r}")
    if normalization in imagenet_values and not enabled:
        raise ValueError(
            "This checkpoint was trained with ImageNet normalization; rerun with "
            "--normalize-present-class-eval"
        )
    if normalization in none_values and enabled:
        raise ValueError(
            "--normalize-present-class-eval disagrees with checkpoint normalization='none'"
        )
    return enabled


def _validate_test_arguments(args: argparse.Namespace, device: torch.device) -> None:
    if args.img_size <= 0:
        raise ValueError("--img-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_test_cases is not None and args.max_test_cases <= 0:
        raise ValueError("--max-test-cases must be positive")
    benchmark_will_run = not (
        args.skip_system_benchmark or args.verbose or args.max_test_cases is not None
    )
    if not benchmark_will_run:
        return
    if args.benchmark_batch_size <= 0:
        raise ValueError("--benchmark-batch-size must be positive")
    if args.benchmark_warmup_steps < 0:
        raise ValueError("--benchmark-warmup-steps cannot be negative")
    if args.benchmark_measure_batches <= 0:
        raise ValueError("--benchmark-measure-batches must be positive")
    if args.single_image_latency_samples < 0:
        raise ValueError("--single-image-latency-samples cannot be negative")
    if args.single_image_warmup_steps < 0:
        raise ValueError("--single-image-warmup-steps cannot be negative")
    if args.benchmark_num_workers < 0:
        raise ValueError("--benchmark-num-workers cannot be negative")
    if args.benchmark_amp and device.type != "cuda":
        raise ValueError("--benchmark-amp requires a CUDA device")


def _environment_notes(device: torch.device) -> dict[str, Any]:
    notes: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        notes["cuda_device_name"] = torch.cuda.get_device_name(device)
    return notes


def _accuracy_loader(dataset, args: argparse.Namespace, device: torch.device) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )


def run_test(args: argparse.Namespace) -> Path | None:
    args.dataset = canonicalize_dataset_name(args.dataset)
    spec = get_dataset_spec(args.dataset)
    checkpoint_path = Path(args.checkpoint).expanduser()
    payload = load_checkpoint_file(checkpoint_path, map_location="cpu")
    metadata = checkpoint_metadata(payload)
    if spec.name == "ACDC":
        fallback_acdc_voxelspacing_zyx(args.acdc_zspacing, "CLI validation")
    device = resolve_device(args.device)
    _validate_test_arguments(args, device)
    seed_everything(args.seed, args.deterministic)

    bilinear = _select_bilinear(args, metadata)
    normalize_cataract = _select_cataract_normalization(args, metadata, spec.name)
    args.bilinear = bilinear

    paths = resolve_dataset_paths(
        spec,
        root=args.root_path,
        volume_root=args.volume_path,
        list_dir=args.list_dir,
    )
    dataset = build_test_dataset(
        spec,
        args.img_size,
        paths=paths,
        fold_id=args.fold_id if spec.name == "ACDC" else None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"{spec.name} test split is empty")
    if spec.name == "ACDC":
        logging.warning(
            "ACDC fold_id=%d is compatibility-only; the supplied active test split is patients 001..020",
            args.fold_id,
        )

    model = UNet(
        n_channels=spec.input_channels,
        n_classes=spec.num_classes,
        bilinear=bilinear,
    ).to(device)
    load_result = load_checkpoint(
        model,
        checkpoint_path,
        mode="test",
        map_location=device,
        dataset=spec.name,
        n_channels=spec.input_channels,
        n_classes=spec.num_classes,
        bilinear=bilinear,
    )
    logging.info("Loaded trained checkpoint: %s", load_result.summary())
    if metadata.get("img_size") is not None and int(metadata["img_size"]) != args.img_size:
        logging.warning(
            "Testing img_size=%d differs from checkpoint img_size=%s",
            args.img_size,
            metadata["img_size"],
        )
    logging.info(
        "Testing dataset=%s cases/frames=%d channels=%d classes=%d img_size=%d "
        "bilinear=%s device=%s roots=%s",
        spec.name,
        len(dataset),
        spec.input_channels,
        spec.num_classes,
        args.img_size,
        bilinear,
        device,
        paths.as_dict(),
    )

    maximum = 1 if args.verbose and args.max_test_cases is None else args.max_test_cases
    if maximum is not None:
        logging.warning(
            "Smoke-test mode: accuracy is limited to %d case(s)/frame(s); this is not an official benchmark",
            maximum,
        )
    prediction_path = Path(args.save_predictions) if args.save_predictions else None
    loader = _accuracy_loader(dataset, args, device)
    if spec.protocol == "volume":
        accuracy = evaluate_volume_loader(
            model,
            loader,
            dataset_name=spec.name,
            num_classes=spec.num_classes,
            class_names=spec.class_names,
            img_size=args.img_size,
            device=device,
            input_channels=spec.input_channels,
            acdc_zspacing=args.acdc_zspacing,
            amp=False,
            save_predictions=prediction_path,
            max_cases=maximum,
        )
    else:
        accuracy = evaluate_cataract_loader(
            model,
            loader,
            num_classes=spec.num_classes,
            class_names=spec.class_names,
            img_size=args.img_size,
            device=device,
            normalize_raw=normalize_cataract,
            input_channels=spec.input_channels,
            amp=False,
            save_predictions=prediction_path,
            max_cases=maximum,
        )
        logging.info("Cataract1k per-class values are diagnostic, not the official headline metric")

    benchmark_skipped = bool(args.skip_system_benchmark or maximum is not None)
    if benchmark_skipped:
        system = BenchmarkResults(
            metrics={},
            notes={
                "system_benchmark_skipped": True,
                "system_benchmark_skip_reason": (
                    "explicit --skip-system-benchmark"
                    if args.skip_system_benchmark
                    else "partial accuracy smoke-test mode"
                ),
            },
        )
    else:
        benchmark_loader = build_benchmark_loader(
            dataset=dataset,
            dataset_name=spec.name,
            img_size=args.img_size,
            input_channels=spec.input_channels,
            batch_size=args.benchmark_batch_size,
            normalize=normalize_cataract if spec.name == "Cataract1k" else False,
            num_workers=args.benchmark_num_workers,
            model=model,
        )
        system = benchmark_segmentation_model(
            model,
            benchmark_loader,
            device=device,
            warmup_steps=args.benchmark_warmup_steps,
            measure_batches=args.benchmark_measure_batches,
            single_image_latency_samples=args.single_image_latency_samples,
            single_image_warmup_steps=args.single_image_warmup_steps,
            repeated_runs=args.repeated_runs,
            autocast=args.benchmark_amp,
            args=args,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    protocol = accuracy["protocol"]
    notes = {
        **system.notes,
        "dataset": spec.name,
        "protocol": protocol,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_structured": load_result.structured,
        "device": str(device),
        "class_names": list(spec.class_names),
        "num_classes": spec.num_classes,
        "input_channels": spec.input_channels,
        "bilinear": bilinear,
        "img_size": args.img_size,
        "normalization": "imagenet" if normalize_cataract else "none",
        "data_root": str(paths.train_root),
        "volume_path": str(paths.volume_root),
        "list_dir": None if paths.list_dir is None else str(paths.list_dir),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "description": args.description,
        "arguments": vars(args),
        "timestamp": timestamp,
        "partial_accuracy_smoke_test": maximum is not None,
        **_environment_notes(device),
    }
    result = BenchmarkResults(metrics={**accuracy, **system.metrics}, notes=notes)
    print(result.pretty())

    if args.no_json:
        logging.info("JSON output disabled by --no-json")
        return None
    destination = (
        Path(args.benchmark_dir)
        / spec.name
        / f"{checkpoint_path.stem}_{args.img_size}_{timestamp}.json"
    )
    saved_path = result.save_json(destination)
    logging.info("Combined accuracy/benchmark JSON saved to %s", saved_path)
    return saved_path


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained U-Net and benchmark it on real medical test samples"
    )
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--root-path", "--root_path", dest="root_path")
    parser.add_argument("--volume-path", "--volume_path", dest="volume_path")
    parser.add_argument("--list-dir", "--list_dir", dest="list_dir")
    parser.add_argument("--checkpoint", "--ckpt", required=True)
    parser.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    parser.add_argument(
        "--bilinear",
        action="store_true",
        default=None,
        help="Use only for raw checkpoints without bilinear metadata",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fold-id", "--fold_id", dest="fold_id", type=int, default=0)
    parser.add_argument(
        "--acdc-zspacing", "--acdc_zspacing", dest="acdc_zspacing", type=float, default=5.0
    )
    parser.add_argument(
        "--normalize-present-class-eval",
        "--normalize_present_class_eval",
        dest="normalize_present_class_eval",
        action="store_true",
    )
    parser.add_argument(
        "--benchmark-dir", "--benchmark_dir", dest="benchmark_dir", default="benchmark"
    )
    parser.add_argument("--benchmark-batch-size", type=int, default=36)
    parser.add_argument("--benchmark-warmup-steps", type=int, default=20)
    parser.add_argument("--benchmark-measure-batches", type=int, default=50)
    parser.add_argument("--single-image-latency-samples", type=int, default=1000)
    parser.add_argument("--single-image-warmup-steps", type=int, default=50)
    parser.add_argument(
        "--repeated-runs", "--repeated_runs", dest="repeated_runs", type=_bounded_repeats, default=1
    )
    parser.add_argument("--benchmark-amp", action="store_true")
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=1)
    parser.add_argument("--benchmark-num-workers", type=int, default=0)
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--save-predictions",
        "--is_savenii",
        dest="save_predictions",
        nargs="?",
        const="predictions",
        default=None,
    )
    parser.add_argument("--verbose", action="store_true", help="Run one-case non-benchmark smoke mode")
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--skip-system-benchmark", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_test(get_args(argv))


if __name__ == "__main__":
    main()
