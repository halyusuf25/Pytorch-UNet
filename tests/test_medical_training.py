import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from train import (
        DEFAULT_CARVANA_LEARNING_RATE,
        DEFAULT_MEDICAL_BASE_LR,
        MEDICAL_MOMENTUM,
        MEDICAL_WEIGHT_DECAY,
        _make_optimizer,
        _medical_polynomial_learning_rate,
        _update_medical_learning_rate,
        get_args,
        run_training,
    )
    from utils.checkpointing import (
        load_checkpoint,
        load_checkpoint_file,
        save_checkpoint,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MedicalOptimizationTests(unittest.TestCase):
    def test_dataset_specific_learning_rate_defaults_and_alias(self):
        for dataset in ("Synapse", "ACDC", "Cataract1k", "Catrakt1k"):
            with self.subTest(dataset=dataset):
                args = get_args(["--dataset", dataset])
                self.assertEqual(args.learning_rate, DEFAULT_MEDICAL_BASE_LR)
                self.assertEqual(args.lambda_, 0.5)
        carvana = get_args(["--dataset", "Carvana"])
        self.assertEqual(
            carvana.learning_rate,
            DEFAULT_CARVANA_LEARNING_RATE,
        )
        explicit = get_args(
            ["--dataset", "Synapse", "--base_lr", "0.025", "--lambda_", "0.3"]
        )
        self.assertEqual(explicit.learning_rate, 0.025)
        self.assertEqual(explicit.lambda_, 0.3)

    def test_medical_optimizer_is_exact_sgd_configuration(self):
        model = torch.nn.Conv2d(1, 2, 1)
        args = Namespace(
            learning_rate=0.01,
            weight_decay=9.0,
            momentum=0.1,
        )
        optimizer = _make_optimizer(model, args, medical=True)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        group = optimizer.param_groups[0]
        self.assertEqual(group["lr"], 0.01)
        self.assertEqual(group["momentum"], MEDICAL_MOMENTUM)
        self.assertEqual(group["weight_decay"], MEDICAL_WEIGHT_DECAY)

    def test_polynomial_learning_rate_updates_every_parameter_group(self):
        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD(
            [{"params": [first], "lr": 0.1}, {"params": [second], "lr": 0.05}],
            lr=0.1,
        )
        self.assertEqual(_medical_polynomial_learning_rate(0.01, 0, 100), 0.01)
        expected_midpoint = 0.01 * (1.0 - 50 / 100) ** 0.9
        self.assertAlmostEqual(
            _medical_polynomial_learning_rate(0.01, 50, 100),
            expected_midpoint,
        )
        updated = _update_medical_learning_rate(
            optimizer,
            base_lr=0.01,
            iter_num=50,
            max_iterations=100,
        )
        self.assertAlmostEqual(updated, expected_midpoint)
        self.assertTrue(
            all(group["lr"] == updated for group in optimizer.param_groups)
        )

    def test_restored_optimizer_lr_continues_reference_step_order(self):
        max_iterations = 6

        def advance(optimizer, start, stop):
            used_learning_rates = []
            for iter_num in range(start, stop):
                used_learning_rates.append(optimizer.param_groups[0]["lr"])
                optimizer.step()
                _update_medical_learning_rate(
                    optimizer,
                    base_lr=0.01,
                    iter_num=iter_num,
                    max_iterations=max_iterations,
                )
            return used_learning_rates

        uninterrupted_parameter = torch.nn.Parameter(torch.tensor(1.0))
        uninterrupted = torch.optim.SGD([uninterrupted_parameter], lr=0.01)
        uninterrupted_lrs = advance(uninterrupted, 0, max_iterations)

        first_parameter = torch.nn.Parameter(torch.tensor(1.0))
        first = torch.optim.SGD([first_parameter], lr=0.01)
        resumed_lrs = advance(first, 0, 3)
        saved_optimizer_state = first.state_dict()

        resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
        resumed = torch.optim.SGD([resumed_parameter], lr=0.5)
        resumed.load_state_dict(saved_optimizer_state)
        resumed_lrs.extend(advance(resumed, 3, max_iterations))

        self.assertEqual(resumed_lrs, uninterrupted_lrs)
        self.assertEqual(uninterrupted_lrs[0], 0.01)
        self.assertEqual(uninterrupted_lrs[1], 0.01)

    def test_checkpoint_round_trip_restores_global_step_without_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "medical.pth"
            source = torch.nn.Conv2d(1, 2, 1)
            source.n_channels = 1
            source.n_classes = 2
            source.bilinear = False
            optimizer = torch.optim.SGD(
                source.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4,
            )
            save_checkpoint(
                path,
                source,
                optimizer=optimizer,
                scheduler=None,
                epoch=3,
                global_step=21,
                best_mean_dice=0.5,
                dataset="ACDC",
            )

            target = torch.nn.Conv2d(1, 2, 1)
            target.n_channels = 1
            target.n_classes = 2
            target.bilinear = False
            target_optimizer = torch.optim.SGD(
                target.parameters(),
                lr=0.02,
                momentum=0.9,
                weight_decay=1e-4,
            )
            result = load_checkpoint(
                target,
                path,
                mode="resume",
                optimizer=target_optimizer,
                scheduler=None,
                dataset="ACDC",
            )
            self.assertTrue(result.optimizer_restored)
            self.assertFalse(result.scheduler_restored)
            self.assertEqual(result.global_step, 21)
            self.assertEqual(result.iter_num, 21)

    def test_medical_training_checkpoint_uses_sgd_poly_and_no_scheduler(self):
        class TwoSliceDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                image = torch.full((1, 16, 16), float(index))
                label = torch.zeros((16, 16), dtype=torch.long)
                label[4:12, 4:12] = 1
                return {
                    "image": image,
                    "label": label,
                    "case_name": f"case_{index}",
                }

        class TinySegmentationModel(torch.nn.Module):
            def __init__(self, n_channels, n_classes, bilinear=False):
                super().__init__()
                self.n_channels = n_channels
                self.n_classes = n_classes
                self.bilinear = bilinear
                self.output = torch.nn.Conv2d(n_channels, n_classes, 1)

            def forward(self, image):
                return self.output(image)

        dataset = TwoSliceDataset()
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            generator=torch.Generator().manual_seed(1234),
        )
        validation_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            generator=torch.Generator().manual_seed(1235),
        )
        spec = SimpleNamespace(
            name="ACDC",
            num_classes=2,
            input_channels=1,
            class_names=("Background", "Foreground"),
            protocol="volume",
            imagenet_normalization=False,
        )
        paths = SimpleNamespace(
            as_dict=lambda: {
                "dataset": "ACDC",
                "train_root": "/synthetic",
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            args = get_args(
                [
                    "--dataset",
                    "ACDC",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--img-size",
                    "16",
                    "--num-workers",
                    "0",
                    "--device",
                    "cpu",
                    "--checkpoint-dir",
                    directory,
                ]
            )
            synthetic_data = (
                spec,
                paths,
                dataset,
                dataset,
                train_loader,
                validation_loader,
                None,
            )
            with (
                patch("train.UNet", TinySegmentationModel),
                patch("train._build_medical_data", return_value=synthetic_data),
                patch(
                    "train._validate_medical",
                    return_value=(0.25, {"protocol": "synthetic"}),
                ),
                patch("train.fallback_acdc_voxelspacing_zyx"),
            ):
                run_training(args)

            checkpoint = load_checkpoint_file(
                Path(directory) / "last_model.pth",
                map_location="cpu",
            )
            self.assertEqual(checkpoint["global_step"], 1)
            self.assertEqual(checkpoint["iter_num"], 1)
            self.assertIsNone(checkpoint["scheduler_state_dict"])
            optimizer_state = checkpoint["optimizer_state_dict"]
            self.assertEqual(optimizer_state["param_groups"][0]["momentum"], 0.9)
            self.assertEqual(
                optimizer_state["param_groups"][0]["weight_decay"],
                1e-4,
            )
            self.assertEqual(
                optimizer_state["param_groups"][0]["lr"],
                DEFAULT_MEDICAL_BASE_LR,
            )
            self.assertEqual(checkpoint["arguments"]["lambda_"], 0.5)


if __name__ == "__main__":
    unittest.main()
