"""Focused tests for channel-aware real-sample U-Net benchmarking."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # Source-only environments may omit project dependencies.
    torch = None

if torch is not None:
    from utils.benchmark import (
        BenchmarkResults,
        collate_for_benchmark,
        count_flops_gflops,
        count_parameters,
        model_size_mb_benchmark,
        runtime_memory_mb_benchmark,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BenchmarkCollationTests(unittest.TestCase):
    def test_grayscale_slice_and_volume_remain_one_channel(self):
        grayscale = np.full((5, 7), 11.0, dtype=np.float32)
        volume = np.stack(
            [np.full((6, 4), depth, dtype=np.float32) for depth in range(5)]
        )

        slice_batch = collate_for_benchmark(
            [{"image": grayscale}], img_size=8, input_channels=1
        )
        volume_batch = collate_for_benchmark(
            [{"image": volume}],
            img_size=8,
            input_channels=1,
            volume_input=True,
        )

        self.assertEqual(tuple(slice_batch.shape), (1, 1, 8, 8))
        self.assertEqual(tuple(volume_batch.shape), (1, 1, 8, 8))
        self.assertTrue(slice_batch.is_contiguous())
        self.assertTrue(volume_batch.is_contiguous())
        self.assertTrue(
            torch.allclose(volume_batch, torch.full_like(volume_batch, 2.0))
        )

    def test_hwc_and_chw_rgb_preserve_channel_order(self):
        hwc = np.empty((4, 6, 3), dtype=np.float32)
        hwc[..., 0], hwc[..., 1], hwc[..., 2] = 21.0, 22.0, 23.0
        chw = torch.empty((3, 7, 5), dtype=torch.float32)
        chw[0].fill_(31.0)
        chw[1].fill_(32.0)
        chw[2].fill_(33.0)

        batch = collate_for_benchmark(
            [{"image": hwc}, {"image": chw}],
            img_size=8,
            input_channels=3,
        )

        self.assertEqual(tuple(batch.shape), (2, 3, 8, 8))
        self.assertTrue(batch.is_contiguous())
        for channel, value in enumerate((21.0, 22.0, 23.0)):
            self.assertTrue(
                torch.allclose(batch[0, channel], torch.full_like(batch[0, channel], value))
            )
        for channel, value in enumerate((31.0, 32.0, 33.0)):
            self.assertTrue(
                torch.allclose(batch[1, channel], torch.full_like(batch[1, channel], value))
            )

    def test_depth_three_volume_is_disambiguated_by_dataset_context(self):
        volume = np.stack(
            [np.full((6, 4), depth, dtype=np.float32) for depth in range(3)]
        )
        batch = collate_for_benchmark(
            [{"image": volume}],
            img_size=8,
            input_channels=1,
            volume_input=True,
        )
        self.assertEqual(tuple(batch.shape), (1, 1, 8, 8))
        self.assertTrue(torch.allclose(batch, torch.ones_like(batch)))

    def test_channel_mismatch_is_rejected_without_gray_to_rgb_expansion(self):
        with self.assertRaisesRegex(ValueError, "not expanded to RGB"):
            collate_for_benchmark(
                [{"image": np.zeros((5, 7), dtype=np.float32)}],
                img_size=8,
                input_channels=3,
            )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BenchmarkCountingTests(unittest.TestCase):
    class TinyConv(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.n_channels = 1
            self.conv = torch.nn.Conv2d(1, 2, kernel_size=3)

        def forward(self, value):
            return self.conv(value)

    def test_parameter_count_matches_torch_sum(self):
        model = self.TinyConv()
        counts = count_parameters(model)
        self.assertEqual(counts.total, sum(value.numel() for value in model.parameters()))
        self.assertEqual(
            counts.trainable,
            sum(value.numel() for value in model.parameters() if value.requires_grad),
        )

    def test_flops_apply_two_times_macs_once(self):
        model = self.TinyConv()
        gflops, _ = count_flops_gflops(model, torch.randn(1, 1, 8, 8))
        # MACs = B * Cin * Cout * Kh * Kw * Hout * Wout = 648.
        self.assertEqual(gflops, (2.0 * 648.0) / 1e9)

    def test_model_size_counts_tensor_element_bytes(self):
        model = self.TinyConv()
        result = model_size_mb_benchmark(model)
        expected = sum(
            tensor.numel() * tensor.element_size()
            for tensor in model.state_dict().values()
            if torch.is_tensor(tensor)
        )
        self.assertEqual(result["total_bytes"], expected)
        self.assertEqual(result["unit"], "MiB")

    def test_cpu_runtime_memory_is_explicitly_unavailable(self):
        result = runtime_memory_mb_benchmark(
            self.TinyConv(), input_size=(1, 8, 8), device="cpu"
        )
        self.assertFalse(result["available"])
        self.assertIsNone(result["peak_allocated_bytes"])
        self.assertIn("CPU", result["reason"])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BenchmarkResultTests(unittest.TestCase):
    @dataclass
    class Diagnostic:
        score: float

    def test_pretty_handles_nested_and_nonfinite_values(self):
        result = BenchmarkResults(
            metrics={
                "mean_dice": 0.5,
                "per_class": {"organ": {"dice": np.float32(0.75)}},
                "undefined_hd95": float("nan"),
            }
        )
        rendered = result.pretty()
        self.assertIn("mean_dice", rendered)
        self.assertIn("per_class", rendered)
        self.assertIn("nan", rendered.lower())

    def test_json_conversion_is_recursive_and_strict(self):
        result = BenchmarkResults(
            metrics={
                "array": np.asarray([1.0, np.inf]),
                "tensor": torch.tensor([2.0, float("nan")]),
                "device": torch.device("cpu"),
                "dtype": torch.float32,
                "diagnostic": self.Diagnostic(score=float("nan")),
            },
            notes={"path": Path("checkpoint.pth"), "shape": (1, 224, 224)},
        )
        converted = result.to_dict()
        self.assertEqual(set(converted), {"metrics", "notes"})
        self.assertEqual(converted["metrics"]["array"], [1.0, None])
        self.assertEqual(converted["metrics"]["tensor"], [2.0, None])
        self.assertEqual(converted["metrics"]["device"], "cpu")
        self.assertEqual(converted["metrics"]["dtype"], "torch.float32")
        self.assertIsNone(converted["metrics"]["diagnostic"]["score"])
        self.assertEqual(converted["notes"]["shape"], [1, 224, 224])

        with tempfile.TemporaryDirectory() as directory:
            output = result.save_json(Path(directory) / "result.json")
            text = output.read_text()
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            self.assertEqual(json.loads(text), converted)


if __name__ == "__main__":
    unittest.main()
