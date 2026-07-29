import math
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    from tools.testing_tiny import _independent_hard_dice
    from tools.tiny_common import (
        HardDiceAccumulator,
        ResizeOnlyTransform,
        TinySampleRecord,
        read_json,
        records_from_document,
        select_tiny_records,
        write_json,
    )
    from tools.training_tiny import classwise_soft_dice_loss
except ImportError:
    torch = None


class _FakeSliceDataset(Dataset):
    def __init__(self):
        self.labels = []
        class_sets = (
            (1,),
            (2,),
            (3,),
            (1, 2),
            (2, 3),
            (1, 3),
            (1,),
            (2,),
            (3,),
            (1, 2, 3),
        )
        for class_ids in class_sets:
            label = np.zeros((16, 16), dtype=np.uint8)
            for offset, class_id in enumerate(class_ids):
                label[2 + offset : 7 + offset, 2 + 3 * offset : 7 + 3 * offset] = class_id
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "image": np.full((16, 16), index, dtype=np.float32),
            "label": self.labels[index].copy(),
            "case_name": f"case_{index}",
        }


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TinyDiagnosticTests(unittest.TestCase):
    def test_classwise_soft_dice_is_near_zero_for_correct_logits(self):
        target = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long)
        logits = torch.full((1, 3, 2, 2), -20.0)
        logits.scatter_(1, target.unsqueeze(1), 20.0)
        loss = classwise_soft_dice_loss(logits, target)
        self.assertLess(float(loss), 1e-6)

    def test_resize_only_preserves_channels_and_is_deterministic(self):
        transform = ResizeOnlyTransform(32, rgb_imagenet=False)
        sample = {
            "image": np.arange(16 * 20, dtype=np.float32).reshape(16, 20),
            "label": np.pad(np.ones((8, 10), dtype=np.uint8), ((4, 4), (5, 5))),
        }
        first = transform(sample)
        second = transform(sample)
        self.assertEqual(tuple(first["image"].shape), (1, 32, 32))
        self.assertEqual(tuple(first["label"].shape), (32, 32))
        self.assertTrue(torch.equal(first["image"], second["image"]))
        self.assertTrue(torch.equal(first["label"], second["label"]))

    def test_rgb_resize_applies_imagenet_normalization(self):
        transform = ResizeOnlyTransform(16, rgb_imagenet=True)
        result = transform(
            {
                "image": np.full((16, 16, 3), 255, dtype=np.uint8),
                "label": np.ones((16, 16), dtype=np.uint8),
            }
        )
        expected = torch.tensor(
            [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]
        )
        self.assertEqual(tuple(result["image"].shape), (3, 16, 16))
        self.assertTrue(torch.allclose(result["image"][:, 0, 0], expected))

    def test_selection_is_seeded_foreground_only_and_covers_classes(self):
        dataset = _FakeSliceDataset()
        transform = ResizeOnlyTransform(16, rgb_imagenet=False)
        first = select_tiny_records(
            dataset,
            transform,
            num_samples=8,
            num_classes=4,
            seed=1234,
        )
        second = select_tiny_records(
            dataset,
            transform,
            num_samples=8,
            num_classes=4,
            seed=1234,
        )
        self.assertEqual(first, second)
        self.assertTrue(all(record.foreground_classes for record in first))
        represented = {
            class_id for record in first for class_id in record.foreground_classes
        }
        self.assertEqual(represented, {1, 2, 3})

    def test_hard_dice_reports_present_class_and_marks_absent_class(self):
        per_class, mean = _independent_hard_dice(
            intersections=np.asarray([5, 2, 0]),
            predicted_counts=np.asarray([5, 4, 1]),
            ground_truth_counts=np.asarray([5, 2, 0]),
        )
        self.assertAlmostEqual(per_class[1], 4 / 6)
        self.assertTrue(math.isnan(per_class[2]))
        self.assertAlmostEqual(mean, 4 / 6)

        accumulator = HardDiceAccumulator.create(3)
        prediction = torch.tensor([[[0, 1], [1, 2]]])
        target = torch.tensor([[[0, 1], [1, 0]]])
        accumulator.update(prediction, target)
        metrics = accumulator.result()
        self.assertEqual(metrics["sample_count"], 1)
        self.assertEqual(metrics["foreground_sample_count"], 1)
        self.assertTrue(math.isnan(metrics["per_class_dice"][2]))

    def test_subset_json_round_trip_validates_records(self):
        document = {
            "dataset": "ACDC",
            "selected_indices": [4, 8],
            "samples": [
                {
                    "index": 4,
                    "case_name": "patient021_frame01",
                    "foreground_classes": [1, 3],
                },
                {
                    "index": 8,
                    "case_name": "patient022_frame02",
                    "foreground_classes": [2],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny_subset.json"
            write_json(path, document)
            loaded = read_json(path)
        records = records_from_document(
            loaded,
            expected_dataset="ACDC",
            num_classes=4,
        )
        self.assertEqual(
            records,
            [
                TinySampleRecord(4, "patient021_frame01", (1, 3)),
                TinySampleRecord(8, "patient022_frame02", (2,)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
