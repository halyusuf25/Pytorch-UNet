import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
    import datasets.dataset_acdc as acdc
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ACDCInterpolationTests(unittest.TestCase):
    def test_random_rotation_uses_cubic_image_and_nearest_label(self):
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        label = (image > 7).astype(np.uint8)

        with patch.object(
            acdc.ndimage,
            "rotate",
            side_effect=lambda array, *_args, **_kwargs: array.copy(),
        ) as rotate:
            output_image, output_label = acdc.random_rotate(image, label)

        self.assertEqual(rotate.call_count, 2)
        self.assertEqual(rotate.call_args_list[0].kwargs["order"], 3)
        self.assertEqual(rotate.call_args_list[1].kwargs["order"], 0)
        self.assertTrue(np.array_equal(output_image, image))
        self.assertTrue(np.array_equal(output_label, label))

    def test_resize_uses_cubic_image_and_nearest_label(self):
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        label = np.zeros((4, 4), dtype=np.uint8)
        label[1:3, 1:3] = 2
        real_zoom = acdc.zoom

        with (
            patch.object(acdc.random, "random", return_value=0.0),
            patch.object(acdc, "zoom", side_effect=real_zoom) as resize,
        ):
            result = acdc.RandomGenerator4ACDC((8, 8))(
                {"image": image, "label": label}
            )

        self.assertEqual(resize.call_count, 2)
        self.assertEqual(resize.call_args_list[0].kwargs["order"], 3)
        self.assertEqual(resize.call_args_list[1].kwargs["order"], 0)
        self.assertEqual(tuple(result["image"].shape), (1, 8, 8))
        self.assertEqual(tuple(result["label"].shape), (8, 8))
        self.assertEqual(result["image"].dtype, torch.float32)
        self.assertEqual(
            set(torch.unique(result["label"]).tolist()),
            {0, 2},
        )


if __name__ == "__main__":
    unittest.main()
