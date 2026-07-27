import math
import unittest

import numpy as np

from utils.medical_metrics import (
    _hd95_empty_mask_penalty,
    _mean_metric_array,
    aggregate_present_class_metrics,
    aggregate_volume_metrics,
    calculate_metric_percase,
    calculate_metric_percase_without_absent_reward,
    extract_acdc_voxelspacing_zyx,
    fallback_acdc_voxelspacing_zyx,
    present_class_frame_metrics,
    validate_acdc_voxelspacing_zyx,
)


class BinaryMedicalMetricTests(unittest.TestCase):
    def test_identical_nonempty_masks(self):
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:4, 2:5] = 1
        dice, hd95, iou = calculate_metric_percase(mask, mask)
        self.assertEqual(dice, 1.0)
        self.assertEqual(hd95, 0.0)
        self.assertEqual(iou, 1.0)

    def test_exactly_one_empty_uses_physical_diagonal(self):
        prediction = np.zeros((3, 4), dtype=np.uint8)
        prediction[1, 1] = 1
        ground_truth = np.zeros_like(prediction)
        spacing = (2.0, 3.0)
        expected = math.sqrt((2 * 2.0) ** 2 + (3 * 3.0) ** 2)
        dice, hd95, iou = calculate_metric_percase(
            prediction, ground_truth, voxelspacing=spacing
        )
        self.assertEqual(dice, 0.0)
        self.assertAlmostEqual(hd95, expected)
        self.assertEqual(iou, 0.0)

    def test_both_empty_protocol_variants(self):
        empty = np.zeros((3, 4), dtype=np.uint8)
        dice, hd95, iou = calculate_metric_percase(empty, empty)
        self.assertEqual((dice, iou), (1.0, 1.0))
        self.assertTrue(math.isnan(hd95))

        values = calculate_metric_percase_without_absent_reward(empty, empty)
        self.assertTrue(all(math.isnan(value) for value in values))

    def test_anisotropic_spacing_changes_hd95(self):
        prediction = np.zeros((5, 5), dtype=np.uint8)
        ground_truth = np.zeros_like(prediction)
        prediction[1, 2] = 1
        ground_truth[2, 2] = 1
        unit_hd95 = calculate_metric_percase(prediction, ground_truth)[1]
        physical_hd95 = calculate_metric_percase(
            prediction, ground_truth, voxelspacing=(5.0, 1.0)
        )[1]
        self.assertAlmostEqual(unit_hd95, 1.0)
        self.assertAlmostEqual(physical_hd95, 5.0)

    def test_diagonal_rejects_spacing_dimensionality_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match mask ndim"):
            _hd95_empty_mask_penalty((3, 4), voxelspacing=(1.0, 1.0, 1.0))


class AggregationTests(unittest.TestCase):
    def test_volume_aggregation_is_case_then_class(self):
        # Class zero has one finite case; class one has two.  A global nanmean
        # would be 1/3, while the required case-then-class result is 1/2.
        cases = np.asarray(
            [
                [[1.0, 10.0, 1.0], [0.0, 2.0, 0.0]],
                [[np.nan, np.nan, np.nan], [0.0, 4.0, 0.0]],
            ],
            dtype=np.float32,
        )
        per_class = _mean_metric_array(cases)
        np.testing.assert_allclose(
            per_class,
            [[1.0, 10.0, 1.0], [0.0, 3.0, 0.0]],
            equal_nan=True,
        )
        result = aggregate_volume_metrics(
            cases, class_names=("Background", "One", "Two")
        )
        self.assertAlmostEqual(result["mean_dice"], 0.5)
        self.assertAlmostEqual(result["mean_hd95"], 6.5)
        self.assertAlmostEqual(result["mean_iou"], 0.5)

    def test_present_class_diagnostics_and_discounting(self):
        label = np.zeros((3, 3), dtype=np.uint8)
        label[0, 0] = 1
        prediction = np.zeros_like(label)
        prediction[2, 2] = 2
        evaluated = present_class_frame_metrics(
            prediction, label, 3, case_name="seq2_0001"
        )
        self.assertFalse(evaluated["discounted_frame"])
        self.assertEqual(evaluated["present_class_ids"], [1])
        self.assertEqual(evaluated["false_positive_absent_class_ids"], [2])
        self.assertEqual(evaluated["frame_metrics"][0], 0.0)
        self.assertAlmostEqual(evaluated["frame_metrics"][1], math.sqrt(8.0))
        self.assertEqual(evaluated["frame_metrics"][2], 0.0)

        background = np.zeros_like(label)
        discounted = present_class_frame_metrics(
            background, background, 3, case_name="seq10_0002"
        )
        self.assertTrue(discounted["discounted_frame"])
        self.assertTrue(np.all(np.isnan(discounted["frame_metrics"])))

        result = aggregate_present_class_metrics(
            [discounted, evaluated],
            3,
            class_names=("Background", "Pupil", "Cornea"),
        )
        self.assertEqual(result["mean_dice"], 0.0)
        self.assertEqual(result["evaluated_frames"], 1)
        self.assertEqual(result["discounted_frames"], 1)
        self.assertEqual(result["class_presence_counts"], {"Pupil": 1, "Cornea": 0})
        self.assertEqual(
            result["false_positive_absent_class_counts"],
            {"Pupil": 0, "Cornea": 1},
        )
        self.assertEqual(list(result["per_case_group"]), ["seq2", "seq10"])


class ACDCSpacingTests(unittest.TestCase):
    def test_spacing_keys_are_normalized_to_zyx(self):
        direct = extract_acdc_voxelspacing_zyx(
            {"voxelspacing_zyx": np.asarray([5.0, 1.2, 1.3])}, "patient001"
        )
        self.assertEqual(direct, (5.0, 1.2000000476837158, 1.2999999523162842))

        xyz = extract_acdc_voxelspacing_zyx(
            {"spacing": np.asarray([1.3, 1.2, 5.0])}, "patient001"
        )
        np.testing.assert_allclose(xyz, (5.0, 1.2, 1.3))

        pixdim = extract_acdc_voxelspacing_zyx(
            {"pixdim": np.asarray([1.0, 1.3, 1.2, 5.0, 0.0])}, "patient001"
        )
        np.testing.assert_allclose(pixdim, (5.0, 1.2, 1.3))

    def test_fallback_and_strict_validation(self):
        self.assertEqual(
            fallback_acdc_voxelspacing_zyx(5.0, "patient001"),
            (5.0, 1.0, 1.0),
        )
        self.assertEqual(
            validate_acdc_voxelspacing_zyx((5.0, 1.0, 1.0), "patient001"),
            (5.0, 1.0, 1.0),
        )
        for invalid in ((1.0, 2.0), (1.0, 2.0, 0.0), (1.0, np.inf, 2.0)):
            with self.assertRaises(ValueError):
                validate_acdc_voxelspacing_zyx(invalid, "patient001")
        with self.assertRaises(ValueError):
            fallback_acdc_voxelspacing_zyx(float("nan"), "patient001")


if __name__ == "__main__":
    unittest.main()
