import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from utils.medical_losses import DiceLoss


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MedicalDiceLossTests(unittest.TestCase):
    @staticmethod
    def _manual_loss(logits, target, weights=None):
        probabilities = torch.softmax(logits, dim=1)
        one_hot = torch.cat(
            [(target == class_id).unsqueeze(1) for class_id in range(logits.shape[1])],
            dim=1,
        ).float()
        if weights is None:
            weights = [1] * logits.shape[1]
        total = 0.0
        for class_id in range(logits.shape[1]):
            score = probabilities[:, class_id]
            truth = one_hot[:, class_id]
            overlap = torch.sum(score * truth)
            target_sum = torch.sum(truth * truth)
            score_sum = torch.sum(score * score)
            class_loss = 1.0 - (
                (2.0 * overlap + 1e-5)
                / (target_sum + score_sum + 1e-5)
            )
            total += class_loss * weights[class_id]
        return total / logits.shape[1]

    def test_matches_reference_classwise_all_class_arithmetic(self):
        logits = torch.tensor(
            [
                [
                    [[2.0, -0.5], [0.0, 0.2]],
                    [[-1.0, 1.0], [0.5, -0.4]],
                    [[0.0, 0.2], [1.5, 0.7]],
                ]
            ],
            dtype=torch.float64,
        )
        target = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.long)
        actual = DiceLoss(3)(logits, target, softmax=True)
        expected = self._manual_loss(logits, target)
        self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-12))

    def test_default_weights_are_equal_and_background_is_included(self):
        logits = torch.tensor(
            [[[[0.0, 0.0]], [[2.0, 2.0]]]],
            dtype=torch.float32,
        )
        target = torch.ones((1, 1, 2), dtype=torch.long)
        loss = DiceLoss(2)(logits, target, softmax=True)
        expected = self._manual_loss(logits, target)
        foreground_only = DiceLoss(2)._dice_loss(
            torch.softmax(logits, dim=1)[:, 1],
            (target == 1).float(),
        )
        self.assertTrue(torch.allclose(loss, expected))
        self.assertGreater(float(loss), float(foreground_only) / 2.0)

    def test_explicit_weights_still_divide_by_number_of_classes(self):
        logits = torch.tensor(
            [[[[1.0]], [[0.0]], [[-1.0]]]],
            dtype=torch.float32,
        )
        target = torch.tensor([[[2]]], dtype=torch.long)
        weights = [0.5, 2.0, 3.0]
        actual = DiceLoss(3)(logits, target, weight=weights, softmax=True)
        expected = self._manual_loss(logits, target, weights)
        self.assertTrue(torch.allclose(actual, expected))

    def test_shape_mismatch_uses_reference_assertion(self):
        inputs = torch.randn(1, 3, 4, 4)
        target = torch.zeros(1, 3, 4, dtype=torch.long)
        with self.assertRaisesRegex(AssertionError, "shape do not match"):
            DiceLoss(3)(inputs, target)

    def test_loss_backpropagates_through_softmax_logits(self):
        logits = torch.randn(2, 4, 3, 3, requires_grad=True)
        target = torch.randint(0, 4, (2, 3, 3))
        loss = DiceLoss(4)(logits, target, softmax=True)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
