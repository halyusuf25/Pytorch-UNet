import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from unet import UNet
    from utils.model_output import _extract_logits


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelChannelTests(unittest.TestCase):
    def test_synapse_unet_shape(self):
        model = UNet(n_channels=1, n_classes=9)
        self.assertEqual(tuple(model(torch.randn(1, 1, 224, 224)).shape), (1, 9, 224, 224))

    def test_acdc_unet_shape(self):
        model = UNet(n_channels=1, n_classes=4)
        self.assertEqual(tuple(model(torch.randn(1, 1, 224, 224)).shape), (1, 4, 224, 224))

    def test_cataract_unet_shape(self):
        model = UNet(n_channels=3, n_classes=5)
        self.assertEqual(tuple(model(torch.randn(1, 3, 224, 224)).shape), (1, 5, 224, 224))

    def test_extract_logits_containers(self):
        logits = torch.randn(1, 4, 8, 8)
        self.assertIs(_extract_logits(logits), logits)
        self.assertIs(_extract_logits((logits, "aux")), logits)
        self.assertIs(_extract_logits([logits]), logits)
        self.assertIs(_extract_logits({"logits": logits}), logits)
        self.assertIs(_extract_logits({"out": logits}), logits)


if __name__ == "__main__":
    unittest.main()
