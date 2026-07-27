import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

try:
    import torch
except ImportError:  # The source-only CI environment may omit project dependencies.
    torch = None

if torch is not None:
    from utils.checkpointing import load_checkpoint, save_checkpoint
    from utils.runtime import capture_rng_state


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointingTests(unittest.TestCase):
    @staticmethod
    def _model(in_channels=1, out_channels=2):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n_channels = in_channels
                self.n_classes = out_channels
                self.bilinear = False
                self.stem = torch.nn.Conv2d(in_channels, 4, 1)
                self.body = torch.nn.Conv2d(4, 4, 1)
                self.head = torch.nn.Conv2d(4, out_channels, 1)

            def forward(self, value):
                return self.head(self.body(self.stem(value)))

        return TinyModel()

    def test_structured_save_and_strict_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            source = self._model()
            optimizer = torch.optim.RMSprop(source.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
            train_generator = torch.Generator().manual_seed(1234)
            validation_generator = torch.Generator().manual_seed(1235)
            save_checkpoint(
                path,
                source,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=7,
                best_mean_dice=0.75,
                dataset="Synapse",
                img_size=224,
                class_names=("Background", "Organ"),
                normalization="none",
                arguments={"seed": 1234},
                extra={
                    "rng_state": capture_rng_state(
                        train_generator=train_generator,
                        validation_generator=validation_generator,
                    )
                },
            )

            target = self._model()
            target_optimizer = torch.optim.RMSprop(target.parameters(), lr=2e-3)
            target_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                target_optimizer, mode="max"
            )
            result = load_checkpoint(
                target,
                path,
                mode="resume",
                optimizer=target_optimizer,
                scheduler=target_scheduler,
                dataset="Synapse",
            )
            self.assertTrue(result.structured)
            self.assertEqual(result.epoch, 7)
            self.assertEqual(result.next_epoch, 8)
            self.assertAlmostEqual(result.best_mean_dice, 0.75)
            self.assertTrue(result.optimizer_restored)
            self.assertTrue(result.scheduler_restored)
            self.assertIsNotNone(result.rng_state)
            for source_value, target_value in zip(
                source.state_dict().values(), target.state_dict().values()
            ):
                self.assertTrue(torch.equal(source_value, target_value))

    def test_raw_module_prefixed_state_and_legacy_mask_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            source = self._model()
            legacy = OrderedDict(
                ("module." + key, value.clone())
                for key, value in source.state_dict().items()
            )
            legacy["mask_values"] = [0, 1]
            torch.save(legacy, path)

            target = self._model()
            result = load_checkpoint(target, path, mode="test", dataset="Synapse")
            self.assertFalse(result.structured)
            self.assertEqual(result.mask_values, [0, 1])
            self.assertFalse(result.partial)

    def test_metadata_mismatch_fails_before_test_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_checkpoint(path, self._model(), dataset="Synapse")
            with self.assertRaisesRegex(ValueError, "dataset"):
                load_checkpoint(self._model(), path, mode="test", dataset="ACDC")

    def test_partial_initialization_reports_every_incompatible_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pth"
            source = self._model(in_channels=1, out_channels=2)
            save_checkpoint(path, source, dataset="Synapse")

            target = self._model(in_channels=3, out_channels=5)
            result = load_checkpoint(
                target,
                path,
                mode="init",
                dataset="Cataract1k",
                allow_partial_init=True,
            )
            self.assertIn("stem.weight", result.shape_mismatches)
            self.assertIn("head.weight", result.shape_mismatches)
            self.assertIn("head.bias", result.shape_mismatches)
            self.assertIn("stem.weight", result.missing_keys)
            self.assertIn("head.weight", result.skipped_keys)
            self.assertIn("dataset", " ".join(result.metadata_mismatches))
            self.assertTrue(result.partial)
            self.assertTrue(torch.equal(source.body.weight, target.body.weight))


if __name__ == "__main__":
    unittest.main()
