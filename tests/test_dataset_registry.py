import unittest

try:
    import torch  # noqa: F401
    from utils.dataset_registry import (
        canonicalize_dataset_name,
        get_dataset_spec,
        resolve_dataset_paths,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DatasetRegistryTests(unittest.TestCase):
    def test_registry_model_contracts(self):
        expected = {
            "Synapse": (1, 9, "volume"),
            "ACDC": (1, 4, "volume"),
            "Cataract1k": (3, 5, "present_class_frame"),
        }
        for name, contract in expected.items():
            spec = get_dataset_spec(name)
            self.assertEqual((spec.input_channels, spec.num_classes, spec.protocol), contract)
            self.assertEqual(len(spec.class_names), spec.num_classes)

    def test_alias_is_canonicalized_immediately(self):
        self.assertEqual(canonicalize_dataset_name("Catrakt1k"), "Cataract1k")

    def test_defaults_and_overrides(self):
        synapse = resolve_dataset_paths("Synapse")
        self.assertEqual(str(synapse.train_root), "/data/halyusuf/data/Synapse/train_npz")
        self.assertEqual(str(synapse.volume_root), "/data/halyusuf/data/Synapse/test_vol_h5")
        self.assertTrue(str(synapse.list_dir).endswith("lists/lists_Synapse"))
        acdc = resolve_dataset_paths("ACDC", root="/tmp/acdc-override")
        self.assertEqual(acdc.train_root, acdc.volume_root)


if __name__ == "__main__":
    unittest.main()
