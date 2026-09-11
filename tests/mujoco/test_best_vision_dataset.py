import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arena_mujoco.best_vision import (
    CLASS_NAMES, DATASET_SCHEMA, VisionConfig, generate_frozen_dataset, sha256_file,
)


class FrozenVisionDatasetTests(unittest.TestCase):
    def test_reuses_json_round_trip_without_regenerating_and_rejects_changed_data(self):
        config = VisionConfig()
        counts = {"train": 1, "validation": 1, "test": 1}
        seeds = {"train": 11, "validation": 22, "test": 33}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = {}
            for split in counts:
                data = root / f"{split}.npy"
                data.write_bytes(f"frozen-{split}".encode())
                splits[split] = {"files": {"images": {
                    "file": data.name, "sha256": sha256_file(data)}}}
            manifest = {
                "schema_version": DATASET_SCHEMA, "classes": list(CLASS_NAMES),
                "vision_config": asdict(config), "split_counts": counts,
                "split_seeds": seeds, "splits": splits,
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            with patch("arena_mujoco.best_vision.MujocoVisionGenerator") as generator:
                result = generate_frozen_dataset(root, config, counts, seeds, root)
                self.assertEqual(result, json.loads(path.read_text()))
                generator.assert_not_called()
                (root / "test.npy").write_bytes(b"replaced")
                with self.assertRaisesRegex(ValueError, "Frozen test images data is missing or changed"):
                    generate_frozen_dataset(root, config, counts, seeds, root)
                generator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
