import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from prepare_benchmark import import_dataset, rectify, speedplus_truth


class ImportTests(unittest.TestCase):
    def test_rectification_preserves_zero_distortion(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        np.testing.assert_array_equal(
            rectify(image, np.array([[10, 0, 8], [0, 10, 8], [0, 0, 1]]), [0] * 5),
            image,
        )

    def test_speedplus_passive_labels_keep_scalar_first_components(self):
        q = [2**-0.5, 0, 0, 2**-0.5]
        truth = speedplus_truth({"q": q, "r": [0, 0, 2]}, "r", "q")
        self.assertEqual(truth["quaternion_wxyz"], q)
        # The official passive DCM transposed maps +body-X into +camera-Y for this q.
        w, x, y, z = truth["quaternion_wxyz"]
        camera_y = 2 * (x * y + w * z)
        self.assertAlmostEqual(camera_y, 1.0)

    def test_import_keeps_seed_independent_from_scoring_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.fromarray(np.zeros((80, 100), dtype=np.uint8)).save(
                root / "frame.jpg"
            )
            content = {
                "labels": [
                    {
                        "filename": "frame.jpg",
                        "q_vbs2tango_true": [1, 0, 0, 0],
                        "r_Vo2To_vbs_true": [9, 8, 7],
                    }
                ],
                "camera": {
                    "cameraMatrix": [[100, 0, 49.5], [0, 100, 39.5], [0, 0, 1]],
                    "distCoeffs": [0] * 5,
                },
                "model": [[i * 0.01, 0, 0] for i in range(6)],
                "seed": {"position_m": [0, 0, 2], "quaternion_wxyz": [1, 0, 0, 0]},
            }
            for name, value in content.items():
                (root / (name + ".json")).write_text(json.dumps(value))
            options = SimpleNamespace(
                images=root,
                labels=root / "labels.json",
                camera=root / "camera.json",
                model=root / "model.json",
                seed=root / "seed.json",
                out=root / "out",
                temporal=False,
                timestamps=None,
                limit=1,
                max_width=640,
                dataset="speedplus",
                position_key="r_Vo2To_vbs_true",
                quaternion_key="q_vbs2tango_true",
            )
            manifest = json.loads(import_dataset(options).read_text())
            self.assertEqual(manifest["initial_pose"], content["seed"])
            self.assertEqual(manifest["frames"][0]["truth"]["position_m"], [9, 8, 7])
            options.temporal = True
            with self.assertRaises(ValueError):
                import_dataset(options)


if __name__ == "__main__":
    unittest.main()
