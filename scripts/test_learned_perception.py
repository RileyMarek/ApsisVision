"""Optional CPU learning tests; CI runs these in a separate dependency job."""

import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Install requirements-learned.txt"
)
class LearnedTests(unittest.TestCase):
    def test_even_a_plausible_unqualified_hypothesis_is_quarantined_from_replay(self):
        import learned_perception as learned

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "output/learned"
            out.mkdir(parents=True)
            speed = root / "output/datasets/speedplus/speedplusv2"
            (speed / "lightbox/images").mkdir(parents=True)
            (speed / "lightbox/images/example.jpg").write_bytes(b"fixture")
            (speed / "camera.json").write_text("{}")
            (out / "heatmap.pt").write_bytes(b"not-loaded")
            (out / "training.json").write_text(
                json.dumps(
                    {
                        "geometry_sha256": "test",
                        "training_hashes": [],
                        "development_hashes": [],
                        "parameters": 101803,
                    }
                )
            )
            sample = {
                "filename": "example.jpg",
                "r_Vo2To_vbs_true": [0, 0, 2],
                "q_vbs2tango_true": [1, 0, 0, 0],
            }
            (root / "output/datasets/selection.json").write_text(
                json.dumps(
                    {
                        "geometry_sha256": "test",
                        "groups": {"speedplus/lightbox/test": [sample]},
                    }
                )
            )
            prediction = {"position_m": [0, 0, 2], "quaternion_wxyz": [1, 0, 0, 0]}
            with (
                patch.object(learned, "ROOT", root),
                patch.object(learned, "HeatmapNet"),
                patch.object(learned.torch, "load", return_value={}),
                patch.object(
                    learned,
                    "loadmat",
                    return_value={"tango3Dpoints": np.zeros((3, 11))},
                ),
                patch.object(
                    learned,
                    "prepared",
                    return_value=(np.zeros((480, 640), np.uint8), np.eye(3)),
                ),
                patch.object(learned, "infer", return_value=(prediction, 11)),
            ):
                learned.evaluate(SimpleNamespace(threads=1, runtime="torch"))
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(
                report["groups"]["speedplus/lightbox/test"]["estimated"], 1
            )
            self.assertFalse(report["gnc_output_enabled"])
            packet = json.loads(
                (out / "speedplus-lightbox-test-torch.jsonl").read_text()
            )
            self.assertIsNone(packet["pose"])

    def test_heatmap_shape_gradients_and_truth_free_inference(self):
        import torch

        from learned_perception import HeatmapNet, heatmap_loss, infer

        torch.set_num_threads(2)
        torch.manual_seed(42)
        network = HeatmapNet()
        logits = network(torch.rand(2, 1, 192, 256))
        self.assertEqual(tuple(logits.shape), (2, 11, 48, 64))
        loss = heatmap_loss(logits, torch.ones(2, 11, 2) * 20)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in network.parameters()
            )
        )
        self.assertEqual(
            list(inspect.signature(infer).parameters),
            ["image", "camera_matrix", "model", "network"],
        )

    def test_heatmap_decode_preserves_pixel_centers_and_abstains_on_uniform_maps(self):
        from learned_perception import decode

        logits = np.zeros((11, 48, 64))
        _, confidence = decode(logits, (480, 640))
        self.assertTrue((confidence < 0.05).all())
        logits[:, 20, 10] = 100
        points, confidence = decode(logits, (480, 640))
        np.testing.assert_allclose(points, np.tile([104.5, 204.5], (11, 1)), atol=1e-5)
        self.assertTrue((confidence > 0.99).all())
        with self.assertRaises(ValueError):
            decode(np.full((11, 48, 64), np.nan), (480, 640))

    def test_development_split_is_disjoint_and_order_independent(self):
        from learned_perception import split_training

        rows = [{"filename": f"frame-{i}.jpg"} for i in range(100)]
        training, development = split_training(rows)
        self.assertEqual((len(training), len(development)), (80, 20))
        self.assertFalse(
            {r["filename"] for r in training} & {r["filename"] for r in development}
        )
        self.assertEqual((training, development), split_training(list(reversed(rows))))


if __name__ == "__main__":
    unittest.main()
