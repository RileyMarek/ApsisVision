"""Crop geometry, visibility imbalance, and stale-prior recovery regressions."""

import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2 as cv
import numpy as np

from reacquisition import Replay


class ReplayTests(unittest.TestCase):
    def test_fresh_selection_is_disjoint_and_cannot_depend_on_pose_labels(self):
        from fresh_audit import choose

        rows = [{"filename": f"image-{i}", "pose": [i, i, i]} for i in range(50)]
        selected = choose(rows, {"image-3", "image-20"})
        changed = [
            {"filename": r["filename"], "pose": [999, 999, 999]} for r in reversed(rows)
        ]
        self.assertEqual(
            [r["filename"] for r in selected],
            [r["filename"] for r in choose(changed, {"image-3", "image-20"})],
        )
        self.assertFalse({r["filename"] for r in selected} & {"image-3", "image-20"})

    def test_outage_expiry_clock_fault_and_global_reacquisition(self):
        class Detector:
            def locate(self, image):
                return {"points": pixels, "ids": np.arange(8)}, "fixture"

        model = np.array(
            [[x, y, z] for z in (-0.1, 0.1) for y in (-0.1, 0.1) for x in (-0.1, 0.1)]
        )
        camera = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
        xyz = model + [0.03, 0.02, 1.8]
        pixels = xyz[:, :2] / xyz[:, 2:] * 300 + [160, 120]
        image = np.zeros((240, 320), np.uint8)
        for point in pixels:
            cv.drawMarker(image, tuple(point.astype(int)), 200, markerSize=7)
        replay = Replay(Detector())
        first = replay.observe(image, camera, model, 0)
        self.assertIsNotNone(first["hypothesis"])
        second = replay.observe(image, camera, model, 0.1)
        self.assertEqual(second["method"], "flow")
        self.assertEqual(replay.observe(None, camera, model, 0.2)["state"], "coasting")
        self.assertEqual(replay.observe(None, camera, model, 3.2)["state"], "lost")
        returned = replay.observe(image, camera, model, 3.3)
        self.assertEqual(returned["method"], "global")
        self.assertIsNotNone(returned["hypothesis"])
        self.assertEqual(returned["global_attempts"], 2)
        changed_camera = camera.copy()
        changed_camera[0, 2] += 1
        self.assertEqual(
            replay.observe(image, changed_camera, model, 3.4)["method"], "global"
        )
        self.assertEqual(
            replay.observe(image, changed_camera, model, 4.0)["method"], "global"
        )
        for timestamp in [3.3, 3.0, float("nan")]:
            bad = replay.observe(image, camera, model, timestamp)
            self.assertEqual(bad["reason"], "timestamp_rejected")
            self.assertIsNone(bad["pose"])
            self.assertIsNone(replay.anchor)
        self.assertFalse(returned["gnc_valid"])
        self.assertIsNone(returned["pose"])
        bad_camera = camera.copy()
        bad_camera[0, 0] = float("nan")
        self.assertEqual(
            replay.observe(image, bad_camera, model, 4.1)["reason"],
            "invalid_calibration",
        )
        self.assertEqual(replay.observe(image, camera, model, 4.2)["method"], "global")


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Install requirements-learned.txt"
)
class CropTests(unittest.TestCase):
    def test_plausible_audit_pose_stays_quarantined_without_invented_camera_times(self):
        import crop_perception as crop

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "output/model"
            out.mkdir(parents=True)
            base = root / "output/datasets/speedplus/speedplusv2"
            (base / "lightbox/images").mkdir(parents=True)
            (base / "lightbox/images/fixture.jpg").write_bytes(b"fixture")
            (base / "camera.json").write_text("{}")
            for name in ["box", "localizer"]:
                (out / (name + ".pt")).write_bytes(b"not loaded")
            (out / "training.json").write_text(
                json.dumps(
                    {
                        "geometry_sha256": "fixture",
                        "training_hashes": [],
                        "development_hashes": [],
                    }
                )
            )
            (root / "output/datasets/selection.json").write_text("{}")
            sample = {
                "filename": "fixture.jpg",
                "r_Vo2To_vbs_true": [0, 0, 2],
                "q_vbs2tango_true": [1, 0, 0, 0],
            }
            selection = {
                "geometry_sha256": "fixture",
                "groups": {"speedplus/lightbox/test": [sample]},
            }
            prediction = {"position_m": [0, 0, 2], "quaternion_wxyz": [1, 0, 0, 0]}
            with (
                patch.object(crop, "ROOT", root),
                patch.object(
                    crop,
                    "context",
                    return_value=(selection, None, None, np.zeros((11, 3))),
                ),
                patch.object(
                    crop,
                    "prepared",
                    return_value=(np.zeros((240, 320), np.uint8), np.eye(3)),
                ),
                patch.object(crop, "Perception") as factory,
            ):
                factory.return_value.infer.return_value = (
                    prediction,
                    {"landmarks": 11},
                )
                crop.evaluate(SimpleNamespace(threads=1, out=out, selection=None))
            packet = json.loads((out / "speedplus-lightbox-test.jsonl").read_text())
            self.assertIsNone(packet["pose"])
            self.assertFalse(packet["temporal_replay_allowed"])
            self.assertNotIn("captured_s", packet)
            self.assertFalse(
                json.loads((out / "report.json").read_text())["gnc_output_enabled"]
            )

    def test_padded_crop_cannot_make_out_of_fov_landmarks_visible(self):
        import torch

        from crop_perception import source_visibility

        flags = torch.tensor([[1, 1, 0, -1]])
        in_source = torch.tensor([[True, False, False, False]])
        self.assertEqual(source_visibility(flags, in_source).tolist(), [[1, 0, 0, -1]])

    def test_pixel_center_roundtrip_and_warp_including_outside_image(self):
        from crop_perception import Crop

        crop = Crop(-15.3, 8.7, 300, 225)
        points = np.array([[0, 0], [100.4, 85.2], [319, 239]])
        np.testing.assert_allclose(
            crop.to_pixels(crop.to_grid(points)), points, atol=1e-12
        )
        image = np.zeros((240, 320), np.uint8)
        image[100, 150] = 255
        warped = crop.image(image)
        y, x = np.unravel_index(warped.argmax(), warped.shape)
        expected = (crop.to_grid([[150, 100]])[0] + 0.5) * 4 - 0.5
        np.testing.assert_allclose([x, y], expected, atol=1)
        for box in [[0.5, 0.5, 0, 0.3], [float("nan"), 0.5, 0.2, 0.2]]:
            with self.assertRaises(ValueError):
                Crop.from_box(box, image.shape)
        with self.assertRaises(ValueError):
            Crop.from_box([0.5, 0.5, 0.5, 1.51], image.shape)
        partial = Crop.from_box([0.5, 0.5, 0.5, 1.51], image.shape, max_extent=2.0)
        np.testing.assert_allclose(
            partial.to_pixels(partial.to_grid(points)), points, atol=1e-12
        )

    def test_balanced_visibility_prevents_all_visible_optimum(self):
        import torch

        from crop_perception import balanced_loss

        heatmap = torch.zeros(1, 11, 48, 64, requires_grad=True)
        logits = torch.zeros(1, 11, requires_grad=True)
        flags = torch.tensor([[1] * 9 + [0, 0]])
        balanced_loss((heatmap, logits), torch.ones(1, 11, 2) * 20, flags).backward()
        self.assertAlmostEqual(
            float(logits.grad[0, :9].sum()), -float(logits.grad[0, 9:].sum()), places=6
        )
        self.assertEqual(float(heatmap.grad[0, 9:].abs().sum()), 0)

    def test_inference_accepts_no_prior_and_blank_cannot_localize(self):
        from crop_perception import Perception

        self.assertEqual(
            list(inspect.signature(Perception.locate).parameters), ["self", "image"]
        )
        instance = Perception.__new__(Perception)
        self.assertEqual(
            instance.locate(np.zeros((240, 320), np.uint8)), (None, "no_contrast")
        )


if __name__ == "__main__":
    unittest.main()
