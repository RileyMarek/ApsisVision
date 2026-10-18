"""Positive controls distinguish an invalid pose solver from failed correspondence."""

import inspect
import io
import unittest
import zipfile

import cv2 as cv
import numpy as np

from envelope_campaign import score, wilson
from fetch_benchmark import RemoteZip
from image_perception import infer, score_pose, solve_correspondences


class IntegrityTests(unittest.TestCase):
    def test_seedless_pnp_positive_control_and_outliers(self):
        rng = np.random.default_rng(42)
        model = rng.uniform(-0.2, 0.2, (11, 3))
        k = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
        pixels, _ = cv.projectPoints(
            model, np.array([0.2, -0.1, 0.3]), np.array([0.05, -0.02, 3.0]), k, None
        )
        pixels = pixels.reshape(-1, 2)
        estimate = solve_correspondences(model, pixels, k)
        self.assertIsNotNone(estimate)
        np.testing.assert_allclose(estimate["position_m"], [0.05, -0.02, 3], atol=1e-5)
        self.assertGreaterEqual(estimate["inliers"], 9)
        pixels[-2:] += 80
        self.assertIsNone(
            solve_correspondences(model, pixels, k),
            "RANSAC subset support cannot bypass all-correspondence MRE admission",
        )
        self.assertIsNone(solve_correspondences(model[:5], pixels[:5], k))

    def test_blank_image_rejects_and_inference_has_no_truth_argument(self):
        self.assertEqual(
            list(inspect.signature(infer).parameters), ["image", "k", "appearance"]
        )
        appearance = {
            "model": np.zeros((11, 3)),
            "descriptors": np.zeros((11, 32), np.uint8),
            "identities": np.arange(11),
        }
        self.assertEqual(
            infer(np.zeros((480, 640), np.uint8), np.eye(3), appearance), (None, 0)
        )
        self.assertEqual(
            score_pose(None, {}), {"position_error_m": None, "attitude_error_deg": None}
        )

    def test_independent_trial_interval_does_not_claim_zero_risk(self):
        self.assertGreater(wilson(0, 128)[1], 0.02)
        self.assertLess(wilson(0, 128)[1], 0.05)
        self.assertGreater(wilson(9, 128)[1], 0.12)

    def test_false_valid_counts_valid_errors_only(self):
        frame = {
            "valid": True,
            "error": 0.2,
            "angle": 0,
            "p": [0, 0, 2],
            "truth_p": [0, 0, 2.2],
            "sigma": [0.01] * 12,
            "t": 6.0,
            "vision_us": 10,
            "filter_us": 5,
        }
        result = score(
            {"frames": [frame, dict(frame, valid=False)], "summary": {}},
            {
                "position_error_m": 0.15,
                "attitude_error_deg": 8,
                "vision_deadline_ms": 100,
                "filter_deadline_ms": 20,
            },
        )
        self.assertEqual(result["false_valid_ticks"], 1)
        self.assertTrue(result["false_valid_trial"])
        self.assertEqual(result["valid_fraction"], 0.5)

    def test_remote_member_verifies_crc_and_declared_size(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("fixture.txt", b"calibration fixture")
        with zipfile.ZipFile(data) as archive:
            item = archive.infolist()[0]
        remote = RemoteZip.__new__(RemoteZip)
        payload = data.getvalue()
        remote.range = lambda start, size: payload[start : start + size]
        self.assertEqual(remote.member(item), b"calibration fixture")
        item.CRC ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            remote.member(item)
        item.file_size = 65 * 1024**2
        with self.assertRaisesRegex(ValueError, "oversized"):
            remote.member(item)


if __name__ == "__main__":
    unittest.main()
