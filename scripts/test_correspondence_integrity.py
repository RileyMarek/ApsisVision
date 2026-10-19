"""Consistency rejects unstable evidence, but cannot certify correlated bias."""

import unittest

import numpy as np

from correspondence_integrity import assess
from image_perception import solve_correspondences


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.geometry = np.array(
            [
                [x, y, z]
                for z in (-0.15, 0.15)
                for y in (-0.25, 0.25)
                for x in (-0.2, 0.2)
            ]
        )
        self.camera = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])

    def project(self, translation):
        xyz = self.geometry + translation
        return xyz[:, :2] / xyz[:, 2:] * 500 + [320, 240]

    def test_repeatable_geometry_passes_with_bounded_fit_count(self):
        pixels = self.project([0.1, 0.05, 3.0])
        pose = solve_correspondences(self.geometry, pixels, self.camera)
        result = assess(self.geometry, [pixels] * 3, self.camera, pose)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["fit_count"], 10)

    def test_displaced_identity_and_duplicate_identity_fail(self):
        pixels = self.project([0.1, 0.05, 3.0])
        pose = solve_correspondences(self.geometry, pixels, self.camera)
        moved = pixels.copy()
        moved[0] += [8, 0]
        result = assess(self.geometry, [pixels, moved, pixels], self.camera, pose)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "crop_correspondence_unstable")
        geometry = self.geometry.copy()
        geometry[0] = geometry[1]
        self.assertEqual(
            assess(geometry, [pixels] * 3, self.camera, pose)["reason"],
            "duplicate_identity",
        )

    def test_correlated_wrong_pose_can_pass_and_must_not_be_called_certified(self):
        pixels = self.project([0.1, 0.05, 4.0])
        pose = solve_correspondences(self.geometry, pixels, self.camera)
        result = assess(self.geometry, [pixels] * 3, self.camera, pose)
        self.assertTrue(result["accepted"])
        self.assertGreater(abs(pose["position_m"][2] - 3.0), 0.9)

    def test_nonfinite_baseline_and_degenerate_camera_fail_closed(self):
        pixels = self.project([0.1, 0.05, 3.0])
        pose = solve_correspondences(self.geometry, pixels, self.camera)
        bad = {**pose, "position_m": [0, 0, float("nan")]}
        self.assertFalse(
            assess(self.geometry, [pixels] * 3, self.camera, bad)["accepted"]
        )
        camera = self.camera.copy()
        camera[0, 0] = 0
        self.assertFalse(assess(self.geometry, [pixels] * 3, camera, pose)["accepted"])


if __name__ == "__main__":
    unittest.main()
