"""Experimental pixel replay with forward/backward flow and global reacquisition.

This adapter emits diagnostic hypotheses only. It never marks a pose GNC-valid.
Global localization accepts no pose prior, and every flow result re-enters PnP.
"""

import math

import cv2 as cv
import numpy as np

from image_perception import solve_correspondences


class Replay:
    def __init__(self, perception):
        self.perception = perception
        self.last_timestamp = None
        self.anchor = None
        self.last_accepted = None
        self.global_attempts = 0

    def observe(self, image, camera, geometry, captured_s):
        if (
            not math.isfinite(captured_s)
            or captured_s < 0
            or (self.last_timestamp is not None and captured_s <= self.last_timestamp)
        ):
            self.anchor = None
            return self.packet("lost", "timestamp_rejected", None, None)
        self.last_timestamp = captured_s
        age = None if self.last_accepted is None else captured_s - self.last_accepted
        if image is None:
            # No image is a feed outage, not a reusable stale camera sample.
            self.anchor = None
            return self.packet(
                "coasting" if age is not None and age <= 0.5 else "lost",
                "camera_unavailable",
                None,
                age,
            )
        if image.ndim != 2 or image.dtype != np.uint8:
            self.anchor = None
            raise ValueError("Expected grayscale uint8 image or explicit missing frame")
        camera = np.asarray(camera, dtype=float)
        geometry = np.asarray(geometry, dtype=float)
        if (
            camera.shape != (3, 3)
            or not np.isfinite(camera).all()
            or camera[0, 0] <= 0
            or camera[1, 1] <= 0
            or not np.array_equal(camera[2], [0, 0, 1])
            or camera[0, 1] != 0
        ):
            self.anchor = None
            return self.packet("lost", "invalid_calibration", None, age)
        if (
            geometry.ndim != 2
            or geometry.shape[1] != 3
            or not 6 <= len(geometry) <= 32
            or not np.isfinite(geometry).all()
        ):
            self.anchor = None
            return self.packet("lost", "invalid_geometry", None, age)
        pose, points, ids = None, None, None
        method = "global"
        if self.anchor is not None:
            previous, old_points, old_ids, timestamp, old_camera, old_geometry = (
                self.anchor
            )
            if (
                image.shape == previous.shape
                and np.array_equal(camera, old_camera)
                and np.array_equal(geometry, old_geometry)
                and 0 < captured_s - timestamp <= 0.25
            ):
                forward, ok, _ = cv.calcOpticalFlowPyrLK(
                    previous, image, old_points.astype(np.float32)[:, None], None
                )
                if forward is not None:
                    backward, back_ok, _ = cv.calcOpticalFlowPyrLK(
                        image, previous, forward, None
                    )
                    if backward is not None:
                        pixels = forward[:, 0]
                        keep = (ok[:, 0] != 0) & (back_ok[:, 0] != 0)
                        keep &= (
                            np.linalg.norm(backward[:, 0] - old_points, axis=1) <= 1.0
                        )
                        keep &= (
                            np.isfinite(pixels).all(1)
                            & (pixels[:, 0] >= 0)
                            & (pixels[:, 0] < image.shape[1])
                            & (pixels[:, 1] >= 0)
                            & (pixels[:, 1] < image.shape[0])
                        )
                        ids, points = old_ids[keep], pixels[keep]
                        pose = solve_correspondences(geometry[ids], points, camera)
                        method = "flow"
        if pose is None:
            self.global_attempts += 1
            localized, _ = self.perception.locate(image)
            if localized is not None:
                ids = localized["ids"]
                points = localized["points"][ids]
                pose = solve_correspondences(geometry[ids], points, camera)
            method = "global"
        if pose is None:
            self.anchor = None
            return self.packet("lost", "no_admitted_hypothesis", None, age, method)
        self.anchor = (
            image.copy(),
            points.copy(),
            ids.copy(),
            captured_s,
            np.array(camera, copy=True),
            geometry.copy(),
        )
        self.last_accepted = captured_s
        return self.packet("tracking", "diagnostic_hypothesis", pose, 0.0, method)

    def packet(self, state, reason, hypothesis, age, method=None):
        return {
            "state": state,
            "reason": reason,
            "measurement_age_s": age,
            "method": method,
            "hypothesis": hypothesis,
            "pose": None,
            "gnc_valid": False,
            "global_attempts": self.global_attempts,
        }
