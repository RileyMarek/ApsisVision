"""Bounded same-image consistency checks, not a calibrated correctness probability.

Correlated semantic errors can pass every check here. Keep hypotheses quarantined;
score wrong emissions AND acquisition coverage on untouched imagery.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from image_perception import solve_correspondences

POLICY = {
    "version": "roi_stability_v2",
    "min_landmarks": 7,
    "max_crop_delta_px": 3.0,
    "max_translation_spread_m": 0.05,
    "max_attitude_spread_deg": 2.0,
    "min_point_separation_px": 1.0,
    "crop_variants": [[0.94, -0.02, 0.02], [1.06, 0.02, -0.02]],
}


def assess(geometry, samples, camera, baseline):
    geometry, samples, camera = (
        np.asarray(x, dtype=float) for x in (geometry, samples, camera)
    )
    report = {"accepted": False, "reason": "malformed_correspondences", "fit_count": 0}
    if (
        geometry.ndim != 2
        or geometry.shape[1] != 3
        or not 7 <= len(geometry) <= 11
        or samples.shape != (3, len(geometry), 2)
        or camera.shape != (3, 3)
        or not all(np.isfinite(x).all() for x in (geometry, samples, camera))
        or camera[0, 0] <= 0
        or camera[1, 1] <= 0
        or not np.allclose(camera[2], [0, 0, 1], atol=1e-12, rtol=0)
    ):
        return report
    if not isinstance(baseline, dict):
        return report
    position = np.asarray(baseline.get("position_m", []), dtype=float)
    quaternion = np.asarray(baseline.get("quaternion_wxyz", []), dtype=float)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(quaternion).all()
        or abs(np.linalg.norm(quaternion) - 1) > 1e-6
    ):
        return report
    if len(np.unique(geometry, axis=0)) != len(geometry):
        return {**report, "reason": "duplicate_identity"}
    if np.linalg.matrix_rank(geometry - geometry.mean(axis=0)) < 3:
        return {**report, "reason": "weak_geometry"}
    for points in samples:
        separation = np.linalg.norm(points[:, None] - points[None], axis=2)
        np.fill_diagonal(separation, np.inf)
        if separation.min() < POLICY["min_point_separation_px"]:
            return {**report, "reason": "coincident_image_points"}
    delta = float(np.linalg.norm(samples[1:] - samples[0], axis=2).max())
    report["max_crop_delta_px"] = delta
    if delta > POLICY["max_crop_delta_px"]:
        return {**report, "reason": "crop_correspondence_unstable"}
    # The caller supplies an already admitted baseline. Every extra fit retains
    # the existing 3 px inlier RMS and all-correspondence mean gates.
    fits = [solve_correspondences(geometry, points, camera) for points in samples[1:]]
    for i in range(len(geometry)):
        keep = np.arange(len(geometry)) != i
        fits.append(solve_correspondences(geometry[keep], samples[0, keep], camera))
    report["fit_count"] = len(fits)
    if any(p is None for p in fits):
        return {**report, "reason": "redundant_pose_rejected"}

    def rotation(pose):
        q = pose["quaternion_wxyz"]
        return Rotation.from_quat([*q[1:], q[0]])

    anchor = rotation(baseline)
    spread = max(
        float(np.linalg.norm(np.asarray(p["position_m"]) - position)) for p in fits
    )
    angle = max(
        float(np.rad2deg((rotation(p) * anchor.inv()).magnitude())) for p in fits
    )
    report.update(max_translation_spread_m=spread, max_attitude_spread_deg=angle)
    if (
        spread > POLICY["max_translation_spread_m"]
        or angle > POLICY["max_attitude_spread_deg"]
    ):
        return {**report, "reason": "pose_sensitivity"}
    return {**report, "accepted": True, "reason": "repeatable_hypothesis"}
