"""Desktop ORB-to-known-landmark baseline with seedless PnP. Evaluation truth is scoring-only.

Training annotations register descriptors to geometry. Inference accepts pixels,
calibration, and a saved appearance map; it has no truth/pose-prior argument.
"""

from __future__ import annotations

import hashlib
import json
import time

import cv2 as cv
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

from fetch_benchmark import ROOT


def prepared(path, calibration):
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    matrix = np.asarray(calibration["cameraMatrix"], np.float64)
    corrected = cv.undistort(
        image, matrix, np.asarray(calibration["distCoeffs"], np.float64)
    )
    scale = 640 / image.shape[1]
    new = cv.resize(
        corrected, (640, round(image.shape[0] * scale)), interpolation=cv.INTER_AREA
    )
    updated = matrix.copy()
    updated[:2] *= scale
    updated[0, 2] = (matrix[0, 2] + 0.5) * scale - 0.5
    updated[1, 2] = (matrix[1, 2] + 0.5) * scale - 0.5
    return new, updated


def features(image):
    detector = cv.ORB_create(
        nfeatures=1500, scaleFactor=1.2, nlevels=8, edgeThreshold=12, fastThreshold=10
    )
    return detector.detectAndCompute(image, None)


def register_training(samples, root, calibration, model):
    descriptors = []
    identities = []
    hashes = []
    for sample in samples:
        path = root / sample["filename"]
        image, k = prepared(path, calibration)
        keypoints, desc = features(image)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if desc is None:
            continue
        q = sample["q_vbs2tango_true"]
        rotation = Rotation.from_quat([*q[1:], q[0]])
        xyz = rotation.apply(model) + np.asarray(sample["r_Vo2To_vbs_true"])
        projected = xyz[:, :2] / xyz[:, 2, None]
        projected = projected @ k[:2, :2].T + k[:2, 2]
        pixels = np.asarray([point.pt for point in keypoints])
        used = set()
        for identity, point in enumerate(projected):
            if xyz[identity, 2] <= 0:
                continue
            distance = np.linalg.norm(pixels - point, axis=1)
            index = int(np.argmin(distance))
            if distance[index] <= 5.0 and index not in used:
                used.add(index)
                descriptors.append(desc[index])
                identities.append(identity)
    if not descriptors:
        raise ValueError("No registered training features")
    return {
        "descriptors": np.asarray(descriptors, np.uint8),
        "identities": np.asarray(identities, np.int32),
        "model": np.asarray(model, np.float64),
        "training_hashes": hashes,
    }


def solve_correspondences(model, pixels, k):
    if len(model) < 6:
        return None
    cv.setRNGSeed(1729)
    ok, rvec, tvec, inliers = cv.solvePnPRansac(
        np.asarray(model, np.float64),
        np.asarray(pixels, np.float64),
        k,
        None,
        iterationsCount=500,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        return None
    ids = inliers.ravel()
    rvec, tvec = cv.solvePnPRefineLM(
        np.asarray(model, np.float64)[ids],
        np.asarray(pixels, np.float64)[ids],
        k,
        None,
        rvec,
        tvec,
    )
    projected, _ = cv.projectPoints(model, rvec, tvec, k, None)
    error = np.linalg.norm(
        projected.reshape(-1, 2)[ids] - np.asarray(pixels)[ids], axis=1
    )
    rotation, _ = cv.Rodrigues(rvec)
    depth = (np.asarray(model) @ rotation.T + tvec.ravel())[:, 2]
    if (
        not np.isfinite(tvec).all()
        or np.any(depth <= 0)
        or np.sqrt(np.mean(error**2)) > 3
    ):
        return None
    q = Rotation.from_matrix(rotation).as_quat()
    return {
        "position_m": tvec.ravel().tolist(),
        "quaternion_wxyz": [float(q[3]), *q[:3].tolist()],
        "inliers": len(ids),
        "rms_px": float(np.sqrt(np.mean(error**2))),
    }


def infer(image, k, appearance):
    keypoints, desc = features(image)
    if desc is None:
        return None, 0
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    distances = np.full((len(desc), len(appearance["model"])), 256.0)
    for identity in range(len(appearance["model"])):
        reference = appearance["descriptors"][appearance["identities"] == identity]
        if len(reference):
            for match in matcher.match(desc, reference):
                distances[match.queryIdx, identity] = match.distance
    ranked = np.argsort(distances, axis=1)
    pairs = []
    for query in range(len(desc)):
        first, second = ranked[query, :2]
        if (
            distances[query, first] < 0.75 * distances[query, second]
            and distances[query, first] <= 64
        ):
            pairs.append((float(distances[query, first]), int(first), query))
    selected = {}
    for _, identity, query in sorted(pairs):
        if identity not in selected:
            selected[identity] = query
    ids = list(selected)
    points = np.asarray([keypoints[selected[i]].pt for i in ids], np.float64)
    return solve_correspondences(appearance["model"][ids], points, k), len(ids)


def score_pose(prediction, truth):
    if prediction is None:
        return {"position_error_m": None, "attitude_error_deg": None}

    def rotation(q):
        return Rotation.from_quat([*q[1:], q[0]])

    return {
        "position_error_m": float(
            np.linalg.norm(
                np.asarray(prediction["position_m"]) - truth["r_Vo2To_vbs_true"]
            )
        ),
        "attitude_error_deg": float(
            np.degrees(
                (
                    rotation(prediction["quaternion_wxyz"])
                    * rotation(truth["q_vbs2tango_true"]).inv()
                ).magnitude()
            )
        ),
    }


def main():
    selected = json.loads((ROOT / "output/datasets/selection.json").read_text())
    speed = ROOT / "output/datasets/speedplus/speedplusv2"
    camera = json.loads((speed / "camera.json").read_text())
    model = loadmat(ROOT / "output/datasets/tangoPoints.mat")["tango3Dpoints"].T.astype(
        np.float64
    )
    training = selected["groups"]["speedplus/synthetic/train"]
    appearance = register_training(training, speed / "synthetic/images", camera, model)
    out = ROOT / "output/real-images"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "appearance.npz", **appearance)
    groups = {}
    for key, samples in selected["groups"].items():
        if key.endswith("/train"):
            continue
        dataset, domain, split = key.split("/")
        root = (
            speed / domain / "images"
            if dataset == "speedplus"
            else ROOT / "output/datasets/shirt" / domain / split / "images"
        )
        calibration = (
            camera
            if dataset == "speedplus"
            else json.loads((ROOT / "output/datasets/shirt/camera.json").read_text())
        )
        rows = []
        packets = []
        step = 5.0 if dataset == "shirt" else 0.1
        for index, sample in enumerate(samples):
            path = root / sample["filename"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in appearance["training_hashes"]:
                raise ValueError("Training/evaluation image leakage")
            image, k = prepared(path, calibration)
            start = time.perf_counter_ns()
            prediction, matches = infer(image, k, appearance)
            elapsed = (time.perf_counter_ns() - start) / 1000
            # Prediction is fixed before the scoring-only annotation is inspected.
            row = {
                "image": sample["filename"],
                "sha256": digest,
                "matches": matches,
                "prediction": prediction,
                "vision_us": elapsed,
                **score_pose(prediction, sample),
            }
            rows.append(row)
            packets.append(
                {
                    "captured_s": index * step,
                    "delivered_s": index * step,
                    "pose": (
                        None
                        if prediction is None
                        else {
                            k: v
                            for k, v in prediction.items()
                            if k in ("position_m", "quaternion_wxyz")
                        }
                    ),
                }
            )
        estimated = [r for r in rows if r["prediction"] is not None]
        errors = [r["position_error_m"] for r in estimated]
        angles = [r["attitude_error_deg"] for r in estimated]
        groups[key] = {
            "frames": len(rows),
            "estimated": len(estimated),
            "failed": len(rows) - len(estimated),
            "position_rms_m": (
                float(np.sqrt(np.mean(np.square(errors)))) if errors else None
            ),
            "attitude_rms_deg": (
                float(np.sqrt(np.mean(np.square(angles)))) if angles else None
            ),
            "results": rows,
        }
        name = key.replace("/", "-")
        (out / (name + ".jsonl")).write_text(
            "".join(json.dumps(p) + "\n" for p in packets)
        )
        print(key, {k: v for k, v in groups[key].items() if k != "results"}, flush=True)
    report = {
        "method": "ORB descriptors registered to 11 Tango landmarks on 500 synthetic training images; seedless EPNP RANSAC; research baseline",
        "no_pose_prior": True,
        "training_images": len(training),
        "registered_descriptors": len(appearance["descriptors"]),
        "opencv": cv.__version__,
        "selection_seed": selected["seed"],
        "geometry_sha256": selected["geometry_sha256"],
        "groups": groups,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
