"""Import explicitly supplied SPEED+/SHIRT images into the Rust benchmark schema.

No downloads, model guessing, seed-from-truth initialization, or inferred sequence timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def rectify(
    pixels: np.ndarray, matrix: np.ndarray, coefficients: list[float]
) -> np.ndarray:
    """Inverse image resampling using the OpenCV five-coefficient Brown-Conrady model."""
    if not any(coefficients):
        return pixels.copy()
    if len(coefficients) != 5:
        raise ValueError("expected [k1,k2,p1,p2,k3] distortion coefficients")
    k1, k2, p1, p2, k3 = coefficients
    height, width = pixels.shape
    v, u = np.indices((height, width), dtype=np.float64)
    x, y = (u - matrix[0, 2]) / matrix[0, 0], (v - matrix[1, 2]) / matrix[1, 1]
    radius = x * x + y * y
    radial = 1 + k1 * radius + k2 * radius**2 + k3 * radius**3
    sx = (
        matrix[0, 0] * (x * radial + 2 * p1 * x * y + p2 * (radius + 2 * x * x))
        + matrix[0, 2]
    )
    sy = (
        matrix[1, 1] * (y * radial + p1 * (radius + 2 * y * y) + 2 * p2 * x * y)
        + matrix[1, 2]
    )
    valid = (
        np.isfinite(sx)
        & np.isfinite(sy)
        & (sx >= 0)
        & (sy >= 0)
        & (sx <= width - 1)
        & (sy <= height - 1)
    )
    sx, sy = np.clip(np.nan_to_num(sx), 0, width - 1), np.clip(
        np.nan_to_num(sy), 0, height - 1
    )
    x0, y0 = sx.astype(int), sy.astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = sx - x0, sy - y0
    result = (
        pixels[y0, x0] * (1 - dx) * (1 - dy)
        + pixels[y0, x1] * dx * (1 - dy)
        + pixels[y1, x0] * (1 - dx) * dy
        + pixels[y1, x1] * dx * dy
    )
    return np.where(valid, np.clip(np.rint(result), 0, 255), 0).astype(np.uint8)


def speedplus_truth(row: dict, position_key: str, quaternion_key: str) -> dict | None:
    if position_key not in row and quaternion_key not in row:
        return None
    q, position = row[quaternion_key], row[position_key]
    if (
        len(q) != 4
        or len(position) != 3
        or not all(math.isfinite(x) for x in [*q, *position])
    ):
        raise ValueError("invalid truth pose")
    if abs(sum(x * x for x in q) - 1) > 1e-4:
        raise ValueError("truth quaternion is not unit length")
    # Stanford's projector transposes its passive quat2dcm(q). That equals the
    # active body-to-camera matrix used by nalgebra with the SAME scalar-first q.
    # Do not conjugate these components merely because the label says vbs2tango.
    return {"position_m": position, "quaternion_wxyz": q}


def import_dataset(options) -> Path:
    images = options.images.resolve()
    labels = json.loads(options.labels.read_text())
    camera = json.loads(options.camera.read_text())
    model = json.loads(options.model.read_text())
    seed = json.loads(options.seed.read_text())
    matrix = np.asarray(camera["cameraMatrix"], dtype=np.float64)
    distortion = np.asarray(camera["distCoeffs"], dtype=np.float64).ravel().tolist()
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
    ):
        raise ValueError("invalid camera matrix")
    if abs(matrix[0, 1]) > 1e-9 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("camera skew or projective intrinsics are unsupported")
    if len(distortion) != 5 or not all(math.isfinite(x) for x in distortion):
        raise ValueError("expected five finite distortion coefficients")
    if not 6 <= len(model) <= 32 or any(
        len(p) != 3 or not all(math.isfinite(x) for x in p) for p in model
    ):
        raise ValueError("supply 6..32 verified model landmarks in meters")
    timestamps = (
        json.loads(options.timestamps.read_text()) if options.timestamps else None
    )
    if options.temporal and timestamps is None:
        raise ValueError(
            "temporal evaluation requires an explicit filename-to-timestamp JSON mapping"
        )
    destination = options.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "images").mkdir(exist_ok=True)
    frames, shape, calibrated, previous = [], None, None, None
    for index, row in enumerate(labels[: options.limit]):
        source = (images / row["filename"]).resolve()
        if not source.is_relative_to(images):
            raise ValueError("image path escapes the supplied image directory")
        with Image.open(source) as image:
            pixels = np.array(image.convert("L"))
        if shape is not None and pixels.shape != shape:
            raise ValueError("image dimensions changed within the camera sequence")
        shape = pixels.shape
        pixels = rectify(pixels, matrix, distortion)
        height, width = shape
        scale = min(1.0, options.max_width / width)
        size = (round(width * scale), round(height * scale))
        sx, sy = size[0] / width, size[1] / height
        calibrated = {
            "width": size[0],
            "height": size[1],
            "fx": matrix[0, 0] * sx,
            "fy": matrix[1, 1] * sy,
            "cx": (matrix[0, 2] + 0.5) * sx - 0.5,
            "cy": (matrix[1, 2] + 0.5) * sy - 0.5,
            "distortion": [0] * 5,
        }
        image = Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR)
        relative = f"images/{index:06}.png"
        image.save(destination / relative)
        time = float(timestamps[row["filename"]]) if options.temporal else index * 0.1
        if (
            not math.isfinite(time)
            or time < 0
            or (previous is not None and time <= previous)
        ):
            raise ValueError("timestamps must increase in the annotation order")
        previous = time
        frames.append(
            {
                "timestamp_s": time,
                "image": relative,
                "truth": speedplus_truth(
                    row, options.position_key, options.quaternion_key
                ),
            }
        )
    if not frames:
        raise ValueError("no images imported")
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            ("labels", options.labels),
            ("camera", options.camera),
            ("model", options.model),
            ("seed", options.seed),
        )
    }
    manifest = {
        "camera": calibrated,
        "landmarks_m": model,
        "initial_pose": seed,
        "temporal": options.temporal,
        "frames": frames,
        "provenance": {
            "dataset": options.dataset,
            "sha256": hashes,
            "timestamps_are_index_only": not options.temporal,
        },
    }
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["speedplus", "shirt"], required=True)
    for name in ("images", "labels", "camera", "model", "seed", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--timestamps", type=Path)
    parser.add_argument("--position-key", default="r_Vo2To_vbs_true")
    parser.add_argument("--quaternion-key", default="q_vbs2tango_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.limit <= 100000 or not 64 <= arguments.max_width <= 4096:
        parser.error("limit must be 1..100000; max-width 64..4096")
    print(import_dataset(arguments))
