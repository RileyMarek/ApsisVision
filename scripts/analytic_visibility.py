"""User-authorized six-face Tango-bus visibility PROXY, not a verified full CAD model.

Derives slanted face planes from points 0..7. Appendage points are tested only
against the bus; appendage volumes/occlusion and illumination are not modeled.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

from fetch_benchmark import ROOT

FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (1, 2, 6, 5),
    (0, 3, 7, 4),
)


def planes(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (11, 3) or not np.isfinite(points).all():
        raise ValueError("Tango proxy requires eleven finite registered landmarks")
    center = points[:8].mean(axis=0)
    normals = []
    offsets = []
    for indices in FACES:
        face = points[list(indices)]
        n = np.cross(face[1] - face[0], face[2] - face[0])
        length = np.linalg.norm(n)
        if length < 1e-9:
            raise ValueError("Degenerate bus face")
        n /= length
        if n @ (face.mean(axis=0) - center) < 0:
            n = -n
        d = -n @ face[0]
        if np.max(np.abs(face @ n + d)) > 1e-6 or np.max(points[:8] @ n + d) > 1e-6:
            raise ValueError("Declared topology is not six planar convex faces")
        normals.append(n)
        offsets.append(d)
    normals, offsets = np.asarray(normals), np.asarray(offsets)
    if np.any(center @ normals.T + offsets >= -1e-6):
        raise ValueError("Bus must have a strictly interior centroid")
    return normals, offsets


def proxy_visibility(points, camera_origin_body):
    normals, offsets = planes(points)
    origin = np.asarray(camera_origin_body, dtype=np.float64)
    if (
        origin.shape != (3,)
        or not np.isfinite(origin).all()
        or np.all(normals @ origin + offsets <= 0)
    ):
        raise ValueError("Camera must be finite and outside the convex bus")
    facing = normals @ origin + offsets > 1e-8
    labels = np.zeros(11, dtype=np.int8)
    for face, visible in zip(FACES, facing, strict=True):
        if visible:
            labels[list(face)] = 1
    for identity in range(8, 11):
        direction = points[identity] - origin
        enter, leave = 0.0, 1.0
        hit = True
        for n, d in zip(normals, offsets, strict=True):
            distance = n @ origin + d
            denominator = n @ direction
            if abs(denominator) < 1e-12:
                if distance > 0:
                    hit = False
                    break
                continue
            boundary = -distance / denominator
            if denominator < 0:
                enter = max(enter, boundary)
            else:
                leave = min(leave, boundary)
            if enter > leave:
                hit = False
                break
        # Intersections strictly before the appendage point are proxy occlusion.
        blocked = hit and leave > 0 and enter < 1 - 1e-6 and enter <= leave
        labels[identity] = 0 if blocked else 1
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, default=ROOT / "output/datasets/selection.json"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "output/visibility-proxy/labels.json"
    )
    options = parser.parse_args()
    destination = options.out.resolve()
    if not destination.is_relative_to(ROOT / "output"):
        parser.error("Labels must remain under output/")
    geometry = ROOT / "output/datasets/tangoPoints.mat"
    points = loadmat(geometry)["tango3Dpoints"].T.astype(np.float64)
    normals, offsets = planes(points)
    selection = json.loads(options.selection.read_text())
    digest = hashlib.sha256(geometry.read_bytes()).hexdigest()
    if selection["geometry_sha256"] != digest:
        raise ValueError("Landmark provenance changed")
    rows = []
    for sample in selection["groups"]["speedplus/synthetic/train"]:
        q = sample["q_vbs2tango_true"]
        origin = (
            -Rotation.from_quat([*q[1:], q[0]]).inv().apply(sample["r_Vo2To_vbs_true"])
        )
        labels = proxy_visibility(points, origin)
        path = (
            ROOT
            / "output/datasets/speedplus/speedplusv2/synthetic/images"
            / sample["filename"]
        )
        rows.append(
            {
                "filename": sample["filename"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "visibility": labels.tolist(),
            }
        )
    report = {
        "schema": 1,
        "source": "User-authorized analytic six-face bus and point-appendage LOS proxy; not verified full CAD",
        "geometry_sha256": digest,
        "proxy_model_sha256": hashlib.sha256(
            json.dumps(
                {
                    "faces": FACES,
                    "normals": normals.tolist(),
                    "offsets": offsets.tolist(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "supervision_kind": "analytic_bus_proxy",
        "foreground_masks_available": False,
        "face_normals": normals.tolist(),
        "face_offsets": offsets.tolist(),
        "limitations": "Bus-face visibility and bus-only appendage occlusion; excludes appendage volumes, other occluders and illumination. No full foreground mask is inferred.",
        "frames": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} proxy training-pool annotations to {destination}")


if __name__ == "__main__":
    main()
