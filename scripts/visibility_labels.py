"""Training-only visibility labels from an explicitly supplied, registered opaque mesh.

No inferred convex hull or guessed normals. Tango CAD is not bundled with Apsis.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

from fetch_benchmark import ROOT
from image_perception import prepared


def validate_mesh(document, landmarks):
    if document.get("units") != "m" or not document.get("source"):
        raise ValueError("Mesh requires metre units and a provenance source")
    vertices = np.asarray(document["vertices_m"], dtype=np.float64)
    faces = np.asarray(document["triangles"])
    mapping = np.asarray(document["landmark_vertex_indices"])
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not 4 <= len(vertices) <= 20000
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Invalid or oversized mesh vertices")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not 4 <= len(faces) <= 40000
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Invalid or oversized triangle indices")
    if (
        mapping.shape != (len(landmarks),)
        or not np.issubdtype(mapping.dtype, np.integer)
        or mapping.min() < 0
        or mapping.max() >= len(vertices)
        or len(np.unique(mapping)) != len(mapping)
    ):
        raise ValueError(
            "An explicit unique mesh vertex mapping is required for every landmark"
        )
    if np.max(np.linalg.norm(vertices[mapping] - landmarks, axis=1)) > 1e-6:
        raise ValueError(
            "Mesh does not register to the landmark geometry within 1 micrometre"
        )
    triangles = vertices[faces]
    if np.any(
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        < 1e-12
    ):
        raise ValueError("Degenerate mesh face")
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError("Visibility requires a closed two-manifold triangle surface")
    return vertices, faces, mapping


def ray_visibility(vertices, faces, landmarks, camera_origin_body, tolerance_m=1e-4):
    """Two-sided opaque ray/triangle intersections. Endpoint hits are not occlusion."""
    origin = np.asarray(camera_origin_body, dtype=np.float64)
    if (
        origin.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(tolerance_m)
        or tolerance_m <= 0
    ):
        raise ValueError("Invalid camera origin or physical tolerance")
    if not np.any(
        (origin < vertices.min(axis=0) - tolerance_m)
        | (origin > vertices.max(axis=0) + tolerance_m)
    ):
        raise ValueError(
            "Camera origin must be demonstrably outside the mesh bounding box"
        )
    triangle = vertices[faces]
    edge1, edge2 = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    result = []
    for landmark in landmarks:
        direction = landmark - origin
        distance = np.linalg.norm(direction)
        if distance <= tolerance_m:
            raise ValueError("Camera overlaps landmark")
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.einsum("ij,ij->i", edge1, h)
        active = np.abs(determinant) > 1e-12
        inverse = np.divide(
            1.0, determinant, out=np.zeros_like(determinant), where=active
        )
        offset = origin - triangle[:, 0]
        u = inverse * np.einsum("ij,ij->i", offset, h)
        q = np.cross(offset, edge1)
        v = inverse * (q @ direction)
        t = inverse * np.einsum("ij,ij->i", edge2, q)
        hits = active & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
        blocked = np.any(hits & (t < 1 - tolerance_m / distance))
        endpoint = np.any(hits & (np.abs(t - 1) <= tolerance_m / distance))
        result.append(0 if blocked else 1 if endpoint else -1)
    return np.asarray(result, dtype=np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "output/visibility/labels.json"
    )
    options = parser.parse_args()
    destination = options.out.resolve()
    if not destination.is_relative_to(ROOT / "output"):
        parser.error("Generated labels must remain under output/")
    if options.mesh.stat().st_size > 8 * 1024**2:
        parser.error("Mesh JSON exceeds 8 MiB bound")
    selection = json.loads((ROOT / "output/datasets/selection.json").read_text())
    mesh = json.loads(options.mesh.read_text())
    if mesh.get("geometry_sha256") != selection["geometry_sha256"]:
        raise ValueError(
            "Mesh provenance must declare the current landmark geometry SHA-256"
        )
    landmarks = loadmat(ROOT / "output/datasets/tangoPoints.mat")["tango3Dpoints"].T
    vertices, faces, _ = validate_mesh(mesh, landmarks)
    samples = selection["groups"]["speedplus/synthetic/train"]
    rows = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    mask_root = destination.parent / "masks"
    mask_root.mkdir(exist_ok=True)
    camera = json.loads(
        (ROOT / "output/datasets/speedplus/speedplusv2/camera.json").read_text()
    )
    for sample in samples:
        q = sample["q_vbs2tango_true"]
        rotation = Rotation.from_quat([*q[1:], q[0]])
        origin = -rotation.inv().apply(sample["r_Vo2To_vbs_true"])
        flags = ray_visibility(vertices, faces, landmarks, origin)
        path = (
            ROOT
            / "output/datasets/speedplus/speedplusv2/synthetic/images"
            / sample["filename"]
        )
        image, k = prepared(path, camera)
        camera_vertices = rotation.apply(vertices) + sample["r_Vo2To_vbs_true"]
        if np.any(camera_vertices[:, 2] <= 0):
            raise ValueError(
                "Cannot make a silhouette mask for a mesh crossing the camera plane"
            )
        pixels = (
            camera_vertices[:, :2] / camera_vertices[:, 2, None] @ k[:2, :2].T
            + k[:2, 2]
        )
        mask = np.zeros_like(image)
        for face in faces:
            cv.fillConvexPoly(mask, np.round(pixels[face]).astype(np.int32), 255)
        mask_name = Path(sample["filename"]).stem + ".png"
        mask_path = mask_root / mask_name
        if not cv.imwrite(str(mask_path), mask):
            raise ValueError("Could not write training mask")
        rows.append(
            {
                "filename": sample["filename"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "visibility": flags.tolist(),
                "mask": "masks/" + mask_name,
                "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            }
        )
    report = {
        "schema": 1,
        "source": mesh["source"],
        "mesh_sha256": hashlib.sha256(options.mesh.read_bytes()).hexdigest(),
        "geometry_sha256": selection["geometry_sha256"],
        "labels": "1 geometric line-of-sight; 0 occluded; -1 unresolved; image bounds applied during training",
        "scope": "selected synthetic training pool only; mesh registration/shape fidelity still require external verification",
        "frames": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} training-pool annotations to {destination}")


if __name__ == "__main__":
    main()
