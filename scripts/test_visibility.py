"""Geometry/provenance checks do not require a flight model or invented Tango mesh."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analytic_visibility import planes, proxy_visibility
from visibility_labels import ray_visibility, validate_mesh


def cube():
    vertices = np.asarray(
        [[x, y, z] for z in [-1.0, 1.0] for y in [-1.0, 1.0] for x in [-1.0, 1.0]]
    )
    quads = [
        [0, 2, 6, 4],
        [1, 5, 7, 3],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 1, 3, 2],
        [4, 6, 7, 5],
    ]
    faces = np.asarray(
        [triangle for a, b, c, d in quads for triangle in [[a, b, c], [a, c, d]]]
    )
    return vertices, faces


class VisibilityGeometryTests(unittest.TestCase):
    def test_tapered_bus_planes_and_appendage_occlusion(self):
        points = np.asarray(
            [
                [-0.5, -0.6, 1],
                [-0.5, 0.6, 1],
                [0.5, 0.6, 1],
                [0.5, -0.6, 1],
                [-0.5, -0.4, 0],
                [-0.5, 0.4, 0],
                [0.5, 0.4, 0],
                [0.5, -0.4, 0],
                [0.8, 0, 0.5],
                [0, 0.9, 0.5],
                [0, -0.9, 0.5],
            ]
        )
        normals, _ = planes(points)
        self.assertGreater(
            abs(normals[4, 2]),
            0.1,
            "Tapered side normal must not be replaced with axis-aligned Y",
        )
        front = proxy_visibility(points, [0, 0, 4])
        np.testing.assert_array_equal(front[:8], [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(front[8], 1)
        self.assertEqual(proxy_visibility(points, [-4, 0, 0.5])[8], 0)
        bad = points.copy()
        bad[1, 2] += 0.1
        with self.assertRaises(ValueError):
            planes(bad)

    def test_closed_cube_occludes_backside_vertices_and_pose_changes_visibility(self):
        vertices, faces = cube()
        document = {
            "units": "m",
            "source": "analytic test cube",
            "vertices_m": vertices.tolist(),
            "triangles": faces.tolist(),
            "landmark_vertex_indices": list(range(8)),
        }
        validate_mesh(document, vertices)
        np.testing.assert_array_equal(
            ray_visibility(vertices, faces, vertices, [0, 0, -4]),
            [1, 1, 1, 1, 0, 0, 0, 0],
        )
        np.testing.assert_array_equal(
            ray_visibility(vertices, faces, vertices, [0, 0, 4]),
            [0, 0, 0, 0, 1, 1, 1, 1],
        )
        document["triangles"] = faces[:-1].tolist()
        with self.assertRaisesRegex(ValueError, "closed"):
            validate_mesh(document, vertices)
        document["triangles"] = faces.tolist()
        with self.assertRaisesRegex(ValueError, "register"):
            validate_mesh(document, vertices + 0.01)
        with self.assertRaises(ValueError):
            ray_visibility(vertices, faces, vertices, [0, 0, 0])


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Install requirements-learned.txt"
)
class VisibilityLearningTests(unittest.TestCase):
    def test_supervision_identity_and_proxy_compositing_fail_closed(self):
        from visibility_perception import load_supervision

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            labels = [1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0]
            document = {
                "schema": 1,
                "source": "test analytic proxy",
                "geometry_sha256": "geometry",
                "proxy_model_sha256": "f" * 64,
                "supervision_kind": "analytic_bus_proxy",
                "frames": [{"sha256": "image", "visibility": labels}],
            }
            path.write_text(json.dumps(document))
            flags, _ = load_supervision(path, ["image"], "geometry")
            self.assertEqual(flags.tolist(), [labels])
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_supervision(path, ["image"], "wrong geometry")
            with self.assertRaisesRegex(ValueError, "Missing"):
                load_supervision(path, ["other image"], "geometry")
            with self.assertRaisesRegex(ValueError, "complete spacecraft"):
                load_supervision(path, ["image"], "geometry", True)
            document["frames"][0]["visibility"] = [-1] * 11
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "known supervision"):
                load_supervision(path, ["image"], "geometry")

    def test_masked_loss_hidden_and_unknown_heatmaps_have_zero_gradient(self):
        import torch

        from visibility_perception import visible_loss

        heatmaps = torch.randn(1, 11, 48, 64, requires_grad=True)
        visibility_logits = torch.zeros(1, 11, requires_grad=True)
        flags = torch.tensor([[1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0]])
        coordinates = torch.ones(1, 11, 2) * 20
        loss = visible_loss(heatmaps, visibility_logits, coordinates, flags)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(heatmaps.grad[0, 1:3].abs().sum()), 0.0)
        self.assertGreater(float(heatmaps.grad[0, 0].abs().sum()), 0.0)
        self.assertEqual(float(visibility_logits.grad[0, 2]), 0.0)
        self.assertGreater(float(visibility_logits.grad[0, 1]), 0.0)
        with self.assertRaises(ValueError):
            visible_loss(
                heatmaps, visibility_logits, coordinates, torch.full_like(flags, -1)
            )

    def test_global_moments_detect_multimodal_spread_and_visibility_head_runs(self):
        import torch

        from visibility_perception import VisibilityNet, spatial_moments

        torch.set_num_threads(2)
        heatmaps, visible = VisibilityNet()(torch.zeros(2, 1, 192, 256))
        self.assertEqual(tuple(heatmaps.shape), (2, 11, 48, 64))
        self.assertEqual(tuple(visible.shape), (2, 11))
        logits = torch.full((1, 1, 48, 64), -100.0)
        logits[0, 0, 20, 10] = 0
        logits[0, 0, 20, 50] = 0
        center, spread = spatial_moments(logits)
        torch.testing.assert_close(center[0, 0], torch.tensor([30.0, 20.0]))
        self.assertGreater(float(spread[0, 0, 0, 0]), 399.0)
        logits[0, 0, 20, 50] = -100
        center, spread = spatial_moments(logits)
        torch.testing.assert_close(center[0, 0], torch.tensor([10.0, 20.0]))
        self.assertLess(float(spread.max()), 1e-5)

    def test_background_augmentation_requires_mask_and_preserves_unknown_labels(self):
        from visibility_perception import augment

        image = np.zeros((192, 256), np.uint8)
        image[40:140, 60:200] = 100
        mask = image > 0
        labels = np.asarray([1, 0, -1])
        coordinates = np.asarray([[80, 80], [100, 100], [120, 120]])
        first, flags = augment(image, coordinates, labels, mask, 42)
        second, _ = augment(image, coordinates, labels, mask, 42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(flags[2], -1)
        self.assertEqual(flags[1], 0)
        self.assertEqual(first.shape, image.shape)
        self.assertFalse(np.array_equal(first, image))
        with self.assertRaises(ValueError):
            augment(image, coordinates, labels, mask.astype(np.uint8), 42)


if __name__ == "__main__":
    unittest.main()
