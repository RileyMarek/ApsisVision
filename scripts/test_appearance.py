"""Sensor augmentation must preserve geometry, unknown labels and reproducibility."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install learned requirements")
class AppearanceTests(unittest.TestCase):
    def test_occluded_landmarks_are_not_trained_as_visible(self):
        import torch

        from appearance import occlude

        image = torch.ones(1, 1, 192, 256)
        xy = torch.tensor([[[10.0, 10.0], [50.0, 40.0], [11.0, 11.0], [12.0, 12.0]]])
        flags = torch.tensor([[1, 1, -1, 0]])
        changed, labels = occlude(
            image, xy, flags, torch.tensor([[0.0, 0.0, 0.5, 0.5]])
        )
        self.assertEqual(labels.tolist(), [[0, 1, -1, 0]])
        self.assertEqual(float(changed[0, 0, 10, 10]), 0.0)
        self.assertEqual(float(changed[0, 0, 180, 240]), 1.0)
        self.assertEqual(flags.tolist(), [[1, 1, -1, 0]])

    def test_seeded_randomization_preserves_coordinates_and_finite_range(self):
        import torch

        from appearance import sensor_appearance

        torch.set_num_threads(2)
        images = (
            torch.linspace(0, 1, 192 * 256).reshape(1, 1, 192, 256).repeat(8, 1, 1, 1)
        )
        xy = torch.full((8, 11, 2), 20.0)
        flags = torch.ones((8, 11), dtype=torch.int64)
        a, ta = sensor_appearance(
            images, (xy, flags), True, torch.Generator().manual_seed(99)
        )
        b, tb = sensor_appearance(
            images, (xy, flags), True, torch.Generator().manual_seed(99)
        )
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(ta[0], xy)
        torch.testing.assert_close(ta[1], tb[1])
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(((a >= 0) & (a <= 1)).all())
        self.assertFalse(torch.equal(images, a))
        boxes = torch.rand(8, 4)
        _, box_target = sensor_appearance(
            images, (boxes,), False, torch.Generator().manual_seed(99)
        )
        torch.testing.assert_close(box_target[0], boxes)


if __name__ == "__main__":
    unittest.main()
