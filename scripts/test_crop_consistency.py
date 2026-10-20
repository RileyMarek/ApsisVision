"""Paired-crop training must preserve coordinate mapping and exclude hidden labels."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install learned requirements")
class ConsistencyTests(unittest.TestCase):
    def test_inverse_transform_roundtrip_and_decoder_equivalence(self):
        import torch

        from crop_consistency import local_centers, to_parent
        from learned_perception import decode

        torch.set_num_threads(2)
        torch.manual_seed(71)
        points = torch.full((2, 11, 2), 20.0)
        scale = torch.tensor([[0.9], [1.1]])
        offset = torch.tensor([[0.1, -0.1], [-0.1, 0.05]])
        size = torch.tensor([64, 48])
        warped = (
            (((points + 0.5) / size * 2 - 1) - offset[:, None]) / scale[:, None] + 1
        ) * size / 2 - 0.5
        torch.testing.assert_close(to_parent(warped, (scale, offset)), points)
        logits = torch.randn(1, 11, 48, 64)
        expected, _ = decode(logits[0].numpy(), (48, 64))
        torch.testing.assert_close(
            local_centers(logits)[0],
            torch.tensor(expected, dtype=torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_finite_gradients_and_hidden_landmarks_have_no_consistency_gradient(self):
        import torch

        from crop_consistency import equivariance_loss

        torch.manual_seed(12)
        a = torch.randn(2, 11, 48, 64, requires_grad=True)
        b = torch.randn(2, 11, 48, 64, requires_grad=True)
        xy = torch.full((2, 11, 2), 20.0)
        flags = torch.ones(2, 11, dtype=torch.long)
        flags[:, -1] = -1
        transform = (torch.ones(2, 1), torch.zeros(2, 2))
        loss = equivariance_loss(a, b, transform, transform, (xy, flags), (xy, flags))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for tensor in (a, b):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertEqual(float(tensor.grad[:, -1].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
