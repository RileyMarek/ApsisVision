"""Data separation and pretrained architecture tests; never download weights in CI."""

import importlib.util
import unittest

from training_pool import select


class PoolTests(unittest.TestCase):
    def test_audit_freeze_binds_configuration_and_inference_code(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import fresh_audit

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            for name in (
                "crop_perception.py",
                "image_perception.py",
                "learned_perception.py",
                "transfer_perception.py",
                "visibility_perception.py",
                "correspondence_integrity.py",
                "requirements-learned.txt",
                "requirements-perception.txt",
            ):
                (root / "scripts" / name).write_text("fixture")
            (root / "training.json").write_text('{"feature_refinement": false}')
            with patch.object(fresh_audit, "ROOT", root):
                before = fresh_audit.pipeline_hashes(root)
                (root / "training.json").write_text('{"feature_refinement": true}')
                self.assertNotEqual(before, fresh_audit.pipeline_hashes(root))
                before = fresh_audit.pipeline_hashes(root)
                (root / "scripts/crop_perception.py").write_text("changed inference")
                self.assertNotEqual(before, fresh_audit.pipeline_hashes(root))

    def test_audit_byte_leak_is_rejected_even_with_a_different_filename(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import training_pool

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "output/datasets/speedplus/speedplusv2/lightbox/images"
            folder.mkdir(parents=True)
            image = folder / "different-name.jpg"
            image.write_bytes(b"same image bytes")
            selection = {
                "groups": {"speedplus/lightbox/test": [{"filename": image.name}]}
            }
            members = [{"sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]
            with patch.object(training_pool, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "overlap an audit"):
                    training_pool.verify_audit_separation(members, [selection])
                image.write_bytes(b"disjoint bytes")
                training_pool.verify_audit_separation(members, [selection])

    def test_expansion_preserves_existing_and_excludes_audits_independent_of_truth(
        self,
    ):
        rows = [{"filename": str(i), "pose": i} for i in range(40)]
        chosen = select(rows, ["0", "1"], {"3", "7"}, 20, 42)
        self.assertEqual([r["filename"] for r in chosen[:2]], ["0", "1"])
        self.assertFalse({r["filename"] for r in chosen} & {"3", "7"})
        changed = [{"filename": r["filename"], "pose": 999} for r in reversed(rows)]
        self.assertEqual(
            [r["filename"] for r in chosen],
            [r["filename"] for r in select(changed, ["0", "1"], {"3", "7"}, 20, 42)],
        )
        with self.assertRaises(ValueError):
            select(rows, ["0"], {"0"}, 20, 42)
        with self.assertRaises(ValueError):
            select(rows + rows, ["0"], set(), 20, 42)
        with self.assertRaises(ValueError):
            select(rows, ["0", "0"], set(), 20, 42)


@unittest.skipUnless(
    importlib.util.find_spec("torchvision"), "Install requirements-learned.txt"
)
class TransferTests(unittest.TestCase):
    def test_inference_rejects_weights_changed_after_freezing(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path

        import torch

        from crop_perception import BoxNet, Perception, VisibilityNet

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            hashes = {}
            for name, model in [("box", BoxNet()), ("localizer", VisibilityNet())]:
                path = root / (name + ".pt")
                torch.save(model.state_dict(), path)
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "training.json").write_text(
                json.dumps({"checkpoint_sha256": hashes})
            )
            Perception(root)
            (root / "localizer.pt").write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "frozen training metadata"):
                Perception(root)
            (root / "training.json").write_text('{"feature_refinement": true}')
            with self.assertRaisesRegex(ValueError, "experiment was rejected"):
                Perception(root)

    def test_storage_counts_buffers_and_mixed_tensor_dtypes(self):
        import torch

        from profile_perception import storage_bytes

        model = torch.nn.Linear(3, 2).half()
        model.register_buffer("fixture", torch.ones(5, dtype=torch.uint8))
        self.assertEqual(storage_bytes({"fixture": model}), {"fixture": 21})

    def test_shapes_finite_gradients_and_portable_state_without_network_download(self):
        import torch

        from crop_perception import balanced_loss
        from transfer_perception import TransferBox, TransferLandmarks

        torch.set_num_threads(2)
        torch.manual_seed(12)
        x = torch.rand(2, 1, 192, 256)
        box = TransferBox()
        boxes = box(x)
        self.assertEqual(tuple(boxes.shape), (2, 4))
        boxes.square().mean().backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in box.parameters()
            )
        )
        model = TransferLandmarks()
        heat, visibility = model(x)
        self.assertEqual(tuple(heat.shape), (2, 11, 48, 64))
        self.assertEqual(tuple(visibility.shape), (2, 11))
        flags = torch.tensor([[1] * 8 + [0] * 3] * 2)
        loss = balanced_loss((heat, visibility), torch.full((2, 11, 2), 20.0), flags)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )
        restored = TransferLandmarks()
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(x)[0], restored(x)[0])


if __name__ == "__main__":
    unittest.main()
