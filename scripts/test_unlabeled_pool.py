"""Unlabeled selection is name-only, deterministic and refuses unsafe identities."""

import unittest

from unlabeled_pool import choose_names


class PoolTests(unittest.TestCase):
    def test_name_only_selection_is_stable_and_excludes_reserved_images(self):
        names = [f"image-{i}.jpg" for i in range(50)]
        a = choose_names(names, {"image-2.jpg", "image-8.jpg"}, 20, 88)
        self.assertEqual(
            a,
            choose_names(list(reversed(names)), {"image-8.jpg", "image-2.jpg"}, 20, 88),
        )
        self.assertFalse(set(a) & {"image-2.jpg", "image-8.jpg"})
        for bad in [["../escape.jpg"], ["a.jpg", "a.jpg"], ["nested\\escape.jpg"]]:
            with self.assertRaises(ValueError):
                choose_names(bad, set(), 1, 88)


if __name__ == "__main__":
    unittest.main()
