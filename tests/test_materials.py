import unittest

import numpy as np

from deeponet_1d.materials import generate_layered_materials


class MaterialTests(unittest.TestCase):
    def test_profiles_are_reproducible_and_physical(self):
        first = generate_layered_materials(
            samples=12, cells=40, min_layers=1, max_layers=3, seed=7
        )
        second = generate_layered_materials(
            samples=12, cells=40, min_layers=1, max_layers=3, seed=7
        )
        np.testing.assert_array_equal(first.sigma_t, second.sigma_t)
        np.testing.assert_array_equal(first.sigma_s, second.sigma_s)
        self.assertTrue(np.all(first.sigma_t > 0.0))
        self.assertTrue(np.all(first.sigma_s >= 0.0))
        self.assertTrue(np.all(first.sigma_s <= first.sigma_t))
        self.assertTrue(np.all(first.sigma_a >= 0.0))

    def test_interface_metadata_matches_layer_count(self):
        profiles = generate_layered_materials(
            samples=20,
            cells=50,
            min_layers=1,
            max_layers=3,
            min_layer_cells=5,
            seed=9,
        )
        for count, interfaces in zip(
            profiles.layer_count, profiles.interfaces, strict=True
        ):
            used = interfaces[interfaces >= 0]
            self.assertEqual(len(used), count - 1)
            self.assertTrue(np.all(np.diff(used) > 0))
            boundaries = np.concatenate(([0], used, [50]))
            self.assertTrue(np.all(np.diff(boundaries) >= 5))


if __name__ == "__main__":
    unittest.main()
