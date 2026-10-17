import unittest

import numpy as np
import torch

from deeponet_1d.publication import (
    SeparateBranchDeepONet,
    bootstrap_mean_interval,
    regime_labels,
)


class PublicationTests(unittest.TestCase):
    def test_separate_branch_output_shape(self):
        model = SeparateBranchDeepONet(
            cells=10, latent_dim=8, encoder_width=6, fusion_width=12
        )
        result = model(torch.randn(4, 30), torch.randn(10, 1))
        self.assertEqual(tuple(result.shape), (4, 10))
        self.assertTrue(torch.isfinite(result).all())

    def test_bootstrap_is_reproducible(self):
        values = np.arange(1.0, 11.0)
        first = bootstrap_mean_interval(values, resamples=500, seed=7)
        second = bootstrap_mean_interval(values, resamples=500, seed=7)
        self.assertEqual(first, second)
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())

    def test_regime_labels(self):
        total = np.ones((3, 4))
        scatter = np.array([[0.1] * 4, [0.5] * 4, [0.9] * 4])
        np.testing.assert_array_equal(
            regime_labels(total, scatter),
            ["absorption_dominated", "moderate", "scattering_dominated"],
        )


if __name__ == "__main__":
    unittest.main()
