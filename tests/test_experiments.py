import unittest

import torch

from deeponet_1d.experiments import ExperimentConfig, _loss, build_model


class ExperimentTests(unittest.TestCase):
    def test_builds_retained_architecture(self):
        model = build_model(ExperimentConfig(latent_dim=16), cells=10)
        output = model(torch.randn(3, 30), torch.randn(10, 1))
        self.assertEqual(tuple(output.shape), (3, 10))

    def test_relative_loss_is_zero_for_exact_prediction(self):
        target = torch.randn(4, 10)
        result = _loss(
            target,
            target,
            loss_name="relative_l2",
            flux_mean=5.0,
            flux_std=2.0,
        )
        self.assertEqual(float(result), 0.0)


if __name__ == "__main__":
    unittest.main()
