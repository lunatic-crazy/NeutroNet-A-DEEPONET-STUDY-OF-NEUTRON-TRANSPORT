import unittest

import numpy as np
from deeponet_1d.learning import (
    ArrayDataset,
    fit_normalization,
    normalize_inputs,
    regression_metrics,
)


def sample_dataset(offset=0.0):
    source = np.arange(24, dtype=float).reshape(4, 6) + 1.0 + offset
    sigma_t = np.full((4, 6), 1.0 + offset)
    sigma_t[::2, 3:] += 0.5
    sigma_s = 0.5 * sigma_t
    flux = 0.7 * source + 0.2 * sigma_t
    return ArrayDataset(
        x=np.linspace(0.05, 0.55, 6),
        source=source,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        scalar_flux=flux,
        interfaces=np.array([[-1], [3], [-1], [3]]),
    )


class LearningTests(unittest.TestCase):
    def test_normalization_uses_only_training_values(self):
        training = sample_dataset()
        test = sample_dataset(offset=100.0)
        statistics = fit_normalization(training)
        branch, trunk, target = normalize_inputs(test, statistics)
        self.assertAlmostEqual(statistics.source.mean, training.source.mean())
        self.assertGreater(branch[:, :6].mean(), 10.0)
        self.assertEqual(branch.shape, (4, 18))
        self.assertEqual(trunk.shape, (6, 1))
        self.assertEqual(target.shape, (4, 6))

    def test_metrics_are_exact_for_perfect_prediction(self):
        truth = np.arange(18, dtype=float).reshape(3, 6) + 1.0
        metrics = regression_metrics(truth.copy(), truth, np.array([[-1], [3], [2]]))
        self.assertEqual(metrics.pop("r2_score"), 1.0)
        for value in metrics.values():
            self.assertEqual(value, 0.0)

    def test_metrics_cover_dimensional_and_normalized_errors(self):
        truth = np.array([[1.0, 2.0], [3.0, 4.0]])
        prediction = np.array([[2.0, 2.0], [2.0, 6.0]])
        metrics = regression_metrics(prediction, truth)
        self.assertAlmostEqual(metrics["mse"], 1.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(1.5))
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["median_absolute_error"], 1.0)
        self.assertAlmostEqual(metrics["max_absolute_error"], 2.0)
        self.assertIn("normalized_rmse", metrics)
        self.assertIn("mean_relative_l1", metrics)
        self.assertIn("r2_score", metrics)


if __name__ == "__main__":
    unittest.main()
