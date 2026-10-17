import tempfile
import unittest
from pathlib import Path

import numpy as np

from deeponet_1d.dataset import generate_dataset, save_dataset
from deeponet_1d.sources import mixed_source_samples


class DatasetTests(unittest.TestCase):
    def test_generation_shapes_and_reproducibility(self):
        arguments = dict(
            samples=3,
            length=1.0,
            cell_width=0.1,
            directions=4,
            min_layers=1,
            max_layers=2,
            seed=123,
        )
        first = generate_dataset(**arguments)
        second = generate_dataset(**arguments)
        self.assertEqual(first.source.shape, (3, 10))
        self.assertEqual(first.scalar_flux.shape, (3, 10))
        np.testing.assert_array_equal(first.source, second.source)
        np.testing.assert_array_equal(first.sigma_t, second.sigma_t)
        np.testing.assert_array_equal(first.scalar_flux, second.scalar_flux)

    def test_mixed_sources_are_balanced_reproducible_and_saved(self):
        arguments = dict(
            samples=14,
            length=1.0,
            cell_width=0.1,
            directions=4,
            source_distribution="mixed",
            seed=321,
        )
        first = generate_dataset(**arguments)
        second = generate_dataset(**arguments)
        np.testing.assert_array_equal(first.source, second.source)
        np.testing.assert_array_equal(first.source_family, second.source_family)
        self.assertTrue(np.all(first.source > 0.0))
        expected = {
            "grf",
            "gaussian_narrow",
            "gaussian_wide",
            "step",
            "sinusoid",
            "multiple_peak",
            "combination",
        }
        self.assertEqual(set(first.source_family.tolist()), expected)
        unique, counts = np.unique(first.source_family, return_counts=True)
        self.assertEqual(len(unique), len(expected))
        np.testing.assert_array_equal(counts, np.full(len(expected), 2))
        with tempfile.TemporaryDirectory() as directory:
            path = save_dataset(first, Path(directory) / "mixed.npz")
            with np.load(path) as archive:
                np.testing.assert_array_equal(
                    archive["source_family"], first.source_family
                )

    def test_mixed_source_labels_describe_distinct_profile_families(self):
        _, sources, labels = mixed_source_samples(
            cells=200, cell_width=0.05, samples=700, seed=20260920
        )

        widths = {}
        for family in ("gaussian_narrow", "gaussian_wide"):
            family_widths = []
            for source in sources[labels == family]:
                baseline = float(source.min())
                half_height = baseline + 0.5 * (float(source.max()) - baseline)
                family_widths.append(int(np.count_nonzero(source >= half_height)))
            widths[family] = float(np.median(family_widths))
        self.assertLess(widths["gaussian_narrow"], widths["gaussian_wide"])

        multiple = sources[labels == "multiple_peak"]
        peak_counts = np.sum(
            (multiple[:, 1:-1] > multiple[:, :-2])
            & (multiple[:, 1:-1] > multiple[:, 2:]),
            axis=1,
        )
        self.assertGreaterEqual(float(np.median(peak_counts)), 2.0)

        family_means = {
            family: sources[labels == family].mean(axis=0)
            for family in np.unique(labels)
        }
        for family in (
            "gaussian_narrow",
            "gaussian_wide",
            "multiple_peak",
            "combination",
        ):
            self.assertFalse(
                np.allclose(family_means[family], family_means["sinusoid"])
            )

    def test_saved_archive_contains_expected_fields(self):
        dataset = generate_dataset(
            samples=2,
            length=1.0,
            cell_width=0.1,
            directions=4,
            seed=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_dataset(dataset, Path(directory) / "sample.npz")
            with np.load(path) as archive:
                self.assertEqual(
                    set(archive.files),
                    {
                        "x",
                        "Q",
                        "Sigma_t",
                        "Sigma_s",
                        "Sigma_a",
                        "phi",
                        "layer_count",
                        "interfaces",
                        "solver_iterations",
                    },
                )


if __name__ == "__main__":
    unittest.main()
