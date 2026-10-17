import unittest

import numpy as np

from scripts.generate_unseen_source_ood import FAMILIES, unseen_source_samples


class UnseenSourceOODTests(unittest.TestCase):
    def test_families_are_balanced_nonnegative_and_reproducible(self):
        arguments = dict(cells=100, cell_width=0.1, samples=60, seed=123)
        x_first, source_first, labels_first = unseen_source_samples(**arguments)
        x_second, source_second, labels_second = unseen_source_samples(**arguments)
        np.testing.assert_array_equal(x_first, x_second)
        np.testing.assert_array_equal(source_first, source_second)
        np.testing.assert_array_equal(labels_first, labels_second)
        self.assertTrue(np.all(source_first >= 0.0))
        unique, counts = np.unique(labels_first, return_counts=True)
        self.assertEqual(set(unique.tolist()), set(FAMILIES.tolist()))
        np.testing.assert_array_equal(counts, np.full(len(FAMILIES), 10))


if __name__ == "__main__":
    unittest.main()
