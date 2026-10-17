import unittest

import numpy as np

from deeponet_1d.transport import solve_sn_slab


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.cells = 20
        self.source = 4.0 + np.random.default_rng(42).random(self.cells)

    def test_scalar_and_constant_array_cross_sections_are_identical(self):
        scalar = solve_sn_slab(
            length=2.0,
            cell_width=0.1,
            sigma_t=1.0,
            sigma_s=0.5,
            source=self.source,
            directions=8,
        )
        arrays = solve_sn_slab(
            length=2.0,
            cell_width=0.1,
            sigma_t=np.ones(self.cells),
            sigma_s=np.full(self.cells, 0.5),
            source=self.source,
            directions=8,
        )
        self.assertTrue(scalar.converged)
        self.assertTrue(arrays.converged)
        np.testing.assert_allclose(
            scalar.scalar_flux, arrays.scalar_flux, rtol=0.0, atol=0.0
        )

    def test_symmetric_problem_has_symmetric_flux(self):
        result = solve_sn_slab(
            length=2.0,
            cell_width=0.1,
            sigma_t=1.0,
            sigma_s=0.5,
            source=np.ones(self.cells),
            directions=8,
        )
        np.testing.assert_allclose(
            result.scalar_flux, result.scalar_flux[::-1], rtol=1e-12, atol=1e-12
        )

    def test_stronger_absorber_reduces_flux_in_center_region(self):
        baseline = solve_sn_slab(
            length=2.0,
            cell_width=0.1,
            sigma_t=1.0,
            sigma_s=0.8,
            source=np.ones(self.cells),
            directions=8,
        )
        total = np.ones(self.cells)
        scatter = np.full(self.cells, 0.8)
        total[8:12] = 2.0
        scatter[8:12] = 0.1
        absorber = solve_sn_slab(
            length=2.0,
            cell_width=0.1,
            sigma_t=total,
            sigma_s=scatter,
            source=np.ones(self.cells),
            directions=8,
        )
        self.assertLess(
            absorber.scalar_flux[8:12].mean(), baseline.scalar_flux[8:12].mean()
        )

    def test_rejects_unphysical_cross_sections(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            solve_sn_slab(
                length=2.0,
                cell_width=0.1,
                sigma_t=1.0,
                sigma_s=1.1,
                source=self.source,
                directions=8,
            )


if __name__ == "__main__":
    unittest.main()
