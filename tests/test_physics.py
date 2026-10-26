"""Physical invariants, exact gradients and cross-discretization checks."""

import unittest
import numpy as np
from photonsmith.physics import (
    solve,
    objective_gradient,
    epsilon_matrix,
    DEFAULT_PARAMS,
)
from photonsmith.independent import solve_fd, solve_fdfd


class MaxwellTests(unittest.TestCase):
    def test_vacuum_exact(self):
        result = solve(material_index=1.0)
        self.assertAlmostEqual(result["T"][5], 1.0, places=11)
        self.assertLess(result["total_R"], 1e-20)
        np.testing.assert_allclose(
            result["amplitudes_t"][5], np.exp(2j * np.pi * 0.9 / 1.55), atol=1e-11
        )

    def test_uniform_binary_coefficient(self):
        orders = np.arange(-5, 6)
        np.testing.assert_allclose(
            epsilon_matrix(1.0, 0.28, orders, 2.0), 4 * np.eye(11), atol=1e-14
        )
        np.testing.assert_allclose(epsilon_matrix(0.0, 0.28, orders, 2.0), np.eye(11), atol=1e-14)

    def test_lossless_power_and_residual(self):
        rng = np.random.default_rng(87)
        for _ in range(4):
            p = np.r_[rng.uniform(0.15, 0.85, 3), rng.random(3)]
            r = solve(p, harmonics=7)
            self.assertAlmostEqual(r["power_balance"], 1.0, places=9)
            self.assertLess(r["residual"], 1e-10)
            self.assertTrue(np.all(r["T"] >= 0))
            self.assertTrue(np.all(r["R"] >= 0))

    def test_translation_and_reflection_symmetry(self):
        p = DEFAULT_PARAMS.copy()
        translated = p.copy()
        translated[3:] = (translated[3:] + 0.217) % 1
        mirrored = p.copy()
        mirrored[3:] = (-mirrored[3:]) % 1
        s = solve(p)
        np.testing.assert_allclose(s["T"], solve(translated)["T"], atol=1e-10)
        np.testing.assert_allclose(s["T"], solve(mirrored)["T"][::-1], atol=1e-10)

    def test_adjoint_against_finite_difference(self):
        p = DEFAULT_PARAMS.copy()
        value, gradient = objective_gradient(p)
        finite = []
        for j in range(6):
            step = np.zeros(6)
            step[j] = 1e-6
            finite.append((solve(p + step)["T"][6] - solve(p - step)["T"][6]) / 2e-6)
        np.testing.assert_allclose(gradient, finite, atol=2e-7, rtol=2e-6)
        self.assertAlmostEqual(gradient[3:].sum(), 0.0, places=9)
        self.assertAlmostEqual(value, solve(p)["T"][6], places=12)

    def test_fd_z_refinement(self):
        reference = solve(harmonics=5)["T"][6]
        coarse = solve_fd(harmonics=5, steps_per_layer=10)["T"][6]
        fine = solve_fd(harmonics=5, steps_per_layer=20)["T"][6]
        self.assertLess(abs(fine - reference), abs(coarse - reference) / 3)
        self.assertLess(abs(fine - reference), 0.0005)

    def test_independent_real_space_solver(self):
        reference = solve(harmonics=11)
        result = solve_fdfd(nx=64, steps_per_layer=30)
        j = np.flatnonzero(result["orders"] == 1)[0]
        self.assertLess(abs(reference["T"][12] - result["T"][j]), 0.003)
        self.assertAlmostEqual(result["power_balance"], 1.0, places=9)
        self.assertLess(result["residual"], 1e-10)

    def test_field_is_complex_and_geometry_binary(self):
        s = solve(fields=True, nx=24, nz=24)
        self.assertEqual(s["field"].shape, (24, 24))
        self.assertTrue(np.iscomplexobj(s["field"]))
        self.assertTrue(np.all(np.isfinite(s["field"])))
        self.assertTrue(set(np.unique(s["epsilon"])).issubset({1.0, 4.0}))

    def test_validation(self):
        for p in (
            [0.1, 0.4, 0.4, 0, 0, 0],
            [0.4, 0.4, 0.4, 0, 0, 2],
            [1, 2],
            [np.nan] * 6,
        ):
            with self.assertRaises(ValueError):
                solve(p)
        for kwargs in (
            {"wavelength": 0},
            {"wavelength": float("nan")},
            {"harmonics": 2.5},
            {"material_index": 0.8},
            {"period": 1.55},
        ):
            with self.assertRaises(ValueError):
                solve(**kwargs)


if __name__ == "__main__":
    unittest.main()
