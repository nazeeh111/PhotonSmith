"""Dataset isolation, learned gradient, geometry constraints and input boundaries."""

import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from photonsmith.physics import solve, DEFAULT_PARAMS
from photonsmith.design import bounds_for, geometry_audit, optimize
from photonsmith.server import parse_config
from photonsmith.surrogate import Surrogate, features

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_public_optimizer_rejects_malformed_geometry(self):
        for params in (0.4, [0.4], [[0.4] * 6], [np.nan] * 6):
            with self.assertRaises(ValueError):
                optimize(params=params, method="random", iterations=1)
        for iterations in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                optimize(method="random", iterations=iterations)

    def test_surrogate_objective_rejects_wrong_channel(self):
        model = Surrogate(ROOT / "experiments/surrogate.npz")
        for order in (-2, 2, True):
            with self.assertRaises(ValueError):
                model.objective_gradient(DEFAULT_PARAMS, 1.55, order)
        with self.assertRaises(ValueError):
            model.objective_gradient([DEFAULT_PARAMS, DEFAULT_PARAMS], 1.55, 1)

    def test_uniform_slab_fresnel_limit(self):
        # Change only the material convolution to homogeneous n=2. The full
        # Maxwell boundary/layer solve is checked against analytic slab optics.
        def uniform(width, shift, orders, material_index=2.0, derivatives=False):
            e = np.eye(len(orders)) * material_index**2
            return (e, np.zeros_like(e), np.zeros_like(e)) if derivatives else e

        with patch("photonsmith.physics.epsilon_matrix", uniform):
            r = solve(material_index=2.0, wavelength=1.55)
        expected = 1 / (1 + ((4 - 1) ** 2 / 16) * np.sin(2 * 2 * np.pi * 0.9 / 1.55) ** 2)
        self.assertAlmostEqual(r["total_T"], expected, places=10)
        self.assertAlmostEqual(r["power_balance"], 1.0, places=10)

    def test_geometry_level_split_and_artifact_provenance(self):
        with np.load(ROOT / "experiments/dataset.npz", allow_pickle=False) as a:
            groups = [set(a[k].tolist()) for k in ("train", "validation", "test")]
            self.assertFalse(
                groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
            )
            self.assertEqual(len(set.union(*groups)), len(a["params"]))
            self.assertEqual(a["powers"].shape[0], len(a["params"]))
            self.assertLess(abs(a["powers"].sum(axis=-1) - 1).max(), 1e-8)
        model = Surrogate(ROOT / "experiments/surrogate.npz")
        actual = hashlib.sha256((ROOT / "experiments/dataset.npz").read_bytes()).hexdigest()
        self.assertEqual(model.metadata["dataset_sha256"], actual)

    def test_surrogate_gradient_matches_finite_difference(self):
        model = Surrogate(ROOT / "experiments/surrogate.npz")
        p = DEFAULT_PARAMS.copy()
        value, g = model.objective_gradient(p, 1.55, 1)
        finite = []
        for j in range(6):
            d = np.zeros(6)
            d[j] = 1e-6
            vp = model.predict(p + d, 1.55)["powers"][0, 2]
            vm = model.predict(p - d, 1.55)["powers"][0, 2]
            finite.append((vp - vm) / 2e-6)
        np.testing.assert_allclose(g, finite, rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(value, model.predict(p, 1.55)["powers"][0, 2], places=10)

    def test_surrogate_translation_gauge_and_power(self):
        model = Surrogate(ROOT / "experiments/surrogate.npz")
        p = DEFAULT_PARAMS.copy()
        q = p.copy()
        q[3:] = (q[3:] + 0.12) % 1
        a = model.predict(p, 1.55, fields=True)
        b = model.predict(q, 1.55, fields=True)
        np.testing.assert_allclose(a["powers"], b["powers"], atol=1e-9)
        self.assertTrue(np.all(a["powers"] >= 0))
        self.assertAlmostEqual(a["powers"].sum(), 1.0, places=10)
        self.assertEqual(a["canonical_field"].shape, (1, 24, 24))
        np.testing.assert_allclose(a["canonical_field"], b["canonical_field"], atol=1e-6)

    def test_minimum_lateral_feature_enforced(self):
        b = bounds_for(0.6)
        self.assertEqual(b[0], (0.25, 0.75))
        audit = geometry_audit([0.2, 0.5, 0.5, 0, 0, 0], 0.6)
        self.assertFalse(audit["lateral_constraint_pass"])
        self.assertFalse(audit["fabrication_ready"])
        with self.assertRaises(ValueError):
            parse_config({"params": [0.2, 0.5, 0.5, 0, 0, 0], "min_feature": 0.6})

    def test_api_invalid_inputs(self):
        for body in (
            [],
            {"wavelength": float("nan")},
            {"wavelength": 1.0},
            {"material_index": float("inf")},
            {"params": [float("nan")] * 6},
            {"target_order": 2},
            {"min_feature": -0.1},
        ):
            with self.assertRaises((ValueError, TypeError)):
                parse_config(body)

    def test_random_search_is_reproducible_and_bounded(self):
        a = optimize(method="random", iterations=3, seed=76)
        b = optimize(method="random", iterations=3, seed=76)
        self.assertEqual(a["params"], b["params"])
        self.assertAlmostEqual(a["efficiency"], b["efficiency"])
        self.assertGreaterEqual(a["efficiency"], a["initial_efficiency"])
        self.assertTrue(a["constraint_audit"]["lateral_constraint_pass"])

    def test_saved_sample_independently_passes(self):
        sample = json.loads((ROOT / "experiments/sample.json").read_text())
        v = sample["validation"]
        self.assertTrue(v["passed"])
        # Fresh solve prevents a stale green JSON flag from serving as proof.
        r = solve(sample["params"], sample["wavelength"], 13)
        index = np.flatnonzero(r["orders"] == sample["target_order"])[0]
        self.assertAlmostEqual(r["T"][index], v["primary_efficiency"], places=8)
        self.assertGreater(v["independent_efficiency"], 0.75)
        self.assertLess(v["complex_field_relative_l2"], 0.1)


if __name__ == "__main__":
    unittest.main()
