"""Power-channel studies are finite-sequence diagnostics, not certificates."""
import unittest
from unittest.mock import patch
import numpy as np
from photonsmith.convergence import spectral_study
from photonsmith.physics import DEFAULT_PARAMS
from photonsmith.quality import NumericalRefusal


class SpectralStudyTests(unittest.TestCase):
    def test_uniform_medium_analytic_solution_and_backend_agreement(self):
        for backend in ('legacy', 'stable'):
            report = spectral_study(material_index=1., orders=[3,5,7], backend=backend)
            self.assertTrue(report['meets_sampled_tolerance'])
            for sample in report['samples']:
                np.testing.assert_allclose(sample['T'], [0.,1.,0.], atol=1e-12)
                np.testing.assert_allclose(sample['R'], 0., atol=1e-12)
        stable = spectral_study(orders=[3,5,7])
        legacy = spectral_study(orders=[3,5,7], backend='legacy')
        for a,b in zip(stable['samples'], legacy['samples']):
            np.testing.assert_allclose(a['T'], b['T'], atol=1e-11)
            np.testing.assert_allclose(a['R'], b['R'], atol=1e-11)

    def test_all_channels_checked_even_when_target_and_balance_constant(self):
        def solve(params, wavelength, m, **kwargs):
            orders = np.arange(-m,m+1); T=np.zeros(len(orders));R=T.copy()
            T[m+1]=.3  # Target unchanged at every order.
            T[m]=.1*m;R[m]=.7-.1*m
            return dict(orders=orders,T=T,R=R,total_T=T.sum(),total_R=R.sum(),
                        power_balance=1.,residual=0.,numerical_admission={})
        with patch('photonsmith.convergence.solve',side_effect=solve):
            result=spectral_study(orders=[1,2,3], tolerance=.1)
        self.assertEqual([s['efficiency'] for s in result['samples']],[.3]*3)
        self.assertAlmostEqual(result['tail_max_channel_power_range'],.2)
        self.assertFalse(result['meets_sampled_tolerance'])

    def test_invalid_study_refused_before_any_solve(self):
        with patch('photonsmith.convergence.solve') as solver:
            for orders in ([1,2],[1,3,2],[1,1,2],[True,2,3],[1,2,16],[1.,2,3]):
                with self.assertRaises(ValueError): spectral_study(orders=orders)
            for value in (float('nan'),float('inf'),0.,-1.):
                with self.assertRaises(ValueError): spectral_study(tolerance=value)
            for field in ('wavelength', 'period', 'layer_height', 'material_index', 'tolerance'):
                for value in (True, np.bool_(True)):
                    with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                        spectral_study(**{field: value})
            with self.assertRaises(ValueError): spectral_study(wavelength=.01,period=2.4)
            with self.assertRaises(ValueError): spectral_study(period=1e300,wavelength=1e-300)
            solver.assert_not_called()

    def test_numerical_refusal_is_never_skipped(self):
        with patch('photonsmith.convergence.solve',side_effect=NumericalRefusal('control refusal')):
            with self.assertRaisesRegex(NumericalRefusal,'control refusal'):
                spectral_study()

    def test_stable_tall_geometry_runs_while_legacy_refuses(self):
        stable=spectral_study(layer_height=1.,orders=[11,13,15])
        self.assertEqual(stable['configuration']['layer_height'],1.)
        self.assertEqual([s['harmonics'] for s in stable['samples']],[11,13,15])
        self.assertTrue(all(abs(s['balance']-1)<1e-10 for s in stable['samples']))
        with self.assertRaises(NumericalRefusal):
            spectral_study(layer_height=1.,orders=[11,13,15],backend='legacy')
