"""Finite-difference verification must retain accuracy at fixed steps and bounds."""
import math
import copy
import unittest
import numpy as np
from photonsmith import gradient_check as check


class GradientCheckTests(unittest.TestCase):
    def test_polynomial_exactness_at_each_kind_of_boundary(self):
        for p in (np.array([.15,.85,.5,0.,1.,.5]),
                  np.array([.15001,.84999,.4,.00001,.99999,.3])):
            for step in check.STEPS:
                result = check.forward_difference(lambda x: float(np.sum(x**4-2*x)), p, step)
                np.testing.assert_allclose(result['gradient'],4*p**3-2,atol=5e-9,rtol=1e-8)
                for c in result['components']:
                    k=c['component']
                    for sample in c['samples']:
                        self.assertGreaterEqual(sample['params'][k],check.LOWER[k])
                        self.assertLessEqual(sample['params'][k],check.UPPER[k])
                    for degree in range(5):
                        moment=math.fsum(w*o**degree for o,w in zip(c['offsets'],c['weights']))
                        self.assertAlmostEqual(moment,1. if degree==1 else 0.,places=12)

    def test_constant_and_signed_exponential_derivatives(self):
        p=np.array([.15,.85,.5,0.,1.,.5])
        signs=np.array([1.,-1.,1.,-1.,1.,-1.])
        for step in check.STEPS:
            constant=check.forward_difference(lambda x: 1.,p,step)
            np.testing.assert_allclose(constant['gradient'],0.,atol=1e-9)
            exponential=check.forward_difference(lambda x: float(np.exp(signs*x).sum()),p,step)
            np.testing.assert_allclose(exponential['gradient'],signs*np.exp(signs*p),atol=2e-8,rtol=2e-8)

    def test_stencil_choice_never_changes_step_to_fit(self):
        self.assertEqual(check.choose_stencil(.15,1e-4,.15,.85),'forward')
        self.assertEqual(check.choose_stencil(.85,1e-4,.15,.85),'backward')
        self.assertEqual(check.choose_stencil(.5,1e-4,.15,.85),'central')
        with self.assertRaises(ValueError):check.choose_stencil(.5,1.,.15,.85)
        with self.assertRaises(ValueError):check.choose_stencil(.1,1e-4,.15,.85)

    def test_nonfinite_samples_are_refused(self):
        with self.assertRaises(ArithmeticError):
            check.forward_difference(lambda p: float('nan'),[.4]*6,1e-4)

    def test_scoring_reconstructs_all_samples_and_diagnostics(self):
        p=[.15,.85,.5,0.,1.,.5]
        record=check.forward_difference(lambda x: float(np.sum(x**4)),p,1e-4)
        record['oracle']='photonsmith.physics.solve:dense-expm-forward-only'
        check.validate_record(p,record)
        for field in ('gradient','second_order_gradient'):
            changed=copy.deepcopy(record);changed[field][0]+=.001
            with self.assertRaises(ValueError):check.validate_record(p,changed)
        changed=copy.deepcopy(record);changed['components'][0]['samples'][0]['params'][1]+=.01
        with self.assertRaises(ValueError):check.validate_record(p,changed)
        changed=copy.deepcopy(record);changed['components'].pop()
        with self.assertRaises(ValueError):check.validate_record(p,changed)


if __name__=='__main__':unittest.main()
