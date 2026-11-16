"""Numerical contract tests for the experimental boundary solver."""
import unittest
import numpy as np
from photonsmith import physics, boundary

class BoundaryTests(unittest.TestCase):
    def test_original_solver_and_gradient(self):
        rng = np.random.default_rng(719)
        for m in (3, 5, 9, 15):
            p = np.r_[rng.uniform(.2,.8,3),rng.uniform(0,1,3)]
            a, da = physics.objective_gradient(p, harmonics=m)
            b, db = boundary.objective_gradient(p, harmonics=m)
            self.assertAlmostEqual(a,b,places=9)
            np.testing.assert_allclose(da,db,atol=2e-8,rtol=2e-8)

    def test_repeated_and_zero_internal_modes(self):
        # Exactly zero +/-1 modal eigenvalues, outside medium away from cutoff.
        n=7; eps=4*np.eye(n)
        for perturb in (0., 1e-9, -1e-9):
            e = eps + perturb*np.eye(n)
            sol=boundary.Solution([boundary.Layer(e,.21)],1.,.5,n_top=1.3,n_bottom=1.1)
            value,g=sol.transmission_gradient(0)
            d=np.diag([0.1,.2,.3,.4,.3,.2,.1])
            fd=[]
            for h in (1e-4,1e-5,1e-6):
                a=boundary.Solution([boundary.Layer(e+h*d,.21)],1.,.5,n_top=1.3,n_bottom=1.1).T[3]
                b=boundary.Solution([boundary.Layer(e-h*d,.21)],1.,.5,n_top=1.3,n_bottom=1.1).T[3]
                fd.append((a-b)/(2*h))
            np.testing.assert_allclose(fd,2*np.vdot(g[0],d).real,atol=2e-8,rtol=2e-6)
            self.assertLess(abs(sol.R.sum()+sol.T.sum()-1),1e-10)

    def test_large_evanescent_exponent(self):
        # Outer modes have kappa*h >700; growing-transfer forms overflow.
        n=65
        s=boundary.Solution([boundary.Layer(4*np.eye(n),2.)],1.55,.51)
        self.assertTrue(np.all(np.isfinite(s.state)))
        self.assertLess(abs(s.T.sum()+s.R.sum()-1),1e-10)
        value,g=s.transmission_gradient(0)
        self.assertTrue(np.all(np.isfinite(g)))
        # Analytic uniform dielectric slab.
        delta=2*np.pi*2*2/1.55
        expected=1/(1+(2**2-1)**2/(4*2**2)*np.sin(delta)**2)
        self.assertAlmostEqual(value,expected,places=10)

    def test_arbitrary_hermitian_direction(self):
        rng=np.random.default_rng(411)
        n=9; z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        e=3*np.eye(n)+.02*(z+z.conj().T)
        z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        d=(z+z.conj().T)/20
        s=boundary.Solution([boundary.Layer(e,.5)],1.55,2.4)
        value,g=s.transmission_gradient(1)
        expected=2*np.vdot(g[0],d).real
        for h in (1e-4,1e-5,1e-6):
            a=boundary.Solution([boundary.Layer(e+h*d,.5)],1.55,2.4).T[n//2+1]
            b=boundary.Solution([boundary.Layer(e-h*d,.5)],1.55,2.4).T[n//2+1]
            self.assertAlmostEqual((a-b)/(2*h),expected,places=7)

    def test_translation(self):
        p=physics.DEFAULT_PARAMS.copy()
        a,da=boundary.objective_gradient(p)
        p[3:]=(p[3:]+.1)%1
        b,db=boundary.objective_gradient(p)
        self.assertAlmostEqual(a,b,places=12)
        np.testing.assert_allclose(da,db,atol=1e-11)
        self.assertLess(abs(sum(da[3:])),1e-11)

    def test_reject_unsupported(self):
        for eps in (np.eye(4),-np.eye(3),np.full((3,3),np.nan),np.eye(3)*(1+1j)):
            with self.assertRaises(ValueError):
                boundary.Solution([boundary.Layer(eps,.2)],1.55,2.4)
        with self.assertRaises(ValueError):
            boundary.Solution([boundary.Layer(np.eye(3),1e8)],1.55,2.4)
        with self.assertRaises(ValueError):
            boundary.Solution([boundary.Layer(np.eye(3),.2)],1.,1.)

if __name__ == '__main__':unittest.main()
