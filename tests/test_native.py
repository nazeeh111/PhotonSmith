"""Native numerical contract and actual optimizer/output behavior."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from photonsmith import native,physics,boundary_workflow
from photonsmith.boundary import Layer,Solution

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:native._load()
        except RuntimeError as e:raise unittest.SkipTest(str(e))

    def test_legacy_values_and_all_gradients(self):
        p=np.array([.27,.64,.52,.19,.43,.77])
        for m in (3,5,9,15):
            a,ga=physics.objective_gradient(p,wavelength=1.43,harmonics=m)
            b,gb=native.objective_gradient(p,wavelength=1.43,harmonics=m)
            self.assertAlmostEqual(a,b,places=10)
            np.testing.assert_allclose(ga,gb,atol=1e-8,rtol=1e-8)

    def test_generic_excited_degenerate_modes(self):
        # A unitary rotation makes exactly degenerate/zero eigenmodes excited.
        rng=np.random.default_rng(881)
        n=7;z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));v,_=np.linalg.qr(z)
        lam=np.array([-4.,0.,0.,1.,1.,3.,5.]);k0=2*np.pi/1.55;kx=2*np.pi*np.arange(-3,4)/2.4
        e=(v*lam)@v.conj().T/k0**2+np.diag(kx**2/k0**2)
        d=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));d=(d+d.conj().T)/20
        s=native.solve([Layer(e,.23)],1.55,2.4,target_order=1)
        r=Solution([Layer(e,.23)],1.55,2.4);f,g=r.transmission_gradient(1)
        self.assertAlmostEqual(s.T[4],f,places=12)
        np.testing.assert_allclose(s.gradient,g,atol=1e-11)
        h=1e-6
        plus=native.solve([Layer(e+h*d,.23)],1.55,2.4).T[4]
        minus=native.solve([Layer(e-h*d,.23)],1.55,2.4).T[4]
        self.assertAlmostEqual((plus-minus)/(2*h),2*np.vdot(s.gradient[0],d).real,places=7)

    def test_thick_uniform_analytic_limit(self):
        for h in (1.,100.,100000.):
            s=native.solve([Layer(4*np.eye(31),h)],1.55,.51,target_order=0)
            phase=2*np.pi*2*h/1.55
            expected=1/(1+(4-1)**2/16*np.sin(phase)**2)
            self.assertAlmostEqual(s.T[15],expected,places=8)
            self.assertLess(abs(s.T.sum()+s.R.sum()-1),1e-10)

    def test_actual_optimizer_and_exclusive_output(self):
        r=boundary_workflow.optimize(wavelength=1.43,target_order=-1,iterations=4)
        self.assertGreaterEqual(r['efficiency'],r['initial_efficiency'])
        self.assertEqual(len(r['history']),r['gradient_evaluations'])
        s=boundary_workflow.simulate(r['params'],1.43)
        self.assertAlmostEqual(r['efficiency'],s['T'][4],places=12)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'result.json'
            boundary_workflow.main(['simulate','--output',str(p)])
            original=p.read_bytes()
            with self.assertRaises(SystemExit):boundary_workflow.main(['simulate','--output',str(p)])
            self.assertEqual(original,p.read_bytes())

    def test_refuse_unsupported_inputs(self):
        for kwargs in ({'harmonics':True},{'harmonics':0},{'target_order':.5},{'wavelength':0},{'material_index':float('nan')}):
            with self.assertRaises(ValueError):native.objective_gradient(physics.DEFAULT_PARAMS,**kwargs)
        for e in (-np.eye(3),np.eye(3)*(1+.1j),np.ones((4,4))):
            with self.assertRaises(ValueError):native.solve([Layer(e,.3)],1.55,2.4)

if __name__=='__main__':unittest.main()
