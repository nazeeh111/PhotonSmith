"""Sharp resonances need algebraic divided differences, even at tiny gaps."""
import json
import unittest
from pathlib import Path
import numpy as np
from photonsmith import boundary, native


class ResonanceTests(unittest.TestCase):
    def test_scalar_pairs_against_70_digit_quotients(self):
        oracle=json.loads((Path(__file__).parent/'data/resonance-oracles.json').read_text())
        for case in oracle['cases']:
            with self.subTest(case=case):
                lr,lt=boundary._loewner(np.array([case['left'],case['right']]),1.,case['eta'])
                for actual,key in ((lr,'reflection'),(lt,'transmission')):
                    expected=complex(*case[key])
                    self.assertLess(abs(actual[0,1]-expected)/max(1,abs(expected)),2e-5)
                    self.assertLess(abs(actual[0,1]-actual[1,0])/max(1,abs(expected)),5e-15)

    def test_excited_sharp_resonance_against_independent_transfer_derivative(self):
        angle=np.pi/8
        u=np.array([[np.cos(angle),-np.sin(angle),0],
                    [np.sin(angle),np.cos(angle),0],[0,0,1.]])
        q=u@np.diag([np.pi**2-4e-6,np.pi**2+4e-6,3.])@u.T
        direction=u@np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])@u.T
        eta=1e7
        epsilon=(q+np.diag([1.,0.,1.]))/eta**2
        layers=[boundary.Layer(epsilon,1.)]
        wavelength=2*np.pi/eta
        reference=boundary.Solution(layers,wavelength,2*np.pi)
        _,gradient=reference.transmission_gradient(0)
        # Independent 70-digit transfer-system differentiation; not native/reference agreement.
        expected=62489.260991
        predicted=2*np.vdot(gradient[0],direction/eta**2).real
        self.assertLess(abs(predicted-expected)/abs(expected),2e-7)
        try:
            native._load()
        except RuntimeError:
            return
        result=native.solve(layers,wavelength,2*np.pi,target_order=0)
        predicted=2*np.vdot(result.gradient[0],direction/eta**2).real
        self.assertLess(abs(predicted-expected)/abs(expected),2e-7)


if __name__=='__main__':
    unittest.main()
