"""Numerical refusal and bounded PR1 reconciliation using synthetic cases."""
import http.client, json, threading, unittest, warnings
from http.server import ThreadingHTTPServer
from unittest.mock import patch
import numpy as np
from scipy.linalg import LinAlgWarning
from photonsmith import physics, design, reporting, server, independent, boundary
from photonsmith.quality import NumericalRefusal, admit_forward


class AdmissionTests(unittest.TestCase):
    def test_known_bad_transfer_refuses_and_stable_replacement_is_finite(self):
        with self.assertRaises(NumericalRefusal):physics.solve(harmonics=15,layer_height=1.)
        with self.assertRaises(NumericalRefusal):physics.objective_gradient(harmonics=15,layer_height=1.)
        r=physics.solve(harmonics=15,layer_height=1.,backend='stable')
        self.assertLess(r['residual'],1e-8);self.assertAlmostEqual(r['power_balance'],1.,places=9)
        self.assertTrue(np.isfinite(r['T']).all())
        with self.assertRaisesRegex(ValueError,'field'):physics.solve(backend='stable',fields=True)

    def test_independent_power_finite_residual_controls(self):
        for residual,t,r in [(np.nan,[1.],[0.]),(1e-7,[1.],[0.]),(0.,[.9],[0.]),(0.,[np.inf],[0.]),(0.,[-.1],[1.1])]:
            with self.subTest(residual=residual,t=t):
                with self.assertRaises(NumericalRefusal):admit_forward(residual,np.array(t),np.array(r))
        self.assertAlmostEqual(admit_forward(1e-10,np.array([.7]),np.array([.3])),1.)
        for config in ({'wavelength':1e-320},{'period':1e-320},{'material_index':1e200},{'layer_height':1e308}):
            with self.subTest(config=config), self.assertRaises(NumericalRefusal):physics.solve(**config)
        with patch.object(physics,'dense_solve',side_effect=lambda a,b,**kw:np.full_like(b,np.nan)):
            with self.assertRaises(NumericalRefusal):physics.solve()
        def warn(a,b,**kw):
            warnings.warn('synthetic ill-conditioning',LinAlgWarning)
        with patch.object(physics,'dense_solve',side_effect=warn):
            with self.assertRaises(NumericalRefusal):physics.solve()

    def test_independent_and_adjoint_failures_do_not_emit_values(self):
        with patch.object(independent,'spsolve',side_effect=lambda a,b:np.full_like(b,np.nan)):
            with self.assertRaises(NumericalRefusal):independent.solve_fdfd(nx=16,steps_per_layer=4)
        original=physics.dense_solve;calls=[0]
        def bad_adjoint(a,b,**kwargs):
            calls[0]+=1
            return original(a,b,**kwargs) if calls[0]==1 else np.full_like(b,np.nan)
        with patch.object(physics,'dense_solve',side_effect=bad_adjoint):
            with self.assertRaises(NumericalRefusal):physics.objective_gradient()
        with patch.object(boundary,'zgbtrs',side_effect=lambda factor,lower,upper,rhs,pivots,**kw:(np.full_like(rhs,np.nan),0)):
            with self.assertRaises(NumericalRefusal):physics.solve(backend='stable')

    def test_finite_wrong_adjoint_is_refused_by_both_backends(self):
        original=physics.dense_solve;calls=[0]
        def corrupt(a,b,**kwargs):
            calls[0]+=1
            value=original(a,b,**kwargs)
            return value if calls[0]==1 else value+0.01
        with patch.object(physics,'dense_solve',side_effect=corrupt):
            with self.assertRaisesRegex(NumericalRefusal,'Adjoint'):physics.objective_gradient()
        original_band=boundary.zgbtrs;calls=[0]
        def corrupt_band(*args,**kwargs):
            calls[0]+=1;value,info=original_band(*args,**kwargs)
            return (value if calls[0]==1 else value+0.01),info
        with patch.object(boundary,'zgbtrs',side_effect=corrupt_band):
            with self.assertRaisesRegex(NumericalRefusal,'Adjoint'):physics.objective_gradient(backend='stable')

    def test_stable_gradient_step_and_fourier_order_convergence(self):
        p=physics.DEFAULT_PARAMS.copy();value,g=physics.objective_gradient(p,harmonics=9,layer_height=1.,backend='stable')
        errors=[]
        for step in (1e-4,1e-5,1e-6):
            d=np.zeros(6);d[0]=step
            a=physics.solve(p+d,harmonics=9,layer_height=1.,backend='stable')['T'][10]
            b=physics.solve(p-d,harmonics=9,layer_height=1.,backend='stable')['T'][10]
            errors.append(abs((a-b)/(2*step)-g[0]))
        self.assertLess(errors[-1],errors[0]/100);self.assertLess(errors[-1],1e-7)
        vals=[physics.solve(p,harmonics=m,layer_height=1.,backend='stable')['T'][m+1] for m in (9,11,13,15)]
        self.assertLess(abs(vals[-1]-vals[-2]),abs(vals[1]-vals[0]))
        self.assertTrue(np.isfinite(value))

    def test_optimizer_refuses_any_bad_trial_and_retry_succeeds(self):
        with patch.object(design,'solve',side_effect=NumericalRefusal('controlled refusal')):
            with self.assertRaises(NumericalRefusal):design.optimize(method='random',iterations=1)
        real=design.solve;count=[0]
        def late(*args,**kwargs):
            count[0]+=1
            if count[0]==2:raise NumericalRefusal('late controlled refusal')
            return real(*args,**kwargs)
        with patch.object(design,'solve',side_effect=late):
            with self.assertRaises(NumericalRefusal):design.optimize(method='random',iterations=2)
        result=design.optimize(method='random',iterations=1,backend='stable')
        self.assertTrue(np.isfinite(result['efficiency']));self.assertEqual(result['backend'],'stable')

    def test_nonfinite_surrogate_cannot_seed_successful_design(self):
        class BadModel:
            def predict(self, params, wavelength):
                return {'powers':np.full((len(params),3),np.nan)}
        with self.assertRaises(NumericalRefusal):
            design.optimize(method='surrogate',iterations=1,model=BadModel())

    def test_actual_http_refusal_then_retry(self):
        local_server=ThreadingHTTPServer(('127.0.0.1',0),server.Handler);thread=threading.Thread(target=local_server.serve_forever,daemon=True);thread.start()
        def post(path='/api/simulate', body=None):
            conn=http.client.HTTPConnection('127.0.0.1',local_server.server_port,timeout=30)
            conn.request('POST',path,json.dumps(body or {}),{'Content-Type':'application/json'})
            response=conn.getresponse();data=json.loads(response.read());conn.close();return response.status,data
        try:
            with patch.object(reporting,'solve',side_effect=NumericalRefusal('controlled refusal')):
                status,data=post();self.assertEqual(status,422);self.assertEqual(data['code'],'numerical-refusal');self.assertNotIn('efficiency',data)
            with patch.object(design,'solve',side_effect=NumericalRefusal('controlled optimizer refusal')):
                status,data=post('/api/optimize',{'method':'random','iterations':1})
                self.assertEqual(status,422);self.assertEqual(data['code'],'numerical-refusal');self.assertNotIn('efficiency',data)
            status,data=post();self.assertEqual(status,200);self.assertTrue(np.isfinite(data['efficiency']))
            self.assertEqual(data['numerical_admission']['status'],'accepted')
        finally:local_server.shutdown();thread.join(timeout=5);local_server.server_close()
