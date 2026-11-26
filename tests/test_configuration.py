"""Scientific inputs must be honored or refused, never silently replaced."""
import contextlib
import http.client
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer

from photonsmith import cli, server
from photonsmith.configuration import configuration_body, parse_config, parse_harmonics, parse_fine
from photonsmith.physics import DEFAULT_PARAMS
from photonsmith.reporting import validate_design


class ConfigurationTests(unittest.TestCase):
    def test_fixed_geometry_aliases_and_wrapper(self):
        default=parse_config({})
        self.assertEqual(default,parse_config({'period':2.4,'layer_height':.3,'min_lateral_bar_gap':.36}))
        self.assertEqual(parse_config({'configuration':{'wavelength':1.6}})['wavelength'],1.6)
        for value in ({'period':3.1},{'layer_height':1.},{'period':float('nan')},
                      {'layer_height':True},{'material_index':10**500},
                      {'min_feature':.36,'min_lateral_bar_gap':.4},
                      {'configuration':{},'params':DEFAULT_PARAMS.tolist()},
                      {'configuration':[]},{'convention':'TM'}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):parse_config(value)
        self.assertEqual(configuration_body({'period':3.1})['period'],3.1) # converge has its own variable geometry contract

    def test_resolution_and_grid_admission(self):
        self.assertEqual(parse_harmonics({'harmonics':3}),3)
        for value in (True,0,16,3.,'3',None):
            with self.assertRaises(ValueError):parse_harmonics({'harmonics':value})
        with self.assertRaises(ValueError):parse_harmonics({'harmonics':3},minimum=7)
        self.assertFalse(parse_fine({'independent':{'nx':96,'steps_per_layer':45}}))
        self.assertTrue(parse_fine({'independent':{'nx':128,'steps_per_layer':60}}))
        for grid in ({'nx':64,'steps_per_layer':30},{'nx':128.,'steps_per_layer':60},{'nx':128,'steps_per_layer':60,'extra':1}):
            with self.assertRaises(ValueError):parse_fine({'independent':grid})
        with self.assertRaises(ValueError):parse_fine({'independent':{'nx':96,'steps_per_layer':45}},fine=True)

    def test_cli_uses_action_resolutions_without_http_import(self):
        result=subprocess.run([sys.executable,'-B','-c',"import sys;import photonsmith.cli;assert 'photonsmith.server' not in sys.modules"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'in.json';output=root/'out.json'
            config.write_text(json.dumps({'configuration':{'harmonics':11,'independent':{'nx':128,'steps_per_layer':60}}}))
            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(cli,'public_simulation',return_value={'test':'simulation'}) as function:
                    cli.main(['simulate','--config',str(config),'--output',str(output)])
                    self.assertEqual(function.call_args.kwargs['harmonics'],11)
                with patch.object(cli,'validate_design',return_value={'test':'validation'}) as function:
                    cli.main(['validate','--config',str(config),'--output',str(root/'validation.json')])
                    self.assertEqual(function.call_args.kwargs['harmonics'],11)
                    self.assertTrue(function.call_args.kwargs['fine'])
                with patch.object(cli,'optimize',return_value={'test':'search'}) as function:
                    cli.main(['optimize','--method','random','--config',str(config),'--output',str(root/'search.json'),'--iterations','1'])
                    self.assertEqual(function.call_args.kwargs['harmonics'],11)
                    self.assertIsNone(function.call_args.kwargs['model'])

    def test_cli_refuses_invalid_physics_before_computation_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'bad.json';output=root/'prior.json';output.write_text('prior result')
            config.write_text(json.dumps({'layer_height':1.}))
            for action in ('simulate','validate','optimize'):
                with patch.object(cli,'public_simulation') as sim,patch.object(cli,'validate_design') as validate,patch.object(cli,'optimize') as optimize,contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:cli.main([action,'--config',str(config),'--output',str(output)])
                    self.assertEqual(error.exception.code,2)
                    unused=root/(action+'-invalid.json')
                    with self.assertRaises(SystemExit) as error:cli.main([action,'--config',str(config),'--output',str(unused)])
                    self.assertEqual(error.exception.code,2)
                    self.assertFalse(unused.exists())
                    for function in (sim,validate,optimize):function.assert_not_called()
            self.assertEqual(output.read_text(),'prior result')
            for action,flag in [('simulate','--fine'),('validate','--iterations'),('converge','--fine')]:
                args=[action,'--output',str(output),flag]+(['1'] if flag=='--iterations' else [])
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:cli.main(args)
                self.assertEqual(error.exception.code,2)

    def test_actual_validation_honors_terminal_order_and_fine_grid(self):
        result=validate_design(DEFAULT_PARAMS,material_index=1.,harmonics=7,fine=True)
        self.assertEqual([r['harmonics'] for r in result['spectral_convergence']],[3,5,7])
        self.assertEqual(result['final_harmonics'],7)
        self.assertEqual(result['grid_convergence'][-1]['nx'],128)
        self.assertEqual(result['grid_convergence'][-1]['steps_per_layer'],60)
        self.assertAlmostEqual(result['primary_efficiency'],0.,places=12)
        self.assertIn('complex_field_relative_l2',result)

    def test_actual_http_refuses_changed_geometry_and_honors_resolution(self):
        class Handler(server.Handler):
            model=None
            def log_message(self,*args):pass
        service=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=service.serve_forever,daemon=True);thread.start()
        def post(body,path='/api/simulate'):
            conn=http.client.HTTPConnection('127.0.0.1',service.server_port,timeout=30)
            conn.request('POST',path,json.dumps(body),{'Content-Type':'application/json'})
            response=conn.getresponse();result=json.loads(response.read());conn.close();return response.status,result
        try:
            with patch.object(server,'public_simulation') as solver:
                for body in ({'period':3.1},{'layer_height':1.},{'harmonics':True}):
                    status,result=post(body);self.assertEqual(status,400);self.assertNotIn('efficiency',result)
                solver.assert_not_called()
            status,result=post({'configuration':{'harmonics':3}})
            self.assertEqual(status,200);self.assertEqual(result['provenance']['harmonics'],3);self.assertEqual(len(result['orders']),7)
            with patch.object(server,'optimize') as optimize:
                status,result=post({'harmonics':3,'method':'random','iterations':1},'/api/optimize')
                self.assertEqual(status,400);optimize.assert_not_called()
            status,result=post({});self.assertEqual(status,200);self.assertEqual(result['provenance']['harmonics'],9)
        finally:
            service.shutdown();thread.join(5);service.server_close();self.assertFalse(thread.is_alive())

    def test_invalid_search_resolution_never_loads_surrogate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'in.json';config.write_text('{"harmonics":true}')
            with patch.object(cli,'Surrogate') as model,contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.main(['optimize','--config',str(config),'--output',str(root/'out.json')])
                self.assertEqual(error.exception.code,2);model.assert_not_called()
                self.assertFalse((root/'out.json').exists())

    def test_bad_variable_geometry_cli_input_is_a_clean_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'bad.json';config.write_text('{"wavelength":"wrong"}')
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                cli.main(['converge','--config',str(config),'--output',str(root/'out.json')])
            self.assertEqual(error.exception.code,2);self.assertFalse((root/'out.json').exists())
