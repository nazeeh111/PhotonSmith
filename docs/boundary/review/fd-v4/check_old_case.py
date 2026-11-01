"""Independent review of a stronger finite-difference oracle on viewed v3 data."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

SOURCE=Path('/private/tmp/photon-corrected-final-public-20261101')
PACKAGE=Path('/private/tmp/photon-corrected-public-env-20261101/lib/python3.12/site-packages')
sys.path.insert(0,str(PACKAGE))
from photonsmith import physics,native

def main():
    for module,name in ((physics,'physics.py'),(native,'native.py')):
        assert Path(module.__file__).read_bytes()==(SOURCE/'photonsmith'/name).read_bytes()
    receipt=json.loads((PACKAGE/'photonsmith/_boundary_native.dylib.json').read_text())
    assert receipt['source_sha256']==hashlib.sha256((SOURCE/'native/boundary.cpp').read_bytes()).hexdigest()
    assert receipt['library_sha256']==hashlib.sha256((PACKAGE/'photonsmith/_boundary_native.dylib').read_bytes()).hexdigest()
    cases=json.loads((SOURCE/'benchmarks/boundary/confirmation-v3/inputs.json').read_text())
    case=next(c for c in cases if c['id']=='v3-m9-09')
    p=np.array(case['params']);m=case['harmonics'];target=case['target_order']
    kwargs={k:case[k] for k in ('wavelength','harmonics','period','layer_height','material_index')}
    def forward(x):return float(physics.solve(x,**kwargs)['T'][m+target])
    with threadpool_limits(1):
        value,dense=physics.objective_gradient(p,target_order=target,**kwargs)
        nv,gradient=native.objective_gradient(p,target_order=target,**kwargs)
        checks=[]
        for h in (1e-4,1e-5,1e-6):
            centered=[];fourth=[];observations=[]
            for k in range(6):
                vals={}
                for offset in (-2,-1,1,2):
                    point=p.copy();point[k]+=offset*h
                    vals[offset]=forward(point)
                g2=(vals[1]-vals[-1])/(2*h)
                g4=(vals[-2]-8*vals[-1]+8*vals[1]-vals[2])/(12*h)
                centered.append(g2);fourth.append(g4)
                observations.append({'component':k,'values':{str(k):v for k,v in vals.items()}})
            centered=np.array(centered);fourth=np.array(fourth)
            checks.append({'step':h,'dense_forward_centered_second_order':centered.tolist(),
                           'dense_forward_centered_fourth_order':fourth.tolist(),
                           'second_order_max_normalized_error':float(np.max(abs(centered-gradient)/np.maximum(1,abs(centered)))),
                           'fourth_order_max_normalized_error':float(np.max(abs(fourth-gradient)/np.maximum(1,abs(fourth)))),
                           'forward_samples':observations})
    result={'scope':'viewed-case diagnosis, not fresh confirmation or human external validation',
            'case':case,'source_sha256':{p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in ('photonsmith/physics.py','photonsmith/native.py')},
            'dense_analytic_gradient':dense.tolist(),'native_gradient':gradient.tolist(),
            'analytic_max_normalized_error':float(np.max(abs(dense-gradient)/np.maximum(1,abs(dense)))),
            'normalization':'max(1,abs(independent finite-difference estimate)), per original finish line',
            'threshold':2e-5,'checks':checks}
    out=Path(__file__).with_name('old-case-fourth-order-diagnostic.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    print(json.dumps([{k:v for k,v in c.items() if k!='forward_samples'} for c in checks],indent=2))

if __name__=='__main__':main()
