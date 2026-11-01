"""Post-confirmation diagnostic, never a replacement for the failed original gate."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path('/Users/nazeeh/Documents/ChatGPT/Continue/engineering-accomplishment-2026-10-29');REV=ROOT/'correction-v2/PhotonSmith';OUT=Path(__file__).parent
sys.path.insert(0,str(REV));from photonsmith import native,physics
def main():
    cases=json.loads((REV/'benchmarks/boundary/confirmation-v3/cases.json').read_text());case=next(c for c in cases if c['case']['id']=='v3-m9-09');c=case['case'];p=np.array(c['params']);m=c['harmonics'];wl=c['wavelength'];target=c['target_order']
    kwargs=dict(wavelength=wl,harmonics=m)
    with threadpool_limits(1):
        nv,ng=native.objective_gradient(p,target_order=target,**kwargs);lv,lg=physics.objective_gradient(p,target_order=target,**kwargs)
        observed={method:{'value':float(rows[0]['value']),'gradient':rows[0]['gradient'],'normalized_error_to_fresh_native':float(np.max(abs(np.array(rows[0]['gradient'])-ng)/np.maximum(1,abs(ng))))} for method,rows in case['observations'].items()}
        fns={'native':lambda x:native.objective_gradient(x,target_order=target,**kwargs)[0],'legacy_dense_forward':lambda x:physics.solve(x,**kwargs)['T'][m+target]};fd={}
        for name,fn in fns.items():
            rows=[]
            for h in [1e-4,5e-5,1e-5,1e-6]:
                values=[];g=[]
                for k in range(6):
                    plus=p.copy();minus=p.copy();plus[k]+=h;minus[k]-=h
                    vp,vm=float(fn(plus)),float(fn(minus));values.append({'component':k,'plus_value':vp,'minus_value':vm});g.append((vp-vm)/(2*h))
                g=np.array(g);errors=abs(g-ng)/np.maximum(1,abs(ng));rows.append({'step':h,'posthoc_extra_step':h==5e-5,'gradient':g.tolist(),'normalized_errors':errors.tolist(),'max_normalized_error':float(errors.max()),'worst_component':int(errors.argmax()),'forward_values':values})
            fd[name]=rows
    richardson={}
    for name,rows in fd.items():
        indexed={r['step']:np.array(r['gradient']) for r in rows};richardson[name]=[]
        for large,small in [(1e-4,5e-5),(1e-4,1e-5),(1e-5,1e-6)]:
            ratio=(large/small)**2;g=(ratio*indexed[small]-indexed[large])/(ratio-1);err=abs(g-ng)/np.maximum(1,abs(ng));richardson[name].append({'large_step':large,'small_step':small,'gradient':g.tolist(),'max_normalized_error':float(err.max())})
    original_fd=case['verification']['finite_differences'];original_errors=[{'step':x['step'],'max_normalized_error':float(np.max(abs(np.array(x['gradient'])-ng)/np.maximum(1,abs(ng))))} for x in original_fd]
    result={'scope':'post-confirmation diagnostic; does not rescore or replace original false gate','case':c,'original_finite_difference_gate':False,'original_fd_errors':original_errors,'fresh_native_value':float(nv),'fresh_native_gradient':ng.tolist(),'fresh_legacy_value':float(lv),'fresh_legacy_gradient':lg.tolist(),'fresh_native_legacy_gradient_max_normalized_error':float(np.max(abs(ng-lg)/np.maximum(1,abs(ng)))),'saved_backend_gradients':observed,'finite_differences':fd,'richardson_diagnostic':richardson,'sources':{str(path.relative_to(REV)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [REV/'native/boundary.cpp',REV/'photonsmith/physics.py',REV/'photonsmith/native.py']}}
    (OUT/'confirmation-v3-fd-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['finite_differences','saved_backend_gradients','sources']},indent=2))
if __name__=='__main__':main()
