"""Observe only this process: peak RSS, thread pools and total task threads.

Process thread count includes idle runtime workers; it is not an active solver
thread count. Peak RSS includes imports/JIT and is not kernel-only allocation.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,ctypes as ct,json,resource,sys,hashlib
from pathlib import Path

def self_threads():
    if sys.platform=='darwin':
        class Task(ct.Structure):
            _fields_=[('v'+str(i),ct.c_uint64) for i in range(6)]+[('i'+str(i),ct.c_int32) for i in range(12)]
        data=Task();lib=ct.CDLL('/usr/lib/libproc.dylib')
        lib.proc_pidinfo.argtypes=[ct.c_int,ct.c_int,ct.c_uint64,ct.c_void_p,ct.c_int]
        lib.proc_pidinfo.restype=ct.c_int
        got=lib.proc_pidinfo(os.getpid(),4,0,ct.byref(data),ct.sizeof(data))
        if got!=ct.sizeof(data):return {'status':'unavailable','returned_bytes':got}
        return {'status':'observed','total_process_threads':data.i9,'running_process_threads':data.i10,'scope':'this process only, includes idle runtime workers'}
    if sys.platform.startswith('linux'):
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('Threads:'):return {'status':'observed','total_process_threads':int(line.split()[1]),'scope':'this process only'}
    return {'status':'unavailable','reason':'unsupported platform'}

def main():
    pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--method',choices=['native','legacy','meent_torch','meent_jax_jit','grcwa_autograd'],required=True);pa.add_argument('--output',type=Path,required=True)
    pa.add_argument('--cases',type=Path)
    args=pa.parse_args()
    if args.output.exists():raise SystemExit('Output exists')
    start=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    before=self_threads()
    from threadpoolctl import threadpool_limits,threadpool_info
    import numpy as np
    from benchmark_boundary import meent_torch,jax_compiled,grcwa_autograd
    from photonsmith import physics,native
    p=physics.DEFAULT_PARAMS;m=15;wl=1.55;case_id='development-default'
    if args.cases:
        case=next(c for c in json.loads(args.cases.read_text()) if c['harmonics']==15)
        p=np.array(case['params']);m=case['harmonics'];wl=case['wavelength'];case_id=case['id']
    with threadpool_limits(1):
        if args.method=='native':f=lambda:native.objective_gradient(p,wavelength=wl,harmonics=m)
        elif args.method=='legacy':f=lambda:physics.objective_gradient(p,wavelength=wl,harmonics=m)
        elif args.method=='meent_jax_jit':
            run=jax_compiled(m);f=lambda:run(p,wl)
        elif args.method=='meent_torch':
            import torch
            torch.set_num_threads(1);torch.set_num_interop_threads(1);f=lambda:meent_torch(p,m,wl)
        else:
            import grcwa
            grcwa.set_backend('autograd');f=lambda:grcwa_autograd(p,m,wl)
        for _ in range(10):v,g=f()
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale=1 if sys.platform=='darwin' else 1024
        result={'method':args.method,'harmonics':m,'case_id':case_id,'params':p.tolist(),'wavelength':wl,'input_manifest_sha256':hashlib.sha256(args.cases.read_bytes()).hexdigest() if args.cases else None,'runs':10,'finite':bool(np.isfinite(v) and np.all(np.isfinite(g))),
                'max_rss_bytes':int(peak*scale),'preimport_max_rss_bytes':int(start*scale),
                'scope':'fresh process; imports and JIT included; process high-water mark, not allocation attributable solely to solver',
                'threads_before':before,'threads_after':self_threads(),'threadpools':threadpool_info()}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
