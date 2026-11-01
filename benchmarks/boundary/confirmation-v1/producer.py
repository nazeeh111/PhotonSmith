"""Frozen-input paired external benchmark. Writes fresh, append-only raw evidence.

Requires explicit optional benchmark dependencies from requirements-boundary.lock.
Never promotes a failed case into a successful or excluded observation.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,importlib.metadata,json,platform,random,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from photonsmith import native,physics


def meent_torch(p,m,wl):
    import torch,meent
    x=torch.tensor(p,dtype=torch.float64,requires_grad=True)
    o=torch.arange(-m,m+1,dtype=torch.float64);d=o[:,None]-o[None,:]
    eps=torch.stack([torch.eye(2*m+1,dtype=torch.complex128)+3*x[j]*torch.sinc(d*x[j])*torch.exp(2j*torch.pi*d*x[j+3]) for j in range(3)])
    obj=meent.call_mee(backend=2,pol=0,phi=None,n_top=1,n_bot=1,fto=[m,0],period=[2.4],wavelength=wl,thickness=[.3]*3,ucell=torch.ones((3,1,1)),type_complex=torch.complex128,device=0)
    res=obj.solve_for_conv(obj.wavelength,eps,eps,torch.linalg.inv(eps))
    value=res.de_ti.ravel()[m+1];g=torch.autograd.grad(value,x)[0]
    return float(value.detach()),g.detach().numpy()


def jax_compiled(m):
    import jax,jax.numpy as jnp
    from meent.on_jax.emsolver.rcwa import RCWAJax
    def objective(p,wl):
        o=jnp.arange(-m,m+1,dtype=jnp.float64);d=o[:,None]-o[None,:]
        eps=jnp.stack([jnp.eye(2*m+1,dtype=jnp.complex128)+3*p[j]*jnp.sinc(d*p[j])*jnp.exp(2j*jnp.pi*d*p[j+3]) for j in range(3)])
        obj=RCWAJax(backend=1,pol=0,phi=None,n_top=1,n_bot=1,fto=[m,0],period=[2.4],wavelength=wl,thickness=[.3]*3,ucell=jnp.ones((3,1,1)),type_complex=jnp.complex128,device=0)
        result=obj.solve_for_conv(obj.wavelength,eps,eps,jnp.linalg.inv(eps))
        return result.de_ti.ravel()[m+1]
    vg=jax.jit(jax.value_and_grad(objective,argnums=0))
    def run(p,wl):
        v,g=vg(jnp.asarray(p,dtype=jnp.float64),jnp.asarray(wl,dtype=jnp.float64))
        return float(v.block_until_ready()),np.asarray(g.block_until_ready())
    return run


def grcwa_autograd(p,m,wl):
    import grcwa,autograd.numpy as a
    from autograd import value_and_grad
    def objective(p):
        obj=grcwa.obj(2*m+2,[2.4,0],[0,.01],1/wl,0,0,verbose=0)
        obj.Add_LayerUniform(0,1)
        for _ in range(3):obj.Add_LayerGrid(.3,64,1)
        obj.Add_LayerUniform(0,1);obj.Init_Setup()
        o=obj.G[:,0];d=o[:,None]-o[None,:];n=obj.nG
        if n!=2*m+1 or np.any(obj.G[:,1]!=0):raise ValueError('Mismatched external retained modes')
        safe=np.where(d==0,1,d)
        for j in range(3):
            coeff=((d==0)*p[j]+(d!=0)*a.sin(np.pi*safe*p[j])/(np.pi*safe))*a.exp(-2j*np.pi*d*p[j+3])
            eps=a.eye(n)+3*coeff;epinv=a.linalg.inv(eps);zero=a.zeros_like(eps)
            ep2=a.concatenate((a.concatenate((eps,zero),axis=1),a.concatenate((zero,eps),axis=1)),axis=0)
            kp=grcwa.rcwa.MakeKPMatrix(obj.omega,1,epinv,obj.kx,obj.ky)
            q,v=grcwa.rcwa.SolveLayerEigensystem(obj.omega,obj.kx,obj.ky,kp,ep2)
            obj.Patterned_epinv_list[j]=epinv;obj.Patterned_ep2_list[j]=ep2
            obj.kp_list[j+1]=kp;obj.q_list[j+1]=q;obj.phi_list[j+1]=v
        obj.MakeExcitationPlanewave(0,0,1,0,order=0)
        _,t=obj.RT_Solve(normalize=1,byorder=1)
        return t[int(np.flatnonzero(o==1)[0])]
    return value_and_grad(objective)(np.asarray(p,dtype=float))


def power_check(p,m,wl):
    import meent
    from photonsmith.boundary import Layer
    o=np.arange(-m,m+1)
    eps=np.array([physics.epsilon_matrix(p[j],p[j+3],o) for j in range(3)])
    s=native.solve([Layer(e,.3) for e in eps],wl,2.4)
    ext=eps.conj()
    obj=meent.call_mee(backend=0,pol=0,phi=None,n_top=1,n_bot=1,fto=[m,0],period=[2.4],wavelength=wl,thickness=[.3]*3,ucell=np.ones((3,1,1)),type_complex=np.complex128)
    r=obj.solve_for_conv(wl,ext,ext,np.linalg.inv(ext))
    return {'max_power_error':float(max(np.max(abs(r.de_ti.ravel()-s.T)),np.max(abs(r.de_ri.ravel()-s.R)))),'balance_error':float(abs(s.R.sum()+s.T.sum()-1)),'residual':s.residual}


def main():
    from threadpoolctl import threadpool_info,threadpool_limits
    import torch,grcwa,jax
    pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--cases',type=Path,required=True);pa.add_argument('--output',type=Path,required=True);pa.add_argument('--repeats',type=int,default=7)
    args=pa.parse_args()
    if args.output.exists():raise SystemExit('Evidence exists; choose a fresh output directory')
    if args.repeats<3:raise SystemExit('At least three paired repeats required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);grcwa.set_backend('autograd')
    cases=json.loads(args.cases.read_text());rng=random.Random(37832)
    source=Path(__file__).resolve().parents[1]
    manifest={'cases_sha256':hashlib.sha256(args.cases.read_bytes()).hexdigest(),'platform':platform.platform(),'python':sys.version,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'sources':{p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in ['photonsmith/physics.py','photonsmith/boundary.py','photonsmith/native.py','native/boundary.cpp','scripts/benchmark_boundary.py']},'native_library':json.loads((source/'photonsmith/_boundary_native.dylib.json').read_text()) if sys.platform=='darwin' else {},'jax_devices':str(jax.devices()),'requested_threads':1,'repeat_count':args.repeats,'warmup':[]}
    manifest_path=args.output/'manifest.json'
    with threadpool_limits(1):
        compiled={}
        # Warm up on an explicitly development geometry, not confirmation cases.
        for m in (3,5,9,15):
            start=time.perf_counter();compiled[m]=jax_compiled(m);v,g=compiled[m](physics.DEFAULT_PARAMS,1.55)
            manifest['warmup'].append({'method':'meent_jax_jit','m':m,'seconds':time.perf_counter()-start,'value':v,'finite':bool(np.all(np.isfinite(g)))})
        for m in (3,5,9,15):
            warm = {'legacy':lambda:physics.objective_gradient(physics.DEFAULT_PARAMS,harmonics=m),
                    'native':lambda:native.objective_gradient(physics.DEFAULT_PARAMS,harmonics=m),
                    'meent_torch':lambda:meent_torch(physics.DEFAULT_PARAMS,m,1.55),
                    'grcwa_autograd':lambda:grcwa_autograd(physics.DEFAULT_PARAMS,m,1.55)}
            for name,fn in warm.items():
                start=time.perf_counter()
                try:
                    v,g=fn();record={'method':name,'m':m,'seconds':time.perf_counter()-start,'value':float(v),'finite':bool(np.all(np.isfinite(g)))}
                except Exception:record={'method':name,'m':m,'error':traceback.format_exc()}
                manifest['warmup'].append(record)
        manifest['threadpools']=threadpool_info();manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        rows=[]
        with (args.output/'raw.jsonl').open('x') as raw:
            for ci,c in enumerate(cases):
                p=np.array(c['params']);m=c['harmonics'];wl=c['wavelength']
                methods={'legacy':lambda:physics.objective_gradient(p,wavelength=wl,harmonics=m),'native':lambda:native.objective_gradient(p,wavelength=wl,harmonics=m),'meent_torch':lambda:meent_torch(p,m,wl),'meent_jax_jit':lambda:compiled[m](p,wl),'grcwa_autograd':lambda:grcwa_autograd(p,m,wl)}
                observations={name:[] for name in methods}
                for rep in range(args.repeats):
                    order=list(methods);rng.shuffle(order)
                    for name in order:
                        start=time.perf_counter_ns();cpu=time.process_time_ns()
                        try:
                            v,g=methods[name]();wall=(time.perf_counter_ns()-start)/1e9;process=(time.process_time_ns()-cpu)/1e9
                            if not np.isfinite(v) or not np.all(np.isfinite(g)):raise ArithmeticError('Nonfinite value/gradient')
                            row={'case':c['id'],'rep':rep,'method':name,'wall':wall,'cpu':process,'value':float(v),'gradient':np.asarray(g).tolist()}
                        except Exception:row={'case':c['id'],'rep':rep,'method':name,'error':traceback.format_exc()}
                        raw.write(json.dumps(row)+'\n');raw.flush();observations[name].append(row)
                verification=power_check(p,m,wl)
                if ci<8:
                    fd=[]
                    for h in (1e-4,1e-5,1e-6):
                        g=[]
                        for k in range(6):
                            plus=p.copy();minus=p.copy();plus[k]+=h;minus[k]-=h
                            g.append((native.objective_gradient(plus,wavelength=wl,harmonics=m)[0]-native.objective_gradient(minus,wavelength=wl,harmonics=m)[0])/(2*h))
                        fd.append({'step':h,'gradient':g})
                    verification['finite_differences']=fd
                rows.append({'case':c,'observations':observations,'verification':verification})
                (args.output/'cases.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(f'{ci+1}/{len(cases)} {c["id"]} power_error={verification["max_power_error"]:.3g}',flush=True)
    print('Completed; summarize raw paired observations without dropping failures',flush=True)

if __name__=='__main__':main()
