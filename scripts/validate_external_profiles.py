"""Independent-author workload geometries; no physical-device claim.

Sources: pinned meent examples in the companion provenance record. Production
meent/grcwa, not the legacy example's aliased convolution assembly, are oracles.
"""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from photonsmith import native
from photonsmith.boundary import Layer

def profile(m,base,rectangles):
    o=np.arange(-m,m+1);d=o[:,None]-o[None,:]
    eps=base**2*np.eye(2*m+1,dtype=complex)
    for index,width_fraction,center_fraction in rectangles:
        eps+=(index**2-base**2)*width_fraction*np.sinc(d*width_fraction)*np.exp(-2j*np.pi*d*center_fraction)
    return eps

def external(layers,wl,period,which):
    import meent,grcwa
    m=(len(layers[0].epsilon)-1)//2;n=2*m+1
    matrices=np.array([l.epsilon for l in layers]);hs=[l.thickness for l in layers]
    if which=='meent':
        e=matrices.conj()
        obj=meent.call_mee(backend=0,pol=0,phi=None,n_top=1,n_bot=1,fto=[m,0],period=[period],wavelength=wl,thickness=hs,ucell=np.ones((len(layers),1,1)),type_complex=np.complex128)
        res=obj.solve_for_conv(wl,e,e,np.linalg.inv(e))
        return res.de_ri.ravel(),res.de_ti.ravel()
    grcwa.set_backend('numpy')
    obj=grcwa.obj(2*m+2,[period,0],[0,period/(100*n)],1/wl,0,0,verbose=0)
    obj.Add_LayerUniform(0,1)
    for h in hs:obj.Add_LayerGrid(h,64,1)
    obj.Add_LayerUniform(0,1);obj.Init_Setup()
    if obj.nG!=n or np.any(obj.G[:,1]):raise ValueError('External mode count differs')
    order=obj.G[:,0].astype(int);perm=order+m
    for j,eps0 in enumerate(matrices):
        eps=eps0[np.ix_(perm,perm)];ei=np.linalg.inv(eps);zero=np.zeros_like(eps)
        ep2=np.block([[eps,zero],[zero,eps]])
        kp=grcwa.rcwa.MakeKPMatrix(obj.omega,1,ei,obj.kx,obj.ky)
        q,v=grcwa.rcwa.SolveLayerEigensystem(obj.omega,obj.kx,obj.ky,kp,ep2)
        obj.Patterned_epinv_list[j]=ei;obj.Patterned_ep2_list[j]=ep2;obj.kp_list[j+1]=kp;obj.q_list[j+1]=q;obj.phi_list[j+1]=v
    obj.MakeExcitationPlanewave(0,0,1,0,order=0)
    r,t=obj.RT_Solve(normalize=1,byorder=1)
    out_r=np.zeros(n);out_t=np.zeros(n);out_r[perm]=r;out_t[perm]=t
    return out_r,out_t

def main():
    from threadpoolctl import threadpool_limits
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    if args.output.exists():raise SystemExit('Preserve existing results; choose fresh output')
    records=[]
    with threadpool_limits(1):
        for m in (3,7,15,31):
            for name,wls in [('uniform-film',np.linspace(.5,2.3,21)),('moharam-TE',np.linspace(.51,2.29,21)),('thick-multi-bar',[.9,.85,.95])]:
                if name=='uniform-film':
                    period=.7;layers=[Layer(3.48**2*np.eye(2*m+1),.46)]
                elif name=='moharam-TE':
                    period=.7;e=profile(m,1.,[(3.48,.3,.5)]);layers=[Layer(e,.46),Layer(e,.66)]
                else:
                    period=.3;layers=[Layer(profile(m,2.0056,[(3.638751670074983,31/300,50/300),(3.638751670074983,49.5/300,200/300)]),.205),Layer(3.638751670074983**2*np.eye(2*m+1),100.)]
                for wl in wls:
                    try:
                        s=native.solve(layers,float(wl),period,target_order=0)
                        row={'workload':name,'harmonics':m,'wavelength':float(wl),'period':period,'native_T':s.T.tolist(),'native_R':s.R.tolist(),'power_balance_error':float(abs(s.T.sum()+s.R.sum()-1)),'residual':s.residual,'reference':{}}
                        for method in ('meent','grcwa'):
                            r,t=external(layers,float(wl),period,method)
                            row['reference'][method]={'max_order_power_error':float(max(np.max(abs(r-s.R)),np.max(abs(t-s.T)))),'T':t.tolist(),'R':r.tolist()}
                        if name=='uniform-film':
                            delta=2*np.pi*3.48*.46/wl
                            analytic=1/(1+(3.48**2-1)**2/(4*3.48**2)*np.sin(delta)**2)
                            row['analytic_transmission_error']=float(abs(s.T.sum()-analytic))
                    except Exception as exc:row={'workload':name,'harmonics':m,'wavelength':float(wl),'error':repr(exc)}
                    records.append(row)
            print('finished external profiles M',m,flush=True)
    result={'scope':'independently authored computational geometry workloads, not measurements or external human validation','source_commit':'bae28024476d4af2826b8412cb8288d79a0630ac','sources':['examples/rcwa/TF_1D.py','examples/rcwa/moharam_1D_TE.py','examples/vector_1d.py'],'adaptations':['lossless scalar TE component only','all lengths converted nm to micrometres for vector case','fixed geometry, sampled wavelength subsets, increased Fourier orders','Moharam global translation chosen center0.5; physical powers invariant','exact full difference-index Fourier coefficients replace aliased legacy example convolution; compare production meent/grcwa'],'records':records}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',len(records),'attempts; failures retained',flush=True)

if __name__=='__main__':main()
