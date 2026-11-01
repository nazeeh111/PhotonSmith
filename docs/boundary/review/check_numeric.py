"""Independent coding-agent checks; not external human validation."""
from pathlib import Path
import hashlib, importlib.util, sys, json, time
import numpy as np
from scipy.linalg import expm, expm_frechet, solve
source=Path('/Users/nazeeh/Documents/ChatGPT/Continue/engineering-accomplishment-2026-10-29/PhotonSmith/photonsmith/boundary.py')
base=Path(__file__).parent
raw=source.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
snapshot=base/('boundary-'+sha+'.py')
if not snapshot.exists(): snapshot.write_bytes(raw)
spec=importlib.util.spec_from_file_location('review_boundary',snapshot)
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)

def independent_transfer(epslist, hs, wavelength, period, target, directions=None, ntop=1., nbot=1.):
 n=len(epslist[0]); m=n//2; k0=2*np.pi/wavelength
 kx=2*np.pi*np.arange(-m,m+1)/period
 kt=np.sqrt(((k0*ntop)**2-kx**2).astype(complex)); kb=np.sqrt(((k0*nbot)**2-kx**2).astype(complex))
 zi=np.zeros((n,n)); eye=np.eye(n)
 p=np.eye(2*n,dtype=complex);dp=np.zeros_like(p)
 for j,(eps,h) in enumerate(zip(epslist,hs)):
  q=k0**2*eps-np.diag(kx**2)
  g=np.block([[zi,eye],[-q,zi]])*h
  if directions is None: bi=expm(g);db=np.zeros_like(bi)
  else:
   dg=np.block([[zi,zi],[-k0**2*directions[j],zi]])*h
   bi,db=expm_frechet(g,dg)
  dp=db@p+bi@dp;p=bi@p
 p11,p12,p21,p22=p[:n,:n],p[:n,n:],p[n:,:n],p[n:,n:]
 d11,d12,d21,d22=dp[:n,:n],dp[:n,n:],dp[n:,:n],dp[n:,n:]
 ikt=1j*np.diag(kt); ikb=1j*np.diag(kb)
 x=p21-ikb@p11;y=p22-ikb@p12
 dx=d21-ikb@d11;dy=d22-ikb@d12
 inc=np.zeros(n);inc[m]=1
 a=x-y@ikt;rhs=-(x+y@ikt)@inc
 r=solve(a,rhs)
 dr=solve(a,-(dx+dy@ikt)@inc-(dx-dy@ikt)@r)
 t=(p11+p12@ikt)@inc+(p11-p12@ikt)@r
 dt=(d11+d12@ikt)@inc+(d11-d12@ikt)@r+(p11-p12@ikt)@dr
 idx=m+target;flux=kb[idx].real/kt[m].real
 return flux*abs(t[idx])**2,2*flux*np.real(t[idx].conj()*dt[idx])

rng=np.random.default_rng(60441);records=[]
for family,values in [('triple_degenerate',[1.,2.,2.,2.,4.,5.,6.]),('triple_zero',[0.,0.,0.,1.,2.,3.,4.]),('mixed_near_degenerate',[0.,1e-11,2e-11,1.,2.,3.,4.])]:
 for case in range(5):
  n=7;k0=2*np.pi/1.37;kx=2*np.pi*np.arange(-3,4)/2.91
  v=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
  eps=(v@np.diag(values)@v.conj().T+np.diag(kx**2))/k0**2
  z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));d=(z+z.conj().T)/15
  eps2=3*np.eye(n)+.03*(z+z.conj().T)
  d2=(z+z.conj().T)/50
  hs=[.28,.17];target=(case%3)-1
  actual=b.Solution([b.Layer(eps,hs[0]),b.Layer(eps2,hs[1])],1.37,2.91)
  f,g=actual.transmission_gradient(target);grad=sum(2*np.vdot(gg,dd).real for gg,dd in zip(g,[d,d2]))
  ref,dref=independent_transfer([eps,eps2],hs,1.37,2.91,target,[d,d2])
  fd=[]
  for step in [1e-4,1e-5,1e-6]:
   plus=b.Solution([b.Layer(eps+step*d,hs[0]),b.Layer(eps2+step*d2,hs[1])],1.37,2.91).T[3+target]
   minus=b.Solution([b.Layer(eps-step*d,hs[0]),b.Layer(eps2-step*d2,hs[1])],1.37,2.91).T[3+target]
   fd.append((plus-minus)/(2*step))
  records.append(dict(family=family,case=case,target=target,value_error=abs(f-ref),gradient_error=abs(grad-dref),fd_errors=[abs(x-grad) for x in fd],balance_error=abs(actual.T.sum()+actual.R.sum()-1),residual=actual.residual))

# Reverse reciprocal propagation: transmission totals agree for asymmetric stacks
# in matching exterior media, and reflected powers need not agree order by order.
for case in range(6):
 n=9;orders=np.arange(-4,5);dd=orders[:,None]-orders[None,:]
 eps=[np.eye(n)+3*w*np.sinc(dd*w)*np.exp(-2j*np.pi*dd*c) for w,c in [(0.23,.17),(.66,.71),(.37,.43)]]
 hs=[.13,.17,.22]
 top=1.2;bottom=1.5
 # For all propagating individual incident modes, reciprocity is tested through
 # independent transfer comparisons instead of reinterpreting total T symmetry.
 actual=b.Solution([b.Layer(e,h) for e,h in zip(eps,hs)],1.51,2.6,n_top=top,n_bottom=bottom)
 for target in [-1,0,1]:
  f=actual.T[4+target]
  ref,_=independent_transfer(eps,hs,1.51,2.6,target,ntop=top,nbot=bottom)
  records.append(dict(family='unequal_exteriors',case=case,target=target,value_error=abs(f-ref),balance_error=abs(actual.T.sum()+actual.R.sum()-1)))

result={'source_sha256':sha,'check':'independent transfer/Frechet comparison at repeated and zero modal eigenvalues','records':records,'max_value_error':max(x['value_error'] for x in records),'max_gradient_error':max(x.get('gradient_error',0) for x in records),'max_balance_error':max(x['balance_error'] for x in records),'limits':'Review-authored finite matrix numerical checks; not external researcher validation, not a continuum proof. Degenerate matrix tests deliberately use general positive Hermitian operators, which need not be Toeplitz convolutions of physical epsilon.'}
(base/'numeric-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
