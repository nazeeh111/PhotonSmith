from pathlib import Path
import hashlib,importlib.util,sys,json
import numpy as np
from scipy.linalg import expm
base=Path(__file__).parent
source=Path('/Users/nazeeh/Documents/ChatGPT/Continue/engineering-accomplishment-2026-10-29/PhotonSmith/photonsmith/boundary.py')
raw=source.read_bytes();sha=hashlib.sha256(raw).hexdigest();snapshot=base/('boundary-'+sha+'.py')
if not snapshot.exists():snapshot.write_bytes(raw)
spec=importlib.util.spec_from_file_location('review_boundary_scatter',snapshot)
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
records=[]
for h in [.1,1.,100.,1e4,1e6,1e7]:
 value=np.array([-9.]);eta=2.
 r,t,dr,dt=b._scattering(value,h,eta)
 kappa=3.;reference=(eta-1j*kappa)/(eta+1j*kappa)
 derivative=1j*eta/(kappa*(eta+1j*kappa)**2)
 # For small h the second interface still matters; thick limits only score h>=100.
 records.append({'kind':'evanescent_thick_limit','h':h,'r':str(r[0]),'dr':str(dr[0]),'r_limit_error':float(abs(r[0]-reference)),'dr_limit_error':float(abs(dr[0]-derivative)),'t_abs':float(abs(t[0])),'scored':h>=100})

# Arbitrary propagating channels in two directions, with transverse-momentum
# reversal for Lorentz reciprocity. This uses the existing factorization only.
rng=np.random.default_rng(88361)
for case in range(4):
 n=11;m=n//2;orders=np.arange(-m,m+1);diff=orders[:,None]-orders[None,:]
 profiles=[(rng.uniform(.2,.8),rng.uniform(0,1),rng.uniform(1.5,3)) for _ in range(3)]
 eps=[np.eye(n)+(index**2-1)*w*np.sinc(diff*w)*np.exp(-2j*np.pi*diff*c) for w,c,index in profiles]
 layers=[b.Layer(e,h) for e,h in zip(eps,[.13,.27,.31])]
 s=b.Solution(layers,1.17,3.1,n_top=1.2,n_bottom=1.6)
 rhs=np.zeros((s.state.size,2*n),complex)
 rhs[np.arange(n),np.arange(n)]=2*s.kz_top/(s.eta+s.kz_top)
 rhs[-n+np.arange(n),n+np.arange(n)]=2*s.kz_bottom/(s.eta+s.kz_bottom)
 ports=s._solve(rhs).reshape(len(s.state),n,2*n)
 left=ports[0]+ports[1];right=ports[-2]+ports[-1]
 left[:,:n]-=np.eye(n);right[:,n:]-=np.eye(n)
 unit_errors=[];reciprocity=[]
 for i in range(n):
  if s.kz_top[i].real>0:
   powers=(np.sum(s.kz_top.real*abs(left[:,i])**2)+np.sum(s.kz_bottom.real*abs(right[:,i])**2))/s.kz_top[i].real
   unit_errors.append(abs(powers-1))
   for j in range(n):
    if s.kz_bottom[j].real>0:
     a=s.kz_bottom[j].real/s.kz_top[i].real*abs(right[j,i])**2
     rev_i=n-1-i;rev_j=n-1-j
     bb=s.kz_top[rev_i].real/s.kz_bottom[rev_j].real*abs(left[rev_i,n+rev_j])**2
     reciprocity.append(abs(a-bb))
  if s.kz_bottom[i].real>0:
   powers=(np.sum(s.kz_top.real*abs(left[:,n+i])**2)+np.sum(s.kz_bottom.real*abs(right[:,n+i])**2))/s.kz_bottom[i].real
   unit_errors.append(abs(powers-1))
 records.append({'kind':'all_propagating_channels','case':case,'unitarity_error':float(max(unit_errors)),'reciprocity_error':float(max(reciprocity)),'channels_tested':len(unit_errors),'pairs_tested':len(reciprocity)})

# Thick propagating physical slab and epsilon derivative against closed form.
for h in [.2,100.,1e4,1e6]:
 eps=4.;n=5;k0=2*np.pi/1.55
 s=b.Solution([b.Layer(eps*np.eye(n),h)],1.55,2.4)
 v,g=s.transmission_gradient(0)
 phi=k0*np.sqrt(eps)*h
 a=(eps-1)**2/(4*eps)
 da=(1-1/eps**2)/4
 denom=1+a*np.sin(phi)**2
 tref=1/denom
 dref=-(da*np.sin(phi)**2+a*np.sin(2*phi)*k0*h/(2*np.sqrt(eps)))/denom**2
 dactual=2*np.trace(g[0]).real
 records.append({'kind':'thick_propagating_slab','h':h,'transmission_error':float(abs(v-tref)),'derivative_error':float(abs(dactual-dref)),'derivative_relative_error':float(abs(dactual-dref)/max(1,abs(dref))),'balance_error':float(abs(s.T.sum()+s.R.sum()-1))})

result={'source_sha256':sha,'records':records,'limits':'Independent coding-agent numerical review only; not outside-human validation. Auxiliary arbitrary-channel tests use the internal linear factorization, while slab and thick-evanescent references are independently derived analytic formulas.'}
(base/'scattering-v3-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
