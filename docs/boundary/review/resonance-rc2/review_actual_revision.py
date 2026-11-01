"""Independent scalar review of actual Python/C++ revision against 70-digit arithmetic."""
import sys,json,math,ctypes as ct,hashlib
from pathlib import Path
import numpy as np
import mpmath as mp
ROOT=Path('/Users/nazeeh/Documents/ChatGPT/Continue/engineering-accomplishment-2026-10-29')
REV=ROOT/'correction-v2/PhotonSmith';OUT=Path(__file__).parent
sys.path.insert(0,str(REV))
from photonsmith import boundary,native
mp.mp.dps=70
def mmodal(z,a):
    z=mp.mpf(z);a=mp.mpf(a)
    c=mp.cos(mp.sqrt(z));s=mp.sin(mp.sqrt(z))/mp.sqrt(z) if z else mp.mpf(1)
    den=c-1j*(a*a+z)*s/(2*a)
    return -1j*(a*a-z)*s/(2*a)/den,1/den
def oracle(x,y,a):
    if x==y:return [complex(mp.diff(lambda z:mmodal(z,a)[k],mp.mpf(x))) for k in (0,1)]
    fx,fy=mmodal(x,a),mmodal(y,a)
    return [complex((fx[k]-fy[k])/(mp.mpf(x)-mp.mpf(y))) for k in (0,1)]
def main():
    lib=ct.CDLL(str(OUT/'review-pair.dylib'));lib.review_pair.argtypes=[ct.c_double]*4+[ct.c_void_p];lib.review_pair.restype=None
    tests=[]
    for a in np.logspace(-8,8,17):
        for base in [0.,1e-4,-1e-4,-1.,-100.,-10000.,1.,math.pi**2,(10*math.pi)**2]:
            tests.append((float(a),base,base,'equal'))
            for gaprel in [2e-16,1e-14,1e-12,1e-10,1e-8,1e-6,8e-6,1e-4]:
                gap=gaprel*max(1,abs(base));x=float(base-gap/2);y=float(base+gap/2)
                if x!=y:tests.append((float(a),x,y,'grid'))
    for a in [1e-8,1.,1e8]:
        for base in [1e6,-1e6,1e10,-1e10,1e14,-1e14,(1e8-1)**2,-(1e8-1)**2]:
            for gaprel in [2e-16,1e-14,1e-12,1e-10,1e-8,1e-6]:
                gap=gaprel*abs(base);x=base-gap/2;y=base+gap/2
                if x!=y and max(abs(x),abs(y))<=1e16:tests.append((a,x,y,'highphase'))
    rows=[]
    for a,x,y,kind in tests:
        lr,lt=boundary._loewner(np.array([x,y]),1.,a);p=[lr[0,1],lt[0,1]]
        out=np.empty(4);lib.review_pair(x,y,1.,a,out.ctypes.data);c=[complex(*out[:2]),complex(*out[2:])]
        exact=oracle(x,y,a)
        for k,component in enumerate(['r','t']):
            for backend,values in [('python',p),('native',c)]:
                error=abs(values[k]-exact[k]);rows.append({'a':a,'x':x,'y':y,'kind':kind,'component':component,'backend':backend,'normalized_error':float(error/max(1,abs(exact[k]))),'relative_error':float(error/max(1e-300,abs(exact[k]))),'oracle':[exact[k].real,exact[k].imag],'actual':[values[k].real,values[k].imag]})
    result={'scope':'independent coding-agent scalar check, no human external validation','test_pairs':len(tests),'component_backend_checks':len(rows),'source_sha256':{str(p.relative_to(REV)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [REV/'native/boundary.cpp',REV/'photonsmith/boundary.py']},'max_normalized_error':max(r['normalized_error'] for r in rows),'worst':sorted(rows,key=lambda r:r['normalized_error'],reverse=True)[:30],'by_backend':{b:max(r['normalized_error'] for r in rows if r['backend']==b) for b in ['python','native']}}
    (OUT/'actual-revision-review-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='worst'},indent=2));print('Worst',result['worst'][0])
if __name__=='__main__':main()
