"""Review-only analytic-derivative integration; no production edit."""
import sys,json,math
from pathlib import Path
import numpy as np
import mpmath as mp
from reproduce import boundary,mquot
mp.mp.dps=70
NODES={n:np.polynomial.legendre.leggauss(n) for n in (8,16)}
def quadrature(x,y,a,n):
    delta=x-y
    pieces=max(1,math.ceil(abs(delta)*max(1,a,1/a)))
    if pieces>10000:raise ValueError('Review budget exceeded')
    s=np.zeros(2,complex);nodes,weights=NODES[n]
    for j in range(pieces):
        pts=y+(j+.5+.5*nodes)*delta/pieces
        f=boundary._scattering(pts,1.,a)
        s+=np.array([weights@f[2],weights@f[3]])/(2*pieces)
    return s,pieces
def main():
    rows=[]
    for a in np.logspace(-8,8,17):
        for base in [0.,1e-4,-1e-4,-1.,-100.,-10000.,1.,math.pi**2,(10*math.pi)**2]:
            for gaprel in [2e-16,1e-14,1e-12,1e-10,1e-8,1e-6,8e-6,1e-4]:
                gap=gaprel*max(1,abs(base));x=float(base-gap/2);y=float(base+gap/2)
                if x==y:continue
                if abs(x-y)<1e-5:
                    q8,p=quadrature(x,y,float(a),8);approx,p=quadrature(x,y,float(a),16);estim=float(np.max(abs(q8-approx)/np.maximum(1,abs(approx))))
                    method='gauss16'
                else:
                    f=boundary._scattering(np.array([x,y]),1.,float(a));approx=[(f[k][0]-f[k][1])/(x-y) for k in (0,1)];p=0;estim=None;method='direct'
                oracle=mquot(x,y,1.,float(a))
                for k,name in enumerate(['r','t']):
                    exact=complex(oracle[k]);error=abs(approx[k]-exact)
                    rows.append({'a':float(a),'base':base,'x':x,'y':y,'method':method,'pieces':p,'estimator':estim,'component':name,'oracle':[exact.real,exact.imag],'approx':[approx[k].real,approx[k].imag],'normalized_error':error/max(1,abs(exact)),'relative_error':error/max(1e-300,abs(exact))})
    result={'count':len(rows),'max_normalized_error':max(x['normalized_error'] for x in rows),'max_quadrature_normalized_error':max(x['normalized_error'] for x in rows if x['method']=='gauss16'),'max_estimator':max(x['estimator'] or 0 for x in rows),'max_pieces':max(x['pieces'] for x in rows),'worst_normalized':sorted(rows,key=lambda r:r['normalized_error'],reverse=True)[:20],'worst_quadrature':sorted((r for r in rows if r['method']=='gauss16'),key=lambda r:r['normalized_error'],reverse=True)[:20]}
    Path(__file__).with_name('quadrature-stress-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2));print('worst',result['worst_normalized'][0]);print('worstquad',result['worst_quadrature'][0])
if __name__=='__main__':main()
