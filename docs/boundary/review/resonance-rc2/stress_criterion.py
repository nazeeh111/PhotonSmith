"""Stress proposed criterion against high-precision scalar divided differences."""
import sys,json,math
from pathlib import Path
import numpy as np
import mpmath as mp
from reproduce import boundary,mmodal,mquot
mp.mp.dps=70
def main():
    threshold=float(sys.argv[1]) if len(sys.argv)>1 else 1e-5
    rows=[]
    for a in np.logspace(-8,8,17):
        for base in [0.,1e-4,-1e-4,-1.,-100.,-10000.,1.,math.pi**2,(10*math.pi)**2]:
            for gaprel in [2e-16,1e-14,1e-12,1e-10,1e-8,1e-6,8e-6,1e-4]:
                gap=gaprel*max(1,abs(base));x=float(base-gap/2);y=float(base+gap/2)
                if x==y:continue
                f=boundary._scattering(np.array([x,y]),1.,float(a));close=abs(x-y)*max(1,a,1/a)<threshold
                if close:
                    mid=boundary._scattering(np.array([(x+y)*.5]),1.,float(a));approx=[mid[2][0],mid[3][0]]
                else:approx=[(f[k][0]-f[k][1])/(x-y) for k in (0,1)]
                oracle=mquot(x,y,1.,float(a))
                for k,name in enumerate(['r','t']):
                    exact=complex(oracle[k]);error=abs(approx[k]-exact)
                    rows.append({'a':float(a),'base':base,'x':x,'y':y,'method':'midpoint' if close else 'direct','component':name,'oracle':[exact.real,exact.imag],'approx':[approx[k].real,approx[k].imag],'normalized_error':error/max(1,abs(exact)),'relative_error':error/max(1e-300,abs(exact))})
    result={'count':len(rows),'max_normalized_error':max(x['normalized_error'] for x in rows),'max_relative_error':max(x['relative_error'] for x in rows),'worst_normalized':sorted(rows,key=lambda r:r['normalized_error'],reverse=True)[:20],'worst_relative':sorted(rows,key=lambda r:r['relative_error'],reverse=True)[:20]}
    Path(__file__).with_name(('criterion-stress-'+str(threshold)+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['worst_normalized','worst_relative']},indent=2));print('worst',result['worst_normalized'][0])
if __name__=='__main__':main()
