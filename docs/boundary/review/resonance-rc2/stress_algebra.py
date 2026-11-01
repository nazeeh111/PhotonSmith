"""Independent pair-algebra verification; no production edit."""
import sys,json,math
from pathlib import Path
import numpy as np
import mpmath as mp
from reproduce import boundary,mquot
mp.mp.dps=70
def poly(coeff,z):return sum(c*z**i for i,c in enumerate(coeff))
def polydiv(coeff,x,y):return sum(c*sum(x**j*y**(i-1-j) for j in range(i)) for i,c in enumerate(coeff) if i)
def sinc(x):return np.sinc(x/np.pi)
def sinhc(x):return 1+x*x/6+x**4/120+x**6/5040+x**8/362880 if abs(x)<.01 else np.sinh(x)/x
def pair(x,y,a,stable_k=False):
    zmid=(x+y)*.5;scale=1
    if max(abs(x),abs(y))<1e-3:
        cc=[1.,-.5,1/24,-1/720,1/40320];ss=[1.,-1/6,1/120,-1/5040,1/362880]
        c1,c2=poly(cc,x),poly(cc,y);s1,s2=poly(ss,x),poly(ss,y)
        dc,ds=polydiv(cc,x,y),polydiv(ss,x,y)
    elif x>0 and y>0:
        xi,xj=np.sqrt(x),np.sqrt(y);u=(xi+xj)*.5;v=(x-y)/(2*(xi+xj))
        c1,c2=np.cos(xi),np.cos(xj);s1,s2=sinc(xi),sinc(xj)
        dc=-.5*sinc(u)*sinc(v);ds=(np.cos(u)*sinc(v)-sinc(u)*np.cos(v))/(2*xi*xj)
        if stable_k:k=(sinc(2*v)-sinc(2*u))/(2*xi*xj)
    elif x<0 and y<0:
        xi,xj=np.sqrt(-x),np.sqrt(-y);u=(xi+xj)*.5;v=-(x-y)/(2*(xi+xj));scale=np.exp(-u)
        c1=.5*(np.exp(v)+np.exp(-2*u-v));c2=.5*(np.exp(-v)+np.exp(-2*u+v))
        s1=.5*(np.exp(v)-np.exp(-2*u-v))/xi;s2=.5*(np.exp(-v)-np.exp(-2*u+v))/xj
        cu=.5*(1+np.exp(-2*u));su=-np.expm1(-2*u)/(2*u)
        dc=-.5*su*sinhc(v);ds=-(cu*sinhc(v)-su*np.cosh(v))/(2*xi*xj)
        if stable_k:k=(-np.expm1(-4*u)/(4*u)-np.exp(-2*u)*sinhc(2*v))/(2*xi*xj)
    else:raise ValueError('Unexpected cross-zero pair outside small polynomial')
    cav=(c1+c2)*.5;sav=(s1+s2)*.5
    if not stable_k or max(abs(x),abs(y))<1e-3:k=ds*cav-sav*dc
    d1=c1-1j*(a*a+x)*s1/(2*a);d2=c2-1j*(a*a+y)*s2/(2*a)
    dd=dc-1j*((a*a+zmid)*ds+sav)/(2*a)
    lr=(.5*s1*s2-1j*((a*a-zmid)*k-.5*(s1*c2+s2*c1))/(2*a))/(d1*d2)
    lt=-scale*dd/(d1*d2)
    return [lr,lt]
def main():
    rows=[]
    for a in np.logspace(-8,8,17):
        for base in [0.,1e-4,-1e-4,-1.,-100.,-10000.,1.,math.pi**2,(10*math.pi)**2]:
            for gaprel in [2e-16,1e-14,1e-12,1e-10,1e-8,1e-6,8e-6,1e-4]:
                gap=gaprel*max(1,abs(base));x=float(base-gap/2);y=float(base+gap/2)
                if x==y:continue
                if abs(x-y)<1e-5:approx=pair(x,y,float(a));method='algebra'
                else:
                    f=boundary._scattering(np.array([x,y]),1.,float(a));approx=[(f[k][0]-f[k][1])/(x-y) for k in (0,1)];method='direct'
                oracle=mquot(x,y,1.,float(a))
                for k,name in enumerate(['r','t']):
                    exact=complex(oracle[k]);error=abs(approx[k]-exact)
                    rows.append({'a':float(a),'base':base,'x':x,'y':y,'method':method,'component':name,'oracle':[exact.real,exact.imag],'approx':[approx[k].real,approx[k].imag],'normalized_error':error/max(1,abs(exact)),'relative_error':error/max(1e-300,abs(exact))})
    result={'count':len(rows),'max_normalized_error':max(x['normalized_error'] for x in rows),'max_algebra_normalized_error':max(x['normalized_error'] for x in rows if x['method']=='algebra'),'max_algebra_relative_error':max(x['relative_error'] for x in rows if x['method']=='algebra'),'worst_normalized':sorted(rows,key=lambda r:r['normalized_error'],reverse=True)[:20],'worst_algebra':sorted((r for r in rows if r['method']=='algebra'),key=lambda r:r['normalized_error'],reverse=True)[:20]}
    Path(__file__).with_name('algebra-stress-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2));print('worst',result['worst_normalized'][0]);print('worstalgebra',result['worst_algebra'][0])
if __name__=='__main__':main()
