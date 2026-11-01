"""Independent high-precision review; never used as production code."""
import sys,json,math
from pathlib import Path
import numpy as np
import mpmath as mp
sys.path.insert(0,'/Users/nazeeh/Documents/ChatGPT/Continue/engineering-accomplishment-2026-10-29/PhotonSmith')
from photonsmith import boundary,native
mp.mp.dps=70
def mmodal(lam,h,eta):
    z=mp.mpf(lam)*h*h;a=mp.mpf(eta)*h
    c=mp.cos(mp.sqrt(z));s=mp.sin(mp.sqrt(z))/mp.sqrt(z) if z else mp.mpf(1)
    den=c-1j*(a*a+z)*s/(2*a)
    return (-1j*(a*a-z)*s/(2*a))/den,1/den
def mquot(x,y,h,a):
    fx=mmodal(x,h,a);fy=mmodal(y,h,a)
    return [(fx[k]-fy[k])/(mp.mpf(x)-mp.mpf(y)) for k in (0,1)]
def pwr_mp(q,d,s):
    q=mp.matrix([[mp.mpf(float(x)) for x in row] for row in q]);d=mp.matrix([[mp.mpf(float(x)) for x in row] for row in d]);q=q+s*d
    n=q.rows;eta=mp.mpf(10)**7
    vals,u=mp.eigsy(q);c=u*mp.diag([mp.cos(mp.sqrt(x)) for x in vals])*u.T
    v=u*mp.diag([mp.sin(mp.sqrt(x))/mp.sqrt(x) for x in vals])*u.T
    kt=mp.diag([mp.sqrt(eta*eta-k*k) for k in [-1,0,1]])
    eye=mp.eye(n);inc=mp.matrix([0,1,0])
    # Transfer at h=1: [C,S;-Q S,C]. Unknowns are r,t.
    lhs=mp.matrix(2*n,2*n);rhs=mp.matrix(2*n,1)
    a=c-1j*v*kt;b=-q*v-1j*c*kt
    rhs0=-(c+1j*v*kt)*inc;rhs1=-(-q*v+1j*c*kt)*inc
    for i in range(n):
        rhs[i]=rhs0[i];rhs[n+i]=rhs1[i]
        for j in range(n):
            lhs[i,j]=a[i,j];lhs[i,n+j]=-eye[i,j]
            lhs[n+i,j]=b[i,j];lhs[n+i,n+j]=-1j*kt[i,j]
    ans=mp.lu_solve(lhs,rhs);return abs(ans[n+1])**2
def main():
    eta=1e7;h=1.;x=math.pi**2-4e-6;y=math.pi**2+4e-6
    lr,lt=boundary._loewner(np.array([x,y]),h,eta);true=mquot(x,y,h,eta)
    direct=boundary._scattering(np.array([x,y]),h,eta)
    rows={'scalar':{'eigenvalues':[x,y],'eta':eta,'midpoint_t':[lt[0,1].real,lt[0,1].imag],'mp_t':[float(mp.re(true[1])),float(mp.im(true[1]))],'relative_midpoint_error':float(abs(complex(lt[0,1])-complex(true[1]))/abs(complex(true[1]))),'relative_direct_error':float(abs((direct[1][0]-direct[1][1])/(x-y)-complex(true[1]))/abs(complex(true[1])))}}
    th=math.pi/8;u=np.array([[math.cos(th),-math.sin(th),0],[math.sin(th),math.cos(th),0],[0,0,1.]])
    q=u@np.diag([x,y,3.])@u.T
    # Offdiagonal derivative in the modal basis specifically exercises Loewner terms.
    d=u@np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])@u.T
    eps=(q+np.diag([1.,0.,1.]))/eta**2
    wl=2*math.pi/eta;period=2*math.pi
    sol=boundary.Solution([boundary.Layer(eps,h)],wl,period);val,g=sol.transmission_gradient(0)
    native_sol=native.solve([boundary.Layer(eps,h)],wl,period,target_order=0)
    native_grad=2*np.vdot(native_sol.gradient[0],d/eta**2).real
    pred=2*np.vdot(g[0],d/eta**2).real
    oracle=float(mp.diff(lambda s:pwr_mp(q,d,s),mp.mpf(0)))
    fds=[]
    for step in [1e-6,1e-7,1e-8,1e-9,1e-10]:
        vals=[boundary.Solution([boundary.Layer(eps+sign*step*d/eta**2,h)],wl,period).T[1] for sign in [1,-1]]
        fds.append({'step':step,'gradient':float((vals[0]-vals[1])/(2*step))})
    rows['full']={'Q':q.tolist(),'direction_Q':d.tolist(),'epsilon':eps.tolist(),'wavelength':wl,'period':period,'value':val,'native_value':float(native_sol.T[1]),'mp_value':float(pwr_mp(q,d,mp.mpf(0))),'python_gradient':float(pred),'native_gradient':float(native_grad),'mp_gradient':oracle,'finite_differences':fds,'positive_epsilon_min':float(np.linalg.eigvalsh(eps).min())}
    path=Path(__file__).with_name('reproduction-v1.json');path.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
