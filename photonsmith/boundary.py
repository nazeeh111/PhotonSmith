"""Stable lossless TE modal scattering and an analytic matrix-function adjoint.

Established scattering, adjoint, and divided-difference techniques, independently
implemented here. Consistent caller units; exp(-i omega t); normal incidence.
There are no eigenvector derivatives, learned approximations, layer-subdivision
parameters, or finite-difference gradients. This is an experimental backend.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, cholesky
from scipy.linalg.lapack import zgbtrf, zgbtrs


def _scattering(values, h, eta):
    """Modal slab reflection/transmission and derivatives in eigenvalue lambda.

    Each port uses the same real reference admittance eta. Zero modal wave
    numbers have analytic limits. For negative eigenvalues all growing
    hyperbolic functions are scaled out before evaluating scattering ratios.
    """
    z = np.asarray(values, dtype=float) * h**2
    a = eta * h
    if not np.all(np.isfinite(z)) or not np.isfinite(a) or max(abs(a), np.sqrt(np.max(abs(z)))) > 1e8:
        raise ValueError("Optical thickness exceeds the supported floating-point phase range.")
    c, s, cp, sp = [np.empty_like(z) for _ in range(4)]
    scale = np.ones_like(z)
    small = np.abs(z) < 1e-3
    x = z[small]
    c[small] = 1 - x/2 + x**2/24 - x**3/720 + x**4/40320
    s[small] = 1 - x/6 + x**2/120 - x**3/5040 + x**4/362880
    cp[small] = -.5 + x/12 - x**2/240 + x**3/10080
    sp[small] = -1/6 + x/60 - x**2/1680 + x**3/90720
    positive = (~small) & (z > 0)
    x = np.sqrt(z[positive])
    c[positive], s[positive] = np.cos(x), np.sin(x)/x
    negative = (~small) & (z < 0)
    x = np.sqrt(-z[negative])
    scale[negative] = np.exp(-x)
    c[negative] = (1 + np.exp(-2*x))/2
    s[negative] = -np.expm1(-2*x)/(2*x)
    nonsmall = ~small
    cp[nonsmall] = -.5*s[nonsmall]
    sp[nonsmall] = (c[nonsmall]-s[nonsmall])/(2*z[nonsmall])
    den = c - 1j*(a*a+z)*s/(2*a)
    num = -1j*(a*a-z)*s/(2*a)
    dd = cp - 1j*(s+(a*a+z)*sp)/(2*a)
    dn = -1j*(-s+(a*a-z)*sp)/(2*a)
    r, t = num/den, scale/den
    dr = h*h*(dn*den-num*dd)/den**2
    dt = -h*h*scale*dd/den**2
    return r, t, dr, dt


def _loewner(values, h, eta):
    """Fréchet derivatives of analytic slab functions in a Hermitian eigenbasis."""
    r, t, _, _ = _scattering(values, h, eta)
    delta = values[:, None]-values[None, :]
    close = np.abs(delta)*h*h < 1e-5
    safe = np.where(close, 1, delta)
    lr = (r[:, None]-r[None, :])/safe
    lt = (t[:, None]-t[None, :])/safe
    midpoint = (values[:, None]+values[None, :])*0.5
    _, _, dr, dt = _scattering(midpoint[close], h, eta)
    lr[close], lt[close] = dr, dt
    return lr, lt


@dataclass
class Layer:
    """Hermitian Fourier convolution of relative permittivity and thickness."""
    epsilon: np.ndarray
    thickness: float


class Solution:
    """Port solution, reusable band factorization and matrix-permittivity VJP."""
    def __init__(self, layers, wavelength, period, n_top=1.0, n_bottom=1.0):
        if not layers or len(layers) > 64:
            raise ValueError("Expected 1..64 lossless layers.")
        scalars = np.asarray([wavelength, period, n_top, n_bottom], dtype=float)
        if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0):
            raise ValueError("Wavelength, period and exterior indices must be finite and positive.")
        first = np.asarray(layers[0].epsilon)
        if first.ndim != 2 or first.shape[0] != first.shape[1]:
            raise ValueError("Permittivity convolutions must be square matrices.")
        n = first.shape[0]
        if n % 2 != 1 or not 3 <= n <= 257:
            raise ValueError("Retain an odd 3..257 Fourier orders, centered on zero.")
        size = 2*(len(layers)+1)
        bandwidth = 3*n-1
        if (3*bandwidth+1)*size*n > 8_000_000:
            raise ValueError("Projected band factorization exceeds the matrix allocation limit.")
        self.n, self.m = n, n//2
        self.k0 = 2*np.pi/wavelength
        self.eta = self.k0
        self.orders = np.arange(-self.m, self.m+1)
        self.kx = 2*np.pi*self.orders/period
        self.kz_top = np.sqrt(((self.k0*n_top)**2-self.kx**2).astype(complex))
        self.kz_bottom = np.sqrt(((self.k0*n_bottom)**2-self.kx**2).astype(complex))
        if (not np.all(np.isfinite(self.kz_top)) or not np.all(np.isfinite(self.kz_bottom))
                or min(np.min(abs(self.kz_top)), np.min(abs(self.kz_bottom))) < 1e-10*self.k0):
            raise ValueError("Nonfinite wave numbers or exact exterior diffraction cutoffs unsupported.")
        self.layers, self.scattering = [], []
        for layer in layers:
            eps = np.asarray(layer.epsilon, dtype=complex)
            h = float(layer.thickness)
            if eps.shape != (n,n) or not np.all(np.isfinite(eps)) or not np.isfinite(h) or h <= 0:
                raise ValueError("Layers need matching finite matrices and positive thickness.")
            if not np.allclose(eps,eps.conj().T,atol=1e-12,rtol=1e-12):
                raise ValueError("Lossless Hermitian permittivity required.")
            eps = (eps+eps.conj().T)*.5
            try:
                cholesky(eps,check_finite=False)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Positive definite permittivity required.") from exc
            q = self.k0**2*eps-np.diag(self.kx**2)
            if not np.all(np.isfinite(q)):
                raise ValueError("Nonfinite Maxwell operator.")
            values,v = eigh(q,check_finite=False)
            r,t,_,_ = _scattering(values,h,self.eta)
            self.layers.append((values,v,h))
            self.scattering.append(((v*r)@v.conj().T,(v*t)@v.conj().T))
        band = np.zeros((3*bandwidth+1,size*n),complex,order='F')
        ri,ci = np.indices((n,n))
        def insert(value,row,col):
            rows,cols=ri+row*n,ci+col*n
            band[2*bandwidth+rows-cols,cols]=value
        eye=np.eye(n)
        rho_top=(self.eta-self.kz_top)/(self.eta+self.kz_top)
        rho_bottom=(self.eta-self.kz_bottom)/(self.eta+self.kz_bottom)
        insert(eye,0,0);insert(-np.diag(rho_top),0,1)
        for j,(r,t) in enumerate(self.scattering):
            # b_j = R a_j + T b_{j+1}; a_{j+1} = T a_j + R b_{j+1}.
            insert(eye,2*j+1,2*j+1)
            insert(-r,2*j+1,2*j);insert(-t,2*j+1,2*j+3)
            insert(eye,2*j+2,2*j+2)
            insert(-t,2*j+2,2*j);insert(-r,2*j+2,2*j+3)
        insert(eye,size-1,size-1)
        insert(-np.diag(rho_bottom),size-1,size-2)
        self.bandwidth=bandwidth
        self.factor,self.pivots,info=zgbtrf(band,bandwidth,bandwidth,overwrite_ab=True)
        if info != 0:
            raise ArithmeticError(f"Maxwell band factorization failed (LAPACK {info}).")
        rhs=np.zeros(size*n,complex)
        rhs[self.m]=2*self.kz_top[self.m]/(self.eta+self.kz_top[self.m])
        self.state=self._solve(rhs).reshape(size,n)
        residual=np.zeros_like(self.state)
        residual[0]=self.state[0]-rho_top*self.state[1]
        for j,(r,t) in enumerate(self.scattering):
            residual[2*j+1]=self.state[2*j+1]-r@self.state[2*j]-t@self.state[2*j+3]
            residual[2*j+2]=self.state[2*j+2]-t@self.state[2*j]-r@self.state[2*j+3]
        residual[-1]=self.state[-1]-rho_bottom*self.state[-2]
        self.residual=float(np.linalg.norm(residual.ravel()-rhs)/np.linalg.norm(rhs))
        self.r=self.state[0]+self.state[1]
        self.r[self.m]-=1
        self.t=self.state[-2]+self.state[-1]
        self.flux_top=self.kz_top.real/self.kz_top[self.m].real
        self.flux_bottom=self.kz_bottom.real/self.kz_top[self.m].real
        self.R=self.flux_top*abs(self.r)**2
        self.T=self.flux_bottom*abs(self.t)**2
        if not np.all(np.isfinite(self.state)):
            raise ArithmeticError("Nonfinite Maxwell solution.")

    def _solve(self,rhs,adjoint=False):
        value,info=zgbtrs(self.factor,self.bandwidth,self.bandwidth,rhs,self.pivots,trans=2 if adjoint else 0)
        if info != 0:
            raise ArithmeticError(f"Maxwell band solve failed (LAPACK {info}).")
        return value

    def transmission_gradient(self,target_order=1):
        """Return F and layer matrices G such that dF=2 Re <G,d epsilon>.

        Eigenvalue multiplicities use analytic divided-difference limits, with
        no eigenvector-gap broadening. Gradients concern Hermitian epsilon only.
        """
        if isinstance(target_order,bool) or int(target_order)!=target_order or abs(target_order)>self.m:
            raise ValueError("Target order must be an integer in the retained basis.")
        q=self.m+int(target_order)
        rhs=np.zeros(self.state.size,complex)
        rhs[-2*self.n+q]=rhs[-self.n+q]=self.flux_bottom[q]*self.t[q]
        adjoint=self._solve(rhs,adjoint=True).reshape(self.state.shape)
        gradients=[]
        for j,(values,v,h) in enumerate(self.layers):
            a,b=v.conj().T@adjoint[2*j+1],v.conj().T@adjoint[2*j+2]
            u,w=v.conj().T@self.state[2*j],v.conj().T@self.state[2*j+3]
            mr=np.outer(a.conj(),u)+np.outer(b.conj(),w)
            mt=np.outer(a.conj(),w)+np.outer(b.conj(),u)
            lr,lt=_loewner(values,h,self.eta)
            g=self.k0**2*(v@(mr*lr+mt*lt).conj()@v.conj().T)
            gradients.append((g+g.conj().T)*.5)
        return float(self.T[q]),gradients


def bar_solution(params,wavelength=1.55,harmonics=5,period=2.4,layer_height=.3,material_index=2.):
    """Adapter for PhotonSmith's original three-bar geometry."""
    from .physics import validate_params,epsilon_matrix
    p=validate_params(params)
    if isinstance(harmonics,bool) or int(harmonics)!=harmonics or not 1<=harmonics<=128:
        raise ValueError("harmonics must be an integer from 1 to 128.")
    if not np.isfinite(material_index) or material_index<1:
        raise ValueError("Material index must be finite and at least one.")
    orders=np.arange(-int(harmonics),int(harmonics)+1)
    layers=[Layer(epsilon_matrix(p[j],p[j+3],orders,material_index),layer_height) for j in range(3)]
    return Solution(layers,wavelength,period)


def objective_gradient(params,wavelength=1.55,target_order=1,harmonics=5,period=2.4,layer_height=.3,material_index=2.):
    from .physics import validate_params,epsilon_matrix
    p=validate_params(params)
    s=bar_solution(p,wavelength,harmonics,period,layer_height,material_index)
    value,geps=s.transmission_gradient(target_order)
    gradient=np.zeros(6)
    for j,g in enumerate(geps):
        _,dw,ds=epsilon_matrix(p[j],p[j+3],s.orders,material_index,True)
        gradient[j]=2*np.vdot(g,dw).real
        gradient[j+3]=2*np.vdot(g,ds).real
    return value,gradient
