"""Explicit optional native backend; build with scripts/build_native.py first.

No automatic compilation, downloads, model inference or silent fallback.
"""
from __future__ import annotations
import ctypes as ct
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

_library = None

def _load():
    global _library
    if _library is None:
        suffix='.dylib' if sys.platform=='darwin' else '.so'
        path=Path(__file__).with_name('_boundary_native'+suffix)
        if not path.is_file():
            raise RuntimeError('Optional native kernel unavailable. Build explicitly with python scripts/build_native.py from a source checkout.')
        lib=ct.CDLL(str(path))
        ptr=ct.c_void_p
        lib.ps_bar.argtypes=[ct.c_int,ptr,ct.c_double,ct.c_double,ct.c_double,ct.c_double,ct.c_int,ptr,ptr,ptr,ct.c_int]
        lib.ps_bar.restype=ct.c_int
        lib.ps_solve.argtypes=[ct.c_int,ct.c_int,ct.c_double,ct.c_double,ct.c_double,ct.c_double,ptr,ptr,ct.c_int,ct.c_int,ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr,ct.c_int]
        lib.ps_solve.restype=ct.c_int
        _library=lib
    return _library

def _int(value,low,high,name):
    if isinstance(value,(bool,np.bool_)) or not np.isfinite(value) or int(value)!=value or not low<=value<=high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}].')
    return int(value)

def objective_gradient(params,wavelength=1.55,target_order=1,harmonics=5,period=2.4,layer_height=.3,material_index=2.):
    m=_int(harmonics,1,128,'harmonics')
    target=_int(target_order,-m,m,'target_order')
    p=np.ascontiguousarray(params,dtype=np.float64)
    if p.shape!=(6,):raise ValueError('Expected six geometry parameters.')
    value=ct.c_double();grad=np.empty(6,dtype=np.float64);error=ct.create_string_buffer(512)
    status=_load().ps_bar(m,p.ctypes.data,wavelength,period,layer_height,material_index,target,ct.byref(value),grad.ctypes.data,error,len(error))
    if status:raise ValueError(error.value.decode('utf-8','replace'))
    if not np.isfinite(value.value) or not np.all(np.isfinite(grad)):
        raise ArithmeticError('Nonfinite native objective or gradient.')
    return value.value,grad

def solve(layers,wavelength,period,n_top=1.,n_bottom=1.,target_order=None):
    if not layers or len(layers)>64:raise ValueError('Expected 1..64 layers.')
    first=np.asarray(layers[0].epsilon)
    if first.ndim!=2 or first.shape[0]!=first.shape[1]:raise ValueError('Expected square layer matrices.')
    n=first.shape[0]
    _int(n,3,257,'matrix order')
    if n%2!=1:raise ValueError('Retain an odd number of Fourier orders.')
    count=len(layers);blocks=2*(count+1)
    if (9*n-2)*blocks*n>8_000_000:raise ValueError('Band allocation limit exceeded.')
    for layer in layers:
        if np.shape(layer.epsilon)!=(n,n):raise ValueError('Layer matrix dimensions differ.')
    target=0 if target_order is None else _int(target_order,-n//2+1,n//2,'target_order')
    eps=np.ascontiguousarray(np.stack([layer.epsilon for layer in layers]),dtype=np.complex128)
    h=np.ascontiguousarray([layer.thickness for layer in layers],dtype=np.float64)
    r=np.empty(n);t=np.empty(n);ar=np.empty(n,dtype=np.complex128);at=np.empty_like(ar)
    state=np.empty((blocks,n),dtype=np.complex128)
    grad=np.empty((count,n,n),dtype=np.complex128) if target_order is not None else np.empty(1,dtype=np.complex128)
    residual=ct.c_double();error=ct.create_string_buffer(512)
    status=_load().ps_solve(n,count,wavelength,period,n_top,n_bottom,eps.ctypes.data,h.ctypes.data,target,int(target_order is not None),r.ctypes.data,t.ctypes.data,ar.ctypes.data,at.ctypes.data,state.ctypes.data,grad.ctypes.data,ct.byref(residual),error,len(error))
    if status:raise ValueError(error.value.decode('utf-8','replace'))
    if not all(np.all(np.isfinite(x)) for x in (r,t,ar,at,state)) or not np.isfinite(residual.value):
        raise ArithmeticError('Nonfinite native Maxwell result.')
    if target_order is not None and not np.all(np.isfinite(grad)):
        raise ArithmeticError('Nonfinite native Maxwell gradient.')
    return SimpleNamespace(R=r,T=t,r=ar,t=at,state=state,gradient=grad if target_order is not None else None,
                           residual=residual.value,orders=np.arange(-n//2+1,n//2+1))
