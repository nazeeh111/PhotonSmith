"""TE scalar Maxwell Fourier-modal solver and exact discrete adjoint.

Units: micrometres. exp(-i omega t); plane wave travels +z. The electric
field is E_perp = E_y, out of the x-z simulation plane. Meep adapters
map propagation to y and therefore use Meep E_z. Dielectrics are lossless,
nonmagnetic and nondispersive. Periodic x; exact exterior radiation.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm, expm_frechet, solve as dense_solve

DEFAULT_PARAMS = np.array([0.45, 0.45, 0.45, 0.15, 0.4, 0.65])
DEFAULTS = dict(period=2.4, layer_height=0.3, material_index=2.0)


def validate_params(params):
    p = np.asarray(params, dtype=float)
    if p.shape != (6,) or not np.all(np.isfinite(p)):
        raise ValueError("Expected six finite parameters: three widths and three shifts.")
    if np.any(p[:3] < 0.15 - 1e-12) or np.any(p[:3] > 0.85 + 1e-12):
        raise ValueError("Widths must lie in [0.15, 0.85] of the period.")
    if np.any(p[3:] < 0) or np.any(p[3:] > 1):
        raise ValueError("Shifts must lie in [0, 1] of the period.")
    return p


def epsilon_matrix(width, shift, orders, material_index=2.0, derivatives=False):
    """Exact Fourier coefficients of a periodic binary rectangular bar.

    Fourier basis exp(i*2*pi*m*x/period). shift is the bar centre.
    """
    d = orders[:, None] - orders[None, :]
    phase = np.exp(-2j * np.pi * d * shift)
    coeff = width * np.sinc(d * width) * phase
    contrast = material_index**2 - 1
    eps = np.eye(len(orders)) + contrast * coeff
    if not derivatives:
        return eps
    dw = contrast * np.cos(np.pi * d * width) * phase
    ds = contrast * coeff * (-2j * np.pi * d)
    return eps, dw, ds


def _system(params, wavelength, harmonics, period, layer_height, material_index):
    p = validate_params(params)
    if not np.isfinite(wavelength) or wavelength <= 0 or not np.isfinite(period) or period <= 0:
        raise ValueError("Wavelength and period must be finite and positive.")
    if (
        not np.isfinite(layer_height)
        or layer_height <= 0
        or not np.isfinite(material_index)
        or material_index < 1
    ):
        raise ValueError("Layer height must be positive and material index at least one.")
    if int(harmonics) != harmonics or not 1 <= harmonics <= 15:
        raise ValueError("harmonics must be an integer from 1 to 15.")
    harmonics = int(harmonics)
    orders = np.arange(-harmonics, harmonics + 1)
    n = len(orders)
    k0 = 2 * np.pi / wavelength
    kx = 2 * np.pi * orders / period
    kz = np.sqrt((k0 * k0 - kx * kx).astype(complex))
    if np.min(np.abs(kz)) < 1e-8:
        raise ValueError("Wavelength lies at a diffraction cutoff; move it slightly.")
    eye = np.eye(n)
    zero = np.zeros((n, n))
    blocks = []
    generators = []
    deps = []
    for j in range(3):
        eps, dw, ds = epsilon_matrix(p[j], p[j + 3], orders, material_index, True)
        q = k0 * k0 * eps - np.diag(kx * kx)
        g = np.block([[zero, eye], [-q, zero]]) * layer_height
        generators.append(g)
        blocks.append(expm(g))
        deps.append((dw, ds))
    a = np.zeros((8 * n, 8 * n), complex)
    b = np.zeros(8 * n, complex)
    a[:n, :n] = 1j * np.diag(kz)
    a[:n, n : 2 * n] = eye
    b[harmonics] = 2j * k0
    for j, pj in enumerate(blocks):
        row = n + 2 * n * j
        col = 2 * n * j
        a[row : row + 2 * n, col : col + 2 * n] = pj
        a[row : row + 2 * n, col + 2 * n : col + 4 * n] = -np.eye(2 * n)
    a[-n:, 6 * n : 7 * n] = -1j * np.diag(kz)
    a[-n:, 7 * n : 8 * n] = eye
    state = dense_solve(a, b, check_finite=False)
    return dict(
        p=p,
        orders=orders,
        n=n,
        k0=k0,
        kx=kx,
        kz=kz,
        a=a,
        b=b,
        state=state,
        generators=generators,
        deps=deps,
        blocks=blocks,
    )


def solve(
    params=DEFAULT_PARAMS,
    wavelength=1.55,
    harmonics=5,
    fields=False,
    nx=96,
    nz=96,
    period=2.4,
    layer_height=0.3,
    material_index=2.0,
):
    """Solve all propagating diffraction efficiencies and optional complex field.

    harmonics=M retains orders -M..M. Evanescent orders carry zero power.
    field[z,x] covers 0.45 um of exterior on each side of the device.
    """
    if fields and (int(nx) != nx or int(nz) != nz or not 8 <= nx <= 1024 or not 8 <= nz <= 1024):
        raise ValueError("Field sample counts must be integers from 8 to 1024.")
    nx, nz = int(nx), int(nz)
    s = _system(params, wavelength, harmonics, period, layer_height, material_index)
    harmonics = int(harmonics)
    n = s["n"]
    state = s["state"]
    incident = np.zeros(n, complex)
    incident[harmonics] = 1
    r = state[:n] - incident
    t = state[6 * n : 7 * n]
    flux = np.maximum(s["kz"].real, 0) / s["k0"]
    T = flux * abs(t) ** 2
    R = flux * abs(r) ** 2
    out = dict(
        orders=s["orders"],
        T=T,
        R=R,
        total_T=float(T.sum()),
        total_R=float(R.sum()),
        power_balance=float(T.sum() + R.sum()),
        amplitudes_t=t,
        amplitudes_r=r,
        residual=float(np.linalg.norm(s["a"] @ state - s["b"]) / np.linalg.norm(s["b"])),
        params=s["p"].copy(),
        wavelength=float(wavelength),
        harmonics=int(harmonics),
        period=float(period),
        layer_height=float(layer_height),
        material_index=float(material_index),
    )
    if fields:
        x = np.linspace(0, period, nx, endpoint=False)
        z = np.linspace(-0.45, 3 * layer_height + 0.45, nz)
        basis = np.exp(1j * np.outer(s["kx"], x))
        f = np.zeros((nz, nx), complex)
        eps = np.ones((nz, nx))
        p = s["p"]
        kz = s["kz"]
        # Eigen decomposition provides inexpensive interior reconstruction; the
        # solved boundary states (not eigenvectors) determine amplitudes.
        eig = []
        for g in s["generators"]:
            vals, vecs = np.linalg.eig(g / layer_height)
            eig.append((vals, vecs, np.linalg.inv(vecs)))
        for iz, zz in enumerate(z):
            if zz < 0:
                fm = incident * np.exp(1j * kz * zz) + r * np.exp(-1j * kz * zz)
            elif zz > 3 * layer_height:
                fm = t * np.exp(1j * kz * (zz - 3 * layer_height))
            else:
                j = min(int(zz / layer_height), 2)
                local = zz - j * layer_height
                vals, v, vi = eig[j]
                fm = (v @ (np.exp(vals * local) * (vi @ state[2 * n * j : 2 * n * (j + 1)])))[:n]
                distance = (x / period - p[j + 3] + 0.5) % 1 - 0.5
                eps[iz, abs(distance) < p[j] / 2] = material_index**2
            f[iz] = fm @ basis
        out.update(x=x, z=z, field=f, epsilon=eps)
    return out


def objective_gradient(
    params=DEFAULT_PARAMS,
    wavelength=1.55,
    target_order=1,
    harmonics=5,
    period=2.4,
    layer_height=0.3,
    material_index=2.0,
):
    """Efficiency and exact real-parameter discrete adjoint gradient.

    A u=b; A^H lambda=c. d efficiency/dp=-2 Re(lambda^H dA/dp u).
    Matrix-exponential Frechet derivatives differentiate each binary layer.
    No geometry or objective finite differences are used in this gradient.
    """
    s = _system(params, wavelength, harmonics, period, layer_height, material_index)
    n = s["n"]
    orders = s["orders"]
    index = np.flatnonzero(orders == target_order)
    if len(index) != 1:
        raise ValueError("Target order is outside the retained Fourier basis.")
    q = int(index[0])
    ratio = max(s["kz"][q].real, 0) / s["k0"]
    t = s["state"][6 * n + q]
    objective = float(ratio * abs(t) ** 2)
    c = np.zeros(8 * n, complex)
    c[6 * n + q] = ratio * t
    adjoint = dense_solve(s["a"].conj().T, c, check_finite=False)
    grad = np.zeros(6)
    for j in range(3):
        start = n + 2 * n * j
        lam = adjoint[start : start + 2 * n]
        u = s["state"][2 * n * j : 2 * n * (j + 1)]
        for k, de in enumerate(s["deps"][j]):
            dg = np.zeros((2 * n, 2 * n), complex)
            dg[n:, :n] = -(s["k0"] ** 2) * de * layer_height
            dp = expm_frechet(s["generators"][j], dg, compute_expm=False)
            grad[j + 3 * k] = -2 * np.real(np.vdot(lam, dp @ u))
    return objective, grad
