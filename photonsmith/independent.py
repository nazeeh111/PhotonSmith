"""Independent finite-difference-in-z, Fourier-Galerkin-in-x Maxwell check.

Unlike physics.py this assembles a sparse second-order differential operator,
uses no layer transfer matrices and no matrix exponentials. The scalar field
is E_perp = E_y in the x-z simulation plane (Meep maps it to E_z). Both formulations
share only the exact analytic Fourier representation of a rectangular bar.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from .physics import validate_params, epsilon_matrix, DEFAULT_PARAMS


def solve_fd(
    params=DEFAULT_PARAMS,
    wavelength=1.55,
    harmonics=7,
    steps_per_layer=40,
    period=2.4,
    layer_height=0.3,
    material_index=2.0,
    fields=False,
    nx=96,
):
    p = validate_params(params)
    if not (
        np.isfinite(wavelength)
        and wavelength > 0
        and np.isfinite(period)
        and period > 0
        and np.isfinite(layer_height)
        and layer_height > 0
        and np.isfinite(material_index)
        and material_index >= 1
    ):
        raise ValueError("Invalid optical configuration.")
    if not 1 <= int(harmonics) <= 15 or int(harmonics) != harmonics:
        raise ValueError("Invalid harmonics.")
    if not 4 <= int(steps_per_layer) <= 500 or int(steps_per_layer) != steps_per_layer:
        raise ValueError("steps_per_layer must be an integer between 4 and 500.")
    harmonics = int(harmonics)
    steps_per_layer = int(steps_per_layer)
    nx = int(nx)
    orders = np.arange(-harmonics, harmonics + 1)
    n = len(orders)
    k0 = 2 * np.pi / wavelength
    kx = 2 * np.pi * orders / period
    kz = np.sqrt((k0 * k0 - kx * kx).astype(complex))
    h = layer_height / steps_per_layer
    nz = 3 * steps_per_layer + 1
    # Nodes at interfaces use the average of the two one-sided epsilon limits.
    # This is the trapezoidal finite-volume treatment of the jump in E''.
    eps = [epsilon_matrix(p[j], p[j + 3], orders, material_index) for j in range(3)]
    eye = np.eye(n)
    a = lil_matrix((nz * n, nz * n), dtype=complex)
    b = np.zeros(nz * n, complex)
    for iz in range(nz):
        if iz == 0:
            e = eps[0]
        elif iz == nz - 1:
            e = eps[2]
        elif iz % steps_per_layer == 0:
            j = iz // steps_per_layer
            e = (eps[j - 1] + eps[j]) / 2
        else:
            e = eps[iz // steps_per_layer]
        q = k0 * k0 * e - np.diag(kx * kx)
        sl = slice(iz * n, (iz + 1) * n)
        a[sl, sl] = q - 2 * eye / h**2
        if iz == 0:
            # Ghost node using centered exact outgoing derivative:
            # E' + i kz E=2 i kz incident.
            a[sl, sl] = q - 2 * eye / h**2 + 2j * np.diag(kz) / h
            a[sl, slice(n, 2 * n)] = 2 * eye / h**2
            b[harmonics] = 4j * k0 / h
        elif iz == nz - 1:
            a[sl, sl] = q - 2 * eye / h**2 + 2j * np.diag(kz) / h
            a[sl, slice((iz - 1) * n, iz * n)] = 2 * eye / h**2
        else:
            a[sl, slice((iz - 1) * n, iz * n)] = eye / h**2
            a[sl, slice((iz + 1) * n, (iz + 2) * n)] = eye / h**2
    aa = a.tocsc()
    u = spsolve(aa, b).reshape(nz, n)
    incident = np.zeros(n, complex)
    incident[harmonics] = 1
    t = u[-1]
    r = u[0] - incident
    flux = np.maximum(kz.real, 0) / k0
    T = flux * abs(t) ** 2
    R = flux * abs(r) ** 2
    out = dict(
        orders=orders,
        T=T,
        R=R,
        total_T=float(T.sum()),
        total_R=float(R.sum()),
        power_balance=float(T.sum() + R.sum()),
        amplitudes_t=t,
        amplitudes_r=r,
        residual=float(np.linalg.norm(aa @ u.ravel() - b) / np.linalg.norm(b)),
        steps_per_layer=int(steps_per_layer),
        harmonics=int(harmonics),
    )
    if fields:
        x = np.linspace(0, period, nx, endpoint=False)
        z = np.linspace(0, 3 * layer_height, nz)
        out.update(x=x, z=z, field=u @ np.exp(1j * np.outer(kx, x)))
    return out


def solve_fdfd(
    params=DEFAULT_PARAMS,
    wavelength=1.55,
    nx=96,
    steps_per_layer=40,
    period=2.4,
    layer_height=0.3,
    material_index=2.0,
    fields=False,
):
    """Fully real-space finite-difference Maxwell verification.

    Independent binary-cell integration, periodic second differences in x,
    centered differences in z, exact radiation for the discrete x stencil.
    Converge BOTH nx and steps_per_layer against continuum Maxwell.
    """
    from scipy.sparse import diags, eye, kron, bmat, csc_matrix

    p = validate_params(params)
    if not (
        np.isfinite(wavelength)
        and wavelength > 0
        and np.isfinite(period)
        and period > 0
        and np.isfinite(layer_height)
        and layer_height > 0
        and np.isfinite(material_index)
        and material_index >= 1
    ):
        raise ValueError("Invalid optical configuration.")
    if (
        int(nx) != nx
        or not 16 <= nx <= 512
        or int(steps_per_layer) != steps_per_layer
        or not 4 <= steps_per_layer <= 500
    ):
        raise ValueError("Invalid grid size.")
    nx = int(nx)
    steps_per_layer = int(steps_per_layer)
    hx = period / nx
    hz = layer_height / steps_per_layer
    nz = 3 * steps_per_layer + 1
    k0 = 2 * np.pi / wavelength
    x = np.arange(nx) * hx
    # Integrate physical bar overlap with each finite-volume x cell. No Fourier
    # coefficients or primary-solver assembly are reused here.
    layers = []
    for j in range(3):
        w = p[j] * period
        center = p[j + 3] * period
        fraction = np.zeros(nx)
        for wrap in (-1, 0, 1):
            left = center + wrap * period - w / 2
            right = center + wrap * period + w / 2
            fraction += (
                np.maximum(0, np.minimum(x + hx / 2, right) - np.maximum(x - hx / 2, left)) / hx
            )
        layers.append(1 + (material_index**2 - 1) * fraction)
    lateral = (
        diags(
            [np.ones(nx - 1), -2 * np.ones(nx), np.ones(nx - 1)],
            [-1, 0, 1],
            format="lil",
        )
        / hx**2
    )
    lateral[0, -1] = lateral[-1, 0] = 1 / hx**2
    lateral = lateral.tocsc()
    # For a Fourier eigenvector, the discrete periodic Laplacian has eigenvalue
    # -4/hx^2 sin^2(pi*m/nx). Its DtN map eliminates the discrete exterior.
    m = np.fft.fftfreq(nx) * nx
    kz = np.sqrt((k0 * k0 - 4 * np.sin(np.pi * m / nx) ** 2 / hx**2).astype(complex))
    fourier = np.fft.fft(np.eye(nx), axis=0)
    dtn = np.fft.ifft(kz[:, None] * fourier, axis=0)
    ii = eye(nx, format="csc")
    off = ii / hz**2
    block_rows = []
    for iz in range(nz):
        if iz == 0:
            e = layers[0]
        elif iz == nz - 1:
            e = layers[-1]
        elif iz % steps_per_layer == 0:
            j = iz // steps_per_layer
            e = (layers[j - 1] + layers[j]) / 2
        else:
            e = layers[iz // steps_per_layer]
        diag = lateral + diags(k0 * k0 * e - 2 / hz**2)
        if iz == 0 or iz == nz - 1:
            diag += csc_matrix(2j * dtn / hz)
        row = {iz: diag}
        if iz > 0:
            row[iz - 1] = off * (2 if iz == nz - 1 else 1)
        if iz < nz - 1:
            row[iz + 1] = off * (2 if iz == 0 else 1)
        block_rows.append(row)
    aa = bmat([[row.get(j) for j in range(nz)] for row in block_rows], format="csc")
    b = np.zeros(nz * nx, complex)
    b[:nx] = 4j * k0 / hz
    u = spsolve(aa, b).reshape(nz, nx)
    t = np.fft.fft(u[-1]) / nx
    r = np.fft.fft(u[0] - 1) / nx
    flux = np.maximum(kz.real, 0) / k0
    T = flux * abs(t) ** 2
    R = flux * abs(r) ** 2
    idx = np.argsort(m)
    out = dict(
        orders=m[idx].astype(int),
        T=T[idx],
        R=R[idx],
        total_T=float(T.sum()),
        total_R=float(R.sum()),
        power_balance=float(T.sum() + R.sum()),
        amplitudes_t=t[idx],
        amplitudes_r=r[idx],
        residual=float(np.linalg.norm(aa @ u.ravel() - b) / np.linalg.norm(b)),
        nx=nx,
        steps_per_layer=steps_per_layer,
    )
    if fields:
        epsilon = np.array([layers[min(iz // steps_per_layer, 2)] for iz in range(nz)])
        out.update(x=x, z=np.arange(nz) * hz, field=u, epsilon=epsilon)
    return out
