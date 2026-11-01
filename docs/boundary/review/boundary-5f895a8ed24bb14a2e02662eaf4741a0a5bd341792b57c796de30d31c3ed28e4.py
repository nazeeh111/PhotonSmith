"""Lossless TE Fourier solver using analytic layer Dirichlet-to-Neumann maps.

Established modal/DtN and matrix-function techniques, implemented independently
here. Units are consistent caller units; time convention is exp(-i omega t).
Only scalar TE, real positive permittivity, and normal incidence are supported.
No eigenvector derivatives, fitted models, or finite-difference gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, cholesky
from scipy.linalg.lapack import zgbtrf, zgbtrs


def _functions(s):
    """sqrt(s)cot(sqrt(s)), sqrt(s)csc(sqrt(s)), and their derivatives.

    Analytic limits at zero; decaying exponentials for negative arguments.
    Positive arguments must stay below the first Dirichlet pole.
    """
    s = np.asarray(s, dtype=float)
    f, g, df, dg = [np.empty_like(s) for _ in range(4)]
    small = np.abs(s) < 1e-3
    x = s[small]
    f[small] = 1 - x / 3 - x**2 / 45 - 2 * x**3 / 945 - x**4 / 4725
    g[small] = 1 + x / 6 + 7 * x**2 / 360 + 31 * x**3 / 15120 + 127 * x**4 / 604800
    df[small] = -1 / 3 - 2 * x / 45 - 2 * x**2 / 315 - 4 * x**3 / 4725
    dg[small] = 1 / 6 + 7 * x / 180 + 31 * x**2 / 5040 + 127 * x**3 / 151200
    pos = (~small) & (s > 0)
    x = np.sqrt(s[pos])
    cot, csc = 1 / np.tan(x), 1 / np.sin(x)
    f[pos], g[pos] = x * cot, x * csc
    df[pos] = (cot - x * csc**2) / (2 * x)
    dg[pos] = csc * (1 - x * cot) / (2 * x)
    neg = (~small) & (s < 0)
    x = np.sqrt(-s[neg])
    coth = 1 / np.tanh(x)
    csch = 2 * np.exp(-x) / (-np.expm1(-2 * x))
    f[neg], g[neg] = x * coth, x * csch
    df[neg] = -(coth - x * csch**2) / (2 * x)
    dg[neg] = -csch * (1 - x * coth) / (2 * x)
    return f, g, df, dg


def _loewner(values, h):
    """Fréchet derivative in the eigenbasis, including repeated eigenvalues."""
    s = values * h**2
    f, g, _, _ = _functions(s)
    delta = s[:, None] - s[None, :]
    close = np.abs(delta) < 1e-5
    safe = np.where(close, 1, delta)
    fd = (f[:, None] - f[None, :]) / safe
    fc = (g[:, None] - g[None, :]) / safe
    midpoint = (s[:, None] + s[None, :]) * 0.5
    _, _, df, dg = _functions(midpoint[close])
    fd[close], fc[close] = df, dg
    return h * fd, h * fc


@dataclass
class Layer:
    """Fourier convolution of relative permittivity and physical thickness."""

    epsilon: np.ndarray
    thickness: float


class Solution:
    """Reusable factorization, order powers, and a matrix-permittivity adjoint."""

    def __init__(self, layers, wavelength, period, n_top=1.0, n_bottom=1.0):
        if not layers or len(layers) > 64:
            raise ValueError("Expected 1..64 lossless layers.")
        scalars = np.asarray([wavelength, period, n_top, n_bottom], dtype=float)
        if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0):
            raise ValueError("Wavelength, period, and exterior indices must be finite and positive.")
        first = np.asarray(layers[0].epsilon)
        if first.ndim != 2 or first.shape[0] != first.shape[1]:
            raise ValueError("Permittivity convolutions must be square matrices.")
        n = first.shape[0]
        if n % 2 != 1 or not 3 <= n <= 257:
            raise ValueError("Retain an odd 3..257 Fourier orders, centered on zero.")
        self.n, self.m = n, n // 2
        self.k0 = 2 * np.pi / wavelength
        self.orders = np.arange(-self.m, self.m + 1)
        self.kx = 2 * np.pi * self.orders / period
        self.kz_top = np.sqrt(((self.k0 * n_top)**2 - self.kx**2).astype(complex))
        self.kz_bottom = np.sqrt(((self.k0 * n_bottom)**2 - self.kx**2).astype(complex))
        if min(np.min(abs(self.kz_top)), np.min(abs(self.kz_bottom))) < 1e-10 * self.k0:
            raise ValueError("Exact exterior diffraction cutoffs are outside this contract.")
        self.layers, self.pieces = [], []
        for index, layer in enumerate(layers):
            eps = np.asarray(layer.epsilon, dtype=complex)
            h = float(layer.thickness)
            if eps.shape != (n, n) or not np.all(np.isfinite(eps)) or not np.isfinite(h) or h <= 0:
                raise ValueError("All layers need matching finite matrices and positive thickness.")
            if not np.allclose(eps, eps.conj().T, atol=1e-12, rtol=1e-12):
                raise ValueError("Lossless Hermitian permittivity required.")
            try:
                cholesky(eps, check_finite=False)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Positive definite permittivity required.") from exc
            q = self.k0**2 * eps - np.diag(self.kx**2)
            values, vectors = eigh(q, check_finite=False)
            # Exact subdivision removes propagating Dirichlet poles. Negative
            # eigenvalues need no subdivision: coth/csch use decaying functions.
            count = max(1, int(np.ceil(h * np.sqrt(max(0, values[-1])) / (0.8 * np.pi))))
            if len(self.pieces) + count > 256 or (len(self.pieces) + count + 1) * n > 16384:
                raise ValueError("Layer subdivision exceeds the bounded solver work limit.")
            hh = h / count
            f, g, _, _ = _functions(values * hh**2)
            d = (vectors * (f / hh)) @ vectors.conj().T
            c = (vectors * (g / hh)) @ vectors.conj().T
            self.layers.append((values, vectors, hh))
            self.pieces.extend([(index, d, c)] * count)
        size = len(self.pieces) + 1
        bandwidth = 2 * n - 1
        if (3 * bandwidth + 1) * size * n > 8_000_000:
            raise ValueError("Projected band factorization exceeds the matrix allocation limit.")
        diagonals = [np.zeros((n, n), complex) for _ in range(size)]
        band = np.zeros((3 * bandwidth + 1, size * n), complex, order="F")
        ri, ci = np.indices((n, n))

        def insert(value, row, col):
            rows, cols = ri + row * n, ci + col * n
            band[2 * bandwidth + rows - cols, cols] = value

        for j, (_, d, c) in enumerate(self.pieces):
            diagonals[j] += d
            diagonals[j + 1] += d
            insert(-c, j, j + 1)
            insert(-c, j + 1, j)
        diagonals[0] -= 1j * np.diag(self.kz_top)
        diagonals[-1] -= 1j * np.diag(self.kz_bottom)
        for j, d in enumerate(diagonals):
            insert(d, j, j)
        self.bandwidth = bandwidth
        self.factor, self.pivots, info = zgbtrf(band, bandwidth, bandwidth, overwrite_ab=True)
        if info != 0:
            raise ArithmeticError(f"Maxwell band factorization failed (LAPACK {info}).")
        rhs = np.zeros(size * n, complex)
        rhs[self.m] = -2j * self.kz_top[self.m]
        self.state = self._solve(rhs).reshape(size, n)
        residual = np.array([d @ u for d, u in zip(diagonals, self.state)])
        for j, (_, _, c) in enumerate(self.pieces):
            residual[j] -= c @ self.state[j + 1]
            residual[j + 1] -= c @ self.state[j]
        self.residual = float(np.linalg.norm(residual.ravel() - rhs) / np.linalg.norm(rhs))
        self.r = self.state[0].copy()
        self.r[self.m] -= 1
        self.t = self.state[-1].copy()
        self.flux_top = self.kz_top.real / self.kz_top[self.m].real
        self.flux_bottom = self.kz_bottom.real / self.kz_top[self.m].real
        self.R = self.flux_top * abs(self.r)**2
        self.T = self.flux_bottom * abs(self.t)**2
        if not np.all(np.isfinite(self.state)):
            raise ArithmeticError("Nonfinite Maxwell solution.")

    def _solve(self, rhs, adjoint=False):
        value, info = zgbtrs(self.factor, self.bandwidth, self.bandwidth,
                            rhs, self.pivots, trans=2 if adjoint else 0)
        if info != 0:
            raise ArithmeticError(f"Maxwell band solve failed (LAPACK {info}).")
        return value

    def transmission_gradient(self, target_order=1):
        """Return efficiency and VJP G with dF = 2 Re <G, d epsilon>.

        Each layer matrix is a separate input. Repeated eigenvalues use the
        derivative of the analytic matrix function, with no gap regularizer.
        """
        if isinstance(target_order, bool) or int(target_order) != target_order or abs(target_order) > self.m:
            raise ValueError("Target order must be an integer in the retained basis.")
        q = self.m + int(target_order)
        rhs = np.zeros(self.state.size, complex)
        rhs[-self.n + q] = self.flux_bottom[q] * self.t[q]
        adjoint = self._solve(rhs, adjoint=True).reshape(self.state.shape)
        contractions = [np.zeros((self.n, self.n), complex) for _ in self.layers]
        divided = [_loewner(values, h) for values, _, h in self.layers]
        for j, (index, _, _) in enumerate(self.pieces):
            _, v, _ = self.layers[index]
            a, b = v.conj().T @ adjoint[j], v.conj().T @ adjoint[j + 1]
            u, w = v.conj().T @ self.state[j], v.conj().T @ self.state[j + 1]
            md = np.outer(a.conj(), u) + np.outer(b.conj(), w)
            mc = np.outer(a.conj(), w) + np.outer(b.conj(), u)
            fd, fc = divided[index]
            contractions[index] -= (md * fd - mc * fc).conj()
        gradients = []
        for (_, v, _), g in zip(self.layers, contractions):
            g = self.k0**2 * (v @ g @ v.conj().T)
            gradients.append((g + g.conj().T) * 0.5)
        return float(self.T[q]), gradients


def bar_solution(params, wavelength=1.55, harmonics=5, period=2.4,
                 layer_height=0.3, material_index=2.0):
    """Compatibility adapter for PhotonSmith's original three-bar geometry."""
    from .physics import validate_params, epsilon_matrix

    p = validate_params(params)
    if isinstance(harmonics, bool) or int(harmonics) != harmonics or not 1 <= harmonics <= 128:
        raise ValueError("harmonics must be an integer from 1 to 128.")
    if not np.isfinite(material_index) or material_index < 1:
        raise ValueError("Material index must be finite and at least one.")
    orders = np.arange(-int(harmonics), int(harmonics) + 1)
    layers = [Layer(epsilon_matrix(p[j], p[j + 3], orders, material_index), layer_height) for j in range(3)]
    return Solution(layers, wavelength, period)


def objective_gradient(params, wavelength=1.55, target_order=1, harmonics=5,
                       period=2.4, layer_height=0.3, material_index=2.0):
    from .physics import validate_params, epsilon_matrix

    p = validate_params(params)
    s = bar_solution(p, wavelength, harmonics, period, layer_height, material_index)
    value, geps = s.transmission_gradient(target_order)
    gradient = np.zeros(6)
    for j, g in enumerate(geps):
        _, dw, ds = epsilon_matrix(p[j], p[j + 3], s.orders, material_index, True)
        gradient[j] = 2 * np.vdot(g, dw).real
        gradient[j + 3] = 2 * np.vdot(g, ds).real
    return value, gradient
