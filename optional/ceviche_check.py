#!/usr/bin/env python3
"""Optional Ceviche API/units smoke test; NOT an independent grating validation.

Run in a separate environment containing the genuine fancompute/ceviche package:
    python optional/ceviche_check.py --output ceviche-smoke.json
No Ceviche code is vendored. The analytic reference is a driven periodic Fourier
mode of the discrete scalar Maxwell system, not a continuum approximation.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        import numpy as np
        from ceviche import fdfd_ez
        from ceviche.constants import C_0, EPSILON_0, MU_0
    except ImportError as exc:
        raise SystemExit(
            "Ceviche unavailable. Install the MIT fancompute/ceviche package in a separate environment; no validation was run."
        ) from exc
    nx, ny, dx = 32, 24, 40e-9
    wavelength_m, epsilon = 1.55e-6, 4.0
    omega = 2 * np.pi * C_0 / wavelength_m
    mx, my = 1, 2
    mode = np.exp(
        2j
        * np.pi
        * (mx * np.arange(nx)[:, None] / nx + my * np.arange(ny)[None, :] / ny)
    )
    k_discrete_sq = (2 * np.sin(np.pi * mx / nx) / dx) ** 2 + (
        2 * np.sin(np.pi * my / ny) / dx
    ) ** 2
    # Ceviche's documented Ez convention: (-Laplacian/mu - eps0*w²*eps)E = i*w*J.
    current = (
        (k_discrete_sq / MU_0 - EPSILON_0 * omega**2 * epsilon) * mode / (1j * omega)
    )
    sim = fdfd_ez(omega, dx, np.full((nx, ny), epsilon), [0, 0])
    _, _, ez = sim.solve(current)
    error = float(np.linalg.norm(ez - mode) / np.linalg.norm(mode))
    result = {
        "test": "periodic-driven-uniform-discrete-mode",
        "scope": "package API, source units and discrete homogeneous operator only; not grating validation",
        "ceviche_version": importlib.metadata.version("ceviche"),
        "relative_field_error": error,
        "pass": error < 1e-8,
        "grid": [nx, ny],
        "dx_m": dx,
        "wavelength_m": wavelength_m,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
