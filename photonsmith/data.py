"""Seeded geometry-level dataset generation. No external data or services."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from .physics import solve


def generate(path, geometries=600, seed=20261025, harmonics=5):
    if not isinstance(geometries, int) or geometries < 20:
        raise ValueError("Use at least 20 geometries for a meaningful train/validation/test split.")
    rng = np.random.default_rng(seed)
    params = np.column_stack(
        [
            rng.uniform(0.15, 0.85, (geometries, 3)),
            np.zeros(geometries),
            rng.uniform(0, 1, (geometries, 2)),
        ]
    )
    # Each geometry has a different wavelength in each of four disjoint bins.
    wavelengths = np.linspace(1.3, 1.7, 5)[:-1] + rng.uniform(0, 0.1, (geometries, 4))
    powers = np.empty((geometries, 4, 6))
    fields = np.empty((geometries, 4, 24, 24), np.complex64)
    balance_error = 0.0
    start = time.perf_counter()
    for i, p in enumerate(params):
        for j, wave in enumerate(wavelengths[i]):
            r = solve(p, wavelength=wave, harmonics=harmonics, fields=True, nx=24, nz=24)
            active = np.isin(r["orders"], [-1, 0, 1])
            powers[i, j] = np.r_[r["T"][active], r["R"][active]]
            fields[i, j] = r["field"]
            balance_error = max(balance_error, abs(r["power_balance"] - 1))
        if (i + 1) % 100 == 0:
            print(
                f"Simulated {i + 1}/{geometries} geometries ({time.perf_counter() - start:.1f}s)",
                flush=True,
            )
    # Split once by geometry, never by wavelength row. Retain the exact indices.
    indices = rng.permutation(geometries)
    a, b = int(0.7 * geometries), int(0.85 * geometries)
    metadata = dict(
        seed=seed,
        geometries=geometries,
        samples=geometries * 4,
        harmonics=harmonics,
        seconds=time.perf_counter() - start,
        max_balance_error=balance_error,
        field_shape=[24, 24],
        wavelength_range=[1.3, 1.7],
        material_index=2.0,
        period=2.4,
        layer_height=0.3,
        response_order=["T-1", "T0", "T+1", "R-1", "R0", "R+1"],
        solver="Fourier-modal TE Maxwell",
        split="geometry-level 70/15/15",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        params=params,
        wavelengths=wavelengths,
        powers=powers,
        fields=fields,
        train=indices[:a],
        validation=indices[a:b],
        test=indices[b:],
        metadata=json.dumps(metadata),
    )
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="experiments/dataset.npz")
    p.add_argument("--geometries", type=int, default=600)
    p.add_argument("--seed", type=int, default=20261025)
    a = p.parse_args()
    print(json.dumps(generate(a.output, a.geometries, a.seed), indent=2))
