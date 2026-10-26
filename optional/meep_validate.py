#!/usr/bin/env python3
"""Optional standalone full-wave Meep FDTD check; not used by PhotonSmith core.

Execute with genuine pymeep in a separate environment:
  python optional/meep_validate.py design.json --output meep-40.json --resolution 40
  python optional/meep_validate.py design.json --output meep-80.json --resolution 80

This file was syntax-checked, but Meep was unavailable during assembly. A run
records raw results, never automatically certifies convergence or fabrication.
Input: {params:[w0,w1,w2,s0,s1,s2], wavelength:1.55, material_index:2.0}.
Units: um; normal incidence in air; real scalar index; fixed 3 x 0.30um layers.
Meep coordinates are x (periodic), y (propagation), z (invariant Ez field).
"""

import argparse
import json
import math
from pathlib import Path
import time


def load_config(path):
    raw = json.loads(Path(path).read_text())
    cfg = raw.get("configuration", raw.get("config", raw))
    params = cfg.get("params")
    if not isinstance(params, list) or len(params) != 6:
        raise ValueError(
            "Expected six params: three width fractions followed by three periodic shift fractions."
        )
    params = [float(v) for v in params]
    wavelength = float(cfg.get("wavelength", 1.55))
    index = float(cfg.get("material_index", 2.0))
    period = float(cfg.get("period", 2.4))
    height = float(cfg.get("layer_height", 0.30))
    if not all(math.isfinite(v) for v in params + [wavelength, index, period, height]):
        raise ValueError("Configuration contains a nonfinite number.")
    if not all(0 < v < 1 for v in params[:3]):
        raise ValueError("Widths must be strictly between zero and one period.")
    if not (
        0.5 <= wavelength <= 5
        and 1 <= index <= 5
        and 0 < period <= 10
        and 0 < height <= 2
    ):
        raise ValueError(
            "Configuration is outside the supported bounded adapter range."
        )
    return {
        "params": params,
        "wavelength": wavelength,
        "material_index": index,
        "period": period,
        "layer_height": height,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=40, help="pixels per um")
    p.add_argument(
        "--pml", type=float, default=1.0, help="absorbing-layer thickness, um"
    )
    p.add_argument(
        "--air", type=float, default=1.5, help="air on each side excluding PML, um"
    )
    p.add_argument(
        "--after-sources",
        type=float,
        default=250.0,
        help="time units um/c after source turns off",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    if not (
        10 <= args.resolution <= 400
        and 0.5 <= args.pml <= 5
        and 1 <= args.air <= 10
        and 20 <= args.after_sources <= 5000
    ):
        p.error(
            "Use resolution 10..400, PML .5..5um, air 1..10um and after-sources 20..5000."
        )
    try:
        import meep as mp
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Genuine pymeep is unavailable; no FDTD result was produced. Use a separate Meep environment, not the unrelated PyPI meep package."
        ) from exc
    if not hasattr(mp, "Simulation"):
        raise SystemExit(
            "Imported meep is not the electromagnetic pymeep package; no simulation was run."
        )
    start = time.perf_counter()
    period, layer_h = cfg["period"], cfg["layer_height"]
    device_h = 3 * layer_h
    sy = device_h + 2 * (args.air + args.pml)
    freq = 1 / cfg["wavelength"]
    # Main coordinates z=0..device_h map to Meep y=-device_h/2..device_h/2.
    y_reflect, y_transmit = (
        -device_h / 2 - 0.35 * args.air,
        device_h / 2 + 0.35 * args.air,
    )
    y_source = -device_h / 2 - 0.75 * args.air
    orders = np.arange(
        -int(math.floor(period / cfg["wavelength"])),
        int(math.floor(period / cfg["wavelength"])) + 1,
    )
    propagating = np.abs(orders * cfg["wavelength"] / period) < 1 - 1e-12
    orders = orders[propagating]
    geometry = []
    for layer in range(3):
        width = cfg["params"][layer] * period
        center_x = (cfg["params"][layer + 3] % 1) * period
        # Use physical x in [0,P) for exported fields, while Meep cell is centered.
        center_x -= period / 2
        center_y = (layer + 0.5) * layer_h - device_h / 2
        for image in (-1, 0, 1):
            x = center_x + image * period
            if x + width / 2 > -period / 2 and x - width / 2 < period / 2:
                geometry.append(
                    mp.Block(
                        center=mp.Vector3(x, center_y),
                        size=mp.Vector3(width, layer_h, mp.inf),
                        material=mp.Medium(index=cfg["material_index"]),
                    )
                )

    def project(sim, mon, component):
        arr = np.asarray(sim.get_dft_array(mon, component, 0)).squeeze()
        x, _, _, weights = sim.get_array_metadata(dft_cell=mon)
        weights = np.asarray(weights).squeeze()
        # Transform to main coordinate x in [0,P), not centered Meep coordinate.
        xs = np.asarray(x) + period / 2
        if arr.ndim != 1 or arr.shape != xs.shape or weights.shape != arr.shape:
            raise RuntimeError(
                "Unexpected monitor layout; refusing to infer sample coordinates."
            )
        return np.array(
            [
                np.sum(weights * arr * np.exp(-2j * np.pi * order * xs / period))
                / period
                for order in orders
            ]
        )

    def run(geo, collect_field=False):
        source = mp.Source(
            mp.GaussianSource(freq, fwidth=0.15 * freq, cutoff=6),
            component=mp.Ez,
            center=mp.Vector3(0, y_source),
            size=mp.Vector3(period, 0),
        )
        sim = mp.Simulation(
            cell_size=mp.Vector3(period, sy),
            geometry=geo,
            sources=[source],
            boundary_layers=[mp.PML(args.pml, direction=mp.Y)],
            k_point=mp.Vector3(),
            resolution=args.resolution,
            dimensions=2,
        )
        refl = sim.add_dft_fields(
            [mp.Ez, mp.Hx],
            freq,
            0,
            1,
            center=mp.Vector3(0, y_reflect),
            size=mp.Vector3(period, 0),
        )
        tran = sim.add_dft_fields(
            [mp.Ez, mp.Hx],
            freq,
            0,
            1,
            center=mp.Vector3(0, y_transmit),
            size=mp.Vector3(period, 0),
        )
        area = None
        if collect_field:
            area = sim.add_dft_fields(
                [mp.Ez],
                freq,
                0,
                1,
                center=mp.Vector3(),
                size=mp.Vector3(period, device_h + 0.5 * args.air),
            )
        sim.run(until_after_sources=args.after_sources)
        result = {
            "re": project(sim, refl, mp.Ez),
            "rh": project(sim, refl, mp.Hx),
            "te": project(sim, tran, mp.Ez),
            "th": project(sim, tran, mp.Hx),
        }
        if area is not None:
            field = np.asarray(sim.get_dft_array(area, mp.Ez, 0)).squeeze()
            xs, ys, _, _ = sim.get_array_metadata(dft_cell=area)
            result["field"] = {
                "x": (np.asarray(xs) + period / 2).tolist(),
                "z": (np.asarray(ys) + device_h / 2).tolist(),
                "real": field.T.real.tolist(),
                "imag": field.T.imag.tolist(),
                "orientation": "rows=z, columns=x; raw DFT, normalize by incident_amplitude for complex comparison",
            }
        sim.reset_meep()
        return result

    reference = run([])
    device = run(geometry, collect_field=True)
    zero = int(np.where(orders == 0)[0][0])
    incident = float(np.real(reference["re"][zero] * np.conj(reference["rh"][zero])))
    if not math.isfinite(incident) or incident <= 0:
        raise RuntimeError(
            "Empty-cell incident power is invalid; no efficiency can be normalized."
        )
    er = device["re"] - reference["re"]
    hr = device["rh"] - reference["rh"]
    transmission = np.real(device["te"] * np.conj(device["th"])) / incident
    reflection = -np.real(er * np.conj(hr)) / incident
    # Do not clip negative values: retain numerical errors for diagnosis.
    result = {
        "status": "executed_not_convergence_certified",
        "solver": "Meep FDTD",
        "meep_version": str(getattr(mp, "__version__", "unknown")),
        "config": cfg,
        "resolution_pixels_per_um": args.resolution,
        "pml_um": args.pml,
        "air_um": args.air,
        "time_after_sources_um_per_c": args.after_sources,
        "orders": orders.tolist(),
        "T": transmission.tolist(),
        "R": reflection.tolist(),
        "transmission": float(transmission.sum()),
        "reflection": float(reflection.sum()),
        "balance": float(transmission.sum() + reflection.sum()),
        "field": device["field"],
        "incident_amplitude": {
            "real": float(reference["re"][zero].real),
            "imag": float(reference["re"][zero].imag),
            "reference_plane_z_um": y_reflect + device_h / 2,
        },
        "transmission_plane_z_um": y_transmit + device_h / 2,
        "runtime_seconds": time.perf_counter() - start,
        "cautions": [
            "Run resolution, PML, air-spacing and time-window convergence studies.",
            "This result alone does not certify design quality, fabrication constraints or 3D behavior.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Wrote " + str(args.output))


if __name__ == "__main__":
    main()
