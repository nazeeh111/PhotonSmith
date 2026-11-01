#!/usr/bin/env python3
"""Bounded three-layer lossless periodic-grating Meep helper.

Derived from the independently exercised PhotonSmith adapter. This helper is
not a general geometry importer or a convergence certificate.

Original adapter description:
Optional standalone full-wave Meep FDTD check; not used by PhotonSmith core.

Execute with genuine pymeep in a separate environment:
  python optional/meep_validate.py design.json --output meep-40.json --resolution 40
  python optional/meep_validate.py design.json --output meep-80.json --resolution 80

This owned derivative was exercised with genuine Meep 1.35.0 and independent
analytic, flux, phase and convergence checks. Each new run still needs its own validation.
Input: {params:[w0,w1,w2,s0,s1,s2], wavelength:1.55, material_index:2.0}.
Units: um; normal incidence in air; real scalar index; fixed 3 x 0.30um layers.
Meep coordinates are x (periodic), y (propagation), z (invariant Ez field).
"""

import sys
sys.dont_write_bytecode = True
import argparse
import json
import math
import os
from pathlib import Path
import time


def load_config(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw,dict): raise ValueError('Input JSON must be an object.')
    cfg = raw.get("configuration", raw.get("config", raw))
    if not isinstance(cfg, dict): raise ValueError('Configuration must be an object.')
    allowed={'params','wavelength','material_index','period','layer_height','polarization','target_order','name','provenance','validation','min_feature'}
    unknown=set(cfg)-allowed
    if unknown: raise ValueError('Unsupported configuration keys: '+', '.join(sorted(unknown)))
    for key in ['absorption','conductivity','dispersion','substrate','geometry','layers']:
        if key in cfg: raise ValueError('Unsupported physical-model field: '+key)
    if cfg.get('polarization', 'TE') != 'TE': raise ValueError('Only electric field perpendicular to the propagation plane (TE) is supported.')
    params = cfg.get("params")
    if not isinstance(params, list) or len(params) != 6:
        raise ValueError(
            "Expected six params: three width fractions followed by three periodic shift fractions."
        )
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in params):
        raise ValueError('All six parameters must be real numbers, not strings/bools.')
    params = [float(v) for v in params]
    for key in ['wavelength','material_index','period','layer_height']:
        if key in cfg and (isinstance(cfg[key],bool) or not isinstance(cfg[key],(int,float))):
            raise ValueError(key+' must be a real number.')
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
    ratio=period/wavelength
    if abs(ratio-round(ratio)) < 1e-9:
        raise ValueError('Exact grazing diffraction cutoffs are unsupported; choose a non-cutoff wavelength.')
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
    p.add_argument('--case', choices=['design','vacuum','slab'], default='design')
    p.add_argument('--translation', type=float, default=0.0)
    args = p.parse_args()
    if not sys.flags.isolated: p.error('Run with the pinned Python executable and -I for isolated imports.')
    if any(k.startswith('DYLD_') or k in ['LD_LIBRARY_PATH','LD_PRELOAD'] for k in os.environ):
        p.error('Dynamic-loader environment overrides are unsupported; run in a clean process environment.')
    if args.output.suffix.lower() != '.json': p.error('Output must use .json; the sibling .npz is reserved for raw arrays.')
    cfg = load_config(args.config)
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        p.error('Output already exists; choose a fresh path. Failed partial outputs are preserved.')
    if not math.isfinite(args.translation): p.error('Translation must be finite.')
    period_pixels=cfg['period']*args.resolution
    if abs(period_pixels-round(period_pixels))>1e-8:
        p.error('period × resolution must be an integer; Meep otherwise changes the physical period. Choose commensurate resolutions.')
    cells=math.ceil(cfg['period']*args.resolution)*math.ceil((3*cfg['layer_height']+2*(args.air+args.pml))*args.resolution)
    if cells > 2_000_000: p.error('Requested grid exceeds the helper two-million-cell scope. Author a separately reviewed simulation.')
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
    original_cfg = json.loads(json.dumps(cfg))
    for j in range(3, 6): cfg['params'][j] = (cfg['params'][j] + args.translation) % 1
    raw_data = {}
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

    if args.case == 'vacuum': geometry = []
    elif args.case == 'slab':
        geometry = [mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, device_h, mp.inf), material=mp.Medium(index=cfg['material_index']))]

    def project(sim, mon, component, tag):
        arr = np.asarray(sim.get_dft_array(mon, component, 0)).squeeze()
        x, _, _, weights = sim.get_array_metadata(dft_cell=mon)
        weights = np.asarray(weights).squeeze()
        raw_data[tag + '_field'] = arr
        raw_data[tag + '_x'] = np.asarray(x)
        raw_data[tag + '_weights'] = weights
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

    def run(geo, collect_field=False, subtract=None, label="reference"):
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
        flux_r = sim.add_flux(freq, 0, 1, mp.FluxRegion(center=mp.Vector3(0,y_reflect),size=mp.Vector3(period,0)))
        flux_t = sim.add_flux(freq, 0, 1, mp.FluxRegion(center=mp.Vector3(0,y_transmit),size=mp.Vector3(period,0)))
        if subtract is not None: sim.load_minus_flux_data(flux_r, subtract)
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
            "re": project(sim, refl, mp.Ez, label+"_re"),
            "rh": project(sim, refl, mp.Hx, label+"_rh"),
            "te": project(sim, tran, mp.Ez, label+"_te"),
            "th": project(sim, tran, mp.Hx, label+"_th"),
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
        result['flux_r'] = mp.get_fluxes(flux_r)[0]
        result['flux_t'] = mp.get_fluxes(flux_t)[0]
        result['flux_data_r'] = sim.get_flux_data(flux_r)
        result['time_final'] = sim.meep_time()
        sim.reset_meep()
        return result

    reference = run([])
    device = run(geometry, collect_field=True, subtract=reference["flux_data_r"], label="device")
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
    kz = np.sqrt((2*np.pi*freq)**2 - (2*np.pi*orders/period)**2)
    z_ref = y_reflect + device_h/2
    z_tran = y_transmit + device_h/2
    inc0 = reference['re'][zero] * np.exp(-1j*2*np.pi*freq*z_ref)
    tc = device['te'] / inc0 * np.exp(-1j*kz*(z_tran-device_h))
    rc = er / inc0 * np.exp(1j*kz*z_ref)
    for label, values in [('reference',reference),('device',device)]:
        for key in ['re','rh','te','th']: raw_data[label + '_projected_' + key] = values[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **raw_data)
    complex_pair = lambda a: {'real': np.real(a).tolist(), 'imag': np.imag(a).tolist()}
    result = {
        'case': args.case,
        'translation_fraction': args.translation,
        'original_config': original_cfg,
        'complex_T_at_z_device_end_incident_at_z0': complex_pair(tc),
        'complex_R_at_z0_incident_at_z0': complex_pair(rc),
        'complex_incident_at_z0': complex_pair(inc0),
        'integrated_flux_T': device['flux_t'] / reference['flux_r'],
        'integrated_flux_R': -device['flux_r'] / reference['flux_r'],
        'empty_integrated_flux_ratio': reference['flux_t'] / reference['flux_r'],
        'reference_integrated_incident_flux': reference['flux_r'],
        'reference_projected_incident_power': incident,
        'final_meep_time': device['time_final'],
        'source_waveform': {'kind':'GaussianSource','frequency':freq,'fwidth':.15*freq,'cutoff':6},
        'coordinates': {'meep_y_equals_physical_z_minus':device_h/2,'source_z_um':y_source+device_h/2,'reflection_z_um':z_ref,'transmission_z_um':z_tran,'transmission_reference_z_um':device_h,'incident_reference_z_um':0},
        'complex_phase_derivation':'inc0=Einc(zr)*exp(-i*k0*zr); t(h)=Et(zt)/inc0*exp(-i*kzm*(zt-h)); r(0)=(Er(zr)-Einc(zr))/inc0*exp(+i*kzm*zr). x=Meep_x+P/2. No fitted phase.',
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
