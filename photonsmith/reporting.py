"""JSON-safe public simulation data and independent verification."""

import numpy as np
from .physics import solve
from .independent import solve_fdfd
from .design import geometry_audit


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def efficiency(result, order):
    return float(result["T"][np.flatnonzero(result["orders"] == order)[0]])


def public_simulation(
    params,
    wavelength=1.55,
    target_order=1,
    material_index=2.0,
    min_feature=0.36,
    harmonics=9,
    spectrum=True,
):
    r = solve(
        params,
        wavelength,
        harmonics,
        fields=True,
        nx=96,
        nz=97,
        material_index=material_index,
    )
    out = dict(
        params=list(params),
        wavelength=wavelength,
        target_order=target_order,
        material_index=material_index,
        min_feature=min_feature,
        orders=r["orders"],
        T=r["T"],
        R=r["R"],
        transmission=r["total_T"],
        reflection=r["total_R"],
        balance=r["power_balance"],
        efficiency=efficiency(r, target_order),
        residual=r["residual"],
        field=dict(
            nx=len(r["x"]),
            nz=len(r["z"]),
            x=r["x"],
            z=r["z"],
            real=r["field"].real,
            imag=r["field"].imag,
            epsilon=r["epsilon"],
        ),
        history=[],
        constraint_audit=geometry_audit(params, min_feature),
        provenance=dict(
            solver="Fourier-modal TE Maxwell",
            harmonics=harmonics,
            polarization="E_perp (E_y in x-z plane), invariant out of plane",
            units="um",
            source="unit-amplitude plane wave at z=0, normal incidence",
            status="simulated; run independent validation for this geometry",
        ),
    )
    if spectrum:
        waves = np.linspace(1.3, 1.7, 21)
        spectra = [solve(params, w, harmonics, material_index=material_index) for w in waves]
        out["spectrum"] = dict(
            wavelength=waves,
            efficiency=[efficiency(s, target_order) for s in spectra],
            transmission=[s["total_T"] for s in spectra],
        )
    return plain(out)


def validate_design(params, wavelength=1.55, target_order=1, material_index=2.0, fine=False):
    resolutions = (
        [(32, 15), (64, 30), (96, 45), (128, 60)] if fine else [(32, 15), (64, 30), (96, 45)]
    )
    spectral = []
    for m in [3, 5, 7, 9, 11, 13]:
        r = solve(params, wavelength, m, material_index=material_index)
        spectral.append(
            dict(
                harmonics=m,
                efficiency=efficiency(r, target_order),
                balance=r["power_balance"],
                residual=r["residual"],
            )
        )
    independent = []
    for nx, nz in resolutions:
        r = solve_fdfd(
            params,
            wavelength,
            nx=nx,
            steps_per_layer=nz,
            fields=bool(fine and nx == 128),
            material_index=material_index,
        )
        independent.append(
            dict(
                nx=nx,
                steps_per_layer=nz,
                efficiency=efficiency(r, target_order),
                transmission=r["total_T"],
                reflection=r["total_R"],
                balance=r["power_balance"],
                residual=r["residual"],
            )
        )
    spectral_delta = float(np.ptp([s["efficiency"] for s in spectral[-3:]]))
    grid_delta = abs(independent[-1]["efficiency"] - independent[-2]["efficiency"])
    cross = abs(independent[-1]["efficiency"] - spectral[-1]["efficiency"])
    result = dict(
        spectral_convergence=spectral,
        grid_convergence=independent,
        primary_efficiency=spectral[-1]["efficiency"],
        independent_efficiency=independent[-1]["efficiency"],
        efficiency_absolute_difference=cross,
        spectral_last_delta=spectral_delta,
        grid_last_delta=grid_delta,
        passed=bool(
            cross < 0.015
            and spectral_delta < 0.003
            and grid_delta < 0.005
            and abs(independent[-1]["balance"] - 1) < 1e-6
            and abs(spectral[-1]["balance"] - 1) < 1e-6
            and spectral[-1]["residual"] < 1e-8
        ),
        criteria=dict(
            cross_solver_abs_max=0.015,
            spectral_last_three_range_max=0.003,
            grid_delta_max=0.005,
            balance_error_max=1e-6,
            primary_residual_max=1e-8,
        ),
        solver="independent real-space finite differences with discrete radiation boundary",
        note="Two scalar TE Maxwell formulations; not a 3D, Meep, fabrication, or measurement verification.",
    )
    if fine:
        primary = solve(
            params,
            wavelength,
            13,
            fields=True,
            nx=128,
            nz=361,
            material_index=material_index,
        )
        # FDFD z spacing .005um; primary includes .45um exterior each side.
        inside = primary["field"][90:271]
        result["complex_field_relative_l2"] = float(
            np.linalg.norm(inside - r["field"]) / np.linalg.norm(inside)
        )
        result["criteria"]["complex_field_relative_l2_max"] = 0.1
        result["passed"] = bool(result["passed"] and result["complex_field_relative_l2"] < 0.1)
    return plain(result)
