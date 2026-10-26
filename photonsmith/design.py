"""Constrained design searches. Every reported final efficiency is Maxwell solved."""

from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from .physics import DEFAULT_PARAMS, solve, objective_gradient, validate_params


def bounds_for(min_feature=0.36, period=2.4):
    if not np.isfinite(min_feature) or not 0.1 <= min_feature <= 0.8:
        raise ValueError("Minimum lateral bar/gap width must be between 0.1 and 0.8 um.")
    f = max(0.15, min_feature / period)
    return [(f, 1 - f)] * 3 + [(0.0, 1.0)] * 3


def geometry_audit(params, min_feature=0.36, period=2.4, layer_height=0.3):
    p = np.asarray(params)
    spans = np.r_[p[:3] * period, (1 - p[:3]) * period]
    return dict(
        min_lateral_bar_gap_um=float(spans.min()),
        requested_lateral_min_um=float(min_feature),
        lateral_constraint_pass=bool(spans.min() >= min_feature - 1e-8),
        layer_height_um=layer_height,
        vertical_layer_height_pass=bool(layer_height >= min_feature),
        junction_minimum_feature="not certified",
        fabrication_ready=False,
        note="Constraint applies within each layer. Inter-layer necks, corners, connectivity, substrate and 3D fabrication are not certified.",
    )


def optimize(
    params=DEFAULT_PARAMS,
    wavelength=1.55,
    target_order=1,
    method="adjoint",
    iterations=20,
    seed=41,
    min_feature=0.36,
    material_index=2.0,
    model=None,
    harmonics=5,
):
    if method not in ("adjoint", "surrogate", "random"):
        raise ValueError("Choose adjoint, surrogate, or random search.")
    if target_order not in (-1, 0, 1) or isinstance(target_order, (bool, np.bool_)):
        raise ValueError("Target diffraction order must be -1, 0, or +1.")
    target_order = int(target_order)
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or iterations < 1
    ):
        raise ValueError("Iterations must be a positive integer.")
    bounds = bounds_for(min_feature)
    lo, hi = np.array(bounds).T
    p = np.clip(validate_params(params), lo, hi)
    rng = np.random.default_rng(seed)
    start = perf_counter()
    history = []
    evaluations = 0
    gradient_evaluations = 0

    def evaluate(x, gradient=False):
        nonlocal evaluations, gradient_evaluations
        if gradient:
            value, g = objective_gradient(
                x, wavelength, target_order, harmonics, material_index=material_index
            )
            gradient_evaluations += 1
        else:
            r = solve(x, wavelength, harmonics, material_index=material_index)
            value = float(r["T"][np.flatnonzero(r["orders"] == target_order)[0]])
            g = None
        evaluations += 1
        history.append(
            dict(
                iteration=evaluations,
                efficiency=float(value),
                seconds=perf_counter() - start,
                gradient=bool(gradient),
                params=np.asarray(x).tolist(),
            )
        )
        return value, g

    initial, _ = evaluate(p)
    best = p.copy()
    best_value = initial
    prediction = None
    if method == "random":
        for _ in range(iterations):
            x = rng.uniform(lo, hi)
            value, _ = evaluate(x)
            if value > best_value:
                best, best_value = x, value
    else:
        if method == "surrogate":
            if model is None:
                raise ValueError("Train or load the supplied surrogate first.")
            if material_index != 2.0 or not 1.3 <= wavelength <= 1.7:
                raise ValueError(
                    "Surrogate supports n=2 and wavelengths 1.3–1.7 um. Choose direct adjoint for other materials."
                )
            candidates = rng.uniform(lo, hi, (1024, 6))
            candidates[0] = p
            predicted = model.predict(candidates, wavelength)["powers"][:, target_order + 1]
            starts = candidates[np.argsort(predicted)[-4:]]
            refined = []
            for x in starts:

                def neural_fun(q):
                    value, gradient = model.objective_gradient(q, wavelength, target_order)
                    return -value, -gradient

                result = minimize(
                    neural_fun,
                    x,
                    jac=True,
                    bounds=bounds,
                    method="L-BFGS-B",
                    options={"maxiter": 60},
                )
                refined.append(result.x)
            for x in refined:
                value, _ = evaluate(x)
                if value > best_value:
                    best, best_value = x, value
            prediction = float(model.predict(best, wavelength)["powers"][0, target_order + 1])

        def objective(x):
            nonlocal best, best_value
            value, g = evaluate(x, True)
            if value > best_value:
                best, best_value = x.copy(), value
            return -value, -g

        result = minimize(
            objective,
            best,
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": iterations, "ftol": 1e-8, "gtol": 1e-5, "maxls": 12},
        )
    return dict(
        params=best.tolist(),
        wavelength=float(wavelength),
        target_order=int(target_order),
        method=method,
        efficiency=float(best_value),
        initial_efficiency=initial,
        runtime_seconds=perf_counter() - start,
        forward_evaluations=evaluations,
        gradient_evaluations=gradient_evaluations,
        history=history,
        prediction_at_warm_start=prediction,
        seed=seed,
        harmonics=harmonics,
        constraint_audit=geometry_audit(best, min_feature),
        budget_note="Iterations are method-specific; wall time, forward evaluations and gradient evaluations are all reported.",
    )
