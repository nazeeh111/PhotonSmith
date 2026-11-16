"""Bounded numerical admission, not a discretization or physics certificate."""
import numpy as np

MAX_RELATIVE_RESIDUAL = 1e-8
MAX_POWER_ERROR = 1e-7


class NumericalRefusal(ValueError):
    """A numerical result failed the declared admission conditions."""


def require_finite(*values):
    if not all(np.all(np.isfinite(value)) for value in values):
        raise NumericalRefusal("Numerical result is nonfinite; simulation refused.")


def require_residual(value, label="Boundary"):
    require_finite(value)
    if value < 0 or value > MAX_RELATIVE_RESIDUAL:
        raise NumericalRefusal(f"{label} relative residual exceeds {MAX_RELATIVE_RESIDUAL:g}; simulation refused.")


def admit_forward(residual, transmission, reflection, *values):
    require_finite(residual, transmission, reflection, *values)
    require_residual(residual)
    if np.any(np.asarray(transmission) < 0) or np.any(np.asarray(reflection) < 0):
        raise NumericalRefusal("Negative diffraction power; simulation refused.")
    power = float(np.sum(transmission) + np.sum(reflection))
    require_finite(power)
    if abs(power - 1.0) > MAX_POWER_ERROR:
        raise NumericalRefusal(f"Lossless power balance exceeds {MAX_POWER_ERROR:g}; simulation refused.")
    return power


def admission_metadata():
    return {"status": "accepted", "relative_residual_max": MAX_RELATIVE_RESIDUAL,
            "lossless_power_error_max": MAX_POWER_ERROR,
            "scope": "finite algebraic solve and energy screen; not a convergence or physical-accuracy certificate"}
