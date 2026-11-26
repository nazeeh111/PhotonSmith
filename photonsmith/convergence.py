"""Auditable Fourier-order studies using either existing Maxwell backend."""
import numpy as np
from .physics import solve, DEFAULT_PARAMS, validate_params
from .reporting import plain


def spectral_study(params=DEFAULT_PARAMS, *, wavelength=1.55, target_order=1,
                   material_index=2.0, period=2.4, layer_height=0.3,
                   orders=(3, 5, 7, 9, 11, 13), backend='stable', tolerance=0.003):
    """Compare all propagating channel powers at the last three sampled orders.

    Meeting a finite sequence tolerance is evidence for that sample sequence,
    not a convergence certificate, independent validation or physical accuracy.
    NumericalRefusal aborts the whole study rather than skipping failed orders.
    """
    params = validate_params(params)
    orders = tuple(orders)
    if (not 3 <= len(orders) <= 15
            or any(isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or not 1 <= m <= 15 for m in orders)
            or any(a >= b for a, b in zip(orders, orders[1:]))):
        raise ValueError('Require 3..15 strictly increasing integer harmonic orders in 1..15.')
    if backend not in ('legacy', 'stable'):
        raise ValueError('Backend must be legacy or stable.')
    if isinstance(tolerance, (bool, np.bool_)) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Tolerance must be a positive finite number.')
    dimensions = [wavelength, period, layer_height, material_index]
    if (any(isinstance(value, (bool, np.bool_)) for value in dimensions)
            or not np.isfinite(dimensions).all()
            or min(wavelength, period, layer_height) <= 0 or material_index < 1):
        raise ValueError('Require finite positive wavelength, period and height, and material index at least one.')
    if isinstance(target_order, (bool, np.bool_)) or not isinstance(target_order, (int, np.integer)) or abs(target_order) > orders[0]:
        raise ValueError('Target order must be an integer included at every sampled resolution.')
    # Every physically propagating vacuum channel must be present even in the
    # smallest basis; energy balance alone cannot establish basis completeness.
    if period / wavelength > orders[0] + 1:
        raise ValueError('Smallest harmonic order omits a propagating diffraction channel.')
    propagating = np.arange(-int(np.ceil(period / wavelength)) + 1, int(np.ceil(period / wavelength)))
    if max(abs(propagating)) > orders[0]:
        raise ValueError('Smallest harmonic order omits a propagating diffraction channel.')
    rows, powers = [], []
    for m in orders:
        r = solve(params, wavelength, m, period=period, layer_height=layer_height,
                  material_index=material_index, backend=backend)
        indices = [int(np.flatnonzero(r['orders'] == order)[0]) for order in propagating]
        channels = np.r_[r['T'][indices], r['R'][indices]]
        powers.append(channels)
        target = int(np.flatnonzero(r['orders'] == target_order)[0])
        rows.append(dict(harmonics=int(m), propagating_orders=propagating,
                         T=r['T'][indices], R=r['R'][indices], efficiency=float(r['T'][target]),
                         transmission=r['total_T'], reflection=r['total_R'],
                         balance=r['power_balance'], residual=r['residual'],
                         numerical_admission=r['numerical_admission']))
    channel_range = float(np.max(np.ptp(np.array(powers[-3:]), axis=0)))
    total_range = float(max(np.ptp([r['transmission'] for r in rows[-3:]]),
                            np.ptp([r['reflection'] for r in rows[-3:]])))
    delta = max(channel_range, total_range)
    return plain(dict(schema='photonsmith-spectral-study-v1', backend=backend,
                      configuration=dict(params=params, wavelength=wavelength, target_order=target_order,
                                         material_index=material_index, period=period, layer_height=layer_height),
                      samples=rows, tail_harmonics=list(orders[-3:]),
                      tail_max_channel_power_range=channel_range, tail_max_total_power_range=total_range,
                      tolerance=tolerance, meets_sampled_tolerance=bool(delta < tolerance),
                      criterion='Strictly below tolerance for each propagating T/R channel and total T/R over the last three samples.',
                      limitations=['Finite sampled-order agreement is not a convergence certificate or error bound.',
                                   'Uses one Fourier-modal formulation, not independent solver validation.',
                                   'Lossless nondispersive 2D TE model; no fabrication or measurement validation.']))
