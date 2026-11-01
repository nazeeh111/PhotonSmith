"""Independent forward-only, fourth-order checks of six geometry derivatives.

No step selection uses observed function values. Every sample is retained.
This is verification code, not part of the production adjoint or timed solve.
"""
import math
import numpy as np
from . import physics

SCHEME = 'dense-fourth-order-v1'
STEPS = (1e-4, 1e-5, 1e-6)
LOWER = np.array([.15, .15, .15, 0., 0., 0.])
UPPER = np.array([.85, .85, .85, 1., 1., 1.])
STENCILS = {
    'central': ([-2, -1, 1, 2], [1/12, -2/3, 2/3, -1/12]),
    'forward': ([0, 1, 2, 3, 4], [-25/12, 4., -3., 4/3, -1/4]),
    'backward': ([0, -1, -2, -3, -4], [25/12, -4., 3., -4/3, 1/4]),
}


def choose_stencil(value, step, lower, upper):
    if not all(math.isfinite(x) for x in (value, step, lower, upper)) or step <= 0:
        raise ValueError('Finite inputs and a positive step required')
    if not lower <= value <= upper:
        raise ValueError('Point outside finite-difference bounds')
    if value-2*step >= lower and value+2*step <= upper:
        return 'central'
    if value+4*step <= upper:
        return 'forward'
    if value-4*step >= lower:
        return 'backward'
    raise ValueError('No fourth-order stencil fits; steps are never adapted')


def forward_difference(forward, params, step):
    p = physics.validate_params(params).copy()
    gradients, second, components = [], [], []
    for k in range(6):
        kind = choose_stencil(float(p[k]), step, float(LOWER[k]), float(UPPER[k]))
        offsets, weights = STENCILS[kind]
        samples, values = [], {}
        for offset in offsets:
            q = p.copy()
            q[k] = p[k]+offset*step
            value = float(forward(q))
            if not math.isfinite(value):
                raise ArithmeticError('Nonfinite independent forward sample')
            values[offset] = value
            samples.append({'offset': offset, 'params': q.tolist(), 'value': value})
        gradients.append(math.fsum(w*values[o] for o,w in zip(offsets,weights))/step)
        if kind == 'central':
            second.append((values[1]-values[-1])/(2*step))
        else:
            sign = 1 if kind == 'forward' else -1
            second.append(sign*math.fsum([-3*values[0],4*values[sign],-values[2*sign]])/(2*step))
        components.append({'component': k, 'kind': kind, 'offsets': offsets,
                           'weights': weights, 'samples': samples})
    return {'step': step, 'scheme': SCHEME, 'gradient': gradients,
            'second_order_gradient': second, 'components': components}


def dense_checks(params, wavelength, harmonics, target_order=1):
    def forward(p):
        result = physics.solve(p, wavelength=wavelength, harmonics=harmonics)
        return result['T'][harmonics+target_order]
    result = []
    for step in STEPS:
        check = forward_difference(forward, params, step)
        check['oracle'] = 'photonsmith.physics.solve:dense-expm-forward-only'
        result.append(check)
    return result


def validate_record(params, record):
    """Refuse incomplete stencils or estimates inconsistent with retained data."""
    if record.get('scheme') != SCHEME or record.get('oracle') != 'photonsmith.physics.solve:dense-expm-forward-only':
        raise ValueError('Unexpected independent finite-difference scheme/oracle')
    step = record['step']
    if step not in STEPS or len(record.get('components',[])) != 6:
        raise ValueError('Missing complete independent finite-difference samples')
    p = physics.validate_params(params)
    for k,c in enumerate(record['components']):
        kind = choose_stencil(float(p[k]),step,float(LOWER[k]),float(UPPER[k]))
        offsets,weights = STENCILS[kind]
        if (c['component']!=k or c['kind']!=kind or c['offsets']!=offsets or c['weights']!=weights
                or len(c['samples'])!=len(offsets)):
            raise ValueError('Finite-difference stencil differs from fixed protocol')
        values=[]
        for offset,sample in zip(offsets,c['samples']):
            q=p.copy();q[k]=p[k]+offset*step
            if sample['offset']!=offset or sample['params']!=q.tolist() or not math.isfinite(sample['value']):
                raise ValueError('Independent finite-difference sample identity differs')
            values.append(sample['value'])
        derivative=math.fsum(w*v for w,v in zip(weights,values))/step
        if not math.isclose(derivative,record['gradient'][k],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Finite-difference gradient differs from retained samples')
        by_offset=dict(zip(offsets,values))
        if kind=='central':
            second=(by_offset[1]-by_offset[-1])/(2*step)
        else:
            sign=1 if kind=='forward' else -1
            second=sign*math.fsum([-3*by_offset[0],4*by_offset[sign],-by_offset[2*sign]])/(2*step)
        if not math.isclose(second,record['second_order_gradient'][k],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Second-order diagnostic differs from retained samples')
