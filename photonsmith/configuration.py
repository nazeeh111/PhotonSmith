"""Scientific inputs shared by the HTTP studio and terminal workflows."""
import math
from .physics import DEFAULT_PARAMS, DEFAULTS, validate_params
from .design import bounds_for

CONVENTION = 'exp(-i omega t), +z incident, E_perp'
DESIGN_KEYS = {'params', 'wavelength', 'target_order', 'material_index', 'min_feature',
               'min_lateral_bar_gap', 'period', 'layer_height', 'harmonics', 'independent', 'convention'}


def configuration_body(body):
    if not isinstance(body, dict):
        raise ValueError('Expected a JSON configuration object.')
    if 'configuration' in body:
        if DESIGN_KEYS.intersection(body):
            raise ValueError('Put scientific inputs in configuration or at the top level, not both.')
        body = body['configuration']
        if not isinstance(body, dict):
            raise ValueError('configuration must be a JSON object.')
    if 'convention' in body and body['convention'] != CONVENTION:
        raise ValueError('Only exp(-i omega t), +z incident, E_perp is supported.')
    return body


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be a finite number.')
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f'{field} must be a finite number.') from None
    if not math.isfinite(result):
        raise ValueError(f'{field} must be a finite number.')
    return result


def parse_config(body):
    body = configuration_body(body)
    for name in ('period', 'layer_height'):
        if name in body and _number(body[name], name) != DEFAULTS[name]:
            raise ValueError(f'The field-bearing studio family requires {name}={DEFAULTS[name]:g} um. '
                             'Use converge or the Python solver for other dimensions.')
    try:
        params = validate_params(body.get('params', DEFAULT_PARAMS)).tolist()
    except (TypeError, OverflowError):
        raise ValueError('Expected six finite geometry parameters.') from None
    wave = _number(body.get('wavelength', 1.55), 'wavelength')
    index = _number(body.get('material_index', 2.0), 'material_index')
    order = body.get('target_order', 1)
    feature = _number(body.get('min_feature', body.get('min_lateral_bar_gap', .36)), 'min_feature')
    if 'min_feature' in body and 'min_lateral_bar_gap' in body:
        if feature != _number(body['min_lateral_bar_gap'], 'min_lateral_bar_gap'):
            raise ValueError('Conflicting min_feature and min_lateral_bar_gap values.')
    if not 1.3 <= wave <= 1.7:
        raise ValueError('Wavelength must be between 1.3 and 1.7 um.')
    if not 1 <= index <= 2.5:
        raise ValueError('Refractive index must be between 1 and 2.5.')
    if order not in (-1, 0, 1) or isinstance(order, bool):
        raise ValueError('Target order must be -1, 0, or +1.')
    bounds = bounds_for(feature)
    if any(v < lo - 1e-10 or v > hi + 1e-10 for v, (lo, hi) in zip(params, bounds)):
        raise ValueError('Geometry violates the selected lateral bar/gap width. Adjust widths or lower the constraint.')
    return dict(params=params, wavelength=wave, target_order=int(order), material_index=index, min_feature=feature)


def parse_harmonics(body, default=9, minimum=1):
    value = configuration_body(body).get('harmonics', default)
    if type(value) is not int or not minimum <= value <= 15:
        raise ValueError(f'harmonics must be an integer from {minimum} to 15 for this operation.')
    return value


def parse_fine(body, fine=False):
    grid = configuration_body(body).get('independent')
    if grid is None:
        return fine
    if not isinstance(grid, dict) or set(grid) != {'nx', 'steps_per_layer'} or any(type(v) is not int for v in grid.values()):
        raise ValueError('independent must specify integer nx and steps_per_layer.')
    if grid == {'nx': 128, 'steps_per_layer': 60}:
        return True
    if grid == {'nx': 96, 'steps_per_layer': 45} and not fine:
        return False
    raise ValueError('Validation supports independent 96/45 or 128/60; --fine requires 128/60.')
