"""Re-run studio exports from a terminal, without a browser."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from .quality import NumericalRefusal
from .configuration import configuration_body, parse_config, parse_harmonics, parse_fine
from .design import optimize
from .reporting import public_simulation, validate_design
from .surrogate import Surrogate
from .resources import asset_root


def _publish_result(path, text):
    """Publish complete bytes once; a collision never replaces prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix='.photonsmith-', suffix='.tmp', delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        # Same-directory hard linking is an atomic create, including when a
        # different writer creates path after the early CLI collision check.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('simulate', 'validate', 'optimize', 'converge'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path, default=asset_root() / 'experiments/sample-config.json')
        command.add_argument('--output', type=Path, required=True)
        if name == 'optimize':
            command.add_argument('--method', choices=['adjoint', 'surrogate', 'random'], default='surrogate')
            command.add_argument('--iterations', type=int, default=20)
            command.add_argument('--backend', choices=['legacy', 'stable'], default='legacy',
                                 help='Legacy (default) or experimental stable, without interior fields')
        elif name == 'validate':
            command.add_argument('--fine', action='store_true', help='Use a 128×181 independent grid and compare complex fields')
        elif name == 'converge':
            command.add_argument('--backend', choices=['legacy', 'stable'], default='stable')
            command.add_argument('--orders', type=int, nargs='+', default=[3,5,7,9,11,13])
            command.add_argument('--tolerance', type=float, default=.003)
            command.add_argument('--independent', action='store_true',
                                 help='Compare with three independent real-space grids')
    args = parser.parse_args(argv)
    if args.action == 'optimize' and args.iterations < 1:
        parser.error('--iterations must be positive')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f'Output already exists: {args.output}. Choose a new output path.')
    raw_bytes = args.config.read_bytes()
    raw = configuration_body(json.loads(raw_bytes))
    if args.action == 'converge':
        from .convergence import spectral_study
        allowed = ('params', 'wavelength', 'target_order', 'material_index', 'period', 'layer_height')
        result = spectral_study(**{k: raw[k] for k in allowed if k in raw},
                                orders=args.orders, backend=args.backend, tolerance=args.tolerance,
                                independent=args.independent)
        result['source_configuration_sha256'] = hashlib.sha256(raw_bytes).hexdigest()
        text = json.dumps(result, indent=2, allow_nan=False) + '\n'
        _publish_result(args.output, text)
        print(f'Saved sampled-order study to {args.output}; meets tolerance: {result["meets_sampled_tolerance"]}')
        if args.independent:
            comparison = result['independent_comparison']
            print(f'Independent comparison meets tolerance: {comparison["meets_comparison_tolerance"]}')
        return
    cfg = parse_config(raw)
    if args.action == "simulate":
        result = public_simulation(**cfg, harmonics=parse_harmonics(raw))
    elif args.action == "validate":
        result = validate_design(
            **{k: v for k, v in cfg.items() if k != "min_feature"},
            fine=parse_fine(raw, args.fine), harmonics=parse_harmonics(raw, default=13, minimum=7)
        )
    else:
        harmonics = parse_harmonics(raw, default=5)
        path = asset_root() / "experiments" / "surrogate.npz"
        result = optimize(
            **cfg, method=args.method, iterations=args.iterations,
            backend=args.backend,
            model=Surrogate(path) if args.method == "surrogate" else None,
            harmonics=harmonics
        )
    _publish_result(args.output, json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved {args.action} result to {args.output}")


def main(argv=None):
    try:
        return _run(argv)
    except (NumericalRefusal, ValueError, TypeError, OverflowError, OSError) as exc:
        print(f"PhotonSmith: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
