"""Re-run studio exports from a terminal, without a browser."""

import argparse
import json
from pathlib import Path
from .server import parse_config
from .design import optimize
from .reporting import public_simulation, validate_design
from .surrogate import Surrogate
from .resources import asset_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["simulate", "validate", "optimize"])
    parser.add_argument(
        "--config", type=Path, default=asset_root() / "experiments/sample-config.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["adjoint", "surrogate", "random"], default="surrogate")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--fine",
        action="store_true",
        help="Use a 128×181 independent grid and compare complex fields",
    )
    args = parser.parse_args(argv)
    raw = json.loads(args.config.read_text())
    raw = raw.get("configuration", raw)
    if "min_lateral_bar_gap" in raw:
        raw["min_feature"] = raw["min_lateral_bar_gap"]
    cfg = parse_config(raw)
    if args.action == "simulate":
        result = public_simulation(**cfg)
    elif args.action == "validate":
        result = validate_design(
            **{k: v for k, v in cfg.items() if k != "min_feature"}, fine=args.fine
        )
    else:
        path = asset_root() / "experiments" / "surrogate.npz"
        result = optimize(
            **cfg, method=args.method, iterations=args.iterations, model=Surrogate(path)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved {args.action} result to {args.output}")


if __name__ == "__main__":
    main()
