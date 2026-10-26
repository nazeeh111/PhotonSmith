"""One entry point for the studio and reproducible command-line operations."""

import os
import sys


def main():
    # Set before importing numerical libraries, including in the installed CLI.
    for key in (
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    args = sys.argv[1:]
    if not args or args[0] == "studio":
        from .server import main as studio

        studio(args[1:] if args else [])
    elif args[0] in ("simulate", "validate", "optimize"):
        from .cli import main as command

        command(args)
    elif args[0] in ("-h", "--help"):
        print(
            "PhotonSmith: local 2D photonic inverse design\n\nUsage:\n  photonsmith [studio] [--port 8765]\n  photonsmith simulate|validate|optimize --output result.json [--config config.json]\n\nUse photonsmith COMMAND --help for command options."
        )
    else:
        raise SystemExit(f"Unknown command {args[0]!r}. Use photonsmith --help.")


if __name__ == "__main__":
    main()
