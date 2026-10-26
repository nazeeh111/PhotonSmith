"""Locate bundled read-only assets without depending on the working directory."""

from pathlib import Path


def asset_root():
    checkout = Path(__file__).resolve().parents[1]
    if (checkout / "web" / "index.html").is_file() and (checkout / "experiments").is_dir():
        return checkout
    bundled = Path(__file__).resolve().parent / "assets"
    if not (bundled / "web" / "index.html").is_file():
        raise FileNotFoundError(
            "PhotonSmith's studio assets are missing. Reinstall the complete package."
        )
    return bundled


def source_root():
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(
            "Regenerating experiments/figures requires a source checkout. Clone the repository and install it with pip install -e '.[train,dev]'."
        )
    return root
