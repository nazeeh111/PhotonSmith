"""Check recorded evidence against data, trained weights and fresh Maxwell solves."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from photonsmith.physics import solve
from photonsmith.surrogate import Surrogate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--refresh-manifests",
        action="store_true",
        help="Refresh hashes only after semantic checks pass",
    )
    a = p.parse_args()
    exp = ROOT / "experiments"
    dataset = np.load(exp / "dataset.npz", allow_pickle=False)
    data_meta = json.loads(str(dataset["metadata"]))
    model = Surrogate(exp / "surrogate.npz")
    assessment = json.loads((exp / "surrogate.json").read_text())
    assert model.metadata == assessment["metadata"]
    assert model.metadata["dataset_sha256"] == digest(exp / "dataset.npz")
    groups = [set(dataset[k].tolist()) for k in ("train", "validation", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert len(set.union(*groups)) == len(dataset["params"])
    ids = dataset["test"]
    params = np.repeat(dataset["params"][ids], dataset["wavelengths"].shape[1], axis=0)
    wave = dataset["wavelengths"][ids].ravel()
    pred = model.predict(params, wave, fields=True)
    error = pred["powers"] - dataset["powers"][ids].reshape(-1, 6)
    truth = dataset["fields"][ids].reshape(-1, 24, 24)
    field_error = np.linalg.norm(
        (pred["canonical_field"] - truth).reshape(len(params), -1), axis=1
    ) / np.linalg.norm(truth.reshape(len(params), -1), axis=1)
    actual = dict(
        power_mae=float(abs(error).mean()),
        power_rmse=float(np.mean(error**2) ** 0.5),
        field_relative_l2_mean=float(field_error.mean()),
    )
    for key, value in actual.items():
        assert abs(value - assessment["held_out_geometry_metrics"][key]) < 1e-7, key
    benchmark = json.loads((exp / "benchmarks.json").read_text())
    assert benchmark["environment"]["dataset_seed"] == data_meta["seed"]
    assert benchmark["environment"]["training_seed"] == model.metadata["seed"]
    assert benchmark["environment"]["search_seeds"] == sorted(
        {r["seed"] for r in benchmark["runs"]}
    )
    for name in (
        "sample.json",
        "designs/1410nm-order-1.json",
        "designs/1630nm-order+1.json",
        "designs/1550nm-refined-config.json",
    ):
        sample = json.loads((exp / name).read_text())
        validation = sample["validation"]
        assert validation["passed"], name
        r = solve(sample["params"], sample["wavelength"], 13)
        index = np.flatnonzero(r["orders"] == sample["target_order"])[0]
        assert abs(r["T"][index] - validation["primary_efficiency"]) < 1e-8, name
    artifact_files = sorted(
        p
        for p in exp.rglob("*")
        if p.is_file()
        and p.suffix in (".npz", ".json")
        and p.name not in ("manifest.json", "benchmark-progress.json")
    )
    if a.refresh_manifests:
        (exp / "manifest.json").write_text(
            json.dumps(
                {"sha256": {str(p.relative_to(exp)): digest(p) for p in artifact_files}}, indent=2
            )
            + "\n"
        )
    for name, expected in json.loads((exp / "manifest.json").read_text())["sha256"].items():
        assert digest(exp / name) == expected, name
    if a.refresh_manifests:
        files = []
        ignored = {".git", ".venv", ".ruff_cache", "__pycache__", ".pytest_cache", "dist", "build"}
        for directory, dirs, names in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in ignored and not d.endswith(".egg-info")]
            for name in names:
                if name in (
                    ".DS_Store",
                    "server.log",
                    "benchmark-progress.json",
                    "source-manifest.json",
                ) or name.endswith((".pyc", ".tmp", ".log")):
                    continue
                files.append(Path(directory) / name)
        (ROOT / "docs/source-manifest.json").write_text(
            json.dumps(
                {
                    "checked": "2026-10-25",
                    "scope": "Public source and retained evidence, excluding environments, build outputs, caches and live logs",
                    "sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)},
                },
                indent=2,
            )
            + "\n"
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "test_cases": len(params),
                "recomputed_metrics": actual,
                "accepted_designs": 4,
                "artifacts_hashed": len(artifact_files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
