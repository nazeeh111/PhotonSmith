"""Reproduce the recorded study with explicit costs and raw per-run results."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
from .physics import solve, DEFAULT_PARAMS
from .surrogate import Surrogate, train
from .data import generate
from .design import optimize
from .reporting import public_simulation, validate_design, efficiency, plain
from .resources import source_root

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"


def benchmark(repeats=2):
    source_root()
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("Benchmark repeats must be a positive integer.")
    EXP.mkdir(exist_ok=True)
    model = Surrogate(EXP / "surrogate.npz")
    data = np.load(EXP / "dataset.npz", allow_pickle=False)
    training = json.loads((EXP / "surrogate.json").read_text())
    data_meta = json.loads(str(data["metadata"]))
    # No benchmark geometry is selected from training/validation/test geometry rows.
    # Objective wavelengths below are fresh continuous values, not a held-out family.
    objectives = [(1.55, 1), (1.41, -1), (1.63, 1)]
    runs = []
    start = time.perf_counter()
    for wave, order in objectives:
        for repeat in range(repeats):
            rng = np.random.default_rng(500 + repeat)
            initial = (
                DEFAULT_PARAMS.copy()
                if repeat == 0
                else np.r_[rng.uniform(0.25, 0.7, 3), rng.uniform(0, 1, 3)]
            )
            # Identical initial geometry and seed within every paired comparison.
            for method, iters in [("random", 32), ("adjoint", 30), ("surrogate", 10)]:
                r = optimize(initial, wave, order, method, iters, seed=500 + repeat, model=model)
                final = solve(r["params"], wave, 13)
                r.update(
                    efficiency_search_M5=r["efficiency"],
                    efficiency=efficiency(final, order),
                    converged_harmonics=13,
                    power_balance=final["power_balance"],
                    objective_split="fresh optimization wavelength/order combination; same physical design family",
                )
                runs.append(r)
                print(
                    f"{wave:.2f}um order{order:+d} seed{500 + repeat} {method}: M13 {r['efficiency']:.4f}, search {r['runtime_seconds']:.3f}s",
                    flush=True,
                )
                (EXP / "benchmark-progress.json").write_text(json.dumps(plain(runs), indent=2))
    # Measure model inference on held-out geometries and compare with forward solve.
    test = data["test"]
    p = np.repeat(data["params"][test], 4, axis=0)
    waves = data["wavelengths"][test].ravel()
    model.predict(p, waves)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        model.predict(p, waves)
        times.append(time.perf_counter() - t)
    t = time.perf_counter()
    for q, w in zip(p[:30], waves[:30]):
        solve(q, w, 5)
    physics_per = (time.perf_counter() - t) / 30
    label_bias = []
    for q, w in zip(p[:20], waves[:20]):
        low = solve(q, w, 5)
        high = solve(q, w, 13)
        mask = np.isin(low["orders"], [-1, 0, 1])
        mh = np.isin(high["orders"], [-1, 0, 1])
        label_bias.extend(
            abs(np.r_[low["T"][mask], low["R"][mask]] - np.r_[high["T"][mh], high["R"][mh]])
        )
    summary = {}
    for method in ("random", "adjoint", "surrogate"):
        rows = [r for r in runs if r["method"] == method]
        summary[method] = dict(
            mean_efficiency=float(np.mean([r["efficiency"] for r in rows])),
            median_search_seconds=float(np.median([r["runtime_seconds"] for r in rows])),
            mean_forward_evaluations=float(np.mean([r["forward_evaluations"] for r in rows])),
            mean_gradient_evaluations=float(np.mean([r["gradient_evaluations"] for r in rows])),
        )
    saved = (
        summary["adjoint"]["median_search_seconds"] - summary["surrogate"]["median_search_seconds"]
    )
    offline_cost = data_meta["seconds"] + training["metadata"]["seconds"]
    report = dict(
        dataset_cost_seconds=data_meta["seconds"],
        training_cost_seconds=training["metadata"]["seconds"],
        model_metrics=training["held_out_geometry_metrics"],
        summary=summary,
        runs=runs,
        objectives=[dict(wavelength=w, target_order=o) for w, o in objectives],
        timing=dict(
            model_batch_size=len(p),
            model_batch_median_seconds=float(np.median(times)),
            model_per_sample_seconds=float(np.median(times) / len(p)),
            forward_M5_per_sample_seconds=physics_per,
            offline_cost_seconds=offline_cost,
            search_only_median_speed_ratio_adjoint_over_ml=summary["adjoint"][
                "median_search_seconds"
            ]
            / summary["surrogate"]["median_search_seconds"],
            approximate_break_even_designs=float(offline_cost / saved) if saved > 0 else None,
            note="Warm local CPU timings. Method-specific iteration budgets; final M13 evaluation and independent verification excluded from search time for every method. Offline training/data are charged separately. No equal-quality general speedup claim.",
        ),
        label_discretization=dict(
            held_out_samples=min(20, len(p)),
            M5_vs_M13_power_mae=float(np.mean(label_bias)),
            max_absolute_power_difference=float(np.max(label_bias)),
        ),
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            numpy=np.__version__,
            threads="BLAS/OMP one thread requested",
            dataset_seed=data_meta["seed"],
            training_seed=training["metadata"]["seed"],
            search_seeds=sorted({r["seed"] for r in runs}),
        ),
        experiment_seconds=time.perf_counter() - start,
        scope=f"{len(objectives)} fresh objective combinations, {repeats} seeds, one six-parameter 2D design family. Small exploratory study, not broad generalization evidence.",
    )
    report["objective_verifications"] = []
    design_dir = EXP / "designs"
    design_dir.mkdir(exist_ok=True)
    for wave, order in objectives[1:]:
        row = max(
            (
                r
                for r in runs
                if r["method"] == "surrogate"
                and r["wavelength"] == wave
                and r["target_order"] == order
            ),
            key=lambda r: r["efficiency"],
        )
        print(
            f"Independent held-out objective check: {wave:.2f}um, order{order:+d}",
            flush=True,
        )
        check = validate_design(row["params"], wave, order)
        design = dict(
            params=row["params"],
            wavelength=wave,
            target_order=order,
            material_index=2.0,
            period=2.4,
            layer_height=0.3,
            min_feature=0.36,
            validation=check,
            method="surrogate",
        )
        report["objective_verifications"].append(design)
        (design_dir / f"{int(round(wave * 1000))}nm-order{order:+d}.json").write_text(
            json.dumps(design, indent=2)
        )
    (EXP / "benchmarks.json").write_text(json.dumps(plain(report), indent=2))
    # Select a fixed public target (1550nm,+1), never silently change the requested objective.
    eligible = [r for r in runs if r["wavelength"] == 1.55 and r["target_order"] == 1]
    winner = max(eligible, key=lambda r: r["efficiency"])
    print("Independently validating selected sample…", flush=True)
    validation = validate_design(winner["params"], fine=True)
    sample = public_simulation(winner["params"])
    sample.update(
        name="1550 nm directional grating",
        validation=validation,
        history=winner["history"],
        runtime_seconds=winner["runtime_seconds"],
        selected_method=winner["method"],
        min_feature=0.36,
    )
    sample["prediction"] = {
        "efficiency": float(model.predict(winner["params"], 1.55)["powers"][0, 2])
    }
    (EXP / "sample.json").write_text(json.dumps(sample, indent=2))
    config = dict(
        schema="photonsmith-config-v1",
        params=winner["params"],
        wavelength=1.55,
        target_order=1,
        period=2.4,
        layer_height=0.3,
        material_index=2.0,
        min_lateral_bar_gap=0.36,
        harmonics=13,
        independent=dict(nx=128, steps_per_layer=60),
        convention="exp(-i omega t), +z incident, E_perp",
        selected_method=winner["method"],
    )
    (EXP / "sample-config.json").write_text(json.dumps(config, indent=2))
    manifest = {
        str(path.relative_to(EXP)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(EXP.rglob("*"))
        if path.is_file()
        and path.suffix in (".npz", ".json")
        and path.name not in ("manifest.json", "benchmark-progress.json")
    }
    (EXP / "manifest.json").write_text(json.dumps(dict(sha256=manifest), indent=2))
    print(json.dumps(dict(summary=summary, validation=validation), indent=2))
    return report


def main():
    source_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--geometries", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.rebuild:
        generate(EXP / "dataset.npz", args.geometries, args.seed)
        train(EXP / "dataset.npz", EXP / "surrogate.npz", args.epochs)
    benchmark(args.repeats)


if __name__ == "__main__":
    main()
