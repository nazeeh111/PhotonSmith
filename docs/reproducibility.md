# Reproduce the experiment

Run these commands from a source checkout using Python 3.12 or 3.13. The installed wheel supports the studio and simulation/design CLI with bundled read-only assets; regenerating scientific evidence intentionally requires a source checkout.

## Environment

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,dev]'
```

`pyproject.toml` defines the portable dependencies. `requirements.lock` records the exact original Mac training environment and its transitive dependencies. The original installed versions are recorded in [runtime.json](runtime.json). The build backend is Hatchling; it does not add a runtime service or framework.

For comparable CPU timing, set `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before Python starts. The `photonsmith` entry point requests one thread when no value is already configured. Report any override and hardware/load conditions with timing results.

## Data and training

```sh
python -m photonsmith.data --geometries 2400 --seed 20261026
python -m photonsmith.surrogate --epochs 1200
```

The first command performs 9,600 full-wave cases: four stratified random wavelengths for each geometry. Dataset seed **20261026** fixes the geometries and the retained geometry-level split. The model uses training seed **17**, training-only mirror augmentation and PCA, and independent validation-selected response/field checkpoints. Test data never fits PCA or network weights.

Files generated: `experiments/dataset.npz`, `dataset.json`, `surrogate.npz`, and `surrogate.json`. Model metadata includes the exact dataset SHA-256. The `.npz` files load with `allow_pickle=False`. Inference needs NumPy/SciPy, not PyTorch.

These commands replace generated files. Preserve evidence you need before rerunning them. Raw bytes and wall times may differ across platforms even with the same seeds; compare measured physical/model metrics, not an expected universal binary hash.

## Benchmark and independent acceptance

```sh
python -m photonsmith.experiment --repeats 2
```

Or perform the full pipeline:

```sh
python -m photonsmith.experiment --rebuild \
  --geometries 2400 --epochs 1200 --seed 20261026 --repeats 2
```

Each paired comparison starts from the same geometry and uses the same search seed. The three targets are 1550 nm/+1, 1410 nm/−1, and 1630 nm/+1; starts use search seeds **500** and **501**. These are held-out optimization queries within the same physical family, not unseen objective classes.

| Method | Budget |
| --- | --- |
| Random | Initial solve and 32 random proposals |
| Direct adjoint | At most 30 L-BFGS-B iterations, analytic physical gradient |
| ML + adjoint | 1,024 neural proposals, four neural refinements, four physical shortlist checks and at most 10 physical adjoint iterations |

Forward/adjoint counts and actual histories are recorded because optimizer iterations are not equivalent costs. Final quality is recomputed at M=13 for all methods. Search timing excludes higher-order final evaluation and independent verification equally. Simulation and training costs are reported separately. Small timing differences on a shared workstation do not establish general acceleration.

Acceptance compares Fourier orders M=3,5,7,9,11,13 and separate real-space grids. The finest sample check uses 128 transverse samples and 60 steps per layer, giving 181 longitudinal nodes. It checks target-power disagreement, recent refinement changes, power balance, primary residual and complex-field relative L2 difference at matching coordinates and phase reference. Tolerances are saved in the report and are acceptance criteria, not rigorous error bounds.

The additional strongest 93.59% example came from a later 20-iteration live-studio refinement. It is retained separately as `experiments/designs/1550nm-refined-config.json`; the original 18-run benchmark is preserved.

## Replays and figures

```sh
photonsmith simulate --config experiments/sample-config.json --output work/replay.json
photonsmith validate --fine --config experiments/designs/1550nm-refined-config.json --output work/fine-check.json
python -m photonsmith.figures
python scripts/render_field_demo.py
```

The simulation CLI also reads actual studio JSON exports through their `configuration` object. Rendering uses saved current results or a physical solve of the saved configuration, not synthetic performance data. The animation is a time-harmonic phase display, not FDTD pulse evolution.

## Checks and package portability

```sh
python -m unittest discover -s tests -v
python scripts/check_artifacts.py
python -m build
python -m pip install --force-reinstall dist/photonsmith-0.1.0-py3-none-any.whl
python tests/package_smoke.py
```

The package smoke test launches isolated Python processes from a temporary directory outside the checkout. It checks bundled data/weights, default CLI simulation, server startup readiness, and the 11 real HTTP workflows. It terminates its server afterward. Test the actual wheel in a separate clean environment when validating a release.

With an already running studio, `python tests/http_smoke.py --output work/http-qa.json` runs only the HTTP checks.

`python scripts/check_artifacts.py --refresh-manifests` first recomputes held-out metrics and accepted-design physical responses, then updates dataset/evidence and source hashes. Refresh hashes after intentional scientific changes; never use a hash refresh as a substitute for physical validation.

## External solvers

The bundled real-space FDFD implementation is the executed independent checker. Ceviche and Meep are optional external integrations, not required dependencies. See [research notes](research.md), [license boundaries](licenses.md) and the scripts' `--help`. The optional Meep result needs its own resolution, PML, air-spacing and duration convergence study before acceptance.
