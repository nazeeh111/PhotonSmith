# Reproduce and challenge the result

The corrected-source measurement is [confirmation-v3](../../benchmarks/boundary/confirmation-v3/) relative to this repository root: `benchmarks/boundary/confirmation-v3`. Its coarse-step finite-difference gate is false; see REPORT.md rather than inferring all-gate success. Its `inputs.json`, `protocol.json`, `raw.jsonl`, `cases.json`, `manifest.json` and `summary.json` are all retained. Earlier v1/v2 results remain historical evidence for their original sources. All inputs are authored numerical geometry workloads unless separately attributed as external examples.

The original production baseline was PhotonSmith commit `be31575fb848ee548501ec912edc15c7c58cd86f`. Its `photonsmith/physics.py` is byte-unchanged in this candidate. The corrected C++ kernel SHA256 is `dd55bc7113b2db50a5e5c4d1de14c8435c856b23377f917ffcb38e3fa6f00a65`; source, Python wrapper and built-library identities are in the manifest/review receipts. The earlier `37d697240a5b3becba4e83622b7484079b4a998af13805b71fa3a43319491547` belongs to historical v1/v2 measurements. Native library bytes depend on compiler/platform and must not be assumed universal.

## Minimal correctness

```sh
python -m pip install -e .
python scripts/build_native.py
python -m unittest discover -s tests -v
python scripts/check_artifacts.py
python scripts/check_boundary_independent.py --output runs/independent-transfer-check
```

The last script preserves a separately authored transfer/exponential-Fréchet oracle and the mixed repeated/zero-mode tests used in review. Only its source/output path wiring was changed for a relocatable source checkout; the original review scripts and receipts remain historical records under `docs/boundary/review`. Some historical scripts retain their original task-local paths; use the supported commands above rather than assuming those archival routing paths are portable.

macOS guarded native check:

```sh
clang++ -std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined native/test_boundary.cpp \
  -framework Accelerate -o /tmp/photonsmith-native-guarded
ASAN_OPTIONS=detect_leaks=0 /tmp/photonsmith-native-guarded
```

On Linux use `-llapack -lblas` instead of Accelerate. The original candidate passed native Linux CI; the corrected candidate requires its own remote execution. The macOS check did not run leak detection; system LAPACK itself is not sanitizer-instrumented.

## External performance and workloads

Use a separate benchmark environment. The lock includes the actual numerical/backend versions and packaging tools; the recorded run manifest is the authority for the original measured environment. PyTorch/JAX installation is only for these comparisons. Normal native use needs neither.

```sh
python -m pip install -r requirements-boundary.lock
python scripts/benchmark_boundary.py \
  --cases benchmarks/boundary/confirmation-v3/inputs.json \
  --output runs/reproduced-confirmation
python scripts/summarize_boundary.py runs/reproduced-confirmation \
  --output runs/reproduced-confirmation/summary.json
python scripts/validate_external_profiles.py --output runs/external-profiles.json
python scripts/boundary_resources.py --method meent_jax_jit \
  --cases benchmarks/boundary/confirmation-v3/inputs.json \
  --output runs/jax-resource-probe.json
```

Repeat the last command with native/legacy/meent_torch/grcwa_autograd and fresh output names. Resource probes inspect only their own process. RSS includes imports/JIT and total process threads include idle workers. They cannot retrospectively measure worker counts in a different timing run.

The benchmark fixes single requested BLAS/Torch thread limits, synchronizes JAX, excludes/records warmup, includes geometry assembly/allocations/FFI, and keeps every attempt. V3 records 50 snapshots in the actual timing process, observing 73–74 total process threads and one running thread at those instants. Total threads include idle runtime workers; snapshots outside timers do not establish a strict single-thread JAX comparison. In historical v2 the timing process's JAX workers were not observed. New numbers on another host should be reported as a new measurement, even if every numerical check matches. Shared-machine scheduling can materially change wall-time ratios.

meent uses the opposite time convention; its Fourier coefficients are conjugated for the same physical geometry. Its specialized 1D TE solve uses `epy_conv` only; unused TM arguments are placeholders, not expensive extra inversions. grcwa is evaluated on the same retained x orders but internally has vector/polarization overhead. No external solver source is patched or bundled into the runtime.

## Independent-author geometry provenance

All three geometry definitions come from [meent commit bae28024476d4af2826b8412cb8288d79a0630ac](https://github.com/kc-ml2/meent/tree/bae28024476d4af2826b8412cb8288d79a0630ac):

- [Uniform film](https://github.com/kc-ml2/meent/blob/bae28024476d4af2826b8412cb8288d79a0630ac/examples/rcwa/TF_1D.py).
- [Moharam TE grating](https://github.com/kc-ml2/meent/blob/bae28024476d4af2826b8412cb8288d79a0630ac/examples/rcwa/moharam_1D_TE.py).
- [Two-bar/thick-layer structure](https://github.com/kc-ml2/meent/blob/bae28024476d4af2826b8412cb8288d79a0630ac/examples/vector_1d.py).

The workload script records unit conversion, scalar-TE selection, sampled wavelengths and increased Fourier orders. The Moharam global translation is immaterial to order powers, and its old aliased coefficient construction is replaced by complete analytic coefficients before comparison with production solvers. These are documented adaptations, not unchanged execution of all original examples or measured device data.

## Reading the receipts

Numerical arrays and raw timings are unchanged in the public-ready evidence copies. A few personal local-path prefixes in manifests are normalized for portability, with the original local manifest SHA256 recorded. Review is by a separate coding agent, not by an outside researcher. The prepared CI workflow and a posted validation request do not establish successful remote execution or human acceptance.
