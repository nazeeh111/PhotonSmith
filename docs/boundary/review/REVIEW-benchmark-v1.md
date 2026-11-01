# Independent audit of benchmark confirmation v1

The saved rows support the reported primary numerical/performance result for this bounded family, with the departures and limits below. V1 is now retained as development evidence while a corrected, freshly seeded v2 benchmark runs. No benchmark or solver was rerun for this audit.

This is independent coding-agent analysis on the same host. It is not outside-human review, independent-machine replication, adoption, or a public preregistration.

## Data and provenance checks

All checked invariants passed:

- 48 unique frozen input cases, 12 each at Fourier M=3,5,9,15.
- 7 repetitions × 5 methods × 48 cases = 1,680 raw rows, with no error, missing or duplicate attempt.
- Raw rows exactly match the observations retained in cases.json.
- The case list and ordering match the frozen input JSON.
- Recorded method order exactly reproduces `random.Random(37832)` and its per-case/per-repeat shuffle.
- All recorded outputs/gradients are finite, and all wall/CPU times positive.
- Every case's period, layer height, refractive index and target order matches the runner's intentionally fixed physical family.
- The legacy physics source hash matches the original public-source snapshot manifest.
- The native source and binary hashes match the build receipt and previously independently reviewed source/library.

Input SHA256: `329ceecb28bed837bf9bdc4841b05e065a29d7e50db17abd35dc09eb75ff0cca`.
Runner SHA256: `2eefca824a28b3cbcadd7c3ff370ec41ae45950d9e669d02c01b0347d02c72fa`, preserved exactly at development/benchmark-boundary-v1.py.
Raw JSONL SHA256: `bdaff3d89b40511ed267e7247251dcebb78c14c1106e09d89e577ef05512b4dc`.
Full evidence hashes and independent calculations are in `benchmark-v1-independent-recomputation.json`; case ratios are in `benchmark-v1-case-ratios.json`.

The external package versions are pinned in the run metadata. A separate **post-run** integrity check hashed 12 decisive meent files and 4 grcwa files and verified all against their installed wheel RECORD hashes. This is useful byte-integrity evidence, but it is not falsely represented as having been captured before the run. See `benchmark-baseline-source-integrity-postrun.json`.

## Independently recomputed primary statistics

Each case uses the median of its seven paired baseline/native wall-time ratios. Overall speedup is the geometric mean of all 48 case medians. Intervals use NumPy `default_rng(37833)`, 10,000 shared paired-case bootstrap resamples and percentile endpoints.

| Comparison | Geometric-mean speedup | 95% case-bootstrap interval |
|---|---:|---:|
| Unmodified public PhotonSmith | 7.8122× | [7.3589, 8.2865] |
| meent JAX JIT specialized 1D TE | 3.3318× | [2.9127, 3.8227] |
| meent Torch specialized 1D TE, v1 adapter | 8.5496× | [7.1032, 10.2191] |
| grcwa autograd vector RCWA on the same 1D orders | 28.8321× | [26.2079, 31.8547] |

JAX is the fastest tested external comparator in every case, including after choosing a per-case oracle among all three external methods. JAX-relative speedups by M=3/5/9/15 are 5.8821× / 4.0732× / 2.5046× / 2.0534×. The original >=5× legacy and >=2× fastest-tested-external aggregate thresholds are numerically met by these saved rows. They are not universal lower bounds, and the M=15 result has less performance headroom than the aggregate.

**The Torch v1 adapter does redundant work:** it eagerly computes an epsilon-matrix inverse that the selected TE solver branch never reads. Do not promote the v1 Torch ratio as a final optimized-baseline claim. The JAX compiler can remove its unused inverse; the primary JAX comparison is not the same eager-inversion problem. Root is correcting the adapters and preserving v1 before evaluating fresh v2 cases.

The bootstrap describes case-level variability within this synthetic six-parameter family on this shared host. It does not include different machines, compilers, thermal states or unrestricted independent problems. Random order and paired medians limit some host variation; they do not establish isolated-machine timing.

## Physical/model and work equivalence

All methods calculate the same three lossless rectangular-bar layers, same normal incidence, wavelength, material contrast, target +1 transmitted order, and retained transverse modes. Analytic Fourier coefficients are supplied directly; no raster-resolution advantage is used. Native computes the physical objective and all six real geometry gradients, including FFI/allocation/setup. The baseline adapters compute the same objective and six gradients. Imports/builds/JIT compilation are excluded from steady timings and warmup durations are separately recorded.

The meent coefficient conjugation accounts for its opposite time convention. Its `pol=0`, `phi=None`, one-row unit-cell shape and `fto=[M,0]` select its specialized scalar 1D TE branch, rather than intentionally sending a 1D problem through a full 2D vector solver. Explicit complex128 and float64 inputs are used. The meent import enables JAX x64 in this installed release. JAX wavelength and geometry are actual dynamic arguments; M is compiled per shape. Both value and gradient are explicitly synchronized before returning to the timer.

Grcwa's retained-mode assertion checks the exact number of x modes and excludes y modes, and target-order selection uses its actual G ordering. Grcwa internally carries both polarizations. Therefore its large speed ratio includes specialization and implementation differences and must not be advertised as a general vector-RCWA speedup or the strongest specialized comparison. Its assembled coefficients and all eigen/scattering calculations remain actual external library code.

## Correctness in the saved confirmation rows

- Maximum full reflected/transmitted-order discrepancy versus meent NumPy: 2.1733e-14.
- Maximum native lossless balance error: 7.7716e-15.
- Maximum native residual: 2.4523e-15.
- Maximum objective discrepancy versus external methods: Torch 5.2128e-10; JAX 1.3878e-14; grcwa 8.0436e-14.
- Maximum componentwise gradient discrepancy normalized by max(1, absolute native component): Torch 3.4221e-8; JAX 2.4925e-13; grcwa 2.0399e-10.
- First-eight-case central finite differences, all six parameters, at steps 1e-4/1e-5/1e-6 have maximum normalized component errors 5.9833e-6 / 5.9849e-8 / 5.3766e-9. All those perturbations stay inside geometry bounds, so no one-sided substitute was required.

The prescribed numerical tolerances are met. The v1 full-order saved check is against meent NumPy only. Grcwa's timed rows save the +1 objective and six gradients, not all outgoing powers. Any statement that *both* external implementations match every outgoing order needs the separate independent-geometry evidence or a v2 all-order record.

## Protocol departures and pending finish-line evidence

1. V1 records no peak-memory observation, although FINISH_LINE requests one. Separate full-process RSS probes are planned. Do not infer peak RSS from matrix size or call a separate probe a measurement from the timing process.
2. Actual timing-process JAX worker counts were not captured. The manifest records requested numerical-thread limits and a Torch OpenMP count, not JAX's worker pool. The protocol explicitly asks to record JAX workers. Root will disclose that departure and supply separately labeled process observations. Do not claim every runtime actually used exactly one worker.
3. The benchmark docstring originally referenced a missing requirements-boundary.lock. A complete pinned file has since been added for reproducibility. Its late addition does not change the already recorded installed versions or timing rows.
4. V1 becomes viewed development evidence after adapter correction. A fresh fixed v2 set is required before calling the corrected result confirmation; do not silently reuse the viewed v1 data for a newly tuned implementation.
5. The three-bar timing workload is authored synthetic input, distinct from externally authored geometry reproduction and actual physical measurement. The remaining independent-workload, changed-objective optimization, public-version/install, and external-human-response gates remain separate.
