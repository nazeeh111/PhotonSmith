# Independent audit of corrected confirmation v2

The saved v2 rows pass the original aggregate speed and numerical thresholds for the declared three-layer, six-parameter, lossless scalar-TE family. This is a local result on a shared Mac, with finite tested comparators. It is not a general RCWA speed record, independent-machine replication, public preregistration, outside-human validation, or completion of the entire finish line.

No full benchmark or solver was rerun for this audit. All calculations below independently read the saved raw evidence.

## Frozen identity and denominator

- Input SHA256: `dd26f909b8dd8e8971b70c7706080ebb5dd11f4dc2ac2ea326afa79e876c1676`.
- Runner SHA256: `f41ea21ba21d77ebde4e6716af91934b97e29da177b253944be2ab5ddb35d11b`.
- Raw JSONL SHA256: `31b8b7c973281cc480fa5ba0170a3670c69f9cdcbd58f25db2e5930db594d2e4`.
- Native C++ SHA256: `37d697240a5b3becba4e83622b7484079b4a998af13805b71fa3a43319491547`.
- Native library SHA256: `98d1ae0c57cb4ddf9bb35fa8b3edc9e0c1c47dd0429b100803de09bd4d408090`.

The input/protocol/run hashes agree. All five recorded producer-source hashes match the inspected files. The native source/library receipt matches the actual source and binary. The legacy numerical source still matches the original public-source snapshot. The input list has 48 unique cases and exactly 12 cases in each M=3,5,9,15 stratum. Every case uses the declared fixed period, material index, layer height and +1 target.

There are exactly 1,680 rows: 48 cases × 7 repetitions × 5 methods. No failed, missing or duplicate attempt was found. The raw rows exactly match cases.json, and method order exactly reproduces the declared per-case/per-repeat random shuffle. Every saved objective/gradient and timing is finite, with positive wall and CPU durations. All cases remain in the denominator, including the one case where native loses to JAX in wall time.

V1 remains preserved as viewed development evidence. The only runner changes from v1 are removal of unused epsilon inversions in the Torch and JAX adapters. V2 uses fresh fixed inputs rather than silently reusing the viewed v1 set. No production numerical source change occurred between the two confirmation attempts.

## Independently reproduced results

Per-case score: median of seven paired baseline/native ratios. Aggregate: geometric mean across all 48 cases. The table intervals exactly reproduce summary.json with NumPy `default_rng(37833)`, advancing the generator once per method to obtain 10,000 paired-case bootstrap samples for that method. An independently computed shared-resample variant is also retained; its small Monte Carlo endpoint differences do not change any gate.

| Comparison | Wall speedup | 95% case-bootstrap interval | CPU-time ratio |
|---|---:|---:|---:|
| Original public PhotonSmith | 13.1208× | [11.0899, 15.8221] | 7.2120× |
| meent JAX JIT specialized 1D TE | 3.7921× | [3.1261, 4.6255] | 3.3417× |
| Corrected meent Torch specialized 1D TE | 12.5276× | [10.4091, 15.0913] | 7.5390× |
| grcwa autograd vector RCWA, same x orders | 65.0740× | [55.3189, 76.3385] | 23.7724× |

JAX is the fastest tested external comparator in aggregate. Choosing the fastest external method separately for each case gives a slightly stricter 3.7810× geometric-mean ratio, also above the target. Native beats JAX in 47/48 case median wall ratios; `m15-03` has ratio 0.4762 and must not be omitted. JAX-relative geometric means by M=3/5/9/15 are 6.7583× / 4.9659× / 2.7735× / 2.2215×.

The original >=5× legacy and >=2× fastest-tested-external aggregate targets, with external interval lower endpoint >1, pass on these rows. The wall/CPU divergence is material evidence of host effects: the 13.12× wall result is not interchangeable with a 13.12× reduction in CPU work. The primary specialized external comparison remains above target on both reported wall and CPU ratios. Do not promote the large vector-grcwa ratio as a general algorithmic speedup.

## Fairness and numerical checks

The substantive v1 audit of Fourier/time conventions, complex precision, retained modes, runtime wavelength argument, actual external code paths, JAX warmup exclusion and value/gradient synchronization still applies. The only two runner changes remove the redundant inversions; the same specialized TE branch remains selected. A deliberate unused epsilon placeholder is safe here because that branch reads `epy_conv` only and the TM-only matrices are never used.

Native includes analytic geometry Fourier assembly, forward and adjoint work, six real geometry derivatives, allocations and Python/native FFI. External methods compute the same objective and six gradients with actual meent or grcwa numerical code. JAX is compiled per M and receives geometry and wavelength dynamically; both returned outputs are synchronized inside the timer. All methods use the same analytic coefficient matrices and retained transverse orders in complex128, with the documented conjugation for meent's time convention. Grcwa internally carries both polarizations, while native and the specialized meent comparison address only scalar TE. This is an implementation-and-specialization comparison, not an isolated proof of a new asymptotic algorithm.

Saved numerical maxima:

- Full reflected/transmitted orders versus meent NumPy: 1.9429e-14.
- Native power-balance error: 8.8818e-15.
- Native residual: 2.0127e-15.
- Maximum normalized external six-gradient discrepancy: 3.0419e-8 (Torch); JAX 2.8916e-13; grcwa 5.9808e-13.
- First-eight-case finite differences at steps 1e-4 / 1e-5 / 1e-6: maximum normalized component discrepancy 5.1137e-6 / 5.1319e-8 / 2.3095e-9. All perturbations stay inside the declared geometry bounds.

These meet the declared numerical thresholds. The timed grcwa records save only the +1 power and six gradients; full-order agreement against both external implementations must be supported by the separate externally authored geometry evidence. That separate study is not silently inferred from the timing table.

## Protocol and delivery limits

1. Numerical thread limits were requested as declared. Actual JAX worker counts in the **timing process** were not measured, despite the protocol asking for that observation. The manifest's device and Torch OpenMP metadata do not establish JAX's actual worker count. Disclose this departure; separate process observations must be labeled separately. Do not claim all runtimes actually used exactly one worker.
2. The first resource-probe script inspected a representative development geometry, although PROTOCOL-v2.memory says frozen v2 cases. Root was notified. Either repeat the separate resource probe on explicit frozen inputs or disclose the representative-input departure. Resource probes are outside the timing result and cannot retroactively measure the timing process.
3. The bootstrap interval covers variation across this authored synthetic family on one host. It does not capture different hardware, compilers, unrestricted profiles or independent authors' workflows. The 47/48 wall win count is more informative than an unsupported every-case claim.
4. Current pyproject includes native sources and requirements-boundary.lock in the sdist, excludes the local compiled native artifact from the pure wheel, and registers photonsmith-boundary. Actual archive contents and installation remain separate checks. The sdist configuration initially lacks an explicit compiled-artifact exclusion, so inspect its contents. A regular pure-wheel install intentionally has no native library; document an editable-source build or an explicit build destination in the installed package before promising that CLI works after ordinary installation.
5. Independent geometry workloads, changed-objective end-to-end optimization, public source/evidence and a substantive outside-human response remain separate finish-line gates. This agent audit cannot satisfy the external-human requirement.

Recomputation artifacts: `audit_benchmark_v2.py`, `benchmark-v2-independent-recomputation.json`, `benchmark-v2-bootstrap-reconciliation.json` and `benchmark-v2-case-ratios.json`.
