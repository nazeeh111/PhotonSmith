# Resolving the finite-difference gate without lowering its standard

The native solver and its adjoint are byte-unchanged from public commit `c259ff80eb4c41b88031d3b7ea6f220c518fe299`. This revision improves the independent numerical measurement of the derivative. It does not tune the solver toward a biased finite-difference answer.

## What changed

The historical v3 check used a second-order centered difference. On one case its largest step, 1e-4, had normalized truncation error 8.55e-5, above the 2e-5 limit. The unchanged dense forward solver reproduced that bias, while its analytic derivative agreed with native within 2.90e-14. Smaller steps and independent diagnostics established second-order truncation. The v3 failed gate remains false in its saved evidence.

The new protocol uses the standard fourth-order estimate

`[f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)] / (12h)`.

It keeps every original step size and the original accuracy limit. Near parameter bounds, fixed fourth-order forward/backward stencils are chosen from geometry and step alone, before observing values. No clipping, wrapping, adaptive step selection or result-dependent stencil choice occurs. All samples come from the unchanged dense matrix-exponential **forward-only** solver, not native or an adjoint-derived approximation. Cancellation-sensitive weighted sums use `math.fsum`.

| Requirement | Historical v3 | New v4 |
|---|---|---|
| Maximum normalized error allowed | 2e-5 | **2e-5, unchanged** |
| Steps | 1e-4, 1e-5, 1e-6 | **All three unchanged** |
| Cases checked by finite differences | First 8 | **All 48** |
| Components per case | All 6 | **All 6** |
| Forward oracle | Native value path | **Independent dense forward path** |
| Error normalization | Analytic-gradient denominator | **Both analytic and independent-FD denominators must pass** |
| Retained evidence | Estimates | **Every coordinate, offset, weight and forward value** |

This increases finite-difference coverage from 144 to 864 coordinate estimates. The scorer independently reconstructs estimates from all 3,456 retained samples and refuses missing cases, steps, changed coordinates, inconsistent derivatives or a lost schema declaration. Constant, signed exponential and quartic controls exercise interior and one-sided boundary formulas. Analytic boundary controls and corrupted-record tests passed independent coding-agent review before fresh cases were scored.

## Fresh confirmation

The plan was frozen before scoring: seed `20261101380`, input SHA256 `db8d5d5ff4abe5146af409474424ad65bd1573ef72ab91cc15e3d1c445e10bd6`. There are 12 cases at each M=3,5,9,15, drawn from the same full supported parameter family. All five numerical methods and all seven paired repeats remain. **All 1,680 timing attempts are retained with no failures or exclusions.** Verification occurs outside performance timers.

All seven declared local gates pass. The worst fourth-order normalized discrepancy is **4.23e-8**, below the unchanged **2e-5** limit. Maximum errors at h=1e-4,1e-5,1e-6 are 7.47e-9, 5.87e-9 and 4.23e-8. Both normalization rules pass at every case/component/step. The retained ordinary second-order diagnostic still exceeds the limit on this new set, at 5.48e-5. Thus the resolution is a more accurate oracle, not a lucky seed or removal of a difficult case.

| Comparator | Wall speedup | Case-bootstrap 95% interval | Process-CPU speedup |
|---|---:|---:|---:|
| Original PhotonSmith | 6.71x | [6.43, 7.00] | 6.66x |
| meent 0.13.2 JAX JIT, fastest tested external aggregate | **2.78x** | **[2.61, 2.97]** | **3.36x** |
| meent 0.13.2 Torch | 6.47x | [5.49, 7.59] | 6.44x |
| grcwa 0.1.2 autograd | 23.76x | [22.76, 24.90] | 23.42x |

Native wins all 48 JAX case median wall comparisons in this set. Shared-host wall timings are conditional; older ratios remain in their historical records. No portable 10x or general RCWA record is claimed. grcwa includes additional internal vector/polarization work. No GPU comparison occurs. Fifty actual process-thread snapshots observed 73–74 total threads, including idle workers; this is not a strict single-thread JAX benchmark.

Separate fresh-process probes on frozen case `v4-m15-01` observed peak RSS, including imports/JIT: native 51.8 MiB, original PhotonSmith 55.8 MiB, meent Torch 279.8 MiB, meent JAX 295.9 MiB and grcwa 64.5 MiB. These are process high-water observations, not kernel-only allocations. They ran after timing, while Meep was active on the shared host; no resource-probe timing is a performance claim. All five attempts succeeded and retain their input identity and own-process thread metadata.

Maximum full-order power disagreement against production meent is 2.41e-14, balance error 1.15e-14 and residual 3.54e-15. Gradient discrepancies against unchanged PhotonSmith, meent JAX, meent Torch and grcwa are at most 1.99e-12, 2.10e-13, 5.02e-8 and 1.40e-12 respectively. An independent coding-agent audit regenerated the input seed, timing order and statistics, reconstructed all 864 estimates and checked 32 newly executed dense forward samples. It found no remaining numerical/provenance finding in this run.

## Reproduce without the large autodiff dependencies

From a source checkout with the normal NumPy/SciPy dependencies:

```sh
python -m pip install -e .
python scripts/build_native.py
python scripts/reproduce_gradient_checks.py \
  --cases benchmarks/boundary/confirmation-v4/inputs.json \
  --output runs/new-independent-gradient-checks.json
```

This performs new dense forward evaluations for every case; it does not reuse saved estimates. JAX, Torch and grcwa are unnecessary for this correctness check. An optional `--case-id` checks a named subset and explicitly reports incomplete plan coverage. Existing output files are refused. To recompute timing statistics from retained raw evidence, use `python -m scripts.summarize_boundary` from the repository root, or install the project first.

[Raw inputs, plan, samples and results](../../benchmarks/boundary/confirmation-v4/) · [Independent audit](review/fd-v4/CONFIRMATION_V4_REVIEW.md)

This establishes bounded numerical evidence. It does not establish independent human validation, adoption, fabrication or an unconditional accuracy guarantee. A separate genuine Meep time-domain study completed all nine frozen control/convergence stages. Its maximum open-channel power discrepancy was 1.02e-4 versus the native reference, below the declared .005 limit; grid refinement was 2.82e-4 below .0025. [Full Meep method, raw data and independent audit](MEEP_V4.md). This is external-solver numerical evidence, not independent human validation.

A related Fourier-convolution correctness contribution, tested and reviewed by a separate coding agent, has been submitted as [meent PR #114](https://github.com/kc-ml2/meent/pull/114). Submission is verified; maintainer acceptance or substantive human review has not yet been established. It is not evidence that meent adopted the PhotonSmith backend.
