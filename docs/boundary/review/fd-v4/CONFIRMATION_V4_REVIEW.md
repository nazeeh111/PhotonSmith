# Independent raw audit of confirmation v4

Disposition: all seven declared local numerical/performance gates pass on the frozen v4 experiment. This is a separate coding-agent audit of actual raw records and independently recomputed statistics, not independent human validation, adoption, physical measurement or a universal solver-accuracy certificate.

## Experiment identity and completeness

The input SHA256 is `db8d5d5ff4abe5146af409474424ad65bd1573ef72ab91cc15e3d1c445e10bd6`. Run inputs are byte-identical to the frozen plan, and the same hash appears in the protocol and manifest. Regeneration from seed `20261101380`, the declared parameter distributions and the seeded final permutation reproduces all 48 case definitions exactly. There are 12 cases each at M=3,5,9,15.

All 1,680 raw attempts are present: 48 cases times seven repeats times five actual methods. Their case/repeat/method Cartesian set is exact, with no missing, duplicate, unexpected or failed observation. Timings are finite and positive; values and all six gradients are finite. The method sequence reproduces the complete seeded randomized shuffle. Each saved case observation equals its corresponding raw row. Native values and gradients are identical across all seven repeats, so selecting its first recorded gradient does not hide a divergent repeat.

The current source bytes match every source identity in the frozen plan and run manifest. The actual native library matches its build receipt; its source SHA256 is `dd55bc7113b2db50a5e5c4d1de14c8435c856b23377f917ffcb38e3fa6f00a65`. The independently reviewed verifier/scorer/runner hashes remain `e3656bec...a4bf7e`, `98649990...2d627`, and `86557eca...9ce767`. No production adjoint change is assigned credit for fixing the finite-difference measurement.

## Independent finite-difference reconstruction

The complete dataset contains 144 case/step records and **864 coordinate derivative estimates**, using **3,456 retained dense-forward values**. All 864 selected stencils are central on this random case set. Boundary rules were separately reviewed and exercised with exact/near-boundary analytic controls; the absence of a random boundary stencil is not evidence that the input family was narrowed.

For every estimate, the audit independently checked:

- Exact case parameters, component, nominal step, offset sequence and mathematical weights.
- Every actual sample coordinate and finite forward value, with all samples inside the fixed bounds.
- Reconstruction of the fourth-order derivative and retained second-order diagnostic from those samples.
- Both normalization rules and their worst-error acceptance decision.

Maximum fourth-order normalized discrepancies are:

| Nominal step | Worst of both normalization rules |
|---|---:|
| 1e-4 | 7.46675e-9 |
| 1e-5 | 5.86616e-9 |
| 1e-6 | 4.23319e-8 |

Every case/component/step is below the unchanged `2e-5` bound. The audit also evaluated the retained weighted values with 60-decimal-digit arithmetic; the largest difference from the stored binary64 weighted reconstruction was `4.16e-11`. Thirty-two independently rerun dense forward samples exactly matched their retained values. These are bounded spot checks of producer behavior, alongside the exhaustive sample/arithmetic audit.

The retained ordinary second-order diagnostic still exceeds the same threshold on v4: maximum `5.47671e-5`. Thus the fourth-order pass is not explained by a fresh seed accidentally avoiding the earlier curvature problem. It is a more accurate numerical oracle at the same nominal steps and unchanged accuracy bound. Historical v3 still records `finite_difference:false`; it was not relabeled as passing.

## Independently recomputed performance

| Comparator | Paired wall geomean | 95% case-bootstrap interval | Process-CPU geomean |
|---|---:|---:|---:|
| Original PhotonSmith | 6.70536x | [6.42881, 6.99529] | 6.65806x |
| meent JAX JIT | 2.77799x | [2.60820, 2.96864] | 3.36051x |
| meent Torch | 6.47291x | [5.49254, 7.58571] | 6.43558x |
| grcwa autograd | 23.76004x | [22.76055, 24.90263] | 23.41709x |

The audit independently formed per-repeat baseline/native ratios, each case's median paired ratio, geometric means, stratified results and 10,000-resample bootstrap intervals. It did not invoke the production summarizer. meent JAX JIT is the fastest tested external aggregate. Native's median paired ratio exceeds one on all 48 cases. All larger grcwa ratios retain its vector/polarization-overhead caveat.

The 50 actual benchmark-process snapshots observe 73–74 total process threads, including idle workers. This is not a strict single-thread JAX experiment or dedicated-host timing certificate. Shared-host wall/CPU differences and the finite tested comparator set still limit portable performance claims.

## Other numerical gates

Maximum all-order power discrepancy is `2.40918e-14`, balance error `1.15463e-14`, and residual `3.54416e-15`. Independent raw gradient-error recomputation gives maximum normalized disagreements `1.99e-12` against original PhotonSmith, `2.09e-13` against meent JAX, `1.39e-12` against grcwa, and `5.02e-8` against meent Torch. These satisfy their unchanged declared bounds.

The independently recomputed gate dictionary exactly matches the saved seven-gate result: legacy speed, external speed, external interval, power, balance, gradients and finite differences are all true. This does not imply every broader project goal has been met.

## Routing failure and limits

The parent preserved a transcription of its observed direct-script import failure in `evidence/confirmation-v4-summary-routing-failure.json`. It is explicitly a parent transcription of session tool output, not an original redirected process logfile. The supported module invocation subsequently produced the summary without changing inputs, numerical sources or thresholds. The audit read that record and independently verified the unchanged source identities; it does not invent an additional original traceback artifact.

Full audit code and numerical output are `audit_confirmation_v4.py` and `confirmation-v4-independent-audit.json`. Existing v3 failure and prior review/diagnostic records remain preserved. No new performance benchmark was run by this reviewer. No source, result or public message was modified by this reviewer.

The fresh local FD gate is resolved through stronger verification, with its procedural change explicit. Public exact-source packaging/CI/reproduction and real outside-engineer/researcher validation remain separate requirements. Meep/FDTD experiments are outside this raw audit's scope and must not be inferred complete from this report.
