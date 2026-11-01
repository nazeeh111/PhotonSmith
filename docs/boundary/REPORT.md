# Current verification and preserved publication history

The latest local evidence is [validation v4](VALIDATION_V4.md): all seven local gates pass using a stronger independent fourth-order oracle at all original step sizes and the unchanged 2e-5 bound. Latest JAX wall speedup is 2.78x [2.61, 2.97]; all 48 cases, 1,680 timing attempts and 864 gradient estimates are retained. The production kernel is unchanged. Historical v3 stays failed. Independent Meep controls/convergence have passed all nine stages and 535 raw-data/identity checks; [the Meep report](MEEP_V4.md) gives the bounded result. Outside-human validation is absent. V4 publication is pending.

The report below is preserved from public commit `c259ff80eb4c41b88031d3b7ea6f220c518fe299` and describes its historical v3/v2 results and pre-publication status. Its 3.19x and 3.79x figures belong to those measurements. Use the v4 page above for the current result.

---

# Measured result and remaining proof

**The full external-validation finish line is not achieved.** The original candidate is public in PR #1 and passed native Linux/macOS and portable-package CI. Subsequent automated review identified a sharp-resonance gradient defect and two reproducibility gaps. See [the correction record](CORRECTIONS.md). The corrected 0.2.0rc2 candidate requires its own measured evidence and exact-source remote checks. Nothing here establishes independent human review, adoption or physical-device performance.

## Corrected-kernel confirmation v3

The corrected source uses C++ SHA256 `dd55bc7113b2db50a5e5c4d1de14c8435c856b23377f917ffcb38e3fa6f00a65`. A fresh 48-case plan was declared before scoring, with 12 cases each at M=3,5,9,15 and seven paired repeats per method. Input SHA256 is `4a6b425ab4d254e9f9b8ab687ce71651297fe715ea6a821a065ea78817d6e2c6`. All 1,680 attempts are retained; none failed or was omitted. [Protocol and raw evidence](../../benchmarks/boundary/confirmation-v3/).

| Comparator | Wall-time speedup | Case-bootstrap 95% interval | Process-CPU speedup |
|---|---:|---:|---:|
| Existing PhotonSmith | 11.29× | [9.89, 12.90] | 7.39× |
| meent 0.13.2 JAX JIT, specialized 1D TE | **3.19×** | **[2.57, 3.90]** | **3.37×** |
| meent 0.13.2 Torch, specialized 1D TE | 14.02× | [10.91, 18.08] | 8.30× |
| grcwa 0.1.2 autograd, same retained x orders | 82.86× | [69.19, 99.37] | 25.44× |

Native wins 44/48 case median wall comparisons against the fastest tested external aggregate, meent JAX JIT. Its JAX speedup by M=3,5,9,15 is 6.88×, 3.81×, 2.80× and 1.40×. The four losses are retained. The much larger grcwa ratio includes its internal vector/polarization overhead and is not a general RCWA record. All timings remain conditional on this shared Apple M4 host; neither the old nor new result establishes a portable 10× advantage.

Maximum full-order power discrepancy against meent is 1.35e-14, balance error 9.33e-15 and residual 2.49e-15. Maximum normalized gradient discrepancies are 1.09e-12 against the unchanged dense PhotonSmith baseline, 1.11e-13 against meent JAX, 6.80e-13 against grcwa and 8.17e-8 against meent Torch.

**One original protocol gate fails.** The maximum finite-difference discrepancy is 8.55e-5 at the deliberately largest step, 1e-4, on `v3-m9-09`; the criterion is 2e-5. At steps 1e-5 and 1e-6 the maxima decrease to 8.55e-7 and 9.77e-9. Independent diagnosis reproduces the coarse-step bias with the unchanged dense forward solver and finds analytic-gradient disagreement only 2.90e-14. Post-hoc Richardson extrapolation reduces the diagnostic discrepancy to about 1.23e-10. This supports central-difference truncation as the explanation; it does not change the original rule. The saved `finite_difference` gate remains **false**, and this is **not an all-gates-passing confirmation**. No case, step or failed criterion is hidden, and no new seed was sampled to chase a passing result.

This run records 50 thread snapshots in the actual benchmark process, with 73–74 total threads and one running thread at the observations. These include idle runtime workers and are snapshots outside the timers, not a strict single-thread JAX guarantee. Requested BLAS/Torch numerical limits remain one. The earlier RSS probes concern the earlier source and are retained as historical evidence, not a new memory measurement.

The corrected source passes 40 local tests and the 46 guarded ASan/UBSan native cases plus four high-precision quotient checks on macOS. The 180 externally authored computational workloads were rerun against meent and grcwa with no failures; maximum order-power differences remain 7.18e-12 and 6.05e-12. The three earlier changed-wavelength/order/material design workloads were also replayed once per method with the corrected source; final M31 target powers agree with a separate real-space discretization to 8.83e-5, 5.67e-4 and 2.81e-4. These are development replays, not another timing confirmation. These checks neither remove the failed coarse-step gate nor supply independent human validation. Updated-source remote CI and public recipient execution remain pending its publication.

## Historical original-kernel confirmation v2

The original pre-implementation targets were at least 5× versus the existing PhotonSmith solver, 2× versus the fastest tested comparable external solver, accurate powers/gradients, independent-author workloads, reproducible public evidence and substantive external engineer validation. Public evidence exists; substantive outside reproduction/review/use is still absent. The table below measures the original kernel, not the numerical correction.

Confirmation v2 fixed 48 geometries and wavelengths before scoring, with 12 cases each at M=3,5,9,15. Each method executes the same scalar TE problem with identical complex128 Fourier coefficients and returns the target efficiency plus all six width/shift derivatives. Seven repeats per case randomize method order. **All 1,680 timed attempts are retained, with no failed, missing, duplicated or excluded results.** Compilation/import warmups are excluded from timing and recorded separately; JAX takes wavelength and geometry as dynamic runtime inputs and is synchronized before stopping each timer.

The main statistic is the geometric mean across cases of each case's median paired baseline/native wall-time ratio. Case-bootstrap intervals use 10,000 resamples. These intervals describe variation across this case set conditional on the measured machine; they do not represent uncertainty across computers.

| Comparator | Wall-time speedup | Case-bootstrap 95% interval | Process-CPU speedup |
|---|---:|---:|---:|
| Existing PhotonSmith, unchanged physical solver | 13.12× | [11.09, 15.82] | 7.21× |
| meent 0.13.2 JAX JIT, specialized 1D TE | **3.79×** | **[3.13, 4.63]** | **3.34×** |
| meent 0.13.2 Torch, specialized 1D TE | 12.53× | [10.41, 15.09] | 7.54× |
| grcwa 0.1.2 autograd, same 1D orders | 65.07× | [55.32, 76.34] | 23.77× |

**JAX is the fastest external aggregate comparator.** Native wins its case median wall comparison in 47/48 cases and loses one. Speedup versus JAX by M=3,5,9,15 is 6.76×,4.97×,2.77×,2.22×. grcwa carries vector/polarization work internally, so its much larger ratio is not the headline comparison or evidence of a general RCWA record. No GPU comparison is claimed. Backend-specific BLAS/LAPACK implementations are part of these measured software systems, not isolated algorithmic effects.

This was a shared Apple M4 workstation. Scheduling affected wall time substantially: a prior preserved 48-case development/confirmation precursor measured 7.81× versus PhotonSmith and 3.33× versus JAX, whereas process-CPU ratios remained approximately 7.1–7.2× and 3.3–3.5×. Do not turn the v2 13.12× into a portable “10× faster everywhere” claim. No quiet dedicated-host replication has occurred.

NumPy/SciPy/BLAS and Torch thread limits were requested as one. **The actual JAX worker count in the timing process was not recorded**, a deviation from the detailed protocol. Separate frozen-input probes observed 72 total JAX-process threads, with one running at the sampled instant; this includes idle workers and is not an active-kernel thread count. This is not a strict single-thread JAX benchmark.

## Accuracy and operating envelope

The following original-kernel accuracy, phase-cost, RSS and development-use records are historical unless explicitly identified as corrected-source replays above. Their source identities remain in the original manifests; they are not reassigned to the corrected kernel.

On v2 cases, maximum all-order reflected/transmitted power disagreement versus production meent NumPy was **1.943e−14**; maximum lossless balance error was **8.89e−15**. The objective and all six derivatives were also compared with actual meent JAX, Torch and grcwa autograd. Worst normalized gradient disagreement was **3.05e−8** (Torch); JAX was below **2.90e−13**. Independent finite-difference steps 1e−4,1e−5,1e−6 are retained for eight frozen cases; the largest-step maximum normalized error was below 5.13e−6.

Three independently authored meent example geometries provided **180 attempted computational workload cases**, all retained: a uniform high-index film, a two-layer high-index grating and a two-bar structure above a 100-µm homogeneous layer. Their parameters and source commit are recorded. Comparisons at M=3,7,15,31 match full reflected/transmitted order powers against both production meent and grcwa to **7.18e−12 / 6.05e−12** maximum absolute error. They are independent-author computational workloads, not experimental measurements or external human validation. The legacy Moharam example's aliased Fourier assembly was replaced with complete difference-index coefficients; this is geometry reuse checked against production solvers, not an assertion that its old script was reproduced unchanged.

Independent coding-agent checks used separately assembled transfer matrices/exponential Fréchet derivatives, excited mixed repeated/zero modes, unequal exterior indices, 64 bidirectional conservation checks and 252 reciprocity pairs. Native/Python comparisons extend through M=31 and thick analytic slabs. ASan/UBSan passed 46 guarded numerical cases plus bounded-error/allocation/input checks. Leak detection was unavailable on macOS and is not claimed. Review findings, source hashes and platform limits are retained.

## Actual use, timing breakdown and memory

The separate CLI runs genuine simulations and L-BFGS-B optimization with the native exact gradient. Three changed-objective/material studies, three repeats per method, produced matching final efficiencies and identical forward counts (29,46,30). Native final M=31 target efficiencies were 0.900379,0.754398,0.490957. Separate real-space FDFD at 128 transverse samples and 181 device nodes disagreed by 0.0000883,0.0005669,0.0002807. These are simulated 2D designs, not fabricated devices. End-to-end development wall ratios were approximately 8.1×–13.9×; they are exploratory because the first study overlapped packaging and the shared-host load was uncontrolled. This is not another held-out performance confirmation.

A supplemental generic-native probe separately measures forward-only and forward-plus-full-layer-matrix-adjoint calls. Matrix construction is excluded, and paired differences estimate incremental adjoint cost. This is **not** a direct isolated backward-phase timer or the primary six-parameter comparison. All 15 pairs per M, including noisy differences, remain in the record. Median forward/forward-plus-adjoint wall times in milliseconds were 0.148/0.203 (M3),0.590/0.652 (M5),1.115/1.424 (M9),5.343/5.730 (M15).

Fresh-process M15 probes on the same frozen v2 input observed peak RSS including imports/JIT: native51.4MiB, oldPhotonSmith56.1MiB, meentTorch270.4MiB, meentJAX294.8MiB, grcwa64.8MiB. These are process high-water marks, not memory attributable solely to kernels. They were measured separately from timing. Earlier default-input probes are retained task-locally as development evidence.

A clean outside-checkout wheel installation plus explicit kernel build from the source distribution reproduced a changed wavelength, material and reverse-order optimization; output overwrite was refused with existing bytes preserved. The source/wheel archives exclude the host-specific dylib. The native source, Python API, numerical suite and guarded harness are retained for reproduction. The existing trained-model dataset, numerical results and GUI physics were not changed.

## What remains

1. Publish this corrected candidate's exact reviewed source/results on the existing PhotonSmith repository after owner approval; run its Linux/macOS native CI and verify remote bytes and a public clean install. The original public candidate's checks do not substitute for this.
2. Obtain substantive reproduction/review/use from an external engineer or researcher, or acceptance of a technically useful upstream contribution. A posted request, stars, generated reviews or CI alone cannot satisfy this.
3. Replicate timing on a quiet independent host before claiming portability or general speed superiority. Absorption, vector/2D periodic polarization, fabrication and hardware measurements remain outside scope.

A defensible **corrected-source local-result draft**, pending its public evidence and external review:

> Engineered a C++/LAPACK photonics solver with analytic matrix-function gradients; measured 3.2× speedup over meent JAX on 48 fixed scalar-TE CPU cases and cross-checked 180 reference workloads against meent and grcwa.

The measured speedup is conditional on the shared host; the report retains the failed coarse-step finite-difference gate and its convergence diagnosis. This draft does not claim an all-gates-passing confirmation.

Do not add “externally validated”, “adopted”, “state of the art”, “new physical device” or a universal 10× claim without the corresponding evidence.

[Reproduction and exact records](REPRODUCE.md) · [Method and prior art](METHOD.md) · [Independent coding-agent audit](review/REVIEW-benchmark-v2.md)
