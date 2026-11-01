# Numerical and evidence corrections after public review

The first public candidate is commit `169a7d169bc6b9034dc23b56e8d14d1bab4a92a3` in [PR #1](https://github.com/nazeeh111/PhotonSmith/pull/1). Its six CI jobs passed. A subsequent automated Codex review found three material gaps that those checks did not cover. Automated review is useful evidence, but is not independent human validation.

## Sharp-resonance adjoints

The original small-gap branch replaced every modal divided difference with a midpoint derivative when `abs(lambda_i-lambda_j)*h² < 1e-5`. This can fail even for positive Hermitian permittivity and phases within the documented limit. At `h=1`, `eta=1e7`, and eigenvalues `pi² ± 4e-6`, the transmission divided difference has about 103% relative error. A rotated, excited three-mode layer returns a power directional derivative near 126,641 instead of the independent 70-digit transfer-system result near 62,489.

The corrected kernel keeps the analytic derivative only for exactly coincident eigenvalues. For distinct close eigenphases it evaluates algebraic divided differences of the entire cosine and sinc functions, then forms reflection/transmission quotients without subtracting nearly identical scattering amplitudes. A common exponential scale handles negative eigenvalues; polynomial divided differences handle zero crossings. No external solver is changed, and no finite-difference derivative is used in production.

A separate coding agent checked 1,509 scalar pairs, comprising 6,036 reflection/transmission and Python/native comparisons against high-precision scalar references. The tested dimensionless reference admittance spans 1e-8 to 1e8, with repeated/zero modes, negative eigenvalues and neighboring phases through 1e8. Maximum normalized discrepancy was 9.74e-6 for Python and 4.16e-6 for native. An additional 1,176 comparisons across five layer thicknesses had maximum normalized discrepancy 1.14e-5. The corrected full native counterexample agreed with the independent transfer derivative to 2.92e-10 relative error. [Review and retained evidence](review/resonance-rc2/REVIEW.md). These are finite tested envelopes, not a universal error certificate. Extreme contrast, phases, eigensystem conditioning and Fourier truncation remain limitations.

The original timing measurements belong to the original kernel. The corrected kernel receives a fresh, predeclared 48-case confirmation v3 with unchanged accuracy/performance thresholds; earlier data and failed approaches are preserved. Its protocol also records actual process-thread observations within the benchmark process. Total process threads include idle workers and do not establish a strict single-thread JAX comparison.

## Incomplete experiment scoring

An interrupted run could previously be summarized as complete because the expected count came from the progressively written `cases.json`. The first eight cases of the 48-case study reproduced this defect and earned all old local gates while omitting 40 planned cases. This does not describe the saved complete v2 run, which contains all 48 cases and 1,680 attempts.

The summarizer now requires the original `inputs.json`, checks its SHA256 against the manifest, and matches the completed case definitions against the entire frozen plan. It verifies the exact Cartesian set of case/repeat/method observations and checks that each summarized observation equals its raw record. Invalid timing/numerical values and missing finite-difference step sweeps are refused. The runner saves the input bytes and planned case IDs before scoring. Regression tests reject interrupted prefixes, missing/changed plans, changed case parameters, duplicate/missing observations and inconsistent timings. Recomputing the existing full v2 dataset still produces its saved summary exactly.

## Material-index replay

Optimization JSON omitted a nondefault material index, although simulation JSON recorded it. The corrected result records `material_index`, period and layer height. A regression test optimizes with index 2.15 and reconstructs the reported efficiency using the saved geometry, wavelength, modes and material. Old optimization JSON must use its originating invocation/protocol to recover the material; the omitted value is not inferred from the geometry or silently filled in.

## Status

The corrected candidate is version 0.2.0rc2. Source and benchmark identities, current checks and publication state belong in the accompanying report. The original candidate's successful CI does not establish successful remote execution of a later revision. An unanswered upstream discussion does not establish external validation or adoption.
