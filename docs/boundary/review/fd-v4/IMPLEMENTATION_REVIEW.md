# Independent prefreeze review of the v4 verifier

Disposition: accepted within the tested numerical and evidence scope below. No production adjoint change is required or implied. This is coding-agent review, not external human validation or a performance result.

## Bound source identities

- `photonsmith/gradient_check.py`: `e3656becfbfafe40074b7e4b10d62ac96f87c57e8971f94b640d899660a4bf7e`
- `scripts/summarize_boundary.py`: `986499900809dbbc275ff931c828c4e4cd8dfddd5423cb6a78a717adeed2d627`
- `scripts/benchmark_boundary.py`: `86557ecae0a1a1280c8c50a0f6079f383f8c6b45a2d0b28355709d2bb49ce767`

These are the actual files reviewed under `validation-v4/PhotonSmith`. This acceptance does not apply to a later uninspected numerical or acceptance-rule change.

## Mathematical and execution checks

The central and one-sided coefficients/signs are correct fourth-order first-derivative formulas. The three nominal step sizes remain `1e-4, 1e-5, 1e-6`; stencil selection uses only the parameter, step and fixed bounds. No observed value selects a favorable side or smaller step. The helper uses `math.fsum`, retains all offsets, binary64 coordinates, weights and forward values, and reconstructs the separately retained second-order diagnostic.

`dense_checks` evaluates only the original dense matrix-exponential forward solver. It does not invoke a native objective or either analytic gradient to produce the finite-difference oracle. The fixed-model benchmark validates period 2.4, layer height .3, index 2 and target +1 before executing this helper, so the helper's default physical model matches the measured input definitions.

The runner applies the independent checks to every case, all six coordinates and every fixed step. Its performance timer stops and records the method result before power and finite-difference verification calls. Verification evaluations therefore do not enter a solver's measured call duration. Shared-host scheduling and any effect of between-case verification work remain ordinary timing limitations, not a dedicated-host claim.

The scorer uses the worse of the native-gradient-normalized and independent-FD-normalized errors for the same unchanged `2e-5` criterion. This is at least as stringent as either normalization alone. Both estimates and errors remain available; no best-step selection occurs. Historical manifests without the new scheme retain their historical calculation, so v3's failed gate is not rewritten.

## Consequential gap found and fixed before freeze

The initial implementation could fall back to historical first-eight-case handling if its `finite_difference_scheme` field disappeared, even with other v4 declaration fields still present. The builder fixed that before the proposed freeze. The reviewed implementation now requires the exact scheme, full planned case IDs and complete step list whenever any v4 declaration key, helper-source identity or per-record scheme appears. A lost scheme cannot downgrade a v4 record into an easier historical check.

## Independent actual-code tests

`review_actual_v4.py` calls the actual helper on exact and near interval boundaries using constant, signed linear, signed quartic and signed exponential functions with independent analytic derivatives. There are 24 function/step/geometry records, covering 144 coordinate derivatives. Maximum normalized discrepancy was `4.93e-10`, below the independently chosen `1e-7` control tolerance. The checks also confirm feasible coordinates, unchanged unrelated coordinates and no mutation of caller inputs.

A schema-only fixture combines existing raw timing identities with explicitly synthetic linear-function verification records. It is not scientific evidence and is not published as a new physical result. The actual evidence validator accepted its complete schema, then rejected seven targeted corruptions:

- Missing independent checks on a case beyond the historical first eight.
- A missing fixed step.
- A lost global scheme with v4 metadata still present.
- A changed sample coordinate.
- A changed retained forward value without a matching derivative.
- An unapproved oracle identity.
- A duplicated step replacing a required one.

All seven corruptions were rejected. The generated fixture directory was disposable and removed; only the explicitly labeled review result remains in `actual-v4-review.json`. The earlier viewed-case dense calculation and raw samples remain in `old-case-fourth-order-diagnostic.json`.

## Remaining proof requirements

Freeze the full new inputs, source identities and protocol before scoring. Retain every case, step, sample and failed attempt. A finite-difference or solver failure must prevent an all-pass claim; the acceptance above does not predict that every future high-curvature case will pass. If mathematical source or acceptance rules change after viewing the new cases, retain that run as development and freeze a fresh confirmation.

The old v3 gate remains failed. A successful fourth-order fresh study can resolve the inaccurate verification measurement without lowering the gradient-accuracy target, but must be described as a changed measurement method, not identical-protocol replication. No continuum, arbitrary-conditioning, fabrication, physical-measurement, universal performance or independent-human-validation certificate follows from this review.
