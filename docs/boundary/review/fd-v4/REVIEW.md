# Independent assessment of the finite-difference gate

This is a coding-agent review, not independent human validation or a new performance experiment.

## Disposition

A predeclared fourth-order finite-difference oracle, using the same nominal steps `1e-4, 1e-5, 1e-6` and the same `2e-5` error bound, is a defensible strengthening of the verification. It uses more independent forward evaluations to reduce approximation bias. It must not retroactively turn the saved second-order v3 gate into a pass. Preserve that failed run and describe v4 as a new experiment with a more accurate measurement procedure.

Changing the production gradient to approach the biased coarse finite difference would be incorrect. The original dense matrix-exponential/Frechet gradient and the current native gradient agree on the failing case to `2.90e-14` normalized error. The dense forward objective reproduces the coarse-step discrepancy, independently of the native adjoint.

## Independent viewed-case calculation

`check_old_case.py` evaluates the unchanged dense `physics.solve` forward objective on the actual `v3-m9-09` geometry. It checks the installed Python bytes against the public source and the actual loaded native library against its build receipt. It does not derive finite differences from native objective calls or invoke the dense analytic gradient inside the finite-difference construction.

| Step | Second-order dense-forward error | Fourth-order dense-forward error |
|---|---:|---:|
| 1e-4 | 8.55119e-5 | 2.77210e-9 |
| 1e-5 | 8.55107e-7 | 1.25734e-10 |
| 1e-6 | 6.21943e-9 | 3.29148e-9 |

These errors use `max(1, abs(independent FD estimate))`, as the initial finish line states. The previous summarizer normalized by the analytic gradient instead. For the new study, requiring both normalizations to meet the unchanged threshold avoids gaining a pass through a denominator change. The larger of the two errors should control acceptance.

The complete scalar forward samples and source identities are retained in `old-case-fourth-order-diagnostic.json`. This is a viewed diagnostic, not held-out confirmation. The first diagnostic attempt stopped because the public source checkout intentionally lacked its excluded optional native library; the corrected routing explicitly used the already-built installed package and verified its bytes/receipt. No source or environment was modified.

## Required stencil and boundary rules

For a component with both `p-2h` and `p+2h` inside the documented bounds, use

```
[f(p-2h) - 8f(p-h) + 8f(p+h) - f(p+2h)] / (12h)
```

This uses a wider aperture than the old stencil, not a smaller hidden step.

For a lower-bound case, if all five `p + j*h` points for `j=0..4` are feasible, use the fourth-order forward formula

```
[-25f(p) + 48f(p+h) - 36f(p+2h) + 16f(p+3h) - 3f(p+4h)] / (12h)
```

For an upper-bound case use the sign-reversed formula at offsets `0,-1,-2,-3,-4`. Width bounds remain `[.15,.85]`; shift bounds remain `[0,1]`. Select a stencil using only the unperturbed parameters, nominal step and bounds before evaluating objectives. Do not clip, wrap, adapt the step or select whichever stencil gives a favorable derivative. If no complete stencil is feasible, refuse explicitly. The planned tiny steps and documented interval widths guarantee a side is available for valid inputs.

Use compensated summation such as `math.fsum` for the weighted numerator. Retain all actual binary64 sample coordinates, weights, function values, failures and nominal steps. A deterministic identifier for every component/step/side is needed. Keep second-order diagnostics too; when a centered diagnostic is geometrically illegal, label any one-sided diagnostic honestly rather than fabricating an outside-domain evaluation.

## Fresh-study requirements

- Freeze the new source and protocol, complete 48-case set, seed, input bytes/hash and step/stencil rules before scoring new cases.
- Check all six coordinates and all three steps on all 48 cases if feasible. This strengthens the earlier first-eight-case FD coverage. No favorable-case or favorable-step selection.
- Generate the FD oracle from the unchanged dense forward-only solver. Keep independent external analytic-gradient comparisons for every timed case. Numerical verification belongs outside performance timers, and no Meep/build job should overlap the timing experiment.
- Require both native-gradient-normalized and FD-normalized errors to be at most `2e-5` for every coordinate/step/case. Counts, completeness and finite-value checks must fail closed. Preserve all performance and power thresholds.
- Retain the old second-order v3 failure, including its raw samples and unchanged gate, alongside new results. Do not describe v4 as identical-protocol replication or erase the fact that the measurement method changed.
- The independently reviewed native solver source need not change to fix this measurement problem. If any mathematical implementation change follows viewed v4 outcomes, preserve v4 as development and declare a fresh confirmation.

## Useful controls

Exercise exact lower/upper and near-boundary parameters such as `[.15,.85,.5,0,1,.5]`. Include constant, linear, quartic and smooth exponential functions with known analytic derivatives and both derivative signs. Verify computed derivatives from retained points/weights, not only method labels. Check feasible coordinates and no mutation of caller inputs. Reject nonfinite functions and malformed or missing sample records. A test that deliberately perturbs one retained function sample should fail evidence reconciliation.

Fourth-order errors can reach the floating-point roundoff floor at the smallest step, so monotonic decrease is not a universal valid acceptance rule. The fixed absolute/normalized accuracy limits must still hold at every step. Multiple steps and independent analytic derivatives provide cross-checks, not a license to select the best result.

This review accepts the mathematical plan within the stated finite tested scope. It does not certify a future implementation, all possible high-Q geometries, continuum accuracy, hardware performance or the external-human-validation gate.
