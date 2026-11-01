# Independent numerical review of the resonance correction

This is a separate coding-agent review. It is not external human validation, adoption, a physical experiment or an independently repeated performance benchmark. No implementation, repository history or public message was changed by this reviewer.

## Finding reproduced

The published midpoint replacement is incorrect for accepted sharp modal resonances. At h=1, eta=1e7 and eigenvalues pi^2 +/-4e-6, its transmission divided difference has relative error 1.026598 against a 70-decimal-digit scalar evaluation at the exact binary64 endpoints. An ordinary quotient has relative error 1.37e-10 in this particular case.

The error affects a complete solve. A three-mode positive-definite Hermitian epsilon matrix, with a pi/8 rotation mixing the close eigenmodes, yields a native/reference transmitted-power directional derivative of approximately +126640.571735. An independent 70-digit transfer-system calculation gives approximately +62489.26099. Centered differences converge toward the transfer result as steps decrease from 1e-6 to 1e-9. The sign depends on the chosen direction. This example is a supported Hermitian operator; it does not establish that every arbitrary Hermitian matrix represents a local fabricated material.

See `reproduce.py`, `reproduction-v1.json`. The transfer oracle constructs the field transfer matrix from C(Q), S(Q), -Q S(Q), C(Q), imposes both exterior radiation conditions, solves for reflected/transmitted fields, and differentiates the resulting power at high precision. It does not reuse PhotonSmith's port adjoint or Loewner implementation.

## Algebra independently checked

For z=lambda*h^2, a=eta*h, C=cos(sqrt(z)), S=sin(sqrt(z))/sqrt(z), D=C-i(a^2+z)S/(2a) and N=-i(a^2-z)S/(2a), the correction evaluates divided differences algebraically rather than treating a finite gap as equality.

For positive pairs, u=(sqrt(zi)+sqrt(zj))/2 and v=(zi-zj)/(2*(sqrt(zi)+sqrt(zj))) preserve the small phase separation. The identities for divided differences of C and S are correct. The expanded reflection numerator is also correct; it avoids cancellation between separately large products involving N and D. For negative pairs, the common fixed exp(-u) scale cancels from reflection and is retained in transmission. This scale is fixed for the pair, so no scale derivative is omitted. Polynomial divided differences handle pairs close to zero. Exact equality continues to use the analytic derivative. The phase-gap branch safely reaches large-phase neighboring eigenvalues; mixed-sign eligible pairs necessarily lie in the small-z polynomial region.

Actual reviewed C++ SHA256: `dd55bc7113b2db50a5e5c4d1de14c8435c856b23377f917ffcb38e3fa6f00a65`.

The tested Python numerical source SHA256 was `58efd3e4719f54250428335806d5219cc47ddb1618de1934b5e2ad38e581c20f`. The builder subsequently changed only its unbounded docstring claim to bounded wording, producing `cb091d9cfbd33ddc86651517c2b81908278a1ed7e0a13aee40341a9f560a3aba`. Replacing only that docstring text reconstructs the exact tested hash; see `docstring-only-check.json`. Numerical acceptance covers this documented-only revision, not an uninspected future mathematical change.

## Actual implementation checks

- `review_actual_revision.py` compares the actual Python `_loewner` and actual C++ `modal`/`close_divided` code, included in a review-only wrapper, against 70-digit arithmetic. It exercises 1,509 pairs and 6,036 backend/component comparisons. The grid spans a=1e-8 through 1e8, zero and repeated values, propagating and evanescent values, machine-neighbor separations, ordinary separations, and high-phase neighbors up to the advertised 1e8 phase limit. Maximum normalized error is 9.73594e-6 in Python and 4.15535e-6 in C++, below the scoped 2e-5 test criterion. `actual-revision-review-v1.json` records these results and source identities.
- `review_scaled_pairs.py` adds 1,176 backend/component comparisons at layer heights 1e-4, 0.3, 1, 3.3 and 1e4, with eta adjusted to hold the dimensionless contrast parameter fixed. Its maximum normalized error is 1.14333e-5, at h=0.3 and a=1e-8 near the tenth resonance. The largest a in this auxiliary grid is 9.99999e7. An initial attempt using exactly 1e8 hit the existing phase guard after nonunit-height rounding and stopped; it was not counted as a numerical pass. See `scaled-pair-review-v1.json`.
- The actual corrected native full-solve derivative is 62489.26096936 versus the independently differentiated transfer value 62489.26098761, relative error 2.92e-10. Python's relative error is 1.13e-9. The native build receipt matched the reviewed source hash. See `actual-full-solve-v1.json`.

These scalar comparisons are checks of the matrix-function derivative coefficients, not independent full-workload benchmarks. The complete objective check independently verifies that the correction reaches the physical observable through the adjoint.

## Rejected simpler routes

Multiplying the midpoint-gap threshold by max(a,1/a) is not sufficient. At threshold 1e-5 it produces up to 0.01825 normalized scalar error in the stress grid because it forces cancellation-prone endpoint subtraction. Thresholds 1e-3 and 1e-2 still leave errors approximately 6.64e-5. All trials are retained as `criterion-stress-*.json` and the original `criterion-stress-v1.json`.

A review-only Gauss derivative-integral prototype with contrast-aware seeded subdivisions also underperformed the algebra here. Its 16-point trial reached 2.57e-5 normalized error and flagged an 8.2e-5 8-versus-16 estimate on the worst case. It is not accepted production code. See `quadrature-stress-v1.json`.

## Regression tests and limits

The added regression file `tests/test_resonance.py` has a useful independent oracle: retained 70-digit scalar quotients plus a complete excited power-gradient check against an independently computed transfer derivative. The scalar fixtures cover the original report, a close pair crossing zero at tiny a, a thick evanescent pair, and a machine-neighbor pair near a higher resonance. The complete check exercises both reference and native implementations when native is available. Existing tests continue to cover exact/near-zero and repeated eigenspaces, arbitrary Hermitian directions, thick slabs and the original geometry gradients. The new material-index replay regression correctly tests the separately identified metadata defect. The missing material field and incomplete-run summarizer are outside this numerical review's primary scope.

The remaining roughly 1e-5 errors at extreme contrast are consistent with double-precision endpoint phase and nondimensionalization roundoff amplified by a very narrow resonance. No uniform accuracy guarantee over every accepted finite parameter, arbitrary conditioning or a below 1e-8 is established by this grid. In particular, a has no explicit positive lower bound in the API. Input-matrix eigensolver error can itself shift a resonance. Keep those floating-point/conditioning limitations explicit. The correction retains the finite spectral gap; it should not be described as resolving arbitrarily narrow resonances without qualification.

Compensated sqrt-phase evaluation could reduce some endpoint roundoff, but is not required to pass the scoped tests above and would not eliminate eigenvalue conditioning. This reviewer does not recommend adding that complexity merely to improve a reported decimal. The existing operating limits are not being narrowed to conceal the reproduced defect.

The original source's timing results cannot be assigned to a changed kernel. A new frozen-input benchmark and actual installation/CI checks of the eventual publication revision are still required. Those are being handled by the parent and are not inferred from this mathematical review.

## Disposition

The reported midpoint-gradient bug is reproduced and the actual algebraic correction fixes it. No remaining consequential algebraic defect was found within the tested envelope. Acceptance is bounded by the source identities, independent tests and conditioning limits above. Public/external human validation remains a separate unfulfilled gate.
