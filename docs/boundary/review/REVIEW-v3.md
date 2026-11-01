# Numerical review of the modal-scattering prototype

Reviewed source SHA256: `8cb56b74d338e261b4880537be898bca6015fde0dc5a9e89e9bdbf3bfb114215`.
This is independent coding-agent review. It does not satisfy the finish line's external-human-validation gate.

No consequential mathematical or numerical defect was found in the checks described below. This is hash-specific numerical acceptance within the exercised region, not a continuum proof, unrestricted stability guarantee, external adoption or whole-project acceptance.

## Independent checks

The separately assembled whole-stack transfer/Frechet derivative reference was rerun against v3 without importing its scattering formulas. Fifteen cases mix incident/output channels with repeated, exactly zero and nearly zero modal eigenspaces. Eighteen additional target-order checks use asymmetric physical bar profiles with unequal exterior indices.

- Maximum target-power difference: 2.1094e-15.
- Maximum directional-permittivity-derivative difference: 1.0322e-15.
- Maximum lossless balance error in these cases: 2.1094e-15.

The mixed-degeneracy cases use general positive Hermitian operators and are explicitly not all physical Toeplitz permittivity convolutions. They test the claimed analytic matrix-function derivative directly, rather than perturbing only uncoupled diagonal modes.

Four additional asymmetric physical stacks were excited from both directions in every propagating transverse channel, using their existing linear factorization. This covered 64 incident-channel conservation checks and 252 reciprocal transmission pairs, including unequal exterior refractive indices and transverse-momentum reversal.

- Maximum all-channel power conservation error: 3.2641e-14.
- Maximum reciprocal normalized-power error: 1.8763e-14.

Closed-form uniform-slab checks cover physical thicknesses 0.2, 100, 10,000 and 1,000,000 in the wavelength-1.55 unit system. Transmission agrees within 3.33e-16; the largest derivative absolute difference is 2.33e-10 on a derivative of order 10^5, with relative difference below 6.2e-16. Unlike v2, these cases do not consume a thickness-dependent subdivision count.

Independent infinite-evanescent-slab limits check modal lambda=-9, reference admittance=2, and thickness through 10^7. At scored thicknesses >=100 the reflection limit error is <=2.3e-16. Its analytic derivative `i eta / [kappa (eta+i kappa)^2]` agrees within 2.7e-10 at the largest thickness. Transmission underflows to zero only in the expected deeply evanescent limit; no growing exponential is evaluated.

Runnable checks, exact source snapshot and full results:

- `check_numeric_v3.py`, `numeric-result-v3.json`, `numeric-check-v3.log`.
- `check_scattering_v3.py`, `scattering-v3-result.json`, `scattering-v3.log`.
- `boundary-8cb56b74d338e261b4880537be898bca6015fde0dc5a9e89e9bdbf3bfb114215.py`.

## Inspection conclusions and remaining limits

The slab reflection/transmission formulas, external reference-port matching and complex adjoint signs are consistent with exp(-i omega t). Both output ports contribute to the transmitted electric-field objective derivative. The matrix VJP correctly uses complex divided differences and then projects to Hermitian input variations. At repeated eigenvalues the function derivative is defined without an eigenvector-gap regularizer.

For evanescent modes, the stored cosine/sinc and their derivatives are all multiplied by the same decay factor. The reflection quotient cancels that factor. The transmission derivative uses the scaled unnormalized denominator derivative consistently; differentiating the arbitrary scaling itself would incorrectly change this representation. The independently derived thick-limit checks exercise this detail.

The method has no layer-subdivision limit, but **thickness is still numerically bounded**: the source refuses optical phase above 10^8, and the global band matrix has an 8-million-complex-entry factorization allocation limit. Exact exterior cutoffs, loss, oblique incidence, TM polarization and arbitrary vector problems remain outside the contract. High-Q resonances can be physically ill-conditioned; observed conservation and residuals in this review do not establish a uniform error guarantee over every admitted input.

For close but distinct eigenvalues, the code substitutes the function derivative at their midpoint once `abs(delta_lambda)*h^2 < 1e-5`. This is a controlled finite-precision approximation, not literal exact arithmetic. The tested errors are well below the finish-line gradient tolerance. Phrase the implementation as an analytic adjoint evaluated in floating point.

The general accepted matrix API still admits positive Hermitian matrices that are not physical Toeplitz convolutions. Document that operator interface separately from physical profile constructors and keep synthetic operator stress checks distinct from external material/geometry workloads.

Performance against the fastest fair external solver, independently authored workloads, changed-objective optimization, public reproducibility and outside-human validation remain separate required gates. None is established by this numerical review. Changes to these source bytes require a scoped follow-up check.
