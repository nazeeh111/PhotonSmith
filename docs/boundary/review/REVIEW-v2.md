# Numerical review of the DtN prototype

Reviewed source SHA256: `5f895a8ed24bb14a2e02662eaf4741a0a5bd341792b57c796de30d31c3ed28e4`.
Reviewed tests SHA256: `341402e0ef099cc9470625be5ebfa1fe76597a0ad25d91780a9b0bea5c67af07`.
This is independent coding-agent review. It is not validation or adoption by an external human researcher and does not satisfy finish-line gate 5.

No consequential numerical defect was found in the exercised operating region. The six supplied tests passed. An independently assembled whole-stack transfer calculation, using SciPy `expm_frechet` for the derivative of that separate formulation, matched 15 cases with excited repeated, exactly zero and nearly zero interior modal eigenvalues. General positive Hermitian matrices deliberately mix eigenvectors with the incident and outgoing channels; they are finite-operator stress tests, not guaranteed physical Toeplitz permittivity profiles. An additional 18 target-order checks used physical asymmetric bar stacks and unequal exterior indices.

Observed maximum absolute differences:

- Target power: 2.3315e-15.
- Directional permittivity derivative: 5.5511e-16.
- Lossless power balance: 1.1102e-15.
- Central finite-difference derivative, across steps 1e-4, 1e-5 and 1e-6: 8.8974e-10.

Source snapshot, runnable independent check and full per-case records are retained beside this review. The positive result is specific to that hash and these cases; it is not whole-method certification or a continuum error bound.

## Material limitations and missing evidence

1. **Thick propagating layers remain bounded by subdivision cost.** Keeping each local positive modal phase below 0.8 pi is mathematically sensible for a DtN representation, but a thick homogeneous layer can hit the 256-piece limit. For example, a 100,000 nm silicon layer at 900 nm, present in the examined external meent reference, requires about 1,011 pieces at refractive index 3.63875. This implementation refuses it. The planned scattering successor is an appropriate way to expand this envelope. Do not claim unrestricted thick-layer support from the large-evanescent-exponent test: large evanescence and large propagating optical thickness are different cases.
2. **The supplied zero-mode regression is weak for the distinguishing gradient claim.** At normal incidence in a uniform layer, the zero +/-1 modes in that test are not excited, and its perturbation is diagonal. The independent mixed-mode tests exercise the off-diagonal divided differences and degenerate subspaces directly. Preserve similarly meaningful coverage in the successor.
3. **Hermitian positive matrix scope is broader than physical local permittivity.** A physical Fourier convolution is Toeplitz with the correct conjugate coefficient symmetry. The general API accepts arbitrary positive Hermitian matrices. That is a valid numerical operator interface, but arbitrary accepted matrices should not be described as independently validated physical materials. A physical-profile constructor and documented operator contract would clarify this.
4. **No performance, external-solver or inverse-design acceptance is established by this review.** Root is handling those independently. Full amplitude/field convention checks, actual reference geometries, exact output/gradient work, memory and numerical-thread records remain necessary. Successor source changes invalidate the current hash-specific numerical acceptance until checked.

## Math inspection

The local DtN maps and outward derivative signs are consistent with the exp(-i omega t) scalar Helmholtz convention. The boundary forcing gives unit incoming amplitude. The banded assembly stores Hermitian interior couplings in both neighboring blocks and uses conjugate-transpose LAPACK solves for the adjoint. The returned matrix VJP convention `dF = 2 Re <G, d epsilon>` is consistent with the objective derivative and real Hermitian perturbations. The small-argument polynomial derivatives and hyperbolic formulas have the expected limits. No eigenvector-gap regularization is introduced; repeated eigenvalues use a matrix-function derivative. Close but distinct eigenvalues use a midpoint derivative approximation, so “analytic adjoint evaluated in finite precision” is more precise than a mathematically exact floating-point derivative claim.
