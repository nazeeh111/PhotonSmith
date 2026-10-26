# Maxwell model and verification

PhotonSmith v1 solves a deliberately bounded but genuinely two-dimensional electromagnetic problem: a normally incident plane wave meets a periodic stack of three patterned dielectric layers. The transverse coordinate is **x**, the propagation coordinate **z**, and the electric field is perpendicular to this plane. In the standard `exp(-iωt)` convention it solves

```
∂²E/∂x² + ∂²E/∂z² + (2π/λ)² εr(x,z) E = 0.
```

For isotropic, nonmagnetic material invariant in the out-of-plane direction, this scalar equation is an exact polarization reduction of Maxwell's equations. It includes all multiple reflections and propagating/evanescent diffraction orders retained by the numerical basis. It is not a ray tracer or an intensity-only surrogate.

The nominal material is lossless, nondispersive, silicon-nitride-like material with refractive index 2.0 in air. This index is a useful telecom idealization, not a characterized deposition process. No substrate, finite out-of-plane thickness, absorption, material dispersion, thermo-optic effects or metal losses are represented. The geometry should not be treated as a fabrication-ready stack.

## Binary geometry and feature rule

The period is 2.4 μm; each layer is 0.3 μm thick. Each layer contains one periodic rectangular bar. Six real parameters are three bar widths and three bar centre positions, expressed as fractions of the period. Widths lie in [0.15, 0.85], guaranteeing a bar and a transverse gap of at least 0.36 μm **within each layer**. Centres wrap periodically. Layer height is 0.3 μm. This is a parameter-family feature constraint, not a two-dimensional minimum-neck or overhang check across layer interfaces. Adjacent offset bars can create narrow overlaps, disconnected regions or unsupported material. The model assumes the prescribed ideal geometry exists; additional manufacturing rules remain future work.

## Primary solver: Fourier modal boundary states

`physics.py` expands E in transverse Fourier orders −M…M. The exact coefficients of a bar of width fraction w and centre c are

```
ε̂d = δd0 + (n²−1) w sinc(dw) exp(−i2πdc).
```

Inside each z-uniform layer, the coupled Fourier amplitudes obey `E'' + Q E=0`, where `Q = k0² ε_convolution − diag(kx²)`. The first-order state `[E,E']` propagates through one layer by a matrix exponential. A single global linear system connects all four layer interfaces, rather than multiplying transfer matrices across the whole stack. This reduces long-product conditioning problems. The interfaces enforce continuity of E and E'. Exact exterior radiation maps use `kz = sqrt(k0²−kx²)` with nonnegative imaginary part. The incident amplitude is one in order zero.

For propagating order m, the power efficiency is `Re(kzm)/k0 |tm|²` (and the analogous reflected amplitude). Evanescent orders carry zero far-field power. Sum T+R should equal one for this lossless model. Small linear residual and power balance test assembly and solution, but do not establish spatial convergence. High Fourier order can still be poorly conditioned; verify by order refinement and independent discretization. Exact grazing diffraction cutoffs are rejected.

### Direct discrete adjoint

For `A(p)u=b`, efficiency F and real parameters p, the implementation solves `Aᴴλ=c` and evaluates `dF/dp = −2 Re(λᴴ(dA/dp)u)`. Width and shift derivatives of the analytic binary Fourier coefficients are exact. SciPy's matrix-exponential Fréchet derivative supplies each layer's matrix derivative. This is an adjoint derivative of the numerical Maxwell solve, not a finite-difference optimizer and not an ML gradient. Its six components are tested against central finite differences solely as an independent derivative check.

## Independent formulations

`independent.solve_fdfd` discretizes **both x and z in real space** using periodic second differences in x and centered finite differences in z. It independently integrates the binary bar overlap over transverse grid cells. It does not call the primary solver, Fourier bar coefficients, matrix exponential or adjoint. A Fourier-diagonal outgoing boundary map eliminates the exterior of this **discrete** x stencil, using its discrete dispersion relation. Its sparse linear system is solved directly. The limit as both grids are refined approaches the continuum radiation problem.

This is an independent implementation and discretization, not an external certified commercial solver. It shares the physical model, material assumptions, normal incidence and geometry parameters with the primary code. Agreement cannot detect an incorrect shared physical assumption. Both grid dimensions must be refined. Flux at finite grid spacing uses the discrete outgoing wave impedance; report grid size with every result.

`independent.solve_fd` is an additional diagnostic: Fourier in x and finite differences in z, with exact Fourier radiation at the endpoints. It shares the analytic Fourier coefficients with the primary method, so it is less independent than real-space FDFD. Its second-order z refinement isolates errors in layer propagation and interface assembly.

## Reproducible numerical checks

Run `python -m unittest discover -s tests -p 'test_physics.py' -v` from the repository. Tests cover the analytic vacuum propagation phase, exact uniform-material Fourier coefficients, conservation on randomized binary geometries, mirror/translation symmetry, adjoint derivatives, second-order z refinement, real-space solver agreement, finite complex fields and rejected invalid configurations. Experiment artifacts additionally record convergence for the delivered optimized geometry; those results, rather than one fixed tolerance, determine design credibility.

For the initial geometry `[.45,.45,.45,.15,.4,.65]` at 1.55 μm, observed checks during development were: adjoint maximum absolute gradient difference about 1.2e-9; R+T within 1e-13 of one; spectral finite-difference T+1 at 10/20/40/80 steps per layer approximately .388319/.387316/.387066/.387003 versus Fourier-modal M5 .386982. Independent real-space grids `(nx,steps/layer)` 32/15, 64/30, 96/45 and 128/60 gave .374016, .382660, .383469 and .383641, compared with Fourier M9 .384566. These are initial-geometry diagnostics, not claims for every generated design. Final experiment JSON is the authority for shipped design results.

A separate derivative-driven feasibility run from this initial geometry found widths `[0.59185587, 0.20410396, 0.20332363]` and centres `[0.22934818, 0.48509091, 0.48872443]`. It directed approximately 85.84% of incident power to transmitted order +1. Fourier M13 gave 0.858394; real-space FDFD 128 × 181 device nodes gave 0.858296. On the same grid the complex field relative L2 difference was 0.00627. This run establishes feasibility; the shipped experiment's own geometry, hashes, convergence and timing remain the reproducible result to cite.

Training at a lower Fourier order has discretization error even before ML. A seed-914 smoke sample of 20 random geometries/wavelengths compared M5 and M13 across six open transmission/reflection channels: mean absolute power difference 0.00313, maximum 0.01602. A surrogate trained on M5 predicts that finite-discretization solver; converged verification is still required. Do not interpret label accuracy as a bound for all geometries or for a trained model.
