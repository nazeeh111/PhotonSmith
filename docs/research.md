# Research foundations and scope

PhotonSmith studies **normal-incidence diffraction steering by a 2D periodic dielectric grating**. A plane wave enters from negative z (the top of the studio plot); three patterned dielectric layers redirect transmitted power into a selected Fourier diffraction order. This is a constrained, six-parameter inverse-design problem, rather than a general photonic-device generator.

All geometry lengths and free-space wavelengths use **micrometres**. The electric field is perpendicular to the simulation plane. The application calls this scalar field **E⊥**; internal legacy identifiers such as `Ez` denote this out-of-plane scalar. For a plotting plane named `(x,z)`, its physical out-of-plane direction is `y`. The scalar equation is the exact reduction of Maxwell's equations for isotropic, nonmagnetic structures invariant along that out-of-plane axis. TE/TM names differ between communities; the field component and invariance define the model unambiguously.

## Why this problem

- Periodicity gives a compact unit cell while retaining genuine interference, reflection, multiple scattering and diffraction.
- The default period is 2.4 µm, with three 0.30 µm thick layers, and a lossless refractive index of 2.0 in air. This is a simplified silicon-nitride-like index near telecom wavelengths, **not measured dispersion for a specified deposition process**. For context, [Arbabi and Goddard, Optics Letters 38, 3878 (2013)](https://opg.optica.org/ol/abstract.cfm?URI=ol-38-19-3878) measured a refractive index of 1.977 ± 0.003 at 1550 nm for their stoichiometric LPCVD film. This supports the scale of the chosen index, not a claim that 2.0 is exact for every film or across the full wavelength interval.
- Layer widths and periodic shifts yield asymmetric 2D structures. Selecting order `+1` or `-1` is a directional response target that an unpatterned slab cannot produce.
- Exact outgoing Fourier conditions remove artificial absorbing-layer parameters from the main model. The separate Meep route uses absorbing boundaries and a time-domain discretization instead.

The present surrogate is a compact finite-dimensional neural response model. Training fixes the first layer shift to zero to remove common lateral translation. Predicted fields are explicitly returned as `canonical_field` with `canonical_shift0=0`; their test metric uses that same canonical geometry. The studio displays fields from the Maxwell solver at the actual edited geometry, not translated surrogate fields. It is **not a Fourier neural operator**, and no invariance to mesh resolution or arbitrary geometry is claimed. Its practical contribution is a measured surrogate-assisted search within this parameterized design family. Its advantage must be assessed after independent physical verification, and training/data cost must be reported separately from per-objective search time.

## Supplied foundations

| Foundation | What it establishes | PhotonSmith's use |
| --- | --- | --- |
| [Ceviche](https://github.com/fancompute/ceviche) | Differentiable FDFD/FDTD electromagnetic computation in NumPy/SciPy, including reverse-mode differentiation | Reference for differentiable Maxwell design. An optional periodic driven-field integration smoke test is provided. Main solver and adjoint are written independently. |
| [Meep](https://github.com/NanoComp/meep) | Open-source full-wave FDTD, periodic media, mode decomposition and adjoint workflows | Optional standalone time-domain validation adapter. It is not imported by the core application. The bundled evidence must distinguish an actually executed result from this available route. |
| [NeuralOperator](https://github.com/NeuralOperator/neuraloperator) | Implementations of Fourier and other neural operators | Architectural research foundation, not a runtime dependency. A compact response surrogate is proportionate to this data and compute budget. |
| [Kovachki et al., JMLR 24 (2023), article 89](https://jmlr.org/papers/v24/21-1524.html), [arXiv:2108.08481](https://arxiv.org/abs/2108.08481) | Neural operators approximate maps between function spaces; includes Fourier operator parameterizations and PDE examples | Motivates learned solution maps. The paper does not establish an accuracy or speed guarantee for this optical problem or this finite-dimensional model. |

The arXiv record is version 6, revised 2 May 2024. The journal publication is 2023. Repository documentation/license files were checked on 25 October 2026; branch URLs are mutable references, not claimed pinned dependencies.

## What numerical validation must establish

1. **Analytic limits:** an empty domain transmits unit power; a uniform slab agrees with its Fresnel/Fabry–Pérot response. This checks signs, normalization and boundary matching.
2. **Power conservation:** for lossless materials, the sum of propagating reflected and transmitted order powers is one, within numerical tolerance. Evanescent amplitudes carry no far-field normal power. Balance alone is insufficient: a consistently wrong solver can conserve power.
3. **Adjoint correctness:** compare the implemented gradient with central finite differences at an interior, asymmetric geometry over several perturbation sizes. A converged optimizer status is not a derivative check.
4. **Discretization convergence:** increase retained Fourier orders and independently refine the validation grid. Use the change in target efficiency, total power and sampled fields to choose acceptable settings. Do not call one coarse-to-fine comparison an error bound.
5. **Independent formulation:** compare full-wave methods with distinct discretizations, not two wrappers around the same linear system. Compare order-resolved power and complex fields at identical coordinates and phase reference. An optional unexecuted Meep script is not validation evidence.
6. **ML generalization:** split by geometry, holding all observations of one geometry together. Evaluate separate held-out wavelength/objective cases. Keep candidate optimization and final verification distinct, since optimization seeks surrogate errors.
7. **Fair search benchmark:** record seeds, objective budget, start geometry, stopping criteria, elapsed time and Maxwell evaluations. Include direct adjoint and random/uniform baselines. Surrogate training and data generation are upfront costs, not free speedup.

## Constraint and fabrication meaning

A width bound can enforce the **minimum lateral bar width and gap in each individual layer**. It does not establish the minimum feature size of the union of three offset layers. Partial overlaps can create small necks, corners or disconnected regions; the fixed 0.30 µm layer height is itself another feature scale. General fabrication clearance, connectivity, etch accessibility and process bias are not certified. Exports are idealized simulation geometries, not fabrication-ready layouts.

## Physical limitations

The study assumes an infinitely repeated unit cell, infinite extent out of plane, normal incidence, a single polarization, linear isotropic nonmagnetic media, and a real constant refractive index. There is no substrate, material absorption/dispersion, finite-aperture edge scattering, source bandwidth integration, fabrication roughness, thermal/nonlinear response or 3D leakage. In particular, high 2D diffraction efficiency does not establish the efficiency of a manufacturable finite 3D device.

## Optional external checks

`optional/ceviche_check.py` checks one periodic driven uniform-field problem against its discrete analytic eigenmode. It tests the installed Ceviche package/API and units; it does **not** validate the grating.

`optional/meep_validate.py` reads an exported JSON geometry and writes order-resolved FDTD powers and sampled complex fields after an empty-cell normalization run. It is a separately launched user control script. It requires a real `pymeep` installation; the similarly named PyPI `meep` package is not a substitute. No Meep installation was found in the checked executable/common conda locations during assembly. Unless a result explicitly records a successful run, this adapter is an unexecuted integration route. Check resolution, absorbing-layer thickness, air spacing and run duration independently before relying on its output.

Useful primary documentation: [Meep mode decomposition](https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/), [Meep Python user interface](https://meep.readthedocs.io/en/latest/Python_User_Interface/), [Ceviche FDFD source](https://github.com/fancompute/ceviche/blob/master/ceviche/fdfd.py).
