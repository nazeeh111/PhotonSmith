# Independent time-domain electromagnetic comparison

This study uses genuine Meep 1.35.0, an established external finite-difference time-domain solver, on the actual final geometry of the existing 1.73 um, index 2.15, target-order -1 inverse-design case. It does not reuse the helper's older accepted 1550 nm result. Meep evolves a spatial/time discretization; PhotonSmith solves a truncated Fourier Maxwell system. Agreement between them checks a materially different numerical formulation.

All inputs, code/runtime identities and tolerances were frozen before the first Meep simulation. Each stage has its own matched empty reference. Reflection uses complex-field subtraction. Fourier projection uses the actual returned coordinates and quadrature weights. Amplitudes are deembedded to common physical planes, without fitting a common phase, clipping channels or rescaling power to force conservation.

## Results and frozen limits

All nine stages completed within the declared 3600-second budget: exact vacuum, uniform slab, patterned device at 40/80/160 pixels per um, doubled post-source time, thicker PML, wider air padding, and a .137-period translation. No failed stage, altered threshold or retried favorable configuration is omitted.

| Check | Observed | Frozen limit |
|---|---:|---:|
| Largest open-channel native/Meep power difference | **0.00010201** | **0.005** |
| Grid 80 to 160, largest channel change | 0.00028151 | 0.0025 |
| Post-source time 250 to 500 | 1.35e-13 | 0.002 |
| PML thickness 1 to 2 um | 1.13e-6 | 0.002 |
| Air padding 1.5 to 3 um | 3.98e-6 | 0.003 |
| Fine-grid lossless balance error | 3.24e-8 | 0.005 |
| Translation, largest power difference | 0.00026933 | 0.005 |
| Vacuum analytic power error | 9.85e-8 | 0.005 |
| Uniform-slab analytic power error | 0.00013102 | 0.005 |

Meep target transmission is **0.491005610540**, compared with **0.490953429499** from native M61. The maximum transmission phase discrepancy is 0.00031414 rad and maximum complex-amplitude discrepancy 0.00017416, below the frozen 0.10 rad and 0.05 limits. Spatial Poynting-flux integration, raw monitor reprojection and translation-phase controls also pass. The native reference retains M21/31/45/61 results; the largest open-channel change between M45 and M61 is 3.12e-6, below its declared 1e-4 check.

A separate coding agent independently reprojected raw NPZ arrays, reconstructed incident normalization and complex reflected/transmitted amplitudes, integrated flux, checked analytic controls and all sensitivity gates, verified journal commands/hashes and all 16,322 frozen runtime file identities. **All 535 checks passed**, with no missing stage. This is an independent coding-agent numerical audit, not external human review or adoption.

## Reproduction and limits

[Frozen plan, inputs, raw monitors, journal and result](../../benchmarks/boundary/meep-v4/) · [Independent audit](review/meep-v4/REVIEW.md) · [Portable helper instructions](../../optional/meep_verified/README.md)

The native kernel is byte-unchanged from public commit `c259ff80eb4c41b88031d3b7ea6f220c518fe299`. Its source hash, actual library receipt and all reference amplitudes are retained. The Meep helpers in `optional/meep_verified` are byte-identical to the executed scripts. Runtime packages are external; no Meep binary is bundled. The exact Mac conda lock and runtime identities are supplied as provenance, not as a portable promise for another operating system.

This validates one nondefault-material, asymmetric reverse-order computational design within a frozen 2D scalar-TE tolerance. It does not prove all geometries, a rigorous continuum error bound, absorption, arbitrary vector/2D-periodic polarization, hardware or fabrication performance. Grid, time, boundary and Fourier refinement remain finite numerical approximations. It supplies external-solver evidence; genuine outside engineer/researcher acceptance remains a separate requirement.
