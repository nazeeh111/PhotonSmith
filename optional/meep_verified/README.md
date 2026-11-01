# Optional independent Meep validation

These two helpers are byte-identical copies of the owned MIT-licensed scripts used for the recorded Meep study. Their hashes are frozen in the execution plan. They extend the older optional adapter with explicit model checks, matched incident normalization, complex reflection subtraction, actual DFT coordinates and quadrature weights, unfitted deembedding, and bounded control/convergence checks. The original `optional/meep_validate.py` is preserved as historical assembly evidence.

No Meep binary is bundled. Meep is external GPL software. This study uses genuine serial Meep 1.35.0 from conda-forge, Python 3.12, isolated imports and single requested numerical threads. See the official [installation documentation](https://meep.readthedocs.io/en/latest/Installation/). The recorded runtime identities and exact Mac recovery lock accompany the evidence; another host must freeze its own runtime identity. An import alone does not validate a simulation.

The supported physical model is three contiguous periodic rectangular dielectric layers in air, normal incidence and scalar TE polarization. It does not import arbitrary geometries, absorbing materials, substrates or fabricated 3D devices. Each stage uses its own matched empty reference. Output paths must be fresh.

With a suitable `MEEP_PY` executable, reproduce from the repository root using:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=runs/meep-reproduction-mpl \
"$MEEP_PY" -I -B optional/meep_verified/validate_grating.py \
  benchmarks/boundary/meep-v4/input.json \
  --native-reference benchmarks/boundary/meep-v4/native-reference.json \
  --thresholds benchmarks/boundary/meep-v4/thresholds.json \
  --resolutions 40 80 160 --timeout 3600 \
  --output-dir runs/meep-reproduction
```

The reported protocol overrides the helper's general default resolution sequence. The plan freezes its thresholds before execution. Failed or interrupted attempts remain available; `--resume` requires the same input, code, thresholds, runtime and known elapsed budget and reuses only hash-verified successful outputs. Never alter thresholds after seeing a failure and call that convergence. Historical journal commands retain original execution paths; use the portable command above for a fresh run.

The native reference retains four Fourier truncations and the original source/build identity. To generate new native amplitudes, use `photonsmith.native.solve` on the physical Fourier matrices and record its complex `t` at the device end, with incident amplitude one at physical z=0. This is the actual wrapper field, distinct from the power array `T`. The basis is exp(+i2*pi*m*x/P), time convention exp(-i*omega*t). Native/Meep phases are compared at common physical planes, without a fitted phase.

Read the final report and all nine stage gates before claiming agreement. The recorded nine-stage study passed; a partial stage set in another run remains incomplete. Numerical Meep agreement is not outside-human validation, adoption, fabrication certification or a universal continuum error bound.
