# Experimental native TE solver

**Validation candidate, 0.2.0rc3. Independent human validation remains open.** The original public candidate passed Linux/macOS CI; automated review then exposed a sharp-resonance gradient defect and two reproducibility gaps. [Corrections and their evidence](CORRECTIONS.md). This revision needs its own public exact-source checks.

The new C++17/LAPACK backend solves supported lossless, one-dimensionally periodic TE Maxwell problems and computes analytic material/geometry gradients. It does not replace the existing solver, trained model, browser studio or archived experiments.

The original kernel's frozen 48-case study on an Apple M4 measured a 3.79× paired wall-time speedup over meent JAX JIT, the fastest tested external comparator. That number belongs to the original source. The production kernel is unchanged from the corrected public commit `c259ff80eb4c41b88031d3b7ea6f220c518fe299`; the stronger independent fourth-order gradient check passes all seven local gates in fresh confirmation v4. Its measured JAX wall speedup is 2.78x [2.61, 2.97]. See [the v4 method and evidence](VALIDATION_V4.md); [results, raw records and caveats](REPORT.md) distinguish source versions. Shared-machine measurements are not portable speed guarantees.

## Build and use

Python 3.12+, NumPy 2.3.5, SciPy 1.18.1, an existing C++17 compiler and BLAS/LAPACK are required. macOS uses the system Accelerate framework. Linux links `-llapack -lblas`; the original candidate passed Linux CI, and this revision requires its own verification.

From a source checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python scripts/build_native.py
photonsmith-boundary simulate --wavelength 1.43 --output runs/native-simulation.json
photonsmith-boundary optimize --wavelength 1.73 --material-index 2.15 \
  --target-order -1 --iterations 20 --output runs/native-design.json
```

Build and result paths must be fresh. Existing paths are refused. Compilation is explicit; importing the package does not install tools, download files or compile code. Missing libraries produce an actionable error, not an approximate fallback.

The portable wheel intentionally **excludes** host-specific native libraries. A normal wheel installation supplies the Python API and legacy application. To use its optional kernel, unpack the matching source distribution and build explicitly into the installed package:

```sh
python path/to/unpacked-source/scripts/build_native.py \
  --output /absolute/path/to/venv/lib/python3.12/site-packages/photonsmith/_boundary_native.dylib
```

Use `.so` on Linux. Determine the actual package location in your environment. Building in a source checkout and then performing a normal non-editable wheel installation does not copy that excluded binary. A fresh outside-checkout wheel installation plus explicit build from the source distribution was tested on macOS.

## Library

```python
from photonsmith.native import objective_gradient
value, gradient = objective_gradient(
    [.24, .63, .51, .18, .53, .79], wavelength=1.73,
    target_order=-1, harmonics=5, material_index=2.15,
)
```

The generic interface accepts a `Layer(epsilon_convolution, thickness)` list through `photonsmith.native.solve`. Set `target_order` to obtain a matrix-permittivity vector-Jacobian product, meaning the derivative contracted against any supported material change. Its convention is `dF = 2 Re sum(conj(G) * d_epsilon)`. Each layer is differentiated independently. Matrix inputs use centered Fourier orders `-M..M`, and Python arrays are row-major. Physical profiles must supply a valid Hermitian Fourier convolution. Passing an arbitrary positive Hermitian matrix does not turn it into a physical local dielectric.

`photonsmith.boundary.Solution` is a readable NumPy/SciPy reference implementation of the same mathematical method. It is useful for inspection and checks, not the source of the native speed claim.

## Scope

Normal incidence, scalar TE polarization, real positive permittivity, planar layers, real positive exterior refractive indices. No absorption, arbitrary 2D periodic/vector polarization, fabrication tolerance, physical-device measurement or safety certification. Exterior diffraction cutoffs are refused. The matrix order, number of layers, band allocation and optical phase are bounded. Phase is limited to `1e8`; this is not an arbitrary-thickness guarantee. Tested reference workloads extend through M=31, while accepted dimensions alone are not evidence of equal validation throughout the input range.

[Mathematical method and prior art](METHOD.md) · [Full measured report](REPORT.md) · [Reproduction](REPRODUCE.md) · [Coding-agent review](review/REVIEW-native.md)
