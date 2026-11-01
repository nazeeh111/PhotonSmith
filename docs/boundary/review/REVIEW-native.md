# Independent review of the optional C++ backend

No consequential buffer, indexing, ABI-use or numerical defect was found in the exercised paths. This is independent coding-agent review of exact local artifacts. It is not review, adoption or reproduction by an outside human engineer and does not satisfy that finish-line gate.

## Exact reviewed artifacts

The manifest `native-reviewed-files.json` pins source and binary bytes:

- `native/boundary.cpp`: `37d697240a5b3becba4e83622b7484079b4a998af13805b71fa3a43319491547`.
- `photonsmith/native.py`: `fc5090ee3ebe0300bc2d8886c9816d3b8c6af5704eb360aa96fe49ad5a8e1f97`.
- `scripts/build_native.py`: `0777a1a6015ad3b22e88725401ddda9dfd7aef5a0139e9de84cfb608b5d719f3`.
- Built `_boundary_native.dylib`: `98d1ae0c57cb4ddf9bb35fa8b3edc9e0c1c47dd0429b100803de09bd4d408090`.
- Python v3 reference: `8cb56b74d338e261b4880537be898bca6015fde0dc5a9e89e9bdbf3bfb114215`.

The build receipt's source and library hashes were checked against the actual current files and both matched. Source snapshots are preserved in this review folder. A fresh Python process loaded that library for the numerical checks.

## Numerical evidence

`check_numeric_native.py` compares the generic native `solve()` against an independently assembled transfer formulation and its `expm_frechet` directional derivative. It includes 15 mixed repeated/zero/near-zero modal cases and 18 target-order comparisons with physical asymmetric profiles and unequal exterior media.

Maximum absolute differences:

- Target power: 2.2204e-15.
- Directional derivative: 1.3878e-16.
- Power balance: 2.2204e-15.

`check_native_equivalence.py` separately compares native outputs, complex amplitudes, port states and full Hermitian matrix gradients against the reviewed Python v3 implementation at M=3, 15 and 31. Maximum power difference is 3.0906e-14, amplitude/state difference 7.5573e-14 and matrix-gradient difference 3.9765e-15. Physical uniform slabs through thickness 10^6 also agree; the largest absolute derivative difference is 2.91e-11 on the thickest case. Seven unsupported/input cases are explicitly refused: even order, loss, indefinite permittivity, nonfinite permittivity, excessive optical phase, target outside the basis and mismatched matrix shapes.

All per-case results, commands and source identities are retained in `numeric-result-native.json`, `native-equivalence-result.json` and their logs. These checks establish local numerical correspondence, not external performance, continuum accuracy or universal stability.

## Native memory and runtime checks

A separate standalone executable was compiled from the exact reviewed C++ snapshot with:

```
clang++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer native_sanitizer.cpp -framework Accelerate -o native_sanitizer
```

It ran 46 numerical cases with guarded real/complex output buffers, spanning 3, 7, 17, 33 and 65 Fourier modes, 1/3/5 layers, physical asymmetry, unequal exterior indices, evanescent target orders and exact zero internal modes. It also checks preserved input matrices, bounded short error messages, allocation-cap rejection and bar-adapter valid/nonfinite inputs. The final run exited successfully with no AddressSanitizer or UndefinedBehaviorSanitizer diagnostic and every canary intact. These instrument the project's C++ code; the prebuilt system Accelerate library itself is not sanitizer-instrumented.

The initial run with `ASAN_OPTIONS=detect_leaks=1` stopped before executing cases because leak detection is unsupported on this platform. Its failed log is preserved. The successful second run used `detect_leaks=0`; **no leak-detection pass is claimed**. Successful log: `native-sanitizer-result-v2.log`. Standalone executable SHA256: `ff61b10d5e881e736c83256e79bb00a11dad3ef2ec80cc56537ac7102c71b6dd`.

## Inspection conclusions and limits

Dimension/count/target checks occur before core matrix allocation, bounded integer products cannot overflow at admitted dimensions, and the band allocation check uses widened arithmetic. The `3n-1` band width covers every neighboring two-port coupling; the Fortran storage row index stays inside its `3*bandwidth+1` allocation. Python constructs contiguous complex128/float64 buffers; the C++ import/export loops correctly convert row-major external matrices to column-major BLAS/LAPACK matrices. C++ exceptions are contained by `noexcept` C entry points. The complex conjugate-transpose adjoint solve, modal derivatives and Hermitian VJP agree with the independent reference.

The raw exported C functions require valid, sufficiently sized caller buffers; they do not constitute a memory-safe interface for arbitrary foreign pointers. The reviewed Python wrapper provides those owned arrays. A future public C API should document that precondition and version the ABI before incompatible changes; the present loader does not dynamically validate a binary ABI version. This review does not establish a defect in the current matching source/library pair.

Only the current macOS/Accelerate build was executed. Linux LAPACK/BLAS linkage, other compilers, public installation, artifact packaging, concurrent builds and prospective ABI migrations remain untested. The explicit builder preserves an existing library and performs no automatic download or compile on import. Preserve the source/binary receipt together and use a fresh process when testing rebuilt native bytes.

The v3 physical restrictions remain: lossless scalar TE, normal incidence, bounded retained modes/layers/band allocation, finite optical phase <=10^8 and exterior cutoffs refused. General positive Hermitian stress operators are not all physical Toeplitz material convolutions. These numerical and sanitizer results do not establish the separate benchmark, independent-workload, inverse-design, publication or external-human-validation gates.
