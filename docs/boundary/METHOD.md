# Method, contribution and limits

The engineering contribution is an original compact CPU implementation and its evaluated operating envelope: stable modal scattering, an analytic matrix-function adjoint, a banded port solve with reused factorization, and explicit Python/CLI interfaces. No learned approximation participates in the exact-solver comparison.

This does **not** claim to invent RCWA, scattering matrices, adjoint optimization, matrix-function derivatives or Dirichlet-to-Neumann methods. Related approaches include [meent](https://github.com/kc-ml2/meent), [grcwa](https://github.com/weiliangjinca/grcwa), [TORCWA](https://github.com/kch3782/torcwa), [fmmax's discussion of eigenvalue degeneracies](https://github.com/facebookresearch/fmmax/blob/main/src/fmmax/utils.py), [2009 DtN grating methods](https://doi.org/10.1364/JOSAA.26.001980), and [2026 matrix-function differentiable RCWA](https://doi.org/10.2528/PIERC25091202). This implementation's useful differentiation must be judged by the code, measurements and external scrutiny, not a literature-priority claim.

For the `exp(-i omega t)` convention and TE polarization, transverse Fourier coefficients obey

```
E''(z) + Q E(z) = 0
Q = k0² epsilon_convolution - diag(kx²)
Q = V diag(lambda) Vᴴ
```

Losslessness makes Q Hermitian. For each modal eigenvalue, define `k=sqrt(lambda)` with nonnegative imaginary part and a real reference port admittance `eta=k0`. A uniform layer's scattering functions are

```
D = cos(kh) - i (eta²+lambda) sin(kh)/(2 eta k)
T = 1/D
R = -i (eta²-lambda) sin(kh)/(2 eta k D)
```

The `sin(kh)/k` limit at zero is h. Polynomial limits handle small arguments. For negative eigenvalues, growing hyperbolic factors cancel analytically before evaluation; only decaying exponentials are computed. Thus high-order evanescent modes do not require exponentially growing transfer matrices. Thick propagating layers do not require internal subdivision to avoid Dirichlet poles. This removes a limitation of the first, retained development prototype.

Layer reflection/transmission matrices are `V diag(R/T) Vᴴ`. Incoming/outgoing amplitudes at neighboring reference ports give a block-banded global system. Exact exterior radiation conditions are matched at the ends. LAPACK band LU is reused for the conjugate-transpose adjoint solve.

The derivative of a matrix function in the Hermitian eigenbasis uses divided differences:

```
L_ij = (f(lambda_i)-f(lambda_j))/(lambda_i-lambda_j)
L_ii = f'(lambda_i)
```

At coincident eigenvalues the analytic derivative replaces a divergent eigenvector-gap formula. Close pairs use a midpoint derivative in floating point. This is a controlled numerical approximation near coincidence, not exact arithmetic or a certified error bound. The code differentiates the assembled Maxwell problem with respect to supported Hermitian permittivity matrices, then contracts analytic bar-width/shift derivatives for the six-parameter API.

The reference Python implementation and native kernel were checked against independent transfer/exponential Fréchet calculations, finite differences, reciprocity, conservation, analytic uniform slabs, and actual external solver packages. These checks do not prove a common physical model correct for a fabricated device, nor establish all possible inputs. Separate coding-agent review is explicitly not independent human validation.
