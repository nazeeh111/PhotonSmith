# Dependency and license boundaries

The PhotonSmith-authored core, browser studio, neural surrogate and independently written numerical methods are distributed under the repository's MIT license. This file records the intended boundary, not legal advice. No third-party solver source or binaries are vendored.

| Component | Role | Upstream license | Distribution/use boundary |
| --- | --- | --- | --- |
| Python | Runtime | PSF license | User's installed interpreter; not redistributed here. |
| NumPy | Core arrays and training algebra | BSD-3-Clause | Installed separately. Retain its notices when redistributing an environment/binary. |
| SciPy | Core linear algebra, sparse solves and optimization | BSD-3-Clause | Installed separately. Binary wheels include additional bundled-library notices. |
| PyTorch | Optional training only | BSD-3-Clause | Installed separately; exported NumPy weights support inference without PyTorch. Its wheel includes additional dependency notices. |
| Matplotlib | Optional reproducible scientific figures | Matplotlib license (PSF-based) | Installed separately, not part of the browser studio. |
| pytest | Optional development test runner | MIT | The same core tests also run with standard-library unittest. |
| Ceviche | Optional API/units smoke test | MIT | Not imported by default and no source copied. A user installing it must retain upstream MIT notices if redistributing it. |
| Meep / pymeep | Optional external FDTD validator | GPL-2.0-or-later | Separate user-installed environment and standalone process. Not linked into the core and no GPL binary/source bundled. |
| NeuralOperator | Research foundation only | MIT | Not imported, copied, or required for the compact surrogate. |
| Browser UI | Core interface | PhotonSmith MIT | Uses authored HTML/CSS/JavaScript and system fonts; no CDN, tracking, billed API, or font-service dependency. |

Primary license sources, inspected 25 October 2026:

- [Ceviche MIT license](https://github.com/fancompute/ceviche/blob/master/LICENSE), copyright 2019–2023 Ceviche Package Contributors.
- [NeuralOperator MIT license](https://github.com/NeuralOperator/neuraloperator/blob/main/LICENSE), copyright 2023 NeuralOperator developers.
- [Meep LICENSE (GPL version 2 text)](https://github.com/NanoComp/meep/blob/master/LICENSE) and [official license/copyright page](https://meep.readthedocs.io/en/latest/License_and_Copyright/). The copyright page specifies version 2 **or later** and clarifies the status of independently written user Python/Scheme control files. C++ programs linked with Meep libraries have a different distribution boundary.
- [NumPy license](https://numpy.org/doc/stable/license.html), [SciPy license](https://scipy.org/legal.html), [Python license](https://docs.python.org/3/license.html).

The optional Meep adapter is written as an independent user control script based on documented public APIs. PhotonSmith exchanges plain JSON configurations/results with its process. Keeping tools separate avoids accidentally bundling a GPL-linked core, but process separation alone is not a universal exemption from license obligations. Anyone distributing Meep or a combined derivative package must review the actual combination and comply with the applicable license.

The numerical experiments and generated data/model artifacts are authored outputs included under the repository license. Cited scientific publications retain their own copyright; their text/figures are not reproduced here. Repository branch links are references, not a claim that their current revision was installed. Exact **executed** runtime versions belong in each generated experiment manifest; available optional integrations must not be listed as successfully executed dependencies.

The installed PyTorch 2.14.1 wheel declares the combined SPDX expression `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`, reflecting bundled dependencies as well as the BSD-licensed project. This inventory does not collapse those binary-wheel obligations into the core project's license. The wheel and environment are excluded from the portable delivery.
