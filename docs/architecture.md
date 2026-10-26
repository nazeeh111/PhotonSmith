# Architecture

PhotonSmith uses ordinary NumPy/SciPy modules and a buildless browser studio. There is no task framework, database, remote inference service or required GPU.

```mermaid
flowchart LR
  U[Target and geometry editor] --> API[Loopback Python API]
  API --> D[Constrained design search]
  N[Trained NumPy surrogate] --> D
  D --> A[Direct discrete Maxwell adjoint]
  D --> F[Fourier-modal Maxwell solver]
  F --> V[Independent real-space FDFD and refinement]
  V --> R[Field, spectrum, trace and evidence]
  R --> U
  R --> X[JSON and SVG exports]
  G[Seeded simulated geometries] --> S[Geometry-level split]
  S --> T[Training-only mirror augmentation and PCA]
  T --> N
  M[Optional standalone Meep process] -. JSON .-> R
```

```mermaid
sequenceDiagram
  participant S as Studio
  participant A as Local API
  participant N as Neural surrogate
  participant F as Primary Maxwell solver
  participant V as Independent FDFD
  S->>A: Target, material, geometry and lateral constraints
  A->>N: Screen candidates and refine four proposals
  A->>F: Check shortlist and perform constrained adjoint optimization
  A->>F: Higher-order field and spectrum
  A->>V: Independent grid and Fourier refinement checks
  A-->>S: Geometry, physical response, trace and validation
```

| Module | Responsibility |
| --- | --- |
| `physics.py` | Analytic binary Fourier coefficients, global boundary-state solve, physical fields and exact discrete adjoint |
| `independent.py` | Separately assembled x/z real-space sparse Maxwell operator; additional spectral-x/FD-z diagnostic |
| `data.py` | Seeded physically simulated dataset and retained geometry-level splits |
| `surrogate.py` | Training, canonical-field PCA, portable NumPy inference and analytic neural gradients |
| `design.py` | Shared lateral bounds; neural-assisted, adjoint and random searches; explicit constraint audit |
| `reporting.py` | Public simulation format and independent convergence acceptance |
| `experiment.py` | Reproducible benchmark, cost accounting, raw traces and selected samples |
| `server.py`, `web/` | Local API, controls, scientifically scaled views and exact exports |
| `__main__.py`, `cli.py` | Installed entry point and exported-configuration replay |
| `resources.py` | Working-directory-independent source/wheel assets |
| `figures.py` | Scientific figures generated from saved experiment data |

The wheel includes the same web, model and experiment assets under `photonsmith/assets/`; source checkouts read the normal top-level folders. Runtime operations treat those bundled assets as read-only and write requested CLI outputs to the explicit destination. Research regeneration commands require a source checkout.

The studio distinguishes a neural estimate, a primary physical solution and an independent numerical acceptance. Edited inputs invalidate previous results. A passed numerical check does not mean the target is globally optimal or fabrication-ready. Search history uses M=5; displayed final fields use M=9; acceptance refines through M=13 and independent spatial grids. That distinction remains in exported provenance.
