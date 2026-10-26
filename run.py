"""Launch the local studio with bounded BLAS thread counts."""

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[key] = "1"
from photonsmith.server import main

if __name__ == "__main__":
    main()
