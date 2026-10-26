# Contributing

Useful contributions make an optical result more reliable, a comparison more reproducible, or the tool easier to use.

1. Install a source checkout with `python -m pip install -e '.[dev]'`. Add `train` only when retraining.
2. Keep changes focused. Describe the concrete problem and resulting behavior.
3. Run `python -m unittest discover -s tests -v` and `python scripts/check_artifacts.py`. For packaging changes, build and run `tests/package_smoke.py` against an installed wheel.
4. For numerical changes, include a minimal geometry, wavelength, physical convention, baseline and convergence evidence. Check the gradient if sensitivities change. Power conservation alone does not establish accuracy.
5. For ML changes, preserve geometry-level splits, train-only preprocessing, raw errors, physical acceptance and offline costs. Avoid selecting checkpoints from test results.
6. For UI changes, inspect real current-source results, stale/error states, keyboard use and a small viewport. Keep neural estimates separate from solved responses.

Please do not replace the retained study with a more flattering result without preserving the comparison, seeds and budgets. New materials, device classes and external solvers need new validation. Optional GPL tools must remain separately installed and clearly identified.

The [reproduction guide](docs/reproducibility.md) documents commands and artifact boundaries. Open an issue with a small reproducible configuration before undertaking a large change.
