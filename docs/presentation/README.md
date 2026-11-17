# Field and spectrum provenance

`field-and-spectrum.png` and the editable SVG render the exact arrays in `../../experiments/designs/interactive-refined-1550nm.json` at source commit `13912d9d640938acf8ce299a152298666ae61d44`.

The left panel uses all 97×96 values in `result.field.real`, with stored x/z coordinates in micrometres. It shows one periodic cell, with equal x/z scale and a symmetric zero-centered diverging color scale. No filtering, smoothing, interpolation, clipping or normalization is applied. A contour at relative permittivity 2.5 marks the binary air/dielectric interface (values 1 and 4). Color represents Re(Ey) relative to unit incident amplitude, not intensity or power.

The right panel uses all 21 stored `result.spectrum.wavelength` / `efficiency` pairs. Wavelength is converted from µm to nm by multiplying by 1000; efficiency is converted to incident-power percentage by multiplying by 100. Lines join sampled solves and do not add intermediate solutions. The percentage axis begins at zero. Raw accessible values remain in that JSON.

The source provenance calls this the final M9 simulation; search used M5. The target-order efficiency at M9 is 0.9358787717897128. M13 validation gives 0.9358595492317339; independent real-space validation gives 0.9358044989554115. These are distinct resolutions/formulations of the same 2D model, not measurements. The README labels them separately. Numerical agreement does not certify Fourier convergence, omitted physics or fabrication readiness.

The older `../field.png` is a different recorded 93.07% figure. It was deliberately not reused as the refined-design hero.

## Regenerate the presentation

With existing NumPy and Matplotlib, run `python docs/presentation/render.py` (desktop) and `python docs/presentation/render.py --mobile` (stacked/narrow) from the repository root. It reads the stored source JSON above and writes the PNG, editable SVG and `manifest.json` beside this note. It performs no new solve, fit or experiment. Matplotlib 3.11.2 and NumPy 2.5.4 were used for these outputs. The manifest records exact inputs, transformations and output hashes.
