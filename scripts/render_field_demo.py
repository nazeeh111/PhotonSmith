"""Render a compact phase-animation GIF from the retained, verified physical solution."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont

ROOT = Path(__file__).resolve().parents[1]


def main():
    exported = json.loads(
        (ROOT / "experiments/designs/interactive-refined-1550nm.json").read_text()
    )
    result = exported["result"]
    f = result["field"]
    real, imag = np.tile(f["real"], (1, 2)), np.tile(f["imag"], (1, 2))
    eps = np.tile(f["epsilon"], (1, 2))
    nz, nx = real.shape
    norm = np.max(np.hypot(real, imag))
    font = findfont("DejaVu Sans")
    title = ImageFont.truetype(font, 25)
    small = ImageFont.truetype(font, 15)
    frames = []
    for phase in np.linspace(0, 2 * np.pi, 40, endpoint=False):
        value = (real * np.cos(phase) + imag * np.sin(phase)) / norm
        weight = np.abs(value) ** 0.68
        positive = np.array([247, 188, 96])
        negative = np.array([60, 158, 190])
        bg = np.array([11, 26, 36])
        colors = np.where(value[:, :, None] >= 0, positive, negative)
        pixels = np.uint8(bg + (colors - bg) * weight[:, :, None])
        frame = Image.new("RGB", (1040, 500), (16, 31, 42))
        image = Image.fromarray(pixels).resize((960, 360), Image.Resampling.BILINEAR)
        frame.paste(image, (40, 90))
        draw = ImageDraw.Draw(frame)
        draw.text((40, 17), "PhotonSmith  /  Maxwell field", font=title, fill=(227, 240, 231))
        eta = 100 * result["validation"]["primary_efficiency"]
        draw.text(
            (40, 54),
            f"1550 nm · transmitted order +1 · {eta:.2f}% verified efficiency",
            font=small,
            fill=(155, 191, 178),
        )
        for z in range(nz):
            for x in range(nx):
                if eps[z, x] <= 1.1:
                    continue
                xx = 40 + x / nx * 960
                zz = 90 + z / nz * 360
                if z == 0 or eps[z - 1, x] <= 1.1:
                    draw.line((xx, zz, xx + 960 / nx, zz), fill=(147, 194, 166), width=1)
                if z == nz - 1 or eps[z + 1, x] <= 1.1:
                    draw.line(
                        (xx, zz + 360 / nz, xx + 960 / nx, zz + 360 / nz),
                        fill=(147, 194, 166),
                        width=1,
                    )
                if x == 0 or eps[z, x - 1] <= 1.1:
                    draw.line((xx, zz, xx, zz + 360 / nz), fill=(147, 194, 166), width=1)
                if x == nx - 1 or eps[z, x + 1] <= 1.1:
                    draw.line(
                        (xx + 960 / nx, zz, xx + 960 / nx, zz + 360 / nz),
                        fill=(147, 194, 166),
                        width=1,
                    )
        draw.text(
            (40, 467),
            "Re(E⊥ exp(−iωt)) · two periodic cells · equal x/z scale · field M9, efficiency M13/FDFD",
            font=small,
            fill=(155, 177, 174),
        )
        frames.append(frame.quantize(colors=128))
    target = ROOT / "docs/field.gif"
    frames[0].save(
        target, save_all=True, append_images=frames[1:], duration=85, loop=0, optimize=True
    )
    print(f"Rendered {target.name}: {len(frames)} phases from retained complex Maxwell field.")


if __name__ == "__main__":
    main()
