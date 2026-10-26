"""Render scientific figures directly from the saved current experiment."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .surrogate import Surrogate
from .resources import source_root

ROOT = Path(__file__).resolve().parents[1]


def main():
    source_root()
    docs = ROOT / "docs"
    exp = ROOT / "experiments"
    sample = json.loads((exp / "sample.json").read_text())
    bench = json.loads((exp / "benchmarks.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#203844",
            "text.color": "#203844",
            "axes.edgecolor": "#a3b4b8",
            "figure.facecolor": "#fafbf7",
            "axes.facecolor": "#fafbf7",
            "savefig.facecolor": "#fafbf7",
        }
    )
    f = sample["field"]
    field = np.array(f["real"])
    eps = np.array(f["epsilon"])
    x = np.array(f["x"])
    z = np.array(f["z"])
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    two = np.tile(field, (1, 2))
    ee = np.tile(eps, (1, 2))
    xx = np.arange(two.shape[1]) * (x[1] - x[0])
    vmax = np.quantile(abs(two), 0.995)
    im = ax.imshow(
        two,
        extent=(0, 4.8, z[-1], z[0]),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    ax.contour(xx, z, ee, levels=[2.5], colors=["#203e36"], linewidths=0.7)
    ax.set(
        xlabel="x (µm), two periodic cells",
        ylabel="z (µm)",
        title=f"PhotonSmith · 1550 nm directional grating · {100 * sample['validation']['primary_efficiency']:.2f}% into +1",
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Re(E⊥), unit incident amplitude")
    fig.savefig(docs / "field.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    s = sample["spectrum"]
    axs[0].plot(
        s["wavelength"],
        np.array(s["efficiency"]) * 100,
        color="#277b67",
        label="Target +1",
    )
    axs[0].plot(
        s["wavelength"],
        np.array(s["transmission"]) * 100,
        color="#869a9c",
        ls="--",
        label="Total transmitted",
    )
    axs[0].set(
        xlabel="Wavelength (µm)",
        ylabel="Incident power (%)",
        title="Spectral response",
        ylim=(0, 100),
    )
    axs[0].legend(fontsize=8)
    v = sample["validation"]
    sp = v["spectral_convergence"]
    gr = v["grid_convergence"]
    axs[1].plot(
        [r["harmonics"] for r in sp],
        100 * np.array([r["efficiency"] for r in sp]),
        "o-",
        color="#277b67",
    )
    axs[1].axhline(
        100 * v["independent_efficiency"],
        color="#bd863f",
        ls="--",
        label="Independent FDFD128",
    )
    axs[1].set(
        xlabel="Retained Fourier half-orders M",
        ylabel="Target efficiency (%)",
        title="Numerical convergence",
    )
    axs[1].legend(fontsize=8)
    methods = ["random", "adjoint", "surrogate"]
    labels = ["Random", "Adjoint", "ML + adjoint"]
    colors = ["#aebcb6", "#40756a", "#bd863f"]
    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = [100 * r["efficiency"] for r in bench["runs"] if r["method"] == method]
        axs[2].bar(i, np.mean(vals), color=color, width=0.55)
        axs[2].scatter(
            i + np.linspace(-0.13, 0.13, len(vals)),
            vals,
            color="#183b40",
            s=12,
            zorder=3,
        )
    axs[2].set(
        xticks=range(3),
        xticklabels=labels,
        ylabel="M13 target efficiency (%)",
        title="Six paired objectives / starts",
        ylim=(0, 100),
    )
    for ax in axs:
        ax.grid(axis="y", alpha=0.15)
    fig.savefig(docs / "evidence.png", dpi=170)
    plt.close(fig)

    a = np.load(exp / "dataset.npz", allow_pickle=False)
    ids = a["test"]
    p = np.repeat(a["params"][ids], 4, axis=0)
    wave = a["wavelengths"][ids].ravel()
    model = Surrogate(exp / "surrogate.npz")
    pred = model.predict(p, wave, fields=True)
    truth = a["powers"][ids].reshape(-1, 6)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    axs[0].scatter(
        truth[:, :3].ravel(),
        pred["powers"][:, :3].ravel(),
        s=3,
        alpha=0.25,
        color="#277b67",
    )
    axs[0].plot([0, 1], [0, 1], ls="--", color="#bd863f")
    axs[0].set(
        xlabel="Maxwell transmission per order",
        ylabel="Neural prediction",
        title="Held-out geometries",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    truefield = a["fields"][ids].reshape(-1, 24, 24)
    error = np.linalg.norm(
        (pred["canonical_field"] - truefield).reshape(len(p), -1), axis=1
    ) / np.linalg.norm(truefield.reshape(len(p), -1), axis=1)
    axs[1].hist(error, bins=25, color="#40756a")
    axs[1].set(
        xlabel="Complex-field relative L2 error",
        ylabel="Held-out wavelength cases",
        title="Canonical fixed-grid field prediction",
    )
    fig.savefig(docs / "surrogate.png", dpi=170)
    plt.close(fig)
    print("Rendered field.png, evidence.png, surrogate.png from current saved artifacts.")


if __name__ == "__main__":
    main()
