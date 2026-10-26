"""Compact finite-dimensional neural surrogates, with NumPy-only inference.

This is deliberately not advertised as a mesh-independent neural operator.
Response uses six-way softmax for nonnegative unit-sum outgoing powers.
Fields use a training-only PCA basis on a fixed 24x24 complex-field grid.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def features(params, wavelengths):
    p = np.atleast_2d(np.asarray(params, float))
    if p.ndim != 2 or p.shape[1] != 6 or not np.all(np.isfinite(p)):
        raise ValueError("Surrogate expects six finite geometry parameters per row.")
    if (
        np.any(p[:, :3] < 0.15 - 1e-10)
        or np.any(p[:, :3] > 0.85 + 1e-10)
        or np.any(p[:, 3:] < 0)
        or np.any(p[:, 3:] > 1)
    ):
        raise ValueError("Surrogate geometry is outside the trained width/shift domain.")
    w = np.broadcast_to(np.asarray(wavelengths, float), (len(p),))
    if not np.all(np.isfinite(w)) or np.any(w < 1.3) or np.any(w > 1.7):
        raise ValueError("Surrogate wavelength must lie in the trained 1.3–1.7 um interval.")
    # Scattering powers are invariant to common lateral translation.
    relative = p[:, 4:6] - p[:, 3:4]
    return np.column_stack(
        [
            (p[:, :3] - 0.5) / 0.35,
            np.sin(2 * np.pi * relative),
            np.cos(2 * np.pi * relative),
            (w - 1.5) / 0.2,
        ]
    )


def _mlp(x, bundle, prefix):
    for i in range(4):
        x = x @ bundle[f"{prefix}_w{i}"].T + bundle[f"{prefix}_b{i}"]
        if i < 3:
            x = np.tanh(x)
    return x


class Surrogate:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as a:
            self.weights = {k: a[k] for k in a.files if k != "metadata"}
            self.metadata = json.loads(str(a["metadata"]))

    def predict(self, params, wavelength=1.55, fields=False):
        x = features(params, wavelength)
        logits = _mlp(x, self.weights, "response")
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        powers = exp / exp.sum(axis=1, keepdims=True)
        result = {"powers": powers}
        if fields:
            coeff = _mlp(x, self.weights, "field") * self.weights["pca_scale"]
            raw = coeff @ self.weights["pca_basis"] + self.weights["field_mean"]
            n = raw.shape[1] // 2
            result["canonical_field"] = (raw[:, :n] + 1j * raw[:, n:]).reshape(-1, 24, 24)
        return result

    def objective_gradient(self, params, wavelength, order):
        # Exact backpropagation through exported tanh network and softmax.
        if order not in (-1, 0, 1) or isinstance(order, (bool, np.bool_)):
            raise ValueError("Target diffraction order must be -1, 0, or +1.")
        order = int(order)
        if np.asarray(params).shape != (6,):
            raise ValueError("An objective gradient requires exactly one six-parameter geometry.")
        x = features(params, wavelength)
        activations = [x]
        for i in range(4):
            x = x @ self.weights[f"response_w{i}"].T + self.weights[f"response_b{i}"]
            if i < 3:
                x = np.tanh(x)
            activations.append(x)
        ex = np.exp(x - x.max(axis=1, keepdims=True))
        prob = ex / ex.sum(axis=1, keepdims=True)
        index = order + 1
        value = prob[0, index]
        g = -value * prob
        g[0, index] += value
        for i in range(3, -1, -1):
            g = g @ self.weights[f"response_w{i}"]
            if i > 0:
                g *= 1 - activations[i] ** 2
        fg = g[0]
        p = np.asarray(params)
        relative = p[4:6] - p[3]
        shift = (
            2
            * np.pi
            * (fg[3:5] * np.cos(2 * np.pi * relative) - fg[5:7] * np.sin(2 * np.pi * relative))
        )
        grad = np.r_[fg[:3] / 0.35, -shift.sum(), shift]
        return float(value), grad


def train(dataset, output, epochs=1400, seed=17):
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("Epochs must be a positive integer.")
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    a = np.load(dataset, allow_pickle=False)
    params = a["params"]
    waves = a["wavelengths"]
    n_wave = waves.shape[1]
    p = np.repeat(params, n_wave, axis=0)
    x = features(p, waves.ravel()).astype("float32")
    powers = a["powers"].reshape(-1, 6).astype("float32")
    complex_field = a["fields"].reshape(len(p), -1)
    raw = np.column_stack([complex_field.real, complex_field.imag]).astype("float32")
    ids = {
        k: (a[k][:, None] * n_wave + np.arange(n_wave)).ravel()
        for k in ["train", "validation", "test"]
    }
    # Exact x-reflection is a Maxwell symmetry at normal incidence. Augment
    # training rows only; mirror powers swap -1/+1 and fields map x -> -x.
    mirror_p = p.copy()
    mirror_p[:, 3:] = (-mirror_p[:, 3:]) % 1
    mirror_x = features(mirror_p, waves.ravel()).astype("float32")
    mirror_powers = powers[:, [2, 1, 0, 5, 4, 3]]
    mirrored_field = (
        a["fields"].reshape(-1, 24, 24)[:, :, (-np.arange(24)) % 24].reshape(len(p), -1)
    )
    mirror_raw = np.column_stack([mirrored_field.real, mirrored_field.imag]).astype("float32")
    raw_train = np.concatenate([raw[ids["train"]], mirror_raw[ids["train"]]])
    x_train = np.concatenate([x[ids["train"]], mirror_x[ids["train"]]])
    powers_train = np.concatenate([powers[ids["train"]], mirror_powers[ids["train"]]])
    start = time.perf_counter()
    mean = raw_train.mean(axis=0)
    # SVD fitted to training geometries only; test data never sets the basis.
    _, singular, basis = np.linalg.svd(raw_train - mean, full_matrices=False)
    rank = min(48, len(basis))
    basis = basis[:rank]
    coeff = (raw - mean) @ basis.T
    coeff_train = (raw_train - mean) @ basis.T
    scale = np.maximum(coeff_train.std(axis=0), 0.05)
    targets = coeff / scale
    tx = torch.from_numpy(x)
    ty = torch.from_numpy(powers)
    tf = torch.from_numpy(targets)
    val_ids = torch.from_numpy(ids["validation"])
    train_x = torch.from_numpy(x_train)
    train_y = torch.from_numpy(powers_train)
    train_f = torch.from_numpy(coeff_train / scale)

    def net(out):
        return nn.Sequential(
            nn.Linear(8, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, out),
        )

    response, field = net(6), net(rank)
    opt = torch.optim.AdamW(
        list(response.parameters()) + list(field.parameters()),
        lr=0.002,
        weight_decay=1e-5,
    )
    best_r = best_f = float("inf")
    state_r = state_f = None
    log = []
    # Fixed full batches make the seed and geometry split easy to reproduce.
    for epoch in range(epochs):
        response.train()
        field.train()
        opt.zero_grad()
        pred = torch.softmax(response(train_x), dim=1)
        lp = ((pred - train_y) ** 2).mean()
        lf = ((field(train_x) - train_f) ** 2).mean()
        loss = lp * 12 + lf * 0.12
        loss.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                vp = torch.softmax(response(tx[val_ids]), dim=1)
                val = float(((vp - ty[val_ids]) ** 2).mean())
                vf = float(((field(tx[val_ids]) - tf[val_ids]) ** 2).mean())
            if val < best_r:
                best_r = val
                state_r = (
                    {k: v.detach().clone() for k, v in response.state_dict().items()},
                    epoch,
                )
            if vf < best_f:
                best_f = vf
                state_f = (
                    {k: v.detach().clone() for k, v in field.state_dict().items()},
                    epoch,
                )
            log.append(
                dict(
                    epoch=epoch,
                    response_train_mse=float(lp.detach()),
                    response_val_mse=val,
                    field_val_mse=vf,
                )
            )
            if epoch % 200 == 0:
                print(f"Epoch {epoch}: validation power RMSE={val**0.5:.4f}", flush=True)
    response.load_state_dict(state_r[0])
    field.load_state_dict(state_f[0])
    bundle = {"field_mean": mean, "pca_basis": basis, "pca_scale": scale}
    for prefix, network in [("response", response), ("field", field)]:
        for i, layer in enumerate(network[::2]):
            bundle[f"{prefix}_w{i}"] = layer.weight.detach().numpy()
            bundle[f"{prefix}_b{i}"] = layer.bias.detach().numpy()
    metadata = dict(
        seed=seed,
        epochs=epochs,
        selected_response_epoch=state_r[1],
        selected_field_epoch=state_f[1],
        seconds=time.perf_counter() - start,
        architecture="8-96-96-96; tanh; softmax6 response and PCA48 complex-field heads",
        kind="finite-dimensional neural surrogate, not a neural operator",
        field_grid=[24, 24],
        canonical_shift0=0.0,
        dataset=str(Path(dataset).name),
        dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        augmentation="training-only exact mirror reflection; no additional solver calls",
        pca_training_variance_retained=float((singular[:rank] ** 2).sum() / (singular**2).sum()),
        domain=dict(
            wavelength=[1.3, 1.7],
            width=[0.15, 0.85],
            material_index=2.0,
            period=2.4,
            layer_height=0.3,
        ),
        training_geometries=len(a["train"]),
        validation_geometries=len(a["validation"]),
        test_geometries=len(a["test"]),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **bundle, metadata=json.dumps(metadata))
    model = Surrogate(output)
    pred = model.predict(p[ids["test"]], waves.ravel()[ids["test"]], fields=True)
    truth = complex_field[ids["test"]].reshape(-1, 24, 24)
    power_error = pred["powers"] - powers[ids["test"]]
    field_error = np.linalg.norm(
        (pred["canonical_field"] - truth).reshape(len(truth), -1), axis=1
    ) / np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
    reconstructed = coeff[ids["test"]] @ basis + mean
    pca_error = np.linalg.norm(reconstructed - raw[ids["test"]], axis=1) / np.linalg.norm(
        raw[ids["test"]], axis=1
    )
    metrics = dict(
        power_mae=float(abs(power_error).mean()),
        power_rmse=float(np.mean(power_error**2) ** 0.5),
        power_p95_absolute_error=float(np.quantile(abs(power_error), 0.95)),
        field_relative_l2_mean=float(field_error.mean()),
        field_relative_l2_median=float(np.median(field_error)),
        field_relative_l2_p95=float(np.quantile(field_error, 0.95)),
        pca_test_relative_l2_mean=float(pca_error.mean()),
        nonnegative=True,
        max_predicted_balance_error=float(abs(pred["powers"].sum(axis=1) - 1).max()),
        test_samples=len(truth),
    )
    report = {
        "metadata": metadata,
        "held_out_geometry_metrics": metrics,
        "learning_curve": log,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(metrics, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="experiments/dataset.npz")
    p.add_argument("--output", default="experiments/surrogate.npz")
    p.add_argument("--epochs", type=int, default=1400)
    a = p.parse_args()
    train(a.dataset, a.output, a.epochs)
