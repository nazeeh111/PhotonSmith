"""Local-only standard-library HTTP studio. No remote services or execution API."""

import argparse
import json
import logging
import mimetypes
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from .physics import DEFAULT_PARAMS, validate_params
from .surrogate import Surrogate
from .design import optimize, bounds_for
from .reporting import public_simulation, validate_design
from .resources import asset_root

ROOT = asset_root()
LOCK = threading.Lock()


def load_json(name, default=None):
    path = ROOT / "experiments" / name
    return json.loads(path.read_text()) if path.exists() else default


def parse_config(body):
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object.")
    params = validate_params(body.get("params", DEFAULT_PARAMS)).tolist()
    wave = float(body.get("wavelength", 1.55))
    index = float(body.get("material_index", 2.0))
    order = body.get("target_order", 1)
    feature = float(body.get("min_feature", 0.36))
    if not np.isfinite(wave) or not 1.3 <= wave <= 1.7:
        raise ValueError("Wavelength must be between 1.3 and 1.7 um.")
    if not np.isfinite(index) or not 1 <= index <= 2.5:
        raise ValueError("Refractive index must be between 1 and 2.5.")
    if order not in (-1, 0, 1) or isinstance(order, bool):
        raise ValueError("Target order must be -1, 0, or +1.")
    bounds = bounds_for(feature)
    if any(v < lo - 1e-10 or v > hi + 1e-10 for v, (lo, hi) in zip(params, bounds)):
        raise ValueError(
            "Geometry violates the selected lateral bar/gap width. Adjust widths or lower the constraint."
        )
    return dict(
        params=params,
        wavelength=wave,
        target_order=int(order),
        material_index=index,
        min_feature=feature,
    )


class Handler(BaseHTTPRequestHandler):
    model = None

    def send_json(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/bootstrap":
            model = load_json("surrogate.json")
            sample = load_json(
                "sample.json",
                dict(
                    name="Starting grating",
                    params=DEFAULT_PARAMS.tolist(),
                    wavelength=1.55,
                    target_order=1,
                ),
            )
            samples = [sample]
            for name, label in (
                ("designs/1410nm-order-1.json", "1410 nm · order −1"),
                ("designs/1630nm-order+1.json", "1630 nm · order +1"),
                ("designs/1550nm-refined-config.json", "1550 nm · refined +1"),
            ):
                design = load_json(name)
                if design:
                    design["name"] = label
                    samples.append(design)
            self.send_json(
                dict(
                    sample=sample,
                    samples=samples,
                    benchmarks=load_json("benchmarks.json", {}),
                    model=model,
                )
            )
            return
        target = (ROOT / "web" / self.path.split("?", 1)[0].lstrip("/")).resolve()
        if self.path.split("?", 1)[0] == "/":
            target = ROOT / "web" / "index.html"
        if not target.is_relative_to(ROOT / "web") or not target.is_file():
            self.send_json({"error": "Not found"}, 404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(target)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        # Protect loopback computation from unrelated browser origins.
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://{self.headers.get('Host')}",):
            self.send_json({"error": "Cross-origin requests are not accepted."}, 403)
            return
        if self.path not in ("/api/simulate", "/api/optimize", "/api/validate"):
            self.send_json({"error": "Not found"}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size <= 0 or size > 16384:
                raise ValueError("Expected a JSON configuration smaller than 16 KiB.")
            body = json.loads(self.rfile.read(size))
            cfg = parse_config(body)
            if not LOCK.acquire(blocking=False):
                self.send_json(
                    {"error": "Another calculation is running. Try again when it finishes."},
                    409,
                )
                return
            try:
                if self.path == "/api/validate":
                    c = {k: v for k, v in cfg.items() if k != "min_feature"}
                    result = validate_design(**c)
                elif self.path == "/api/simulate":
                    result = public_simulation(**cfg)
                    if self.model is not None and cfg["material_index"] == 2.0:
                        result["prediction"] = {
                            "efficiency": float(
                                self.model.predict(cfg["params"], cfg["wavelength"])["powers"][
                                    0, cfg["target_order"] + 1
                                ]
                            ),
                            "status": "surrogate estimate; forward value is shown as the result",
                        }
                else:
                    iterations = body.get("iterations", 20)
                    if (
                        not isinstance(iterations, int)
                        or isinstance(iterations, bool)
                        or not 1 <= iterations <= 60
                    ):
                        raise ValueError("Iterations must be an integer from 1 to 60.")
                    search = optimize(
                        **cfg,
                        method=body.get("method", "surrogate"),
                        iterations=iterations,
                        model=self.model,
                    )
                    cfg["params"] = search["params"]
                    result = public_simulation(**cfg)
                    result.update(
                        {
                            k: v
                            for k, v in search.items()
                            if k not in ("efficiency", "params", "wavelength")
                        }
                    )
                    c = {k: v for k, v in cfg.items() if k != "min_feature"}
                    result["validation"] = validate_design(**c)
                    if self.model is not None and cfg["material_index"] == 2.0:
                        result["prediction"] = {
                            "efficiency": float(
                                self.model.predict(cfg["params"], cfg["wavelength"])["powers"][
                                    0, cfg["target_order"] + 1
                                ]
                            ),
                            "status": "surrogate estimate at final geometry; Maxwell value is authoritative",
                        }
                    result["provenance"]["history_harmonics"] = search["harmonics"]
                    result["provenance"]["status"] = (
                        "Final M9 simulation; independent validation attached. Search used M5."
                    )
            finally:
                LOCK.release()
            self.send_json(result)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception:
            logging.exception("Calculation failed")
            self.send_json(
                {"error": "Calculation failed. See the local terminal for diagnostics."},
                500,
            )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765, help="Local port; 0 selects an available port")
    a = p.parse_args(argv)
    if not 0 <= a.port < 65536:
        p.error("port must be between 0 and 65535 (0 selects an available port)")
    model_path = ROOT / "experiments" / "surrogate.npz"
    if model_path.exists():
        Handler.model = Surrogate(model_path)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    except OSError as error:
        p.exit(
            1,
            f"Cannot start PhotonSmith on port {a.port}: {error}. Use --port 0 to select an available port.\n",
        )
    print(f"PhotonSmith studio: http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPhotonSmith stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
