"""Exercise a running local studio: python tests/http_smoke.py --output docs/http-qa.json."""

import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--output", type=Path, default=Path("docs/http-qa.json"))
    a = p.parse_args()
    if not a.url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("Smoke test is restricted to a loopback server.")
    opener = build_opener(ProxyHandler({}))
    rows = []

    def req(path, data=None, expected=200, headers=None):
        start = time.perf_counter()
        body = None if data is None else json.dumps(data).encode()
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        request = Request(a.url + path, data=body, headers=h)
        try:
            with opener.open(request, timeout=90) as r:
                status = r.status
                raw = r.read()
        except HTTPError as e:
            status = e.code
            raw = e.read()
        assert status == expected, (path, status, raw[:300])
        rows.append(
            dict(
                path=path,
                method="GET" if data is None else "POST",
                status=status,
                seconds=time.perf_counter() - start,
            )
        )
        return json.loads(raw) if raw.startswith(b"{") else raw

    req("/")
    boot = req("/api/bootstrap")
    cfg = {
        k: boot["sample"][k]
        for k in (
            "params",
            "wavelength",
            "target_order",
            "material_index",
            "min_feature",
        )
    }
    sim = req("/api/simulate", cfg)
    assert sim["efficiency"] > 0.75 and abs(sim["balance"] - 1) < 1e-7
    assert len(sim["field"]["real"]) == 97 and len(sim["spectrum"]["wavelength"]) == 21
    validation = req("/api/validate", cfg)
    assert validation["passed"]
    result = req("/api/optimize", {**cfg, "method": "surrogate", "iterations": 5})
    assert (
        result["history"]
        and result["validation"]
        and result["constraint_audit"]["lateral_constraint_pass"]
    )
    req("/api/simulate", {**cfg, "wavelength": float("nan")}, 400)
    req(
        "/api/optimize",
        {**cfg, "material_index": 1.8, "method": "surrogate", "iterations": 5},
        400,
    )
    req("/api/optimize", {**cfg, "iterations": 999}, 400)
    req("/api/simulate", cfg, 403, {"Origin": "https://example.invalid"})
    req("/../pyproject.toml", expected=404)
    req("/api/unknown", {}, 404)
    report = dict(
        passed=True,
        checks=len(rows),
        rows=rows,
        model_dataset_sha256=boot["model"]["metadata"]["dataset_sha256"],
        simulated_efficiency=sim["efficiency"],
        independent_validation=validation,
        note="Actual HTTP requests against the running loopback application; no browser rendering claim.",
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(f"{len(rows)} HTTP checks passed; saved {a.output}")


if __name__ == "__main__":
    main()
