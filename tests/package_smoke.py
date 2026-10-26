"""Test an installed wheel from outside the checkout, including the real HTTP API."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.request import build_opener, ProxyHandler
from urllib.error import URLError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", default=sys.executable, help="Interpreter with the wheel installed"
    )
    args = parser.parse_args()
    interpreter = str(Path(args.python).absolute())
    smoke = Path(__file__).with_name("http_smoke.py").resolve()
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    with tempfile.TemporaryDirectory(prefix="photonsmith-package-") as temp:
        output = Path(temp) / "simulation.json"
        subprocess.run(
            [interpreter, "-I", "-m", "photonsmith", "simulate", "--output", str(output)],
            cwd=temp,
            env=env,
            check=True,
        )
        result = json.loads(output.read_text())
        assert result["efficiency"] > 0.75 and abs(result["balance"] - 1) < 1e-7
        with (Path(temp) / "server.log").open("w") as log:
            server = subprocess.Popen(
                [interpreter, "-I", "-m", "photonsmith", "studio", "--port", "0"],
                cwd=temp,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                opener = build_opener(ProxyHandler({}))
                deadline = time.monotonic() + 60
                while True:
                    if server.poll() is not None:
                        raise RuntimeError((Path(temp) / "server.log").read_text())
                    match = re.search(
                        r"PhotonSmith studio: http://127\.0\.0\.1:(\d+)",
                        (Path(temp) / "server.log").read_text(),
                    )
                    if match is None:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "Installed studio did not announce a free port within 60 seconds"
                            )
                        time.sleep(0.1)
                        continue
                    port = int(match.group(1))
                    try:
                        with opener.open(
                            f"http://127.0.0.1:{port}/api/bootstrap", timeout=1
                        ) as response:
                            assert json.load(response)["model"]["metadata"]["dataset_sha256"]
                        break
                    except (URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "Installed studio did not become ready within 60 seconds"
                            )
                        time.sleep(0.1)
                subprocess.run(
                    [
                        sys.executable,
                        str(smoke),
                        "--url",
                        f"http://127.0.0.1:{port}",
                        "--output",
                        str(Path(temp) / "http.json"),
                    ],
                    cwd=temp,
                    env=env,
                    check=True,
                )
            finally:
                server.terminate()
                server.wait(timeout=15)
    print(
        "Installed-wheel CLI, bundled assets and 11 real HTTP checks passed outside the checkout."
    )


if __name__ == "__main__":
    main()
