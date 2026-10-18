"""Local-only demo server. Recomputes scenarios with the Rust executable; no shell commands."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
LOCK = threading.Lock()
LIMITS = {
    "noise": (0, 25),
    "occlusion": (0, 0.9),
    "false_detections": (0, 40),
    "calibration_bias": (-0.15, 0.15),
    "delay_ms": (0, 2000),
    "blackout_start": (0, 50),
    "blackout_duration": (0, 8),
    "glare": (0, 3),
}


def scenario_args(body: dict) -> list[str]:
    if not isinstance(body, dict) or set(body) - LIMITS.keys():
        raise ValueError("unknown scenario fields")
    result = []
    for key, value in body.items():
        lower, upper = LIMITS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key}: expected a number")
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key}: outside [{lower}, {upper}]")
        if key == "false_detections" and int(value) != value:
            raise ValueError("false_detections must be an integer")
        result.extend(
            [
                "--" + key.replace("_", "-"),
                str(int(value) if key == "false_detections" else value),
            ]
        )
    # Preserve the complete outage and two seconds of recovery even near the end of playback.
    frames = max(
        1000,
        math.ceil(
            (body.get("blackout_start", 8) + body.get("blackout_duration", 3) + 2) * 50
        ),
    )
    result.extend(["--frames", str(frames)])
    return result


def command(destination: Path, args: list[str]) -> list[str]:
    native = (
        ROOT
        / "target"
        / "release"
        / ("apsis-cli.exe" if os.name == "nt" else "apsis-cli")
    )
    if native.exists():
        return [str(native), "--out", str(destination), *args]
    if os.name == "nt":
        linux_root = "/mnt/" + ROOT.drive[0].lower() + ROOT.as_posix()[2:]
        relative = destination.relative_to(ROOT).as_posix()
        return [
            "wsl",
            "-d",
            "Ubuntu-24.04",
            "--",
            linux_root + "/target/release/apsis-cli",
            "--out",
            linux_root + "/" + relative,
            *args,
        ]
    raise FileNotFoundError("Build first: cargo build --release -p apsis-cli")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUTPUT), **kwargs)

    def allowed(self) -> bool:
        hosts = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (
            origin is None or origin in {"http://" + h for h in hosts}
        )

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def do_GET(self):
        if not self.allowed():
            self.send_error(403)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/run" or not self.allowed():
            self.send_error(403)
            return
        if self.headers.get_content_type() != "application/json":
            self.send_error(415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("invalid request size")
            args = scenario_args(json.loads(self.rfile.read(length)))
        except (ValueError, TypeError) as exc:
            self.send_error(400, str(exc))
            return
        if not LOCK.acquire(blocking=False):
            self.send_error(409, "A scenario is already running")
            return
        try:
            run_id = uuid.uuid4().hex
            destination = OUTPUT / "runs" / run_id
            completed = subprocess.run(
                command(destination, args),
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if completed.returncode:
                self.send_error(500, "Rust scenario failed; inspect the server console")
                print(completed.stderr, flush=True)
                return
            response = json.dumps({"url": f"/runs/{run_id}/index.html"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(exc, flush=True)
            self.send_error(503, "Rust executable unavailable or scenario timed out")
        finally:
            LOCK.release()


class DemoServer(ThreadingHTTPServer):
    allow_reuse_address = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18765)
    options = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    server = DemoServer(("127.0.0.1", options.port), Handler)
    print(f"Apsis demo: http://127.0.0.1:{options.port}", flush=True)
    server.serve_forever()
