"""Bounded loopback HTTP check of an already-built demo and scenario API."""

import json
import threading
import urllib.error
import urllib.request
from urllib.parse import urljoin

from serve import DemoServer, Handler


def main():
    server = DemoServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def fetch(path, payload=None, origin=None):
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        if origin:
            headers["Origin"] = origin
        request = urllib.request.Request(
            urljoin(base, path),
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise AssertionError(f"Unexpected HTTP status: {response.status}")
            return response.read()

    try:
        if b"<canvas" not in fetch("/"):
            raise AssertionError("Generated viewer has no canvas")
        data = json.loads(fetch("/data.json"))
        if not data["frames"] or not fetch("/" + data["frames"][0]["scene"]).startswith(
            b"\x89PNG"
        ):
            raise AssertionError("Missing camera playback")
        for payload, origin, expected in [
            ({"noise": -1}, None, 400),
            ({}, "https://example.invalid", 403),
        ]:
            try:
                fetch("/api/run", payload, origin)
            except urllib.error.HTTPError as error:
                if error.code != expected:
                    raise
            else:
                raise AssertionError("Invalid scenario request was accepted")
        result = json.loads(fetch("/api/run", {"noise": 2, "blackout_duration": 3}))
        if not result["url"].startswith("/runs/") or not result["url"].endswith(
            "/index.html"
        ):
            raise AssertionError("Unexpected scenario playback URL")
        if b"<canvas" not in fetch(result["url"]):
            raise AssertionError("Scenario viewer is missing")
        scenario = json.loads(fetch(result["url"].replace("index.html", "data.json")))
        if (
            scenario["config"]["noise"] != 2
            or scenario["config"]["blackout_duration"] != 3
        ):
            raise AssertionError("Scenario controls did not reach the simulator")
        print(
            json.dumps(
                {
                    "http_smoke": "passed",
                    "scenario_frames": len(scenario["frames"]),
                    "valid_fraction": scenario["summary"]["valid_fraction"],
                }
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
