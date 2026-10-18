"""Collect host storage, constrained-stack smoke results, and static per-function stack displacements."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True)


def cfi_frames(assembly: str) -> list[dict]:
    """Only accept functions whose CFI remains based on the stack pointer."""
    rows = []
    symbol = None
    maximum = 8
    supported = True
    for line in assembly.splitlines():
        match = re.search(r"^([A-Za-z_][\w.$]*):$", line)
        if match and not match[1].startswith(".L"):
            symbol, maximum, supported = match[1], 8, True
        offset = re.search(r"\.cfi_def_cfa_offset\s+(\d+)", line)
        if offset:
            maximum = max(maximum, int(offset[1]))
        if (
            ".cfi_def_cfa_register" in line
            or ".cfi_escape" in line
            or ".cfi_adjust_cfa_offset" in line
        ):
            supported = False
        if ".cfi_endproc" in line and symbol:
            rows.append(
                {
                    "symbol": symbol,
                    "static_frame_bytes": maximum - 8 if supported else None,
                }
            )
            symbol = None
    return rows


def main() -> None:
    version = run("rustc", "-vV")
    host = next(
        line.split(": ", 1)[1]
        for line in version.splitlines()
        if line.startswith("host:")
    )
    memory = json.loads(
        run(
            "cargo",
            "run",
            "--release",
            "-p",
            "apsis-core",
            "--example",
            "memory_profile",
        )
    )
    report = {
        "compiler": version,
        "host": host,
        "platform": platform.platform(),
        "storage_and_stack_smoke": memory,
        "limitations": "Host-specific per-function prologue stack displacement, with red zone disabled. Excludes callees, interrupts, runtime, and whole-call-chain peak. Not an ARM or flight stack bound.",
        "crates": {},
    }
    if host != "x86_64-unknown-linux-gnu":
        raise SystemExit(
            "CFI parser currently supports x86_64-unknown-linux-gnu only; storage smoke test completed"
        )
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    for crate in ("apsis-math", "apsis-ekf", "apsis-vision", "apsis-core"):
        run(
            "cargo",
            "rustc",
            "--release",
            "-p",
            crate,
            "--lib",
            "--",
            "--emit=asm",
            "-C",
            "no-redzone=yes",
            "-C",
            "force-unwind-tables=yes",
        )
        candidates = list(
            (target / "release" / "deps").glob(crate.replace("-", "_") + "-*.s")
        )
        if not candidates:
            raise RuntimeError(f"no assembly for {crate}")
        assembly = max(candidates, key=lambda path: path.stat().st_mtime)
        rows = cfi_frames(assembly.read_text())
        measured = [row for row in rows if row["static_frame_bytes"] is not None]
        if not measured:
            raise RuntimeError(f"no supported stack frames for {crate}")
        report["crates"][crate] = {
            "maximum_single_frame_bytes": max(
                row["static_frame_bytes"] for row in measured
            ),
            "unmeasured_functions": len(rows) - len(measured),
            "functions": rows,
        }
    destination = ROOT / "output" / "embedded-evidence.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
