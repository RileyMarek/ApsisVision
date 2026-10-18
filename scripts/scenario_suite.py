"""Run seeded image-failure experiments and audit the recorded navigation contract.

Integrity checks are regression guards, not an accuracy or flight-safety certificate.
Severe scenarios deliberately have no success-rate requirement: failures stay visible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from serve import ROOT, command, scenario_args

CASES = ROOT / "scenarios" / "demo.json"


def percentile(values: list[float], fraction: float) -> float | None:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1] if ordered else None


def audit(recording: dict, contract: str) -> dict:
    """Inspect every tick, including failures and absent estimates; no truth feedback."""
    frames, summary, config = (
        recording["frames"],
        recording["summary"],
        recording["config"],
    )
    violations = Counter()
    state_counts, rejections = Counter(), Counter()
    errors, valid_errors, angles = [], [], []
    coverage, longest_lost, lost = 0, 0, 0
    previous = -1.0
    for f in frames:
        state_counts[f["state"]] += 1
        if f["reason"] != "none":
            rejections[f["reason"]] += 1
        if not math.isfinite(f["t"]) or f["t"] <= previous:
            violations["nonmonotonic_time"] += 1
        previous = f["t"]
        if f["valid"] and (
            f["state"] not in {"tracking", "coasting"}
            or f["age"] is None
            or not 0 <= f["age"] <= 0.5 + 1e-9
            or f["p"] is None
        ):
            violations["invalid_health_contract"] += 1
        if not f["available"] and f["used"]:
            # These experiments have fixed delivery delay, not a buffered replay.
            violations["update_during_blackout"] += 1
        if f["p"] is not None:
            coverage += 1
            if f["error"] is None or f["angle"] is None:
                violations["unscored_estimate"] += 1
            for key, size in (("p", 3), ("v", 3), ("w", 3), ("q", 4), ("sigma", 12)):
                values = f[key]
                if (
                    values is None
                    or len(values) != size
                    or not all(math.isfinite(v) for v in values)
                ):
                    violations["nonfinite_or_incomplete_state"] += 1
            if f["q"] and abs(sum(v * v for v in f["q"]) - 1) > 1e-6:
                violations["quaternion_norm"] += 1
            if f["sigma"] and any(v < 0 for v in f["sigma"]):
                violations["negative_sigma"] += 1
            if f["valid"] and f["sigma"] and any(v > 0.5 for v in f["sigma"][:3]):
                violations["unbounded_valid_sigma"] += 1
        for key, collection in (("error", errors), ("angle", angles)):
            value = f[key]
            if value is not None:
                if not math.isfinite(value) or value < 0:
                    violations["invalid_score"] += 1
                else:
                    collection.append(value)
        if f["valid"] and f["error"] is not None:
            valid_errors.append(f["error"])
        lost = lost + 1 if f["state"] == "lost" else 0
        longest_lost = max(longest_lost, lost)
    # Apply the same declared tracking guards to every case, but only baseline
    # tracking is a CI acceptance contract. Stress failures are characterization.
    limits = {
        "position_rms_m": 0.08,
        "worst_position_m": 0.15,
        "attitude_rms_deg": 8.0,
        "recovery_s": 1.0,
    }
    tracking_failures = [
        key
        for key, limit in limits.items()
        if summary.get(key) is None or not 0 <= summary[key] <= limit
    ]
    if summary["valid_fraction"] < 0.7:
        tracking_failures.append("valid_coverage")
    if contract == "tracking":
        violations.update("baseline_" + key for key in tracking_failures)
    elif contract == "reject_stale":
        if summary["accepted"] or any(f["valid"] or f["p"] for f in frames):
            violations["accepted_stale_stream"] += 1
        if not rejections["stale_measurement"]:
            violations["stale_stream_not_observed"] += 1
    elif contract != "integrity":
        raise ValueError(f"Unknown contract: {contract}")
    return {
        "contract": contract,
        "contract_passed": not violations,
        "tracking_guard_passed": not tracking_failures,
        "tracking_guard_failures": tracking_failures,
        "violations": dict(violations),
        "seed": config["seed"],
        "config": config,
        "frames": len(frames),
        "state_frames": dict(state_counts),
        "rejection_ticks": dict(rejections),
        "estimate_coverage": coverage / len(frames),
        "longest_lost_s": longest_lost * 0.02,
        "position_p95_m": percentile(errors, 0.95),
        "attitude_p95_deg": percentile(angles, 0.95),
        "valid_position_worst_m": max(valid_errors, default=None),
        **summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "suite")
    options = parser.parse_args()
    cases = json.loads(CASES.read_text())
    selected = options.cases or list(cases)
    if set(selected) - cases.keys() or any(
        s < 0 or s > 2**64 - 1 for s in options.seeds
    ):
        parser.error("Unknown case or seed outside u64 range")
    destination = options.out.resolve()
    if not destination.is_relative_to(ROOT / "output"):
        parser.error("--out must be inside this checkout's output directory")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in selected:
        case = cases[case_id]
        for seed in options.seeds:
            run_dir = destination / f"{case_id}-{seed}"
            # Explicit defaults prevent inheriting another run's configuration.
            args = scenario_args(case["conditions"]) + [
                "--seed",
                str(seed),
                "--no-images",
            ]
            try:
                subprocess.run(
                    command(run_dir, args),
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"{case_id}/{seed}: Rust failed: {exc.stderr}"
                ) from exc
            raw = (run_dir / "data.json").read_bytes()
            result = audit(json.loads(raw), case["contract"])
            result.update(
                case=case_id, recording_sha256=hashlib.sha256(raw).hexdigest()
            )
            rows.append(result)
            print(
                f"{case_id}/{seed}: contract={'PASS' if result['contract_passed'] else 'FAIL'} "
                f"tracking={'PASS' if result['tracking_guard_passed'] else 'MISS'} "
                f"valid={result['valid_fraction']:.1%} "
                f"worst_m={result['worst_position_m']} recovery_s={result['recovery_s']}",
                flush=True,
            )
    report = {
        "scope": "Synthetic 20 s trials. Integrity PASS does not imply accurate tracking.",
        "case_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "tracking_guards": {
            "position_rms_m_max": 0.08,
            "worst_position_m_max": 0.15,
            "attitude_rms_deg_max": 8,
            "recovery_s_max": 1,
            "valid_fraction_min": 0.7,
        },
        "runs": rows,
        "contract_passed": all(r["contract_passed"] for r in rows),
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "case",
        "seed",
        "contract",
        "contract_passed",
        "tracking_guard_passed",
        "estimate_coverage",
        "valid_fraction",
        "position_rms_m",
        "position_p95_m",
        "worst_position_m",
        "valid_position_worst_m",
        "attitude_rms_deg",
        "attitude_p95_deg",
        "worst_attitude_deg",
        "recovery_s",
        "longest_lost_s",
        "accepted",
        "camera_frames",
        "vision_p95_us",
        "filter_p95_us",
        "recording_sha256",
    ]
    with (destination / "report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0 if report["contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
