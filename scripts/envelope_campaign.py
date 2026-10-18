"""Independent seeded trials; frame-level scores are correlated and are not Bernoulli trials."""

import argparse
import json
import math
import random
import subprocess
from pathlib import Path

from scipy.stats import chi2

from serve import ROOT, command


def wilson(failures, trials):
    if not trials:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = failures / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return [max(0.0, center - radius), min(1.0, center + radius)]


def score(recording, limits):
    frames = recording["frames"]
    valid = [f for f in frames if f["valid"]]
    false = [
        f
        for f in valid
        if f["error"] is None
        or f["angle"] is None
        or f["error"] > limits["position_error_m"]
        or f["angle"] > limits["attitude_error_deg"]
    ]
    coverage = [
        abs(f["p"][i] - f["truth_p"][i]) <= 3 * f["sigma"][i]
        for f in valid
        for i in range(3)
    ]
    # Exactly one predeclared tick per independent trial avoids pretending 50 Hz
    # correlated filter outputs are independent chi-square samples.
    anchor = min(frames, key=lambda f: abs(f["t"] - 6.0))
    return {
        "valid_ticks": len(valid),
        "false_valid_ticks": len(false),
        "false_valid_trial": bool(false),
        "valid_fraction": len(valid) / len(frames),
        "position_3sigma_coverage": sum(coverage) / len(coverage) if coverage else None,
        "anchor_nees_12": anchor.get("nees_12"),
        "anchor_has_estimate": anchor["p"] is not None,
        "vision_deadline_misses": sum(
            f["vision_us"] > limits["vision_deadline_ms"] * 1000 for f in frames
        ),
        "filter_deadline_misses": sum(
            f["filter_us"] > limits["filter_deadline_ms"] * 1000 for f in frames
        ),
        "summary": recording["summary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--out", type=Path, default=ROOT / "output/envelope")
    options = parser.parse_args()
    if not 1 <= options.trials <= 1000:
        parser.error("trials must be 1..1000")
    out = options.out.resolve()
    if not out.is_relative_to(ROOT / "output"):
        parser.error("output must remain under output/")
    out.mkdir(parents=True, exist_ok=True)
    envelope = json.loads((ROOT / "scenarios/envelope.json").read_text())
    rng = random.Random(options.seed)
    rows = []
    keys = {
        "range_m": "range_m",
        "tumble_scale": "angular_speed_rad_s",
        "phase_rad": "initial_roll_phase_rad",
        "noise": "noise_adu",
        "occlusion": "occlusion_fraction",
        "calibration_bias": "focal_bias_fraction",
        "light_gain": "light_gain",
        "sun_angle_rad": "sun_angle_rad",
        "glare": "glare",
        "delay_ms": "delivery_delay_ms",
    }
    for trial in range(options.trials):
        config = {key: rng.uniform(*envelope[source]) for key, source in keys.items()}
        config["tumble_scale"] /= math.sqrt(0.035**2 + 0.05**2 + 0.14**2)
        config["false_detections"] = rng.randint(*envelope["false_features"])
        config["seed"] = rng.randrange(2**32)
        args = ["--no-images"]
        for key, value in config.items():
            args += ["--" + key.replace("_", "-"), str(value)]
        destination = out / f"trial-{trial:04}"
        completed = subprocess.run(
            command(destination, args),
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        result = score(
            json.loads((destination / "data.json").read_text()),
            envelope["screening_limits"],
        )
        result.update(trial=trial, config=config)
        rows.append(result)
        if trial % 8 == 0:
            print(f"completed {trial+1}/{options.trials}", flush=True)
    false = sum(r["false_valid_trial"] for r in rows)
    interval = wilson(false, len(rows))
    nees = [r["anchor_nees_12"] for r in rows if r["anchor_nees_12"] is not None]
    dof = 12 * len(nees)
    consistency = {
        "estimated_trials": len(nees),
        "total_trials": len(rows),
        "mean_nees_12": sum(nees) / len(nees) if nees else None,
        "expected_mean": 12,
        "independent_trial_95_bounds": (
            [
                float(chi2.ppf(0.025, dof) / len(nees)),
                float(chi2.ppf(0.975, dof) / len(nees)),
            ]
            if nees
            else None
        ),
        "caveat": "Missing estimates reported separately; conditioning on successful acquisition can bias consistency results.",
    }
    cover = [
        r["position_3sigma_coverage"]
        for r in rows
        if r["position_3sigma_coverage"] is not None
    ]
    bounds = consistency["independent_trial_95_bounds"]
    mean = consistency["mean_nees_12"]
    limits = envelope["screening_limits"]
    passes = {
        "false_valid_rate": interval[1]
        <= limits["false_valid_trial_wilson95_upper_max"],
        "availability": all(
            r["valid_fraction"] >= limits["valid_fraction_min"] for r in rows
        ),
        "uncertainty": bool(cover)
        and len(nees) == len(rows)
        and sum(cover) / len(cover) >= limits["position_marginal_3sigma_coverage_min"]
        and bounds[0] <= mean <= bounds[1],
        "host_deadlines": not any(
            r["vision_deadline_misses"] or r["filter_deadline_misses"] for r in rows
        ),
    }
    report = {
        "envelope": envelope,
        "seed": options.seed,
        "trials": len(rows),
        "false_valid_trials": false,
        "false_valid_trial_wilson95": interval,
        "consistency": consistency,
        "mean_valid_trial_3sigma_coverage": sum(cover) / len(cover) if cover else None,
        "screening_gates": passes,
        "envelope_validated": all(passes.values()),
        "runs": rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["runs", "envelope"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
