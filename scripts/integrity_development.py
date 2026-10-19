"""Score conservative correspondence screening on synthetic DEVELOPMENT data only."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from correspondence_integrity import POLICY
from crop_perception import Perception, context
from fetch_benchmark import ROOT
from image_perception import prepared, score_pose
from learned_perception import split_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    options = parser.parse_args()
    if (
        not options.model_dir.resolve().is_relative_to(ROOT / "output")
        or not 1 <= options.threads <= 8
    ):
        parser.error("Model must be under output/; threads 1..8")
    torch.set_num_threads(options.threads)
    metadata = json.loads((options.model_dir / "training.json").read_text())
    selection, speed, calibration, geometry = context()
    if metadata.get("training_selection"):
        path = (ROOT / metadata["training_selection"]).resolve()
        if (
            not path.is_relative_to(ROOT / "output")
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != metadata["training_selection_sha256"]
        ):
            raise ValueError("Development pool changed")
        selection = json.loads(path.read_text())
    _, samples = split_training(selection["groups"]["speedplus/synthetic/train"])
    perception = Perception(options.model_dir)
    rows = []
    for sample in samples:
        path = speed / "synthetic/images" / sample["filename"]
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            not in metadata["development_hashes"]
        ):
            raise ValueError("Image is not in frozen development split")
        image, camera = prepared(path, calibration)
        start = time.perf_counter_ns()
        prediction, diagnostics = perception.infer(image, camera, geometry)
        rows.append(
            {
                "image": sample["filename"],
                "estimated": prediction is not None,
                "diagnostics": diagnostics,
                "vision_ms": (time.perf_counter_ns() - start) / 1e6,
                **score_pose(prediction, sample),
                "baseline_score": score_pose(
                    diagnostics.get("baseline_pose", prediction), sample
                ),
            }
        )
    emitted = [r for r in rows if r["estimated"]]
    correct = sum(
        r["position_error_m"] <= 0.15 and r["attitude_error_deg"] <= 8 for r in emitted
    )
    baseline = [
        r["baseline_score"]
        for r in rows
        if r["baseline_score"]["position_error_m"] is not None
    ]
    report = {
        "scope": "Development diagnostic; same-image repeatability is not correctness or calibrated uncertainty",
        "policy": POLICY if perception.integrity != "off" else "off",
        "frames": len(rows),
        "raw_hypotheses": sum(
            r["diagnostics"].get("raw_hypothesis", r["estimated"]) for r in rows
        ),
        "estimated": len(emitted),
        "baseline_within_15cm_8deg": sum(
            r["position_error_m"] <= 0.15 and r["attitude_error_deg"] <= 8
            for r in baseline
        ),
        "baseline_position_worst_m": max(
            (r["position_error_m"] for r in baseline), default=None
        ),
        "newly_recovered": sum(
            r["estimated"]
            and not r["diagnostics"].get("raw_hypothesis", r["estimated"])
            for r in rows
        ),
        "within_15cm_8deg": correct,
        "incorrect_emitted": len(emitted) - correct,
        "correct_coverage": correct / len(rows),
        "position_worst_m": max((r["position_error_m"] for r in emitted), default=None),
        "attitude_worst_deg": max(
            (r["attitude_error_deg"] for r in emitted), default=None
        ),
        "reasons": dict(Counter(r["diagnostics"]["reason"] for r in rows)),
        "vision_p95_ms": float(np.percentile([r["vision_ms"] for r in rows], 95)),
        "gnc_output_enabled": False,
        "deployment_approved": False,
        "results": rows,
    }
    (options.model_dir / "development-integrity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
