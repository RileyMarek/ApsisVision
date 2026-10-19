"""Measure desktop CPU inference and model storage; never extrapolate to flight hardware."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from crop_perception import Perception, context
from fetch_benchmark import ROOT
from image_perception import prepared
from learned_perception import split_training


def storage_bytes(models):
    """Logical checkpoint tensor bytes, including buffers and actual dtypes."""
    return {
        name: sum(t.numel() * t.element_size() for t in model.state_dict().values())
        for name, model in models.items()
    }


def peak_working_set():
    if os.name == "nt":

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t)
                for name in [
                    "peak_working_set",
                    "working_set",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                ]
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not query(
            kernel.GetCurrentProcess(), ctypes.byref(counters), ctypes.sizeof(counters)
        ):
            raise OSError(
                ctypes.get_last_error(), "Cannot query current-process memory"
            )
        return counters.peak_working_set
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "output/transfer-perception"
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--weight-budget-mib", type=float)
    options = parser.parse_args()
    if (
        not options.model_dir.resolve().is_relative_to(ROOT / "output")
        or not 5 <= options.samples <= 100
        or not 1 <= options.threads <= 8
        or not np.isfinite(options.deadline_ms)
        or options.deadline_ms <= 0
        or (
            options.weight_budget_mib is not None
            and (
                not np.isfinite(options.weight_budget_mib)
                or options.weight_budget_mib <= 0
            )
        )
    ):
        parser.error(
            "Models must be under output/; samples 5..100, threads 1..8, positive budgets"
        )
    torch.set_num_threads(options.threads)
    metadata = json.loads((options.model_dir / "training.json").read_text())
    selection, speed, camera, geometry = context()
    if metadata.get("training_selection"):
        path = (ROOT / metadata["training_selection"]).resolve()
        if (
            not path.is_relative_to(ROOT / "output")
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != metadata["training_selection_sha256"]
        ):
            raise ValueError("Development-pool provenance changed")
        selection = json.loads(path.read_text())
    _, rows = split_training(selection["groups"]["speedplus/synthetic/train"])
    reference = Perception(options.model_dir)
    sizes = storage_bytes({"box": reference.box, "localizer": reference.localizer})
    samples = []
    for i, row in enumerate(rows[: options.samples]):
        path = speed / "synthetic/images" / row["filename"]
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            not in metadata["development_hashes"]
        ):
            raise ValueError("Profile image is not in the frozen development set")
        image, k = prepared(path, camera)
        if i == 0:
            for _ in range(3):
                reference.infer(image, k, geometry)
        start = time.perf_counter_ns()
        pose, diagnostics = reference.infer(image, k, geometry)
        samples.append(
            {
                "vision_ms": (time.perf_counter_ns() - start) / 1e6,
                "hypothesis": pose is not None,
                "landmarks": diagnostics["landmarks"],
            }
        )
    values = [s["vision_ms"] for s in samples]
    report = {
        "scope": "Desktop CPU, batch one, FP32. Decoder/rectification/I/O and scheduling excluded. Peak process memory includes Python/PyTorch, not standalone activation memory or stack.",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "threads": options.threads,
        "samples": len(samples),
        "model_sha256": {
            name: hashlib.sha256(
                (options.model_dir / (name + ".pt")).read_bytes()
            ).hexdigest()
            for name in sizes
        },
        "tensor_storage_bytes": sizes,
        "total_tensor_storage_bytes": sum(sizes.values()),
        "checkpoint_file_bytes": {
            name: (options.model_dir / (name + ".pt")).stat().st_size for name in sizes
        },
        "peak_host_process_working_set_bytes": peak_working_set(),
        "latency_ms": {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": max(values),
        },
        "deadline_ms": options.deadline_ms,
        "host_deadline_met": max(values) <= options.deadline_ms,
        "candidate_weight_budget_mib": options.weight_budget_mib,
        "candidate_weight_budget_met": (
            sum(sizes.values()) <= options.weight_budget_mib * 1024**2
            if options.weight_budget_mib is not None
            else None
        ),
        "flight_hardware_measured": False,
        "activation_peak_measured": False,
        "stack_measured": False,
        "deployment_approved": False,
        "results": samples,
    }
    (options.model_dir / "host-profile.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
