"""Small supervised Tango heatmap experiment; no runtime truth or pose prior.

CPU-only reference, not an embedded-qualified or lighting-invariant model.
Train/dev use only the selected synthetic training pool. Audit images never select weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import cv2 as cv
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as functional

from fetch_benchmark import ROOT
from image_perception import prepared, score_pose, solve_correspondences

WIDTH, HEIGHT = 256, 192
GRID_W, GRID_H = 64, 48
SEED = 1729


def block(input_channels, output_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            input_channels,
            3,
            stride,
            1,
            groups=input_channels,
            bias=False,
        ),
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, output_channels, 1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
    )


class HeatmapNet(nn.Module):
    """Depthwise encoder/decoder with spatial skip connections; eleven semantic channels."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 5, 2, 2), nn.ReLU())
        self.s2 = nn.Sequential(block(16, 32, 2), block(32, 32))
        self.s3 = nn.Sequential(block(32, 64, 2), block(64, 64))
        self.s4 = nn.Sequential(block(64, 96, 2), block(96, 96))
        self.s5 = nn.Sequential(block(96, 128, 2), block(128, 128))
        self.u4, self.u3, self.u2 = block(224, 96), block(160, 64), block(96, 48)
        self.head = nn.Conv2d(48, 11, 1)

    def forward(self, image):
        x2 = self.s2(self.stem(image))
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        x5 = self.s5(x4)
        x = self.u4(
            torch.cat(
                (
                    functional.interpolate(
                        x5, size=x4.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    x4,
                ),
                1,
            )
        )
        x = self.u3(
            torch.cat(
                (
                    functional.interpolate(
                        x, size=x3.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    x3,
                ),
                1,
            )
        )
        x = self.u2(
            torch.cat(
                (
                    functional.interpolate(
                        x, size=x2.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    x2,
                ),
                1,
            )
        )
        return self.head(x)


def image_tensor(image):
    resized = cv.resize(image, (WIDTH, HEIGHT), interpolation=cv.INTER_AREA)
    return torch.from_numpy(resized.copy()).to(torch.float32)[None] / 255.0


def decode(logits, image_shape):
    """Local probability mass is an uncalibrated score, never a safety probability."""
    if tuple(logits.shape) != (11, GRID_H, GRID_W) or not np.isfinite(logits).all():
        raise ValueError("Expected finite eleven-channel 48x64 heatmaps")
    probabilities = np.exp(logits - logits.max(axis=(1, 2), keepdims=True))
    probabilities /= probabilities.sum(axis=(1, 2), keepdims=True)
    points, confidence = [], []
    for heatmap in probabilities:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y0, y1, x0, x1 = (
            max(0, y - 2),
            min(GRID_H, y + 3),
            max(0, x - 2),
            min(GRID_W, x + 3),
        )
        patch = heatmap[y0:y1, x0:x1]
        mass = patch.sum()
        yy, xx = np.mgrid[y0:y1, x0:x1]
        points.append(
            [
                ((xx * patch).sum() / mass + 0.5) * image_shape[1] / GRID_W - 0.5,
                ((yy * patch).sum() / mass + 0.5) * image_shape[0] / GRID_H - 0.5,
            ]
        )
        confidence.append(float(mass))
    return np.asarray(points), np.asarray(confidence)


def infer(image, camera_matrix, model, network):
    with torch.inference_mode():
        logits = network(image_tensor(image)[None])[0].numpy()
    points, confidence = decode(logits, image.shape)
    ids = np.flatnonzero(
        confidence >= 0.05
    )  # fixed before audit; not calibrated confidence
    prediction = solve_correspondences(model[ids], points[ids], camera_matrix)
    return prediction, len(ids)


def split_training(rows):
    ranked = sorted(
        rows,
        key=lambda r: hashlib.sha256((str(SEED) + r["filename"]).encode()).hexdigest(),
    )
    if len(ranked) < 20:
        raise ValueError("At least twenty training-pool images required")
    cut = int(len(ranked) * 0.8)
    return ranked[:cut], ranked[cut:]


def training_data(rows, root, calibration, model):
    images, coordinates, hashes = [], [], []
    for sample in rows:
        path = root / sample["filename"]
        image, k = prepared(path, calibration)
        q = sample["q_vbs2tango_true"]
        transformed = (
            Rotation.from_quat([*q[1:], q[0]]).apply(model) + sample["r_Vo2To_vbs_true"]
        )
        if np.any(transformed[:, 2] <= 0):
            raise ValueError("Training landmark behind the camera")
        pixels = transformed[:, :2] / transformed[:, 2, None] @ k[:2, :2].T + k[:2, 2]
        xy = (pixels + 0.5) * np.array(
            [GRID_W / image.shape[1], GRID_H / image.shape[0]]
        ) - 0.5
        images.append(image_tensor(image))
        coordinates.append(xy)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return (
        torch.stack(images),
        torch.tensor(np.asarray(coordinates), dtype=torch.float32),
        hashes,
    )


def heatmap_loss(logits, coordinates):
    y, x = torch.meshgrid(torch.arange(GRID_H), torch.arange(GRID_W), indexing="ij")
    squared = (x - coordinates[..., 0, None, None]) ** 2 + (
        y - coordinates[..., 1, None, None]
    ) ** 2
    targets = torch.exp(-squared / (2 * 1.5**2))
    targets /= targets.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    inside = (
        (coordinates[..., 0] >= 0)
        & (coordinates[..., 0] < GRID_W)
        & (coordinates[..., 1] >= 0)
        & (coordinates[..., 1] < GRID_H)
    )
    cross_entropy = -(
        targets.flatten(2) * functional.log_softmax(logits.flatten(2), dim=-1)
    ).sum(-1)
    if not inside.any():
        raise ValueError("Training batch contains no in-frame landmarks")
    return cross_entropy[inside].mean()


def train(options):
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(options.threads)
    selection = json.loads((ROOT / "output/datasets/selection.json").read_text())
    speed = ROOT / "output/datasets/speedplus/speedplusv2"
    calibration = json.loads((speed / "camera.json").read_text())
    geometry = loadmat(ROOT / "output/datasets/tangoPoints.mat")[
        "tango3Dpoints"
    ].T.astype(np.float64)
    training, development = split_training(
        selection["groups"]["speedplus/synthetic/train"]
    )
    images, labels, hashes = training_data(
        training, speed / "synthetic/images", calibration, geometry
    )
    dev_images, dev_labels, dev_hashes = training_data(
        development, speed / "synthetic/images", calibration, geometry
    )
    if set(hashes) & set(dev_hashes):
        raise ValueError("Train/development image overlap")
    network = HeatmapNet()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.0001)
    out = ROOT / "output/learned"
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    start = time.perf_counter()
    for epoch in range(options.epochs):
        network.train()
        total = 0.0
        for ids in torch.randperm(len(images)).split(16):
            image = images[ids].clone()
            # Photometric augmentation only: no truth-informed crop at inference.
            gain = 0.5 + torch.rand(len(ids), 1, 1, 1)
            gamma = 0.6 + torch.rand(len(ids), 1, 1, 1)
            image = (image.pow(gamma) * gain + torch.randn_like(image) * 0.015).clamp(
                0, 1
            )
            optimizer.zero_grad()
            loss = heatmap_loss(network(image), labels[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        network.eval()
        with torch.inference_mode():
            dev = sum(
                float(
                    heatmap_loss(
                        network(dev_images[i : i + 16]), dev_labels[i : i + 16]
                    )
                )
                * len(dev_images[i : i + 16])
                for i in range(0, len(dev_images), 16)
            ) / len(dev_images)
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": total / len(images),
                "development_loss": dev,
            }
        )
        if not np.isfinite(dev):
            raise ValueError("Nonfinite development loss")
        if dev < best:
            best = dev
            torch.save(network.state_dict(), out / "heatmap.pt")
        print(history[-1], flush=True)
    metadata = {
        "seed": SEED,
        "epochs": options.epochs,
        "torch": torch.__version__,
        "threads": options.threads,
        "parameters": sum(p.numel() for p in network.parameters()),
        "training_hashes": hashes,
        "development_hashes": dev_hashes,
        "geometry_sha256": selection["geometry_sha256"],
        "input": [1, 1, HEIGHT, WIDTH],
        "heatmaps": [1, 11, GRID_H, GRID_W],
        "best_development_loss": best,
        "elapsed_s": time.perf_counter() - start,
        "history": history,
        "limits": "400 training / 100 development images; supervised geometric vertices, no visibility labels or calibrated confidence; no hardware evidence",
    }
    (out / "training.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def evaluate(options):
    torch.set_num_threads(options.threads)
    out = ROOT / "output/learned"
    metadata = json.loads((out / "training.json").read_text())
    selection = json.loads((ROOT / "output/datasets/selection.json").read_text())
    if selection["geometry_sha256"] != metadata["geometry_sha256"]:
        raise ValueError("Geometry provenance changed")
    used = set(metadata["training_hashes"] + metadata["development_hashes"])
    geometry = loadmat(ROOT / "output/datasets/tangoPoints.mat")[
        "tango3Dpoints"
    ].T.astype(np.float64)
    network = HeatmapNet()
    network.load_state_dict(
        torch.load(out / "heatmap.pt", map_location="cpu", weights_only=True)
    )
    network.eval()
    if options.runtime != "torch":
        import onnxruntime as ort

        runtime_options = ort.SessionOptions()
        runtime_options.intra_op_num_threads = options.threads
        filename = "heatmap-int8.onnx" if options.runtime == "int8" else "heatmap.onnx"
        session = ort.InferenceSession(
            str(out / filename), runtime_options, providers=["CPUExecutionProvider"]
        )

        def network(value):
            return torch.from_numpy(session.run(None, {"image": value.numpy()})[0])

    groups = {}
    for name, samples in selection["groups"].items():
        if name.endswith("/train"):
            continue
        dataset, domain, split = name.split("/")
        root = ROOT / "output/datasets" / dataset
        base = root / "speedplusv2" if dataset == "speedplus" else root
        camera = json.loads((base / "camera.json").read_text())
        folder = (
            base / domain / "images"
            if dataset == "speedplus"
            else base / domain / split / "images"
        )
        rows, packets = [], []
        for index, sample in enumerate(samples):
            path = folder / sample["filename"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in used:
                raise ValueError("Audit image overlaps training/development")
            image, k = prepared(path, camera)
            start = time.perf_counter_ns()
            prediction, matches = infer(image, k, geometry, network)
            elapsed = (time.perf_counter_ns() - start) / 1000
            # No evaluation annotation has entered inference or model selection.
            rows.append(
                {
                    "image": sample["filename"],
                    "sha256": digest,
                    "matches": matches,
                    "prediction": prediction,
                    "vision_us": elapsed,
                    **score_pose(prediction, sample),
                }
            )
            t = index * (5.0 if dataset == "shirt" else 0.1)
            # Unqualified candidates are diagnostics only. Never replay them as GNC measurements.
            packets.append({"captured_s": t, "delivered_s": t, "pose": None})
        estimated = [row for row in rows if row["prediction"] is not None]
        groups[name] = {
            "frames": len(rows),
            "estimated": len(estimated),
            "failed": len(rows) - len(estimated),
            "position_rms_m": (
                float(np.sqrt(np.mean([r["position_error_m"] ** 2 for r in estimated])))
                if estimated
                else None
            ),
            "attitude_rms_deg": (
                float(
                    np.sqrt(np.mean([r["attitude_error_deg"] ** 2 for r in estimated]))
                )
                if estimated
                else None
            ),
            "vision_p95_us": float(np.percentile([r["vision_us"] for r in rows], 95)),
            "results": rows,
        }
        (out / (name.replace("/", "-") + "-" + options.runtime + ".jsonl")).write_text(
            "".join(json.dumps(p) + "\n" for p in packets), encoding="utf-8"
        )
        print(
            name, {k: v for k, v in groups[name].items() if k != "results"}, flush=True
        )
    report = {
        "method": "Compact supervised depthwise heatmap encoder/decoder + fixed seedless geometric PnP gate",
        "runtime": options.runtime,
        "checkpoint_sha256": hashlib.sha256(
            (out / "heatmap.pt").read_bytes()
        ).hexdigest(),
        "parameters": metadata["parameters"],
        "no_pose_prior": True,
        "confidence_calibrated": False,
        "gnc_output_enabled": False,
        "packet_policy": "All poses null until independent perception/integrity qualification; raw PnP hypotheses remain in scored diagnostics only",
        "int8_validated": False,
        "hardware_verified": False,
        "groups": groups,
    }
    (
        out
        / (
            "report.json"
            if options.runtime == "torch"
            else "report-" + options.runtime + ".json"
        )
    ).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runtime", choices=["torch", "onnx", "int8"], default="torch")
    options = parser.parse_args()
    if not 1 <= options.epochs <= 100 or not 1 <= options.threads <= 8:
        parser.error("epochs must be 1..100 and threads 1..8")
    (train if options.mode == "train" else evaluate)(options)


if __name__ == "__main__":
    main()
