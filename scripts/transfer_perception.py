"""Pretrained desktop acquisition reference; no audit labels or inference pose prior.

ResNet-18 ImageNet features with separate box and pyramid landmark models. FP32
training on an explicitly supplied synthetic-training pool. Not an embedded model.
"""

import argparse
import hashlib
import json
import os
import shutil
import ssl
import time
import urllib.request
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import truststore
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from crop_perception import balanced_loss, context, dataset, jitter, source_visibility
from fetch_benchmark import ROOT
from learned_perception import SEED, split_training
from visibility_perception import load_supervision


class Encoder(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.model = resnet18(weights=None)
        if weights is not None:
            self.model.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True)
            )
        self.model.fc = nn.Identity()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, gray):
        x = (gray.repeat(1, 3, 1, 1) - self.mean) / self.std
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        a = m.layer1(x)
        b = m.layer2(a)
        c = m.layer3(b)
        d = m.layer4(c)
        return a, b, c, d


class TransferBox(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.encoder = Encoder(weights)
        self.head = nn.Sequential(
            # Inputs are fixed at 192x256; the encoder produces 6x8 features.
            # Fixed pooling also supports deterministic CUDA backward.
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(512 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, image):
        return self.head(self.encoder(image)[-1])


class TransferLandmarks(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.encoder = Encoder(weights)
        self.lateral = nn.ModuleList(nn.Conv2d(c, 64, 1) for c in (64, 128, 256, 512))
        self.smooth = nn.ModuleList(
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()) for _ in range(3)
        )
        self.heatmap = nn.Conv2d(64, 11, 1)
        self.visibility = nn.Sequential(
            nn.AvgPool2d((6, 8)), nn.Flatten(), nn.Linear(512, 11)
        )

    def forward(self, image):
        features = self.encoder(image)
        p = self.lateral[3](features[3])
        for i in (2, 1, 0):
            p = self.smooth[i](
                F.interpolate(p, size=features[i].shape[-2:], mode="nearest")
                + self.lateral[i](features[i])
            )
        return self.heatmap(p), self.visibility(features[3])


def pretrained(destination):
    url = ResNet18_Weights.IMAGENET1K_V1.url
    path = destination / url.rsplit("/", 1)[1]
    destination.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(
            url, context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), timeout=60
        ) as response:
            data = response.read(64 * 1024**2 + 1)
        if len(data) > 64 * 1024**2:
            raise ValueError("Pretrained weight transfer exceeded 64 MiB")
        digest = hashlib.sha256(data).hexdigest()
        if not digest.startswith(path.stem.rsplit("-", 1)[1]):
            raise ValueError("Official pretrained weight hash-prefix mismatch")
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(path.stem.rsplit("-", 1)[1]):
        raise ValueError("Cached pretrained weights failed hash check")
    return path, url, digest


def fit(network, inputs, labels, development, options, path, localizer):
    from appearance import sensor_appearance
    from crop_consistency import equivariance_loss

    device = torch.device(options.device)
    network.to(device)
    encoder = list(network.encoder.parameters())
    heads = [p for n, p in network.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": 0.0001 * (0.3 if options.initialize else 1)},
            {"params": heads, "lr": 0.001 * (0.3 if options.initialize else 1)},
        ],
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, options.epochs, eta_min=0.00001
    )
    criterion = balanced_loss if localizer else F.smooth_l1_loss
    best = float("inf")
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(options.epochs):
        network.train()
        total = 0.0
        consistency_total = 0.0
        start = time.perf_counter()
        for ids in torch.randperm(len(inputs)).split(options.batch):
            image = inputs[ids].clone()
            target = tuple(t[ids].clone() for t in labels)
            paired = localizer and getattr(options, "consistency_weight", 0) > 0
            if paired:
                image_a, xy_a, transform_a = jitter(
                    image, target[0], return_transform=True
                )
                image_b, xy_b, transform_b = jitter(
                    image, target[0], return_transform=True
                )
                image = torch.cat([image_a, image_b])
                target = (torch.cat([xy_a, xy_b]), torch.cat([target[1], target[1]]))
                transform_a = tuple(t.to(device) for t in transform_a)
                transform_b = tuple(t.to(device) for t in transform_b)
            elif localizer:
                image, xy = jitter(image, target[0])
                target = (xy, target[1])
            image = image.to(device)
            target = tuple(t.to(device) for t in target)
            if options.appearance == "sensor":
                image, target = sensor_appearance(image, target, localizer)
            else:
                gamma = 0.6 + torch.rand(len(image), 1, 1, 1, device=device) * 0.8
                image = (
                    image.pow(gamma)
                    * (0.7 + torch.rand(len(image), 1, 1, 1, device=device) * 0.6)
                    + torch.randn_like(image) * 0.01
                ).clamp(0, 1)
            optimizer.zero_grad(set_to_none=True)
            output = network(image)
            loss = criterion(output, *target)
            if paired:
                n = len(ids)
                consistency = equivariance_loss(
                    output[0][:n],
                    output[0][n:],
                    transform_a,
                    transform_b,
                    tuple(t[:n] for t in target),
                    tuple(t[n:] for t in target),
                )
                loss = loss + options.consistency_weight * consistency
                consistency_total += float(consistency.detach()) * n
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        scheduler.step()
        network.eval()
        dev_x, dev_y = development
        with torch.inference_mode():
            clean_loss, sensor_loss = 0.0, 0.0
            for i in range(0, len(dev_x), options.batch):
                image = dev_x[i : i + options.batch].to(device)
                target = tuple(v[i : i + options.batch].to(device) for v in dev_y)
                clean_loss += float(criterion(network(image), *target)) * len(image)
                if options.appearance == "sensor":
                    generator = torch.Generator(device=device).manual_seed(
                        SEED + 100 + i
                    )
                    altered, altered_target = sensor_appearance(
                        image, target, localizer, generator
                    )
                    sensor_loss += float(
                        criterion(network(altered), *altered_target)
                    ) * len(image)
            clean_loss /= len(dev_x)
            sensor_loss = (
                sensor_loss / len(dev_x)
                if options.appearance == "sensor"
                else clean_loss
            )
            dev = (clean_loss + sensor_loss) / 2
        if not np.isfinite(dev):
            raise ValueError("Nonfinite development loss")
        row = {
            "epoch": epoch + 1,
            "training_loss": total / len(inputs),
            "training_consistency_loss": consistency_total / len(inputs),
            "development_loss": dev,
            "development_clean_loss": clean_loss,
            "development_sensor_loss": sensor_loss,
            "elapsed_s": time.perf_counter() - start,
        }
        history.append(row)
        if dev < best:
            best = dev
            # Portable tensor-only checkpoint; inference never needs a GPU.
            torch.save(
                {k: v.detach().cpu() for k, v in network.state_dict().items()}, path
            )
        print(path.stem, row, flush=True)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    network.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history, peak


def train(options):
    if (options.out / "training.json").exists():
        raise ValueError("Frozen experiment already exists; use a new output directory")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(options.threads)
    cv.setNumThreads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if options.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select cpu explicitly")
    _, speed, camera, geometry = context()
    selection = json.loads(options.training_selection.read_text())
    if (
        selection["geometry_sha256"]
        != hashlib.sha256(
            (ROOT / "output/datasets/tangoPoints.mat").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("Training geometry changed")
    rows = selection["groups"]["speedplus/synthetic/train"]
    training, development = split_training(rows)
    print(
        f"Preparing {len(training)} training and {len(development)} development images",
        flush=True,
    )
    a = dataset(training, speed, camera, geometry, max_extent=2.0)
    b = dataset(development, speed, camera, geometry, max_extent=2.0)
    if set(a[4]) & set(b[4]):
        raise ValueError("Train/development image overlap")
    verified = {
        r["sha256"] for source in selection["provenance"] for r in source["members"]
    }
    if set(a[4] + b[4]) != verified:
        raise ValueError("Training-pool image provenance changed")
    flags, _ = load_supervision(options.labels, a[4], selection["geometry_sha256"])
    dev_flags, _ = load_supervision(options.labels, b[4], selection["geometry_sha256"])
    excluded = {
        "training": int(((flags == 1) & ~a[5]).sum()),
        "development": int(((dev_flags == 1) & ~b[5]).sum()),
    }
    flags = source_visibility(flags, a[5])
    dev_flags = source_visibility(dev_flags, b[5])
    options.out.mkdir(parents=True, exist_ok=True)
    initialization = None
    if options.initialize:
        parent = json.loads((options.initialize / "training.json").read_text())
        if (
            parent.get("architecture") != "resnet18_fpn"
            or parent["training_hashes"] != a[4]
            or parent["development_hashes"] != b[4]
            or parent["visibility_sha256"]
            != hashlib.sha256(options.labels.read_bytes()).hexdigest()
            or parent["geometry_sha256"] != selection["geometry_sha256"]
        ):
            raise ValueError(
                "Warm start must use the same verified geometry, supervision and train/development split"
            )
        initialization = {
            "parent": options.initialize.resolve().relative_to(ROOT).as_posix(),
            "metadata_sha256": hashlib.sha256(
                (options.initialize / "training.json").read_bytes()
            ).hexdigest(),
            "checkpoint_sha256": parent["checkpoint_sha256"],
        }
        weights, url, digest = (
            None,
            parent["pretrained"]["source"],
            parent["pretrained"]["sha256"],
        )
    else:
        weights, url, digest = pretrained(options.out / "pretrained")

    def initialize(network, name):
        if options.initialize:
            path = options.initialize / (name + ".pt")
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != initialization["checkpoint_sha256"][name]
            ):
                raise ValueError("Warm-start checkpoint changed")
            network.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
        return network

    start = time.perf_counter()
    box = initialize(TransferBox(weights), "box")
    if options.reuse_initial_box:
        shutil.copy2(options.initialize / "box.pt", options.out / "box.pt")
        box_history, box_peak = [], None
    else:
        box_history, box_peak = fit(
            box, a[0], (a[1],), (b[0], (b[1],)), options, options.out / "box.pt", False
        )
    del box
    torch.manual_seed(SEED + 1)
    localizer = initialize(TransferLandmarks(weights), "localizer")
    localizer_history, localizer_peak = fit(
        localizer,
        a[2],
        (a[3], flags),
        (b[2], (b[3], dev_flags)),
        options,
        options.out / "localizer.pt",
        True,
    )
    metadata = {
        "architecture": "resnet18_fpn",
        "initialization": initialization,
        "appearance_policy": options.appearance,
        "consistency_weight": options.consistency_weight,
        "consistency_source_sha256": hashlib.sha256(
            (ROOT / "scripts/crop_consistency.py").read_bytes()
        ).hexdigest(),
        "box_reused": options.reuse_initial_box,
        "localizer_network_batch": options.batch
        * (2 if options.consistency_weight > 0 else 1),
        "appearance_source_sha256": hashlib.sha256(
            (ROOT / "scripts/appearance.py").read_bytes()
        ).hexdigest(),
        "checkpoint_selection": (
            "mean of clean and fixed-seed randomized synthetic development loss"
            if options.appearance == "sensor"
            else "clean development loss"
        ),
        "checkpoint_sha256": {
            name: hashlib.sha256(
                (options.out / (name + ".pt")).read_bytes()
            ).hexdigest()
            for name in ("box", "localizer")
        },
        "training_hashes": a[4],
        "development_hashes": b[4],
        "geometry_sha256": selection["geometry_sha256"],
        "visibility_sha256": hashlib.sha256(options.labels.read_bytes()).hexdigest(),
        "training_selection_sha256": hashlib.sha256(
            options.training_selection.read_bytes()
        ).hexdigest(),
        "training_selection": options.training_selection.resolve()
        .relative_to(ROOT)
        .as_posix(),
        "crop_max_extent": 2.0,
        "pretrained": {
            "source": url,
            "sha256": digest,
            "task": "ImageNet classification, not satellite pose",
        },
        "epochs": options.epochs,
        "seed": SEED,
        "localizer_seed": SEED + 1,
        "device": options.device,
        "gpu": torch.cuda.get_device_name() if options.device == "cuda" else None,
        "torch": torch.__version__,
        "batch": options.batch,
        "threads": options.threads,
        "box_history": box_history,
        "localizer_history": localizer_history,
        "gpu_peak_allocated_bytes": {"box": box_peak, "localizer": localizer_peak},
        "source_fov_supervision": True,
        "source_fov_visible_excluded": excluded,
        "elapsed_s": time.perf_counter() - start,
        "parameters": sum(p.numel() for p in localizer.parameters()),
        "thresholds": {
            "local_mass_min": 0.05,
            "visibility_min": 0.5,
            "all_point_mean_px_max": 3,
            "crop_padding": 1.35,
        },
        "deployment_approved": False,
        "limits": "Desktop FP32 reference; proxy visibility only; no flight-hardware, INT8 or radiation qualification",
    }
    (options.out / "training.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-selection",
        type=Path,
        default=ROOT / "output/datasets/training-2000.json",
    )
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "output/visibility-2000/labels.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "output/transfer-perception")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--appearance", choices=["baseline", "sensor"], default="baseline"
    )
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--reuse-initial-box", action="store_true")
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    options = parser.parse_args()
    if (
        not options.out.resolve().is_relative_to(ROOT / "output")
        or not options.training_selection.resolve().is_relative_to(ROOT / "output")
        or not 1 <= options.epochs <= 100
        or not 1 <= options.threads <= 8
        or not 2 <= options.batch <= 32
        or (
            options.initialize
            and not options.initialize.resolve().is_relative_to(ROOT / "output")
        )
        or (options.reuse_initial_box and not options.initialize)
        or not 0 <= options.consistency_weight <= 1
    ):
        parser.error(
            "Output must remain under output/; epochs 1..100, threads 1..8, batch 2..32"
        )
    train(options)


if __name__ == "__main__":
    main()
