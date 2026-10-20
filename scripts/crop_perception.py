"""Coarse-to-fine Tango experiment. Training may use labels; inference never does.

Separate image-only box regression and ROI landmark localization. All poses remain
diagnostics, never navigation packets. No checkpoint is flight-qualified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.nn import functional as F

from fetch_benchmark import ROOT, SOURCES
from image_perception import prepared, score_pose, solve_correspondences
from learned_perception import (
    GRID_H,
    GRID_W,
    SEED,
    decode,
    image_tensor,
    split_training,
    training_data,
)
from visibility_perception import VisibilityNet, load_supervision, visible_loss


@dataclass(frozen=True)
class Crop:
    """Continuous source-pixel bounds with exact OpenCV pixel-center mapping."""

    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_box(cls, box, shape, padding=1.35, max_extent=1.5):
        box = np.asarray(box, dtype=float)
        if (
            box.shape != (4,)
            or not np.isfinite(box).all()
            or padding < 1
            or not 1 <= max_extent <= 2
        ):
            raise ValueError("Expected finite normalized center/size box")
        cx, cy, w, h = box
        if not (
            0 <= cx <= 1
            and 0 <= cy <= 1
            and 0.02 <= w <= max_extent
            and 0.02 <= h <= max_extent
        ):
            raise ValueError("Box outside supported image envelope")
        height = max(h * shape[0], w * shape[1] * 0.75, 24) * padding
        width = height * 4 / 3
        return cls(cx * shape[1] - width / 2, cy * shape[0] - height / 2, width, height)

    def to_grid(self, pixels):
        return (np.asarray(pixels) + 0.5 - [self.x, self.y]) * [
            GRID_W / self.width,
            GRID_H / self.height,
        ] - 0.5

    def to_pixels(self, grid):
        return (
            (np.asarray(grid) + 0.5) * [self.width / GRID_W, self.height / GRID_H]
            + [self.x, self.y]
            - 0.5
        )

    def image(self, image):
        sx, sy = 256 / self.width, 192 / self.height
        matrix = np.array(
            [[sx, 0, (0.5 - self.x) * sx - 0.5], [0, sy, (0.5 - self.y) * sy - 0.5]]
        )
        return cv.warpAffine(
            image,
            matrix,
            (256, 192),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
        )


class BoxNet(nn.Module):
    """Spatial feature map retained; no pose or landmark prior at inference."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = 1
        for out in (16, 32, 48, 64, 96):
            layers.extend([nn.Conv2d(channels, out, 3, 2, 1), nn.ReLU()])
            channels = out
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(96 * 6 * 8, 128), nn.ReLU(), nn.Linear(128, 4)
        )

    def forward(self, image):
        return self.head(self.encoder(image))


def balanced_loss(output, xy, flags):
    """Retain masked CE, replace majority-dominated BCE by equal class means."""
    heatmaps, logits = output
    known = flags >= 0
    inside = (
        (xy[..., 0] >= 0)
        & (xy[..., 0] < GRID_W)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] < GRID_H)
    )
    target = (flags == 1) & inside
    base = visible_loss(heatmaps, logits, xy, flags)
    bce = F.binary_cross_entropy_with_logits(
        logits[known], target[known].float(), reduction="none"
    )
    classes = [
        bce[target[known] == label].mean()
        for label in (False, True)
        if (target[known] == label).any()
    ]
    return base - bce.mean() + torch.stack(classes).mean()


def context():
    selection = json.loads((ROOT / "output/datasets/selection.json").read_text())
    speed = ROOT / "output/datasets/speedplus/speedplusv2"
    calibration = json.loads((speed / "camera.json").read_text())
    geometry_path = ROOT / "output/datasets/tangoPoints.mat"
    if (
        hashlib.sha256(geometry_path.read_bytes()).hexdigest()
        != selection["geometry_sha256"]
    ):
        raise ValueError("Geometry identity mismatch")
    geometry = loadmat(geometry_path)["tango3Dpoints"].T.astype(np.float64)
    return selection, speed, calibration, geometry


def dataset(rows, speed, calibration, geometry, max_extent=1.5):
    images, xy, hashes = training_data(
        rows, speed / "synthetic/images", calibration, geometry
    )
    # Labels construct supervised training targets only, never an inference crop.
    normalized = (xy.numpy() + 0.5) / [GRID_W, GRID_H]
    low, high = normalized.min(1), normalized.max(1)
    boxes = np.concatenate([(low + high) / 2, high - low], axis=1)
    crops, crop_xy = [], []
    for row, coordinates, box in zip(rows, normalized, boxes, strict=True):
        image, _ = prepared(speed / "synthetic/images" / row["filename"], calibration)
        crop = Crop.from_box(box, image.shape, max_extent=max_extent)
        crops.append(image_tensor(crop.image(image)))
        crop_xy.append(
            crop.to_grid(coordinates * [image.shape[1], image.shape[0]] - 0.5)
        )
    targets = np.concatenate([boxes[:, :2], np.log(boxes[:, 2:])], axis=1)
    return (
        images,
        torch.tensor(targets, dtype=torch.float32),
        torch.stack(crops),
        torch.tensor(np.asarray(crop_xy), dtype=torch.float32),
        hashes,
        (xy[..., 0] >= 0)
        & (xy[..., 0] < GRID_W)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] < GRID_H),
    )


def source_visibility(flags, in_source):
    """A padded ROI cannot make an out-of-FOV landmark observable. Preserve unknowns."""
    if flags.shape != in_source.shape or in_source.dtype != torch.bool:
        raise ValueError("Expected source-FOV mask for every visibility flag")
    return torch.where((flags == 1) & ~in_source, torch.zeros_like(flags), flags)


def jitter(images, xy, return_transform=False):
    """Training-only affine crop jitter with corresponding label transformation."""
    scale = torch.rand(len(images), 1) * 0.3 + 0.85
    offset = (torch.rand(len(images), 2) - 0.5) * 0.16
    matrix = torch.zeros(len(images), 2, 3)
    matrix[:, 0, 0] = scale[:, 0]
    matrix[:, 1, 1] = scale[:, 0]
    matrix[:, :, 2] = offset
    grid = F.affine_grid(matrix, images.shape, align_corners=False)
    changed = F.grid_sample(images, grid, align_corners=False)
    normal = (xy + 0.5) / torch.tensor([GRID_W, GRID_H]) * 2 - 1
    normal = (normal - offset[:, None]) / scale[:, None]
    coordinates = (normal + 1) * torch.tensor([GRID_W, GRID_H]) / 2 - 0.5
    if return_transform:
        return changed, coordinates, (scale, offset)
    return changed, coordinates


def fit(
    network,
    inputs,
    labels,
    loss_fn,
    development,
    epochs,
    threads,
    path,
    localizer=False,
):
    torch.set_num_threads(threads)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=0.00005
    )
    best, history = float("inf"), []
    for epoch in range(epochs):
        network.train()
        total = 0.0
        for ids in torch.randperm(len(inputs)).split(16):
            image = inputs[ids].clone()
            target = tuple(value[ids].clone() for value in labels)
            if localizer:
                image, target_xy = jitter(image, target[0])
                target = (target_xy, target[1])
            gamma = 0.6 + torch.rand(len(ids), 1, 1, 1) * 0.8
            image = (
                image.pow(gamma) * (0.7 + torch.rand(len(ids), 1, 1, 1) * 0.6)
                + torch.randn_like(image) * 0.01
            ).clamp(0, 1)
            optimizer.zero_grad()
            loss = loss_fn(network(image), *target)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        scheduler.step()
        network.eval()
        dev_x, dev_y = development
        with torch.inference_mode():
            value = sum(
                float(
                    loss_fn(network(dev_x[i : i + 16]), *(v[i : i + 16] for v in dev_y))
                )
                * len(dev_x[i : i + 16])
                for i in range(0, len(dev_x), 16)
            ) / len(dev_x)
        if not np.isfinite(value):
            raise ValueError("Nonfinite development loss")
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": total / len(inputs),
                "development_loss": value,
            }
        )
        if value < best:
            best = value
            torch.save(network.state_dict(), path)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(path.stem, history[-1], flush=True)
    return history


def train(options):
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    selection, speed, camera, geometry = context()
    training, development = split_training(
        selection["groups"]["speedplus/synthetic/train"]
    )
    a = dataset(training, speed, camera, geometry)
    b = dataset(development, speed, camera, geometry)
    if set(a[4]) & set(b[4]):
        raise ValueError("Training and development overlap")
    flags, _ = load_supervision(options.labels, a[4], selection["geometry_sha256"])
    dev_flags, _ = load_supervision(options.labels, b[4], selection["geometry_sha256"])
    excluded = {
        "training": int(((flags == 1) & ~a[5]).sum()),
        "development": int(((dev_flags == 1) & ~b[5]).sum()),
    }
    flags = source_visibility(flags, a[5])
    dev_flags = source_visibility(dev_flags, b[5])
    out = options.out
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    reused_box = None
    if options.reuse_box is not None:
        prior = json.loads((options.reuse_box / "training.json").read_text())
        if (
            prior["training_hashes"] != a[4]
            or prior["development_hashes"] != b[4]
            or prior["geometry_sha256"] != selection["geometry_sha256"]
        ):
            raise ValueError(
                "Box checkpoint was not trained on this exact train/development selection"
            )
        reused_box = hashlib.sha256(
            (options.reuse_box / "box.pt").read_bytes()
        ).hexdigest()
        shutil.copyfile(options.reuse_box / "box.pt", out / "box.pt")
        box = BoxNet()
        box.load_state_dict(
            torch.load(out / "box.pt", map_location="cpu", weights_only=True)
        )
        box_history = prior["box_history"]
    else:
        box = BoxNet()
        box_history = fit(
            box,
            a[0],
            (a[1],),
            F.smooth_l1_loss,
            (b[0], (b[1],)),
            options.epochs,
            options.threads,
            out / "box.pt",
        )
    # Keep landmark initialization independent of whether unchanged box training is reused.
    torch.manual_seed(SEED + 1)
    localizer = VisibilityNet()
    heatmap_history = fit(
        localizer,
        a[2],
        (a[3], flags),
        balanced_loss,
        (b[2], (b[3], dev_flags)),
        options.epochs,
        options.threads,
        out / "localizer.pt",
        True,
    )
    metadata = {
        "seed": SEED,
        "localizer_seed": SEED + 1,
        "reused_box_sha256": reused_box,
        "epochs": options.epochs,
        "training_hashes": a[4],
        "development_hashes": b[4],
        "geometry_sha256": selection["geometry_sha256"],
        "visibility_sha256": hashlib.sha256(options.labels.read_bytes()).hexdigest(),
        "box_history": box_history,
        "localizer_history": heatmap_history,
        "parameters": {
            "box": sum(v.numel() for v in box.parameters()),
            "localizer": sum(v.numel() for v in localizer.parameters()),
        },
        "elapsed_s": time.perf_counter() - start,
        "training_crops": "Annotation-derived box with padding 1.35 and affine jitter; inference uses BoxNet pixels only",
        "source_fov_supervision": True,
        "source_fov_visible_excluded": excluded,
        "thresholds": {
            "local_mass_min": 0.05,
            "visibility_min": 0.5,
            "all_point_mean_px_max": 3,
            "crop_padding": 1.35,
        },
        "deployment_approved": False,
        "torch": torch.__version__,
    }
    (out / "training.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


class Perception:
    """Stateless global acquisition. An inaccurate/missing pose prior cannot steer it."""

    def __init__(self, directory):
        metadata = json.loads((directory / "training.json").read_text())
        self.crop_extent = metadata.get("crop_max_extent", 1.5)
        self.integrity = metadata.get("correspondence_integrity", "off")
        if self.integrity not in ("off", "roi_stability_v2", "geometry_sensitivity_v1"):
            raise ValueError("Unsupported correspondence integrity policy")
        if metadata.get("feature_refinement"):
            raise ValueError("Image-corner refinement experiment was rejected")
        if metadata.get("architecture") == "resnet18_fpn":
            from transfer_perception import TransferBox, TransferLandmarks

            self.box, self.localizer = TransferBox(), TransferLandmarks()
        else:
            self.box, self.localizer = BoxNet(), VisibilityNet()
        for network, name in [(self.box, "box"), (self.localizer, "localizer")]:
            path = directory / (name + ".pt")
            if metadata.get("checkpoint_sha256") and (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != metadata["checkpoint_sha256"][name]
            ):
                raise ValueError("Checkpoint differs from frozen training metadata")
            network.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            network.eval()

    def locate(self, image):
        if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 24:
            raise ValueError("Expected grayscale uint8 camera image")
        if image.std() < 1:
            return None, "no_contrast"
        with torch.inference_mode():
            output = self.box(image_tensor(image)[None])[0].numpy()
            box = np.concatenate([output[:2], np.exp(np.clip(output[2:], -10, 2))])
            try:
                crop = Crop.from_box(box, image.shape, max_extent=self.crop_extent)
            except ValueError:
                return None, "box_out_of_envelope"
            heatmaps, visibility = self.localizer(image_tensor(crop.image(image))[None])
        # Local mass preserves peaks; a global expectation can fall between modes.
        points, mass = decode(heatmaps[0].numpy(), (GRID_H, GRID_W))
        pixels = crop.to_pixels(points)
        eligible = (visibility[0].sigmoid().numpy() > 0.5) & (mass >= 0.05)
        eligible &= (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < image.shape[1])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < image.shape[0])
        )
        return {
            "box": box.tolist(),
            "points": pixels,
            "ids": np.flatnonzero(eligible),
            "mass": mass,
        }, "localized"

    def infer(self, image, camera, geometry):
        localized, reason = self.locate(image)
        if localized is None:
            return None, {"reason": reason, "landmarks": 0}
        ids = localized["ids"]
        pose = solve_correspondences(geometry[ids], localized["points"][ids], camera)
        raw_hypothesis = pose is not None
        baseline_diagnostics = {"raw_hypothesis": raw_hypothesis, "baseline_pose": pose}
        integrity = None
        if self.integrity == "geometry_sensitivity_v1" and pose is not None:
            from correspondence_integrity import assess

            integrity = assess(
                geometry[ids],
                [localized["points"][ids]],
                camera,
                pose,
                probe_crops=False,
            )
            if not integrity["accepted"]:
                return None, {
                    "reason": integrity["reason"],
                    "landmarks": len(ids),
                    **baseline_diagnostics,
                    "integrity": integrity,
                }
        if self.integrity == "roi_stability_v2":
            from correspondence_integrity import POLICY, assess

            if len(ids) < POLICY["min_landmarks"]:
                return None, {
                    "reason": "insufficient_redundancy",
                    "landmarks": len(ids),
                    **baseline_diagnostics,
                }
            base = Crop.from_box(
                localized["box"], image.shape, max_extent=self.crop_extent
            )
            crops = []
            for scale, dx, dy in POLICY["crop_variants"]:
                width, height = base.width * scale, base.height * scale
                crops.append(
                    Crop(
                        base.x + (0.5 + dx) * base.width - width / 2,
                        base.y + (0.5 + dy) * base.height - height / 2,
                        width,
                        height,
                    )
                )
            with torch.inference_mode():
                heatmaps, visibility = self.localizer(
                    torch.stack([image_tensor(crop.image(image)) for crop in crops])
                )
            samples = [localized["points"][ids]]
            stable = np.ones(len(ids), dtype=bool)
            for index, crop in enumerate(crops):
                xy, mass = decode(heatmaps[index].numpy(), (GRID_H, GRID_W))
                stable &= (visibility[index].sigmoid().numpy()[ids] > 0.5) & (
                    mass[ids] >= 0.05
                )
                pixels = crop.to_pixels(xy)[ids]
                stable &= (
                    (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < image.shape[1])
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < image.shape[0])
                )
                stable &= (
                    np.linalg.norm(pixels - samples[0], axis=1)
                    <= POLICY["max_crop_delta_px"]
                )
                samples.append(pixels)
            # Selection uses image-space repeatability only, never a fitted pose,
            # reprojection residual, covariance or evaluation annotation.
            excluded = int((~stable).sum())
            if stable.sum() < POLICY["min_landmarks"]:
                return None, {
                    "reason": "insufficient_stable_correspondences",
                    "landmarks": int(stable.sum()),
                    "excluded_unstable_landmarks": excluded,
                    **baseline_diagnostics,
                }
            ids = ids[stable]
            samples = np.asarray(samples)[:, stable]
            pose = solve_correspondences(geometry[ids], samples[0], camera)
            if pose is None:
                return None, {
                    "reason": "stable_pnp_rejected",
                    "landmarks": len(ids),
                    **baseline_diagnostics,
                }
            integrity = assess(geometry[ids], samples, camera, pose)
            integrity["excluded_unstable_landmarks"] = excluded
            if not integrity["accepted"]:
                return None, {
                    "reason": integrity["reason"],
                    "landmarks": len(ids),
                    **baseline_diagnostics,
                    "integrity": integrity,
                }
        return pose, {
            "reason": "hypothesis" if pose else "pnp_rejected",
            "landmarks": len(ids),
            "box": localized["box"],
            "integrity": integrity,
            **baseline_diagnostics,
        }


def evaluate(options):
    torch.set_num_threads(options.threads)
    selection, _, _, geometry = context()
    metadata = json.loads((options.out / "training.json").read_text())
    frozen = {
        n: hashlib.sha256((options.out / (n + ".pt")).read_bytes()).hexdigest()
        for n in ("box", "localizer")
    }
    audit_hashes = {}
    if options.selection is not None:
        selection = json.loads(options.selection.read_text())
        if selection.get("frozen_model_sha256") != frozen:
            raise ValueError(
                "Fresh audit requires the exact preselected frozen checkpoints"
            )
        if selection.get("frozen_pipeline_sha256"):
            from fresh_audit import pipeline_hashes

            if selection["frozen_pipeline_sha256"] != pipeline_hashes(options.out):
                raise ValueError(
                    "Fresh audit requires the exact frozen inference pipeline"
                )
        audit_hashes = {
            (source["source"], row["member"]): row["sha256"]
            for source in selection["provenance"]
            for row in source["members"]
        }
    if metadata["geometry_sha256"] != selection["geometry_sha256"]:
        raise ValueError("Geometry provenance changed")
    used = set(metadata["training_hashes"] + metadata["development_hashes"])
    perception = Perception(options.out)
    groups = {}
    for name, samples in selection["groups"].items():
        if name.endswith("/train"):
            continue
        dataset_name, domain, split = name.split("/")
        base = ROOT / "output/datasets" / dataset_name
        base = base / "speedplusv2" if dataset_name == "speedplus" else base
        camera = json.loads((base / "camera.json").read_text())
        folder = (
            base / domain / "images"
            if dataset_name == "speedplus"
            else base / domain / split / "images"
        )
        rows = []
        for sample in samples:
            path = folder / sample["filename"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if (
                options.selection
                and audit_hashes.get(
                    (
                        SOURCES[dataset_name],
                        path.relative_to(
                            ROOT / "output/datasets" / dataset_name
                        ).as_posix(),
                    )
                )
                != digest
            ):
                raise ValueError(
                    "Audit image no longer matches verified archive provenance"
                )
            if digest in used:
                raise ValueError("Audit image overlaps training or model selection")
            image, k = prepared(path, camera)
            start = time.perf_counter_ns()
            prediction, diagnostics = perception.infer(image, k, geometry)
            # Scoring annotations are read only after inference.
            rows.append(
                {
                    "image": sample["filename"],
                    "sha256": digest,
                    "prediction": prediction,
                    "diagnostics": diagnostics,
                    "vision_us": (time.perf_counter_ns() - start) / 1000,
                    **score_pose(prediction, sample),
                    "baseline_score": score_pose(
                        diagnostics.get("baseline_pose", prediction), sample
                    ),
                }
            )
        valid = [r for r in rows if r["prediction"] is not None]
        baseline_valid = [
            r for r in rows if r["baseline_score"]["position_error_m"] is not None
        ]
        groups[name] = {
            "frames": len(rows),
            "estimated": len(valid),
            "baseline_estimated": len(baseline_valid),
            "baseline_within_15cm_8deg": sum(
                r["baseline_score"]["position_error_m"] <= 0.15
                and r["baseline_score"]["attitude_error_deg"] <= 8
                for r in baseline_valid
            ),
            "baseline_position_worst_m": max(
                (r["baseline_score"]["position_error_m"] for r in baseline_valid),
                default=None,
            ),
            "within_15cm_8deg": sum(
                r["position_error_m"] <= 0.15 and r["attitude_error_deg"] <= 8
                for r in valid
            ),
            "position_rms_m": (
                float(np.sqrt(np.mean([r["position_error_m"] ** 2 for r in valid])))
                if valid
                else None
            ),
            "attitude_rms_deg": (
                float(np.sqrt(np.mean([r["attitude_error_deg"] ** 2 for r in valid])))
                if valid
                else None
            ),
            "position_worst_m": max(
                (r["position_error_m"] for r in valid), default=None
            ),
            "attitude_worst_deg": max(
                (r["attitude_error_deg"] for r in valid), default=None
            ),
            "rejection_reasons": {
                reason: sum(r["diagnostics"]["reason"] == reason for r in rows)
                for reason in sorted({r["diagnostics"]["reason"] for r in rows})
            },
            "vision_p95_us": float(np.percentile([r["vision_us"] for r in rows], 95)),
            "results": rows,
        }
        print(
            name, {k: v for k, v in groups[name].items() if k != "results"}, flush=True
        )
        (
            options.out
            / (
                name.replace("/", "-")
                + ("-fresh" if options.selection else "")
                + ".jsonl"
            )
        ).write_text(
            "".join(
                json.dumps(
                    {
                        "audit_index": i,
                        "temporal_replay_allowed": False,
                        "pose": None,
                    }
                )
                + "\n"
                for i in range(len(rows))
            ),
            encoding="utf-8",
        )
    report = {
        "method": "Image-only learned box -> padded crop -> balanced visibility/local heatmap peak -> unchanged strict PnP",
        "no_pose_prior": True,
        "architecture": metadata.get("architecture", "small_crop_reference"),
        "correspondence_integrity": metadata.get("correspondence_integrity", "off"),
        "runtime": {
            "torch": torch.__version__,
            "opencv": cv.__version__,
            "numpy": np.__version__,
        },
        "gnc_output_enabled": False,
        "temporal_replay_allowed": False,
        "deployment_approved": False,
        "model_sha256": frozen,
        "selection_sha256": hashlib.sha256(
            (options.selection or ROOT / "output/datasets/selection.json").read_bytes()
        ).hexdigest(),
        "groups": groups,
    }
    (
        options.out / ("fresh-report.json" if options.selection else "report.json")
    ).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def development(options):
    """Separate box coverage from landmark quality; labels are scoring-only here."""
    torch.set_num_threads(options.threads)
    selection, speed, camera, geometry = context()
    metadata = json.loads((options.out / "training.json").read_text())
    if metadata.get("training_selection"):
        source = (ROOT / metadata["training_selection"]).resolve()
        if (
            not source.is_relative_to(ROOT / "output")
            or hashlib.sha256(source.read_bytes()).hexdigest()
            != metadata["training_selection_sha256"]
        ):
            raise ValueError("Development training-pool provenance changed")
        selection = json.loads(source.read_text())
    _, rows = split_training(selection["groups"]["speedplus/synthetic/train"])
    _, _, teacher_crops, crop_xy, hashes, source_in_frame = dataset(
        rows, speed, camera, geometry, max_extent=metadata.get("crop_max_extent", 1.5)
    )
    if (
        hashes != metadata["development_hashes"]
        or hashlib.sha256(options.labels.read_bytes()).hexdigest()
        != metadata["visibility_sha256"]
    ):
        raise ValueError("Development or visibility identities changed")
    flags, _ = load_supervision(options.labels, hashes, selection["geometry_sha256"])
    flags = source_visibility(flags, source_in_frame)
    perception = Perception(options.out)
    teacher_squared, count, tp, tn, fp, fn = 0.0, 0, 0, 0, 0, 0
    containment, eligible, pnp = [], [], 0
    pose_errors = []
    localized_frames = 0
    predicted_errors = []
    landmark_errors = [[] for _ in range(11)]
    with torch.inference_mode():
        for i, sample in enumerate(rows):
            heatmaps, visible = perception.localizer(teacher_crops[i : i + 1])
            xy, _ = decode(heatmaps[0].numpy(), (GRID_H, GRID_W))
            truth_xy = crop_xy[i].numpy()
            inside = (
                (truth_xy[:, 0] >= 0)
                & (truth_xy[:, 0] < GRID_W)
                & (truth_xy[:, 1] >= 0)
                & (truth_xy[:, 1] < GRID_H)
            )
            target = (flags[i].numpy() == 1) & inside
            predicted = visible[0].sigmoid().numpy() > 0.5
            known = flags[i].numpy() >= 0
            tp += int((predicted & target & known).sum())
            tn += int((~predicted & ~target & known).sum())
            fp += int((predicted & ~target & known).sum())
            fn += int((~predicted & target & known).sum())
            teacher_squared += float((((xy - truth_xy) ** 2).sum(1)[target] * 16).sum())
            count += int(target.sum())
            image, k = prepared(speed / "synthetic/images" / sample["filename"], camera)
            localized, _ = perception.locate(image)
            if localized is None:
                containment.append(0.0)
                eligible.append(0)
                continue
            localized_frames += 1
            # Derive scoring coordinates after image-only inference has completed.
            from scipy.spatial.transform import Rotation

            q = sample["q_vbs2tango_true"]
            xyz = (
                Rotation.from_quat([*q[1:], q[0]]).apply(geometry)
                + sample["r_Vo2To_vbs_true"]
            )
            expected = xyz[:, :2] / xyz[:, 2:] @ k[:2, :2].T + k[:2, 2]
            roi = Crop.from_box(
                localized["box"], image.shape, max_extent=perception.crop_extent
            )
            grid = roi.to_grid(expected)
            contained = (
                (grid[:, 0] >= 0)
                & (grid[:, 0] < GRID_W)
                & (grid[:, 1] >= 0)
                & (grid[:, 1] < GRID_H)
            )
            containment.append(float(contained.mean()))
            ids = localized["ids"]
            eligible.append(len(ids))
            visible_ids = np.flatnonzero(flags[i].numpy() == 1)
            errors = np.linalg.norm(
                localized["points"][visible_ids] - expected[visible_ids], axis=1
            )
            predicted_errors.extend(errors.tolist())
            for identity, error in zip(visible_ids, errors, strict=True):
                landmark_errors[identity].append(float(error))
            hypothesis = solve_correspondences(
                geometry[ids], localized["points"][ids], k
            )
            if hypothesis is not None:
                pnp += 1
                pose_errors.append(score_pose(hypothesis, sample))
    report = {
        "scope": f"{len(rows)} development images; proxy geometry labels; teacher crop metric is diagnostic, not inference performance",
        "images": len(rows),
        "mean_landmark_crop_containment": float(np.mean(containment)),
        "mean_eligible_landmarks": float(np.mean(eligible)),
        "raw_pose_hypotheses": pnp,
        "image_only_localized_frames": localized_frames,
        "per_landmark_visible_errors_prepared_px": [
            {
                "identity": identity,
                "observations": len(errors),
                "median_px": float(np.median(errors)) if errors else None,
                "p95_px": float(np.percentile(errors, 95)) if errors else None,
                "within_5px": sum(error <= 5 for error in errors),
            }
            for identity, errors in enumerate(landmark_errors)
        ],
        "hypotheses_within_15cm_8deg": sum(
            row["position_error_m"] <= 0.15 and row["attitude_error_deg"] <= 8
            for row in pose_errors
        ),
        "hypothesis_position_rms_m": (
            float(
                np.sqrt(np.mean([row["position_error_m"] ** 2 for row in pose_errors]))
            )
            if pose_errors
            else None
        ),
        "hypothesis_position_worst_m": max(
            (row["position_error_m"] for row in pose_errors), default=None
        ),
        "hypothesis_attitude_worst_deg": max(
            (row["attitude_error_deg"] for row in pose_errors), default=None
        ),
        "teacher_crop_visible_keypoint_rms_256x192_px": (
            float(np.sqrt(teacher_squared / count)) if count else None
        ),
        "image_only_crop_visible_keypoint_rms_prepared_px": (
            float(np.sqrt(np.mean(np.square(predicted_errors))))
            if predicted_errors
            else None
        ),
        "teacher_crop_visibility_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "teacher_crop_visibility_balanced_accuracy": (
            ((tp / (tp + fn) + tn / (tn + fp)) / 2) if tp + fn and tn + fp else None
        ),
        "deployment_approved": False,
    }
    (options.out / "development.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


def overfit(options):
    """Eight TRAINING images only; no checkpoint writes or audit model selection."""
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(options.threads)
    selection, speed, camera, geometry = context()
    rows, _ = split_training(selection["groups"]["speedplus/synthetic/train"])
    _, _, images, xy, hashes, source_in_frame = dataset(
        rows[:8], speed, camera, geometry
    )
    flags, _ = load_supervision(options.labels, hashes, selection["geometry_sha256"])
    flags = source_visibility(flags, source_in_frame)
    network = VisibilityNet()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.003, weight_decay=0.0001)
    history = []
    start = time.perf_counter()
    for step in range(options.steps):
        network.train()
        optimizer.zero_grad()
        loss = balanced_loss(network(images), xy, flags)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()
        if step % 50 == 0 or step == options.steps - 1:
            history.append({"step": step + 1, "loss": float(loss.detach())})
            print("overfit", history[-1], flush=True)
    network.eval()
    with torch.inference_mode():
        heatmaps, visibility = network(images)
        predicted = np.stack(
            [decode(value.numpy(), (GRID_H, GRID_W))[0] for value in heatmaps]
        )
        target = flags.numpy() == 1
        errors = np.linalg.norm((predicted - xy.numpy()) * 4, axis=-1)[target]
        known = flags >= 0
        classification = visibility.sigmoid() > 0.5
        correct = classification == (flags == 1)
    report = {
        "scope": "Eight deterministic training images, annotation-derived crops, no augmentation, random initialization; NOT generalization or image-only acquisition",
        "training_hashes": hashes,
        "steps": options.steps,
        "history": history,
        "visible_keypoint_rms_crop_input_px": float(np.sqrt(np.mean(errors**2))),
        "visible_keypoint_pck5": float(np.mean(errors <= 5)),
        "visibility_accuracy": float(correct[known].float().mean()),
        "elapsed_s": time.perf_counter() - start,
        "audit_checkpoints_modified": False,
        "deployment_approved": False,
    }
    options.out.mkdir(parents=True, exist_ok=True)
    (options.out / "overfit-control.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("training_hashes", "history")
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "development", "evaluate", "overfit"])
    parser.add_argument("--out", type=Path, default=ROOT / "output/crop-perception")
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "output/visibility-proxy/labels.json"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--reuse-box",
        type=Path,
        help="Training-only reuse of a box checkpoint with identical data provenance",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--steps", type=int, default=400, help="Training-only overfit control steps"
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="Fresh, frozen-model audit selection; evaluate only",
    )
    options = parser.parse_args()
    options.out = options.out.resolve()
    if options.selection is not None and options.mode != "evaluate":
        parser.error("Custom audit selection cannot enter training/development")
    if (
        not options.out.is_relative_to(ROOT / "output")
        or not 1 <= options.epochs <= 100
        or not 1 <= options.threads <= 8
        or not 1 <= options.steps <= 1000
    ):
        parser.error(
            "Output must be under output/; epochs 1..100, threads 1..8, steps 1..1000"
        )
    {
        "train": train,
        "development": development,
        "evaluate": evaluate,
        "overfit": overfit,
    }[options.mode](options)


if __name__ == "__main__":
    main()
