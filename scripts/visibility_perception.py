"""Visibility-supervised model/loss and spatial moments; no invented Tango labels.

These are training building blocks. No visibility-qualified checkpoint is bundled.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from learned_perception import GRID_H, GRID_W, HeatmapNet


def load_supervision(path, hashes, geometry_sha256, include_masks=False):
    document = json.loads(path.read_text())
    if (
        document.get("schema") != 1
        or document.get("geometry_sha256") != geometry_sha256
        or not document.get("source")
        or len(document.get("mesh_sha256", document.get("proxy_model_sha256", "")))
        != 64
    ):
        raise ValueError("Visibility provenance/schema does not match this geometry")
    index = {row["sha256"]: row for row in document["frames"]}
    if include_masks and document.get("supervision_kind") == "analytic_bus_proxy":
        raise ValueError(
            "Bus-only proxy has no complete spacecraft foreground mask; compositing would erase appendages"
        )
    if len(index) != len(document["frames"]):
        raise ValueError("Duplicate visibility image identities")
    flags = []
    masks = []
    for digest in hashes:
        if digest not in index:
            raise ValueError(
                "Missing visibility annotation for a training/development image"
            )
        row = index[digest]
        values = np.asarray(row["visibility"])
        if (
            values.shape != (11,)
            or not np.isin(values, [-1, 0, 1]).all()
            or not (values >= 0).any()
        ):
            raise ValueError(
                "Visibility requires eleven explicit labels, with some known supervision"
            )
        flags.append(values)
        if include_masks:
            mask_path = (path.parent / row["mask"]).resolve()
            if (
                not mask_path.is_relative_to(path.parent.resolve())
                or hashlib.sha256(mask_path.read_bytes()).hexdigest()
                != row["mask_sha256"]
            ):
                raise ValueError("Mask provenance mismatch or escaping path")
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            if mask is None or not np.isin(mask, [0, 255]).all():
                raise ValueError("Expected binary rectified foreground mask")
            masks.append(
                cv.resize(mask, (256, 192), interpolation=cv.INTER_NEAREST) > 0
            )
    return torch.tensor(np.asarray(flags), dtype=torch.int8), masks


class VisibilityNet(HeatmapNet):
    def __init__(self):
        super().__init__()
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 11)
        )

    def forward(self, image):
        features = self.features(image)
        return self.head(features), self.visibility_head(features)


def spatial_moments(logits, beta=1.0):
    """Differentiable global expectation and spread in heatmap pixel units.

    Multimodal heatmaps may average into empty space. Spread is not calibrated R.
    """
    if (
        logits.ndim != 4
        or logits.shape[-2:] != (GRID_H, GRID_W)
        or not torch.isfinite(logits).all()
        or not np.isfinite(beta)
        or beta <= 0
    ):
        raise ValueError("Expected finite BxKx48x64 logits and positive beta")
    y, x = torch.meshgrid(
        torch.arange(GRID_H, device=logits.device, dtype=logits.dtype),
        torch.arange(GRID_W, device=logits.device, dtype=logits.dtype),
        indexing="ij",
    )
    grid = torch.stack([x.flatten(), y.flatten()], -1)
    probability = functional.softmax(logits.flatten(2) * beta, dim=-1)
    center = probability @ grid
    difference = grid - center[..., None, :]
    covariance = torch.einsum(
        "bkn,bkni,bknj->bkij", probability, difference, difference
    )
    return center, covariance


def visible_loss(heatmaps, visibility_logits, coordinates, visibility):
    if (
        heatmaps.shape[:2] != visibility_logits.shape
        or visibility.shape != visibility_logits.shape
        or coordinates.shape != (*visibility.shape, 2)
    ):
        raise ValueError("Landmark training shapes disagree")
    if not torch.all((visibility == -1) | (visibility == 0) | (visibility == 1)):
        raise ValueError("Visibility must be -1 unknown, 0 occluded, or 1 visible")
    if (
        not torch.isfinite(coordinates).all()
        or not torch.isfinite(heatmaps).all()
        or not torch.isfinite(visibility_logits).all()
    ):
        raise ValueError(
            "Training tensors must be finite, including unknown-landmark coordinates"
        )
    inside = (
        (coordinates[..., 0] >= 0)
        & (coordinates[..., 0] < GRID_W)
        & (coordinates[..., 1] >= 0)
        & (coordinates[..., 1] < GRID_H)
    )
    known = visibility >= 0
    if not known.any():
        raise ValueError("Cannot train visibility from entirely unknown labels")
    target_visible = (visibility == 1) & inside
    classification = functional.binary_cross_entropy_with_logits(
        visibility_logits[known], target_visible[known].to(heatmaps.dtype)
    )
    y, x = torch.meshgrid(
        torch.arange(GRID_H, device=heatmaps.device),
        torch.arange(GRID_W, device=heatmaps.device),
        indexing="ij",
    )
    squared = (x - coordinates[..., 0, None, None]) ** 2 + (
        y - coordinates[..., 1, None, None]
    ) ** 2
    target = torch.exp(-squared / (2 * 1.5**2))
    target /= target.sum((-1, -2), keepdim=True).clamp_min(1e-12)
    per_landmark = -(
        target.flatten(2) * functional.log_softmax(heatmaps.flatten(2), -1)
    ).sum(-1)
    regression = (
        per_landmark[target_visible].mean()
        if target_visible.any()
        else heatmaps.sum() * 0
    )
    return regression + classification


def augment(image, coordinates, visibility, foreground, seed):
    """Training-only grayscale compositing; a supplied mask protects target identity.

    Gamma/glare are visual stress augmentation, not calibrated orbital lighting.
    """
    if (
        image.ndim != 2
        or foreground.shape != image.shape
        or foreground.dtype != np.bool_
    ):
        raise ValueError("Explicit binary foreground mask required")
    rng = np.random.default_rng(seed)
    h, w = image.shape
    background = np.zeros_like(image, dtype=np.float32)
    if rng.random() < 0.5:
        yy, xx = np.mgrid[:h, :w]
        horizon = ((xx - w * rng.uniform(0.1, 0.9)) ** 2 + (yy - h * 1.8) ** 2) < (
            h * 1.5
        ) ** 2
        background[horizon] = rng.uniform(20, 120)
    stars = rng.random((h, w)) < 0.001
    background[stars] = rng.uniform(80, 255, stars.sum())
    composite = np.where(foreground, image, background) / 255.0
    composite = np.clip(
        composite ** rng.uniform(0.2, 3.0) * rng.uniform(0.5, 1.5), 0, 1
    )
    yy, xx = np.mgrid[:h, :w]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    glare = np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(2, 12) ** 2)
    ) * rng.uniform(0, 1.5)
    composite = np.clip(composite + glare * foreground, 0, 1)
    labels = np.asarray(visibility).copy()
    if rng.random() < 0.5:
        x0, y0 = int(rng.uniform(0, w * 0.8)), int(rng.uniform(0, h * 0.8))
        x1, y1 = min(w, x0 + max(1, w // 5)), min(h, y0 + max(1, h // 5))
        composite[y0:y1, x0:x1] = 0
        covered = (
            (coordinates[:, 0] >= x0)
            & (coordinates[:, 0] < x1)
            & (coordinates[:, 1] >= y0)
            & (coordinates[:, 1] < y1)
        )
        labels[covered & (labels == 1)] = 0
    return (composite * 255).round().astype(np.uint8), labels


def main():
    """Characterize the frozen head against DEVELOPMENT proxy labels, not real visibility truth."""
    from scipy.io import loadmat

    from fetch_benchmark import ROOT
    from learned_perception import split_training, training_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "output/visibility-proxy-model"
    )
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "output/visibility-proxy/labels.json"
    )
    options = parser.parse_args()
    metadata = json.loads((options.model_dir / "training.json").read_text())
    label_hash = hashlib.sha256(options.labels.read_bytes()).hexdigest()
    if metadata.get("visibility_source_sha256") != label_hash:
        raise ValueError("Development visibility provenance differs from training")
    selection = json.loads((ROOT / "output/datasets/selection.json").read_text())
    _, development = split_training(selection["groups"]["speedplus/synthetic/train"])
    camera = json.loads(
        (ROOT / "output/datasets/speedplus/speedplusv2/camera.json").read_text()
    )
    model = loadmat(ROOT / "output/datasets/tangoPoints.mat")["tango3Dpoints"].T
    images, coordinates, hashes = training_data(
        development,
        ROOT / "output/datasets/speedplus/speedplusv2/synthetic/images",
        camera,
        model,
    )
    if hashes != metadata["development_hashes"]:
        raise ValueError("Development image identity mismatch")
    flags, _ = load_supervision(options.labels, hashes, metadata["geometry_sha256"])
    network = VisibilityNet()
    network.load_state_dict(
        torch.load(
            options.model_dir / "heatmap.pt", map_location="cpu", weights_only=True
        )
    )
    network.eval()
    torch.set_num_threads(4)
    tp = fp = fn = correct = known_count = visible_count = 0
    squared = 0.0
    eligible = 0
    with torch.inference_mode():
        for start in range(0, len(images), 16):
            heatmap, logits = network(images[start : start + 16])
            center, spread = spatial_moments(heatmap)
            predicted = logits.sigmoid() > 0.5
            xy = coordinates[start : start + 16]
            target = (
                (flags[start : start + 16] == 1)
                & (xy[..., 0] >= 0)
                & (xy[..., 0] < GRID_W)
                & (xy[..., 1] >= 0)
                & (xy[..., 1] < GRID_H)
            )
            known = flags[start : start + 16] >= 0
            tp += int((predicted & target & known).sum())
            fp += int((predicted & ~target & known).sum())
            fn += int((~predicted & target & known).sum())
            correct += int(((predicted == target) & known).sum())
            known_count += int(known.sum())
            squared += float(((center - xy).square().sum(-1)[target] * 16).sum())
            visible_count += int(target.sum())
            eligible += int(
                (predicted & (spread.diagonal(dim1=-2, dim2=-1).sum(-1) < 16)).sum()
            )
    report = {
        "scope": "Development analytic-proxy labels only; no verified real visibility labels or hardware evidence",
        "images": len(images),
        "known_landmarks": known_count,
        "visibility_accuracy": correct / known_count,
        "visibility_precision": tp / (tp + fp) if tp + fp else None,
        "visibility_recall": tp / (tp + fn) if tp + fn else None,
        "geometrically_visible_landmarks": visible_count,
        "visible_keypoint_rms_model_input_px": (
            float(np.sqrt(squared / visible_count)) if visible_count else None
        ),
        "mean_keypoints_passing_visibility_and_spread": eligible / len(images),
        "confidence_calibrated": False,
        "deployment_approved": False,
        "labels_sha256": label_hash,
    }
    (options.model_dir / "development-visibility.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
