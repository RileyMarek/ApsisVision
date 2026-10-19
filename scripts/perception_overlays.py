"""Development-only landmark contact sheet; truth appears only after inference."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from crop_perception import Crop, Perception, context
from fetch_benchmark import ROOT
from image_perception import prepared
from learned_perception import split_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "output/transfer-perception"
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    options = parser.parse_args()
    if (
        not options.model_dir.resolve().is_relative_to(ROOT / "output")
        or not 1 <= options.samples <= 12
        or not 1 <= options.threads <= 8
    ):
        parser.error("Models must be under output/; samples 1..12, threads 1..8")
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
            raise ValueError("Development-pool provenance changed")
        selection = json.loads(path.read_text())
    _, rows = split_training(selection["groups"]["speedplus/synthetic/train"])
    perception = Perception(options.model_dir)
    tiles = []
    for sample in rows[: options.samples]:
        path = speed / "synthetic/images" / sample["filename"]
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            not in metadata["development_hashes"]
        ):
            raise ValueError("Overlay image is not in the frozen development set")
        image, camera = prepared(path, calibration)
        localized, reason = perception.locate(image)
        # Annotations enter only this diagnostic renderer, after prediction.
        q = sample["q_vbs2tango_true"]
        xyz = (
            Rotation.from_quat([*q[1:], q[0]]).apply(geometry)
            + sample["r_Vo2To_vbs_true"]
        )
        truth = xyz[:, :2] / xyz[:, 2, None] @ camera[:2, :2].T + camera[:2, 2]
        predicted, scored = (cv.cvtColor(image, cv.COLOR_GRAY2BGR) for _ in range(2))
        if localized is not None:
            crop = Crop.from_box(
                localized["box"], image.shape, max_extent=perception.crop_extent
            )
            low = tuple(np.rint([crop.x, crop.y]).astype(int))
            high = tuple(
                np.rint([crop.x + crop.width, crop.y + crop.height]).astype(int)
            )
            cv.rectangle(predicted, low, high, (150, 130, 80), 1)
            for identity, point in enumerate(localized["points"]):
                color = (
                    (150, 230, 80) if identity in localized["ids"] else (100, 100, 240)
                )
                if np.isfinite(point).all() and np.max(np.abs(point)) < 100000:
                    pixel = tuple(np.rint(point).astype(int))
                    cv.drawMarker(predicted, pixel, color, cv.MARKER_CROSS, 9, 1)
                    cv.putText(
                        predicted,
                        str(identity),
                        (pixel[0] + 4, pixel[1] - 4),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        color,
                        1,
                        cv.LINE_AA,
                    )
        for identity, point in enumerate(truth):
            pixel = tuple(np.rint(point).astype(int))
            cv.drawMarker(scored, pixel, (80, 190, 250), cv.MARKER_CROSS, 9, 1)
            cv.putText(
                scored,
                str(identity),
                (pixel[0] + 4, pixel[1] - 4),
                cv.FONT_HERSHEY_SIMPLEX,
                0.35,
                (80, 190, 250),
                1,
                cv.LINE_AA,
            )
        pair = np.hstack([predicted, scored])
        label = np.zeros((32, pair.shape[1], 3), np.uint8)
        cv.putText(
            label,
            f"{sample['filename']}  |  predicted: {reason} (green eligible / red excluded)    |    annotation projection: scoring only",
            (8, 21),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv.LINE_AA,
        )
        tiles.append(np.vstack([label, pair]))
    result = np.vstack(tiles)
    destination = options.model_dir / "development-overlays.jpg"
    if not cv.imwrite(str(destination), result, [cv.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError("Cannot save development overlay contact sheet")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
