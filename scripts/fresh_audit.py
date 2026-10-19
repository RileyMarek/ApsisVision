"""Fetch 100 disjoint audit images only after freezing both crop-model checkpoints.

Selection uses filenames, not pose/lighting/accuracy. Uses the existing bounded,
CRC-verified Stanford archive reader. Does not modify the training selection.
"""

import argparse
import hashlib
import json
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_benchmark import ROOT, SOURCES, RemoteZip


def checkpoints(directory):
    return {
        name: hashlib.sha256((directory / (name + ".pt")).read_bytes()).hexdigest()
        for name in ("box", "localizer")
    }


def choose(rows, excluded, count=20):
    available = [r for r in rows if r["filename"] not in excluded]
    if len(available) < count:
        raise ValueError("Insufficient disjoint audit images")
    return sorted(
        available,
        key=lambda r: hashlib.sha256(
            ("20261022/" + r["filename"]).encode()
        ).hexdigest(),
    )[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "output/crop-perception"
    )
    options = parser.parse_args()
    metadata = options.model_dir / "training.json"
    if not metadata.is_file():
        parser.error("Freeze training before selecting fresh audit images")
    frozen = checkpoints(options.model_dir)
    original = json.loads((ROOT / "output/datasets/selection.json").read_text())
    selections = {}
    for dataset, domain, split in [
        ("speedplus", "synthetic", "validation"),
        ("speedplus", "lightbox", "test"),
        ("speedplus", "sunlamp", "test"),
        ("shirt", "roe1", "lightbox"),
        ("shirt", "roe2", "lightbox"),
    ]:
        group = f"{dataset}/{domain}/{split}"
        root = ROOT / "output/datasets" / dataset
        labels = (
            root / "speedplusv2" / domain / (split + ".json")
            if dataset == "speedplus"
            else root / domain / (domain + ".json")
        )
        # Synthetic train and validation can share the same image namespace.
        excluded = {
            r["filename"]
            for key, rows in original["groups"].items()
            if key.startswith(f"{dataset}/{domain}/")
            for r in rows
        }
        selections[group] = choose(json.loads(labels.read_text()), excluded)
    provenance = []
    for dataset in ("speedplus", "shirt"):
        remote = RemoteZip(SOURCES[dataset], budget=128 * 1024**2)
        with zipfile.ZipFile(remote) as archive:
            entries = {item.filename: item for item in archive.infolist()}
        requests = []
        for group, rows in selections.items():
            prefix, domain, split = group.split("/")
            if prefix == dataset:
                folder = (
                    f"speedplusv2/{domain}/images/"
                    if dataset == "speedplus"
                    else f"{domain}/{split}/images/"
                )
                requests.extend(folder + row["filename"] for row in rows)

        def fetch(name):
            path = ROOT / "output/datasets" / dataset / name
            entry = entries[name]
            if not path.exists():
                data = remote.member(entry)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            data = path.read_bytes()
            if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
                raise ValueError("Cached image failed archive size/CRC verification")
            return {"member": name, "sha256": hashlib.sha256(data).hexdigest()}

        with ThreadPoolExecutor(max_workers=4) as pool:
            members = list(pool.map(fetch, requests))
        provenance.append(
            {
                "source": remote.url,
                "transferred_bytes": remote.transferred,
                "members": members,
            }
        )
        print(dataset, len(members), remote.transferred, flush=True)
    if frozen != checkpoints(options.model_dir):
        raise ValueError("Checkpoint changed while selecting audit; freeze it first")
    report = {
        "seed": 20261022,
        "groups": selections,
        "geometry_sha256": original["geometry_sha256"],
        "frozen_model_sha256": frozen,
        "provenance": provenance,
        "limits": "100 disjoint images; SHIRT frames are correlated, not independent trials; no model selection using this audit",
    }
    destination = ROOT / "output/datasets/fresh-selection.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
