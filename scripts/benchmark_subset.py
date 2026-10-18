"""Reproduce a small, fixed, disjoint benchmark selection without tuning on held-out labels."""

import hashlib
import json
import random
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

from fetch_benchmark import ROOT, SOURCES, RemoteZip


def main():
    selections = {}
    speed = ROOT / "output/datasets/speedplus/speedplusv2"
    rng = random.Random(1729)
    for domain, labels, count in (
        ("synthetic", "train", 500),
        ("synthetic", "validation", 40),
        ("lightbox", "test", 40),
        ("sunlamp", "test", 40),
    ):
        rows = json.loads((speed / domain / (labels + ".json")).read_text())
        chosen = rng.sample(rows, count)
        selections[f"speedplus/{domain}/{labels}"] = chosen
    for sequence in ("roe1", "roe2"):
        root = ROOT / "output/datasets/shirt" / sequence
        rows = json.loads((root / (sequence + ".json")).read_text())
        # Consecutive frames preserve documented cam_step=5 s; no invented timing.
        selections[f"shirt/{sequence}/lightbox"] = rows[:20]
    for dataset in ("speedplus", "shirt"):
        remote = RemoteZip(SOURCES[dataset])
        with zipfile.ZipFile(remote) as archive:
            entries = {i.filename: i for i in archive.infolist()}
        requests = []
        for key, rows in selections.items():
            if not key.startswith(dataset + "/"):
                continue
            _, domain, split = key.split("/")
            prefix = (
                f"speedplusv2/{domain}/images/"
                if dataset == "speedplus"
                else f"{domain}/{split}/images/"
            )
            requests += [prefix + r["filename"] for r in rows]

        def fetch(name):
            path = ROOT / "output/datasets" / dataset / name
            if not path.exists():
                data = remote.member(entries[name])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            data = path.read_bytes()
            if (
                len(data) != entries[name].file_size
                or zlib.crc32(data) != entries[name].CRC
            ):
                raise ValueError(f"Cached image failed archive size/CRC check: {name}")
            return {
                "member": name,
                "sha256": hashlib.sha256(data).hexdigest(),
            }

        with ThreadPoolExecutor(max_workers=6) as pool:
            provenance = list(pool.map(fetch, requests))
        out = ROOT / "output/datasets" / dataset
        (out / "subset-provenance.json").write_text(
            json.dumps(
                {
                    "source": remote.url,
                    "selection_seed": 1729,
                    "members": provenance,
                    "transferred_bytes": remote.transferred,
                },
                indent=2,
            )
        )
        print(
            f"{dataset}: verified {len(provenance)} selected images, {remote.transferred} transferred bytes",
            flush=True,
        )
    # Author-provided geometry; data only, never executable code.
    geometry = ROOT / "output/datasets/tangoPoints.mat"
    remote = RemoteZip(SOURCES["shirt"])
    url = "https://raw.githubusercontent.com/tpark94/spnv2/main/core/utils/models/tangoPoints.mat"
    with urllib.request.urlopen(url, context=remote.context, timeout=40) as response:
        data = response.read(1024 * 1024)
    geometry.write_bytes(data)
    (ROOT / "output/datasets/selection.json").write_text(
        json.dumps(
            {
                "seed": 1729,
                "groups": selections,
                "geometry_source": url,
                "geometry_sha256": hashlib.sha256(data).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
