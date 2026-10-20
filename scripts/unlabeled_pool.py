"""Reserve disjoint real-camera pixels without reading per-frame pose annotations."""

import argparse
import hashlib
import json
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_benchmark import ROOT, SOURCES, RemoteZip


def choose_names(names, excluded, count, seed):
    if len(names) != len(set(names)) or any(
        not isinstance(name, str)
        or "/" in name
        or "\\" in name
        or name in ("", ".", "..")
        for name in names
    ):
        raise ValueError("Expected distinct image basenames")
    available = sorted(
        set(names) - set(excluded),
        key=lambda name: hashlib.sha256(f"{seed}/{name}".encode()).hexdigest(),
    )
    if not 1 <= count <= 100 or len(available) < count:
        raise ValueError("Insufficient disjoint images or invalid pool size")
    return available[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-domain", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--exclude-selection", type=Path, action="append", default=[])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "output/datasets/unlabeled-adaptation.json"
    )
    options = parser.parse_args()
    if (
        not 1 <= options.per_domain <= 100
        or not options.out.resolve().is_relative_to(ROOT / "output")
        or options.out.exists()
    ):
        parser.error("Pool size 1..100/domain; choose a new file under output/")
    original_path = ROOT / "output/datasets/selection.json"
    paths = [original_path, *options.exclude_selection]
    selections = [json.loads(path.read_text()) for path in paths]
    excluded_hashes = {
        row["sha256"]
        for selection in selections
        for source in selection.get("provenance", [])
        for row in source["members"]
    }
    for dataset in SOURCES:
        initial = json.loads(
            (ROOT / "output/datasets" / dataset / "subset-provenance.json").read_text()
        )
        excluded_hashes.update(row["sha256"] for row in initial["members"])
    groups, provenance = {}, []
    for dataset in ("speedplus", "shirt"):
        remote = RemoteZip(SOURCES[dataset], budget=128 * 1024**2)
        with zipfile.ZipFile(remote) as archive:
            entries = {item.filename: item for item in archive.infolist()}
        domains = (
            [("lightbox", "test"), ("sunlamp", "test")]
            if dataset == "speedplus"
            else [("roe1", "lightbox"), ("roe2", "lightbox")]
        )
        requests = []
        for domain, split in domains:
            group = f"{dataset}/{domain}/{split}"
            prefix = (
                f"speedplusv2/{domain}/images/"
                if dataset == "speedplus"
                else f"{domain}/{split}/images/"
            )
            # Archive names only: no train/test pose-annotation JSON is opened.
            names = [
                name[len(prefix) :]
                for name in entries
                if name.startswith(prefix) and name.endswith((".jpg", ".png"))
            ]
            excluded_names = {
                row["filename"]
                for selection in selections
                for key, rows in selection["groups"].items()
                if key.startswith(f"{dataset}/{domain}/")
                for row in rows
            }
            chosen = choose_names(
                names,
                excluded_names,
                options.per_domain,
                str(options.seed) + "/" + group,
            )
            groups[group] = [{"filename": name} for name in chosen]
            requests.extend(prefix + name for name in chosen)

        def fetch(name):
            folder = ROOT / "output/datasets" / dataset
            path = (folder / name).resolve()
            entry = entries[name]
            if (
                not path.is_relative_to(folder.resolve())
                or entry.file_size > 16 * 1024**2
            ):
                raise ValueError("Image escapes dataset directory or exceeds 16 MiB")
            if not path.exists():
                data = remote.member(entry)
                if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
                    raise ValueError("Image failed archive size/CRC")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if (
                len(data) != entry.file_size
                or zlib.crc32(data) != entry.CRC
                or digest in excluded_hashes
            ):
                raise ValueError("Image changed or overlaps an excluded selection")
            return {"member": name, "sha256": digest}

        with ThreadPoolExecutor(max_workers=4) as pool:
            members = list(pool.map(fetch, requests))
        excluded_hashes.update(row["sha256"] for row in members)
        provenance.append(
            {
                "source": remote.url,
                "transferred_bytes": remote.transferred,
                "members": members,
            }
        )
        print(dataset, len(members), remote.transferred, flush=True)
    hashes = [row["sha256"] for source in provenance for row in source["members"]]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate image bytes in unlabeled pool")
    manifest = {
        "role": "reserved_unlabeled_adaptation_pixels",
        "used_for_training": False,
        "pose_annotations_included": False,
        "seed": options.seed,
        "groups": groups,
        "geometry_sha256": selections[0]["geometry_sha256"],
        "excluded_selection_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        ],
        "provenance": provenance,
        "limits": "Reserved only; not used in this run. Future evaluation must exclude this pool. No real-image pose labels were read for selection.",
    }
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(options.out, flush=True)


if __name__ == "__main__":
    main()
