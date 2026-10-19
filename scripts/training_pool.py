"""Expand only SPEED+ synthetic TRAINING data; preserve every audit selection."""

import argparse
import hashlib
import json
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_benchmark import ROOT, SOURCES, RemoteZip


def select(rows, existing, excluded, count, seed):
    index = {row["filename"]: row for row in rows}
    if (
        len(index) != len(rows)
        or len(existing) != len(set(existing))
        or not set(existing) <= index.keys()
        or set(existing) & excluded
    ):
        raise ValueError(
            "Training identities are duplicate, absent or overlap an audit"
        )
    remaining = sorted(
        index.keys() - set(existing) - excluded,
        key=lambda name: hashlib.sha256(f"{seed}/{name}".encode()).hexdigest(),
    )
    names = list(existing) + remaining[: count - len(existing)]
    if len(names) != count or count < len(existing):
        raise ValueError("Invalid training-pool size")
    return [index[name] for name in names]


def verify_audit_separation(members, selections):
    training_hashes = {row["sha256"] for row in members}
    for selection in selections:
        for group, rows in selection["groups"].items():
            if group.endswith("/train"):
                continue
            dataset, domain, split = group.split("/")
            base = ROOT / "output/datasets" / dataset
            folder = (
                base / "speedplusv2" / domain / "images"
                if dataset == "speedplus"
                else base / domain / split / "images"
            )
            for row in rows:
                path = (folder / row["filename"]).resolve()
                if (
                    not path.is_relative_to(folder.resolve())
                    or hashlib.sha256(path.read_bytes()).hexdigest() in training_hashes
                ):
                    raise ValueError(
                        "Training image bytes overlap an audit or escape its image directory"
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--exclude-selection", type=Path, action="append", default=[])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "output/datasets/training-2000.json"
    )
    options = parser.parse_args()
    if not 500 <= options.count <= 5000 or not options.out.resolve().is_relative_to(
        ROOT / "output"
    ):
        parser.error("count must be 500..5000; output must remain under output/")
    original = json.loads((ROOT / "output/datasets/selection.json").read_text())
    exclusions = [
        original,
        *(json.loads(path.read_text()) for path in options.exclude_selection),
    ]
    if any(s["geometry_sha256"] != original["geometry_sha256"] for s in exclusions):
        raise ValueError("Audit geometry changed")
    excluded = {
        r["filename"]
        for s in exclusions
        for group, rows in s["groups"].items()
        if group.startswith("speedplus/synthetic/") and not group.endswith("/train")
        for r in rows
    }
    speed = ROOT / "output/datasets/speedplus/speedplusv2"
    rows = json.loads((speed / "synthetic/train.json").read_text())
    chosen = select(
        rows,
        [r["filename"] for r in original["groups"]["speedplus/synthetic/train"]],
        excluded,
        options.count,
        options.seed,
    )
    remote = RemoteZip(SOURCES["speedplus"], budget=768 * 1024**2)
    with zipfile.ZipFile(remote) as archive:
        entries = {item.filename: item for item in archive.infolist()}
    prefix = "speedplusv2/synthetic/images/"
    excluded_hashes = {
        r["sha256"]
        for s in exclusions
        for source in s.get("provenance", [])
        for r in source["members"]
    }

    def fetch(sample):
        name = prefix + sample["filename"]
        entry = entries[name]
        folder = ROOT / "output/datasets/speedplus/speedplusv2/synthetic/images"
        path = (ROOT / "output/datasets/speedplus" / name).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Training image path escapes its image directory")
        if not path.exists():
            data = remote.member(entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        data = path.read_bytes()
        if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
            raise ValueError("Training image failed archive size/CRC verification")
        digest = hashlib.sha256(data).hexdigest()
        if digest in excluded_hashes:
            raise ValueError("Training image bytes overlap an excluded audit")
        return {"member": name, "sha256": digest}

    with ThreadPoolExecutor(max_workers=6) as pool:
        members = []
        for row in pool.map(fetch, chosen):
            members.append(row)
            if len(members) % 100 == 0:
                print(
                    f"verified {len(members)}/{len(chosen)} training images", flush=True
                )
    if len({r["sha256"] for r in members}) != len(members):
        raise ValueError("Duplicate image bytes in training pool")
    verify_audit_separation(members, exclusions)
    report = {
        "seed": options.seed,
        "groups": {"speedplus/synthetic/train": chosen},
        "geometry_sha256": original["geometry_sha256"],
        "original_selection_sha256": hashlib.sha256(
            (ROOT / "output/datasets/selection.json").read_bytes()
        ).hexdigest(),
        "excluded_selection_sha256": [
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in options.exclude_selection
        ],
        "provenance": [
            {
                "source": remote.url,
                "transferred_bytes": remote.transferred,
                "members": members,
            }
        ],
        "scope": "Synthetic training split only. No audit images used for training or checkpoint selection.",
        "audit_hashes_disjoint": True,
    }
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(options.out, remote.transferred, flush=True)


if __name__ == "__main__":
    main()
