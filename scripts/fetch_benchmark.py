"""Fetch selected members of official public benchmark ZIPs with verified HTTP ranges.

Archives and images stay under ignored output/. Dataset licenses remain separate
from Apsis source licensing. Never fetch an entire archive as a range fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import ssl
import struct
import threading
import urllib.request
import zipfile
import zlib
from pathlib import Path

import truststore

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "speedplus": "https://stacks.stanford.edu/file/wv398fc4383/speedplusv2.zip",
    "shirt": "https://stacks.stanford.edu/file/zq716br5462/shirtv1.zip",
}


class RemoteZip(io.RawIOBase):
    def __init__(self, url: str, budget: int = 512 * 1024**2):
        self.url, self.budget, self.transferred, self.position = url, budget, 0, 0
        self.lock = threading.Lock()
        self.context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), context=self.context, timeout=40
        ) as response:
            self.size = int(response.headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (
            0 if whence == 0 else self.position if whence == 1 else self.size
        )
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        result = self.range(self.position, size)
        self.position += len(result)
        return result

    def range(self, start, size):
        with self.lock:
            if (
                size < 0
                or start < 0
                or start + size > self.size
                or self.transferred + size > self.budget
            ):
                raise ValueError("Invalid range or transfer budget exceeded")
            self.transferred += size
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{start+size-1}"}
        )
        with urllib.request.urlopen(
            request, context=self.context, timeout=40
        ) as response:
            if response.status != 206:
                raise ValueError(
                    "Server ignored byte range; full archive download refused"
                )
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("Truncated or oversized byte range")
        return data

    def member(self, item: zipfile.ZipInfo):
        if item.flag_bits & 1 or item.file_size > 64 * 1024**2:
            raise ValueError("Encrypted or oversized archive member")
        header = self.range(item.header_offset, 30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid local ZIP header")
        name_size, extra_size = struct.unpack_from("<HH", header, 26)
        data = self.range(
            item.header_offset + 30 + name_size + extra_size, item.compress_size
        )
        if item.compress_type == zipfile.ZIP_DEFLATED:
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(data, item.file_size + 1)
            if not decoder.eof:
                raise ValueError("Incomplete compressed member")
        elif item.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Unsupported ZIP compression")
        if len(data) != item.file_size or zlib.crc32(data) != item.CRC:
            raise ValueError("ZIP size/CRC verification failed")
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--members", nargs="*")
    parser.add_argument("--out", type=Path)
    options = parser.parse_args()
    out = (options.out or ROOT / "output" / "datasets" / options.dataset).resolve()
    if not out.is_relative_to(ROOT / "output"):
        parser.error("Destination must be under output/")
    out.mkdir(parents=True, exist_ok=True)
    remote = RemoteZip(SOURCES[options.dataset])
    with zipfile.ZipFile(remote) as archive:
        entries = archive.infolist()
    index = {item.filename: item for item in entries if not item.is_dir()}
    (out / "archive-index.json").write_text(json.dumps(list(index), indent=2))
    if options.list:
        print(
            json.dumps(
                {
                    "files": len(index),
                    "examples": list(index)[:12],
                    "metadata": [
                        n for n in index if not n.endswith((".jpg", ".png", ".jpeg"))
                    ][:80],
                    "index_bytes": remote.transferred,
                },
                indent=2,
            )
        )
        return
    provenance = []
    for name in options.members or []:
        target = (out / name).resolve()
        if not target.is_relative_to(out) or name not in index:
            raise ValueError("Unknown or escaping archive member")
        data = remote.member(index[name])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        provenance.append(
            {
                "member": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    (out / "download-provenance.json").write_text(
        json.dumps(
            {
                "source": remote.url,
                "transferred_bytes": remote.transferred,
                "members": provenance,
            },
            indent=2,
        )
    )
    print(f"Verified {len(provenance)} members; transferred {remote.transferred} bytes")


if __name__ == "__main__":
    main()
