#!/usr/bin/env python3
"""Stream a size-limited LibriTTS-R evaluation sample from OpenSLR.

The upstream archives are larger than the intended sample. This program reads a
gzip tar stream and stops after extracting approximately --budget-mb MiB. It
does not save the complete archive. Existing files are left intact on reruns.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import sys
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath
from typing import BinaryIO

DEFAULT_URLS = (
    "https://openslr.trmal.net/resources/141/test_clean.tar.gz",
    "https://openslr.elda.org/resources/141/test_clean.tar.gz",
    "https://openslr.magicdatatech.com/resources/141/test_clean.tar.gz",
)
ALLOWED_SUFFIXES = (".wav", ".normalized.txt", ".original.txt", ".tsv")


class CountingReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.source.read(size)
        self.bytes_read += len(chunk)
        return chunk


def safe_destination(root: Path, archive_name: str) -> Path:
    relative = PurePosixPath(archive_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive path: {archive_name}")
    destination = root.joinpath(*relative.parts).resolve()
    resolved_root = root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise ValueError(f"archive member escapes destination: {archive_name}")
    return destination


def wanted(name: str) -> bool:
    return name.endswith(ALLOWED_SUFFIXES)


def stream_sample(url: str, output: Path, budget_bytes: int, timeout: float) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "vocal-acoustics/0.1"})
    context = ssl.create_default_context()
    extracted = 0
    files = 0
    audio_files = 0
    started = time.time()

    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        counter = CountingReader(response)
        with tarfile.open(fileobj=counter, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not wanted(member.name):
                    continue
                if extracted > 0 and extracted + member.size > budget_bytes:
                    break
                destination = safe_destination(output, member.name)
                source = archive.extractfile(member)
                if source is None:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.stat().st_size == member.size:
                    # The stream still needs to consume the member, but avoid rewriting it.
                    while source.read(1024 * 1024):
                        pass
                else:
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    try:
                        with temporary.open("wb") as target:
                            shutil.copyfileobj(source, target, length=1024 * 1024)
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                extracted += member.size
                files += 1
                audio_files += member.name.endswith(".wav")
                if files % 100 == 0:
                    print(
                        f"extracted {extracted / 1024**2:.1f} MiB "
                        f"({audio_files} audio files)",
                        file=sys.stderr,
                    )

    return {
        "source_url": url,
        "dataset": "LibriTTS-R",
        "openslr_id": "SLR141",
        "license": "CC BY 4.0",
        "requested_budget_bytes": budget_bytes,
        "extracted_bytes": extracted,
        "downloaded_compressed_bytes": counter.bytes_read,
        "file_count": files,
        "audio_file_count": audio_files,
        "elapsed_seconds": round(time.time() - started, 3),
        "note": "Size-limited prefix sample; not an official upstream split.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/libritts-r-sample"))
    parser.add_argument("--budget-mb", type=float, default=250.0,
                        help="maximum extracted MiB (default: 250)")
    parser.add_argument("--url", action="append",
                        help="archive URL; repeat for fallback mirrors")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="network timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.budget_mb <= 0:
        raise SystemExit("--budget-mb must be positive")
    budget_bytes = int(args.budget_mb * 1024 * 1024)
    urls = tuple(args.url) if args.url else DEFAULT_URLS
    errors = []
    report = None
    for url in urls:
        try:
            print(f"trying {url}", file=sys.stderr)
            report = stream_sample(url, args.output, budget_bytes, args.timeout)
            break
        except (OSError, tarfile.TarError, ValueError) as error:
            errors.append(f"{url}: {error}")
            print(f"mirror failed: {error}", file=sys.stderr)
    if report is None:
        print("download failed on every mirror:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1

    manifest = args.output / "sample_manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
