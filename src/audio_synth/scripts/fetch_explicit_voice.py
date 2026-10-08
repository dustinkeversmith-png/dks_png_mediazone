#!/usr/bin/env python3
"""Fetch and export a real 24 kHz English FastSpeech2/HiFi-GAN explicit voice.

Ordinary HTTPS downloads are sequential and cached. Tiny statistics/config files
are read with standard ZIP byte ranges instead of downloading a 400 MB training
snapshot. Runtime synthesis remains entirely local.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from fetch_piper_voice import CMUDICT_URL, download

BASE = "https://paddlespeech.cdn.bcebos.com/Parakeet/released_models"
ARCHIVES = {
    "fastspeech2_vctk_onnx_1.1.0.zip": ("fastspeech2", "daa47612d643556514e5b93af9032db345eb8e3fecd7771c2af4f7585a08e0df"),
    "hifigan_vctk_onnx_1.1.0.zip": ("hifigan", "a0e24082f409cd67c9cd96ed10e8d059a98b6f752dc6a8b83c05d502d2ce30a5"),
}
STATS_URL = BASE + "/fastspeech2/fastspeech2_vctk_ckpt_1.2.0.zip"
STAT_HASHES = {
    "pitch_stats.npy": "522903b544c0c6a88f92861aac7ef889e54d499a4be40e5813820a3f0af94b78",
    "energy_stats.npy": "f321ec03e9c7c3816ce9a55e3e4eb1b9680d0b4d6dc62a41b28ec74bbd0185cf",
    "speech_stats.npy": "fa0e318d4387d94d3ad1f188d0da5c45309ea5e3b8e301f9d27c4c2f4969be84",
    "default.yaml": "a38ae9f2f6bb0d72e47931bd18c75e2005bd25c73f4d127de0266c66e339d207",
}


def digest(path: Path) -> str:
    with path.open("rb") as source: return hashlib.file_digest(source, "sha256").hexdigest()


class RemoteZip(io.RawIOBase):
    def __init__(self, url: str):
        self.url, self.position = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
        self.requests, self.received = 0, 0

    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2): raise ValueError("invalid seek")
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if target < 0: raise ValueError("negative seek")
        self.position = target
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0: return b""
        request = urllib.request.Request(self.url, headers={"User-Agent": "vocal-acoustics/0.2",
            "Range": f"bytes={self.position}-{self.position + size - 1}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206: raise RuntimeError("publisher did not honor standard HTTP byte ranges")
            data = response.read(size + 1)
        if len(data) != size: raise RuntimeError("incomplete archive byte range")
        self.position += size; self.requests += 1; self.received += size
        return data


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "model_assets/explicit_neural")
    parser.add_argument("--cache", type=Path, default=root / "third_party/downloads")
    parser.add_argument("--source-dir", type=Path, default=root / "build/pretrained-inspect/vctk")
    args = parser.parse_args()
    # Validate export dependency before downloading weights.
    from export_paddlespeech_explicit import export
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for filename, (kind, expected) in ARCHIVES.items():
        url = f"{BASE}/{kind}/{filename}"
        archive = args.cache / filename
        if download(url, archive) != expected: raise RuntimeError(f"release checksum mismatch: {filename}")
        provenance[filename] = {"url": url, "sha256": expected}
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                name = Path(member.filename).name
                if name in ("fastspeech2_vctk.onnx", "hifigan_vctk.onnx", "phone_id_map.txt", "speaker_id_map.txt"):
                    destination = args.source_dir / name
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    with bundle.open(member) as source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    os.replace(temporary, destination)
    missing = [name for name, expected in STAT_HASHES.items()
               if not (args.source_dir / name).is_file() or digest(args.source_dir / name) != expected]
    if missing:
        remote = RemoteZip(STATS_URL)
        with zipfile.ZipFile(remote) as bundle:
            for name in missing:
                candidates = [item for item in bundle.infolist() if Path(item.filename).name == name]
                if len(candidates) != 1 or candidates[0].file_size > 100_000:
                    raise RuntimeError(f"unexpected publisher statistics entry: {name}")
                data = bundle.read(candidates[0])
                if hashlib.sha256(data).hexdigest() != STAT_HASHES[name]: raise RuntimeError(f"statistics checksum mismatch: {name}")
                (args.source_dir / name).write_bytes(data)
        print(f"Statistics: {remote.received} bytes in {remote.requests} ordinary range requests")
    export(args.source_dir, args.output)
    dictionary = args.output / "cmudict.dict"
    shared = root / "model_assets/piper/cmudict.dict"
    dictionary_source = "existing local dictionary"
    if not dictionary.exists() and shared.is_file():
        shutil.copyfile(shared, dictionary)
        dictionary_source = str(shared)
    elif not dictionary.exists():
        download(CMUDICT_URL, dictionary)
        dictionary_source = CMUDICT_URL
    shutil.copyfile(root / "docs/EXPLICIT_VOICE_MODEL_CARD.md", args.output / "MODEL_CARD.md")
    for name in ("pitch_stats.npy", "energy_stats.npy", "speech_stats.npy", "default.yaml", "phone_id_map.txt"):
        shutil.copyfile(args.source_dir / name, args.output / name)
    sources = {name: {"url": STATS_URL, "sha256": value} for name, value in STAT_HASHES.items()}
    manifest = {"model": "PaddleSpeech FastSpeech2 VCTK + HiFi-GAN VCTK", "archives": provenance,
                "statistics": sources, "dictionary_source": dictionary_source,
                "sample_rate": 24000, "hop_length": 300, "frontend": "CMUdict ARPAbet, publisher IDs",
                "sha256": {path.name: digest(path) for path in args.output.iterdir() if path.is_file() and path.name != "download_manifest.json"},
                "export": "Retained trained weights; exact duration Gather expansion, frame-level 1x1 variance embeddings, separate trained predictor; no retraining.",
                "source_docs": "https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md"}
    (args.output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Trained explicit assets ready: {args.output.resolve()}")


if __name__ == "__main__": main()
