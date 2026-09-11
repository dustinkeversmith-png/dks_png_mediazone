#!/usr/bin/env python3
"""Fetch a Piper voice, CMUdict, and optional ONNX Runtime C++ SDK."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

VOICES = {
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "en_US-hfc_male-medium": "en/en_US/hfc_male/medium",
}
VOICE_REPOSITORY = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
ORT_VERSION = "1.26.0"
ORT_URL = (
    f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/"
    f"onnxruntime-win-x64-{ORT_VERSION}.zip"
)
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"


def download(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    if path.exists():
        with path.open("rb") as existing:
            for block in iter(lambda: existing.read(1024 * 1024), b""):
                digest.update(block)
        print(f"using existing {path}")
        return digest.hexdigest()
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vocal-acoustics/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as target:
            total = int(source.headers.get("Content-Length", "0"))
            received = 0
            while block := source.read(1024 * 1024):
                target.write(block)
                digest.update(block)
                received += len(block)
                if total:
                    print(f"{path.name}: {received * 100 / total:.0f}%", end="\r")
        os.replace(temporary, path)
        print(f"{path.name}: complete")
        return digest.hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def prepare_voice(output: Path, voice: str) -> dict[str, str]:
    model = output / f"{voice}.onnx"
    config_path = output / f"{voice}.onnx.json"
    card = output / f"{voice}.MODEL_CARD"
    root = f"{VOICE_REPOSITORY}/{VOICES[voice]}"
    hashes = {
        model.name: download(f"{root}/{model.name}?download=true", model),
        config_path.name: download(f"{root}/{config_path.name}?download=true", config_path),
        card.name: download(f"{root}/MODEL_CARD?download=true", card),
    }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with (output / f"{voice}.tokens.tsv").open("w", encoding="utf-8", newline="\n") as tokens:
        tokens.write("# Unicode codepoint\tcomma-separated model IDs\n")
        for symbol, ids in config["phoneme_id_map"].items():
            codepoint = ord(symbol)
            tokens.write(f"{codepoint:04X}\t{','.join(str(value) for value in ids)}\n")
    inference = config.get("inference", {})
    properties = {
        "sample_rate": config["audio"]["sample_rate"],
        "num_speakers": config["num_speakers"],
        "noise_scale": inference.get("noise_scale", 0.667),
        "length_scale": inference.get("length_scale", 1.0),
        "noise_w": inference.get("noise_w", 0.8),
        "voice": config.get("espeak", {}).get("voice", "en-us"),
    }
    (output / f"{voice}.properties").write_text(
        "".join(f"{key}={value}\n" for key, value in properties.items()), encoding="utf-8"
    )
    return hashes


def prepare_ort(destination: Path, cache: Path) -> tuple[Path, str]:
    archive = cache / f"onnxruntime-win-x64-{ORT_VERSION}.zip"
    digest = download(ORT_URL, archive)
    expected = destination / "include" / "onnxruntime_cxx_api.h"
    if not expected.exists():
        extraction = destination.parent / (destination.name + ".extracting")
        if extraction.exists():
            shutil.rmtree(extraction)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extraction)
        roots = [path for path in extraction.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("unexpected ONNX Runtime archive layout")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(roots[0]), destination)
        shutil.rmtree(extraction)
    return destination, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--ort-root", type=Path, default=Path("third_party/onnxruntime"))
    parser.add_argument("--cache", type=Path, default=Path("third_party/downloads"))
    parser.add_argument("--voice", action="append", choices=sorted(VOICES),
                        help="voice to fetch; repeat to fetch multiple (default: two contrasting voices)")
    parser.add_argument("--skip-ort", action="store_true")
    args = parser.parse_args()

    selected_voices = args.voice or list(VOICES)
    hashes = {"cmudict.dict": download(CMUDICT_URL, args.models / "cmudict.dict")}
    for voice in selected_voices:
        hashes.update(prepare_voice(args.models, voice))
    ort_root = None
    if not args.skip_ort:
        ort_root, ort_hash = prepare_ort(args.ort_root, args.cache)
        hashes[f"onnxruntime-win-x64-{ORT_VERSION}.zip"] = ort_hash
    manifest = {
        "voices": selected_voices,
        "voice_repository": "https://huggingface.co/rhasspy/piper-voices",
        "cmudict_repository": "https://github.com/cmusphinx/cmudict",
        "onnxruntime_version": None if args.skip_ort else ORT_VERSION,
        "onnxruntime_root": None if ort_root is None else str(ort_root.resolve()),
        "sha256": hashes,
    }
    manifest_path = args.models / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")
    if ort_root:
        print(f"configure with -DVA_ENABLE_ONNX_RUNTIME=ON -DONNXRUNTIME_ROOT={ort_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
