#!/usr/bin/env python3
"""Fetch the compact public English Matcha-TTS + HiFi-GAN v2 pair sequentially."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

from fetch_piper_voice import CMUDICT_URL, download

ACOUSTIC_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2"
ACOUSTIC_ARCHIVE_SHA256 = "ea75702da7456a8b1874728278a835220dc8a26f4e8bd93c83bf53dc27679845"
VOCODER_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx"
VOCODER_SHA256 = "a41d404cce7924493540238da5b30a4bc14b6ddaf1a37f3c79fa4f59548c19f0"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "model_assets/matcha")
    parser.add_argument("--cache", type=Path, default=root / "third_party/downloads")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.cache / "matcha-icefall-en_US-ljspeech.tar.bz2"
    digest = download(ACOUSTIC_URL, archive)
    if digest != ACOUSTIC_ARCHIVE_SHA256:
        raise RuntimeError("Matcha archive checksum differs from the pinned public release; inspect it before use")
    # Extract only known regular files, without accepting archive paths or links.
    filenames = {"model-steps-3.onnx": "acoustic_generator.onnx", "tokens.txt": "tokens.txt", "README.md": "UPSTREAM_README.md"}
    with tarfile.open(archive, "r:bz2") as bundle:
        for original, target in filenames.items():
            members = [member for member in bundle.getmembers() if member.isfile() and Path(member.name).name == original]
            if len(members) != 1:
                raise RuntimeError(f"unexpected archive layout for {original}")
            destination = args.output / target
            if not destination.exists() or destination.stat().st_size != members[0].size:
                stream = bundle.extractfile(members[0])
                if stream is None: raise RuntimeError(f"cannot extract {original}")
                temporary = destination.with_suffix(destination.suffix + ".part")
                try:
                    with stream, temporary.open("wb") as output:
                        shutil.copyfileobj(stream, output)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
    vocoder_hash = download(VOCODER_URL, args.output / "vocoder_hifigan.onnx")
    if vocoder_hash != VOCODER_SHA256:
        raise RuntimeError("vocoder checksum differs from the pinned HiFi-GAN v2 download")
    local_dictionary = root / "model_assets/piper/cmudict.dict"
    dictionary = args.output / "cmudict.dict"
    if local_dictionary.exists() and not dictionary.exists():
        shutil.copyfile(local_dictionary, dictionary)
    else:
        download(CMUDICT_URL, dictionary)
    # Convert the publisher's single-codepoint vocabulary into the shared frontend's format.
    with (args.output / "tokens.tsv").open("w", encoding="utf-8", newline="\n") as output:
        output.write("# Unicode codepoint\tmodel ID (publisher Matcha vocabulary)\n")
        for line in (args.output / "tokens.txt").read_text(encoding="utf-8").splitlines():
            symbol, value = line.rsplit(" ", 1)
            if len(symbol) != 1:
                raise RuntimeError(f"unsupported multi-codepoint token: {symbol!r}")
            output.write(f"{ord(symbol):04X}\t{int(value)}\n")
    hashes = {}
    for path in args.output.iterdir():
        if path.is_file() and path.name != "download_manifest.json":
            with path.open("rb") as stream: hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"model": "matcha-icefall-en_US-ljspeech", "vocoder": "hifigan_v2",
                "sources": [ACOUSTIC_URL, VOCODER_URL, CMUDICT_URL], "archive_sha256": digest,
                "vocoder_sha256": vocoder_hash, "sha256": hashes,
                "native_sample_rate": 22050, "output_sample_rate": 24000,
                "note": "Original ONNX graphs; no retraining or graph rewrites. CMUdict frontend used by this runner."}
    (args.output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Matcha assets: {args.output.resolve()}")


if __name__ == "__main__":
    main()
