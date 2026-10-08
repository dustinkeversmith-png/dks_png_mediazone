#!/usr/bin/env python3
"""Stage matching acoustic/vocoder exports and token vocabulary under model_assets.

Sources can be local files or HTTPS URLs. No pretrained model URLs are assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from fetch_piper_voice import CMUDICT_URL, download


def stage(source: str, destination: Path) -> str:
    if source.startswith("https://"):
        if destination.exists():
            raise FileExistsError(f"refusing to reuse {destination} for a possibly different source; choose a new --output")
        return download(source, destination)
    original = Path(source).resolve()
    if not original.is_file():
        raise FileNotFoundError(f"model source file does not exist: {original}")
    if original != destination.resolve():
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}; choose a new --output")
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            shutil.copyfile(original, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    with destination.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", required=True, help="compatible acoustic-generator ONNX path or HTTPS URL")
    parser.add_argument("--vocoder", required=True, help="matching 24 kHz vocoder ONNX path or HTTPS URL")
    parser.add_argument("--tokens", required=True, help="the acoustic export's IPA/special token map TSV path or HTTPS URL")
    parser.add_argument("--dictionary", default=CMUDICT_URL, help="CMUdict path or HTTPS URL")
    parser.add_argument("--output", type=Path, default=root / "model_assets/explicit_neural")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {"acoustic_generator.onnx": args.acoustic, "vocoder_hifigan.onnx": args.vocoder,
               "tokens.tsv": args.tokens, "cmudict.dict": args.dictionary}
    hashes = {name: stage(source, args.output / name) for name, source in sources.items()}
    manifest = {"sources": sources, "sha256": hashes,
                "note": "User-supplied exports; verify training, licenses, vocabulary, mel convention and hop length match."}
    (args.output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Staged explicit pipeline assets: {args.output.resolve()}")


if __name__ == "__main__":
    main()
