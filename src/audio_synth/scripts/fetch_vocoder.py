#!/usr/bin/env python3
"""Fetch only the small public HiFi-GAN v2 ONNX vocoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fetch_piper_voice import download

VOCODER_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx"
VOCODER_SHA256 = "a41d404cce7924493540238da5b30a4bc14b6ddaf1a37f3c79fa4f59548c19f0"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "model_assets/vocoders")
    args = parser.parse_args()
    model = args.output / "hifigan_v2.onnx"
    digest = download(VOCODER_URL, model)
    if digest != VOCODER_SHA256:
        raise RuntimeError("vocoder checksum differs from the pinned public export")
    manifest = {
        "model": "hifigan_v2", "source": VOCODER_URL,
        "sha256": {model.name: digest}, "native_sample_rate": 22050,
        "hop_length": 256, "mel_bins": 80,
        "note": "Original pretrained vocoder only. Requires compatible acoustic mel; not a 24 kHz export.",
    }
    (args.output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Vocoder asset: {model.resolve()}")


if __name__ == "__main__":
    main()
