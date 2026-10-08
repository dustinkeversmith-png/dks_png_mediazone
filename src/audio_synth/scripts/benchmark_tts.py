#!/usr/bin/env python3
"""Measure local CLI inference and process latency; no network or model downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

SHORT = "Hello from the explicit speech pipeline. We can control pitch and energy."
LONG = (
    "The purpose of this experiment is to compare how local speech synthesis handles a longer passage. "
    "The voice reads several sentences in sequence while the program records the time spent generating audio. "
    "We keep the model files on disk and use the same number of worker threads for each run. "
    "Pitch, energy, and duration controls belong to the acoustic stage, while a separate vocoder turns its output into a waveform. "
    "These measurements describe this computer and these exact exports. They do not predict the performance of a different trained model."
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=root / "build/onnx/Release")
    parser.add_argument("--piper-assets", type=Path, default=root / "model_assets/piper")
    parser.add_argument("--voice", default="en_US-lessac-medium")
    parser.add_argument("--explicit-assets", type=Path)
    parser.add_argument("--include-fixture", action="store_true", help="Measure untrained test graphs separately from trained speech")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=root / "artifacts/benchmarks")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [("piper", "trained", args.bin_dir / "piper-tts.exe", args.piper_assets,
             ["--voice", args.voice], [args.voice + ".onnx", args.voice + ".properties", args.voice + ".tokens.tsv", "cmudict.dict"])]
    if args.explicit_assets:
        jobs.append(("explicit", "user-supplied exports; verify training provenance", args.bin_dir / "explicit-tts.exe",
                     args.explicit_assets, [], ["acoustic_generator.onnx", "vocoder_hifigan.onnx", "tokens.tsv", "cmudict.dict"]))
        for name in ("prosody_predictor.onnx", "pipeline.properties", "download_manifest.json"):
            if (args.explicit_assets / name).is_file(): jobs[-1][-1].append(name)
    if args.include_fixture:
        # The tiny graph accepts arbitrary IDs but has no learned token embeddings.
        # A complete frontend lets identical texts exercise transport/shape overhead.
        fixture = root / "build/benchmark-fixture"
        fixture.mkdir(parents=True, exist_ok=True)
        for name in ("acoustic_generator.onnx", "vocoder_hifigan.onnx"):
            shutil.copyfile(root / "tests/fixtures/explicit" / name, fixture / name)
        shutil.copyfile(args.piper_assets / "cmudict.dict", fixture / "cmudict.dict")
        shutil.copyfile(args.piper_assets / (args.voice + ".tokens.tsv"), fixture / "tokens.tsv")
        jobs.append(("explicit-fixture", "untrained test graphs; output is not speech", args.bin_dir / "explicit-tts.exe",
                     fixture, [], ["acoustic_generator.onnx", "vocoder_hifigan.onnx", "tokens.tsv", "cmudict.dict"]))
    result = {
        "utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_processors": os.cpu_count(), "threads": args.threads,
        "repeats": args.repeats, "warmup_processes_per_case": 1,
        "method": "Fresh process/session per sample. One discarded process warms OS file caches; not steady-state session latency. wall_ms includes loading, frontend, inference, WAV and diagnostics.",
        "explicit_trained_status": "measured user-supplied assets" if args.explicit_assets else "unavailable: no compatible trained acoustic/vocoder pair supplied",
        "matcha_status": "removed at user request; no current measurements",
        "cases": [],
    }
    for engine, provenance, exe, assets, extra, asset_names in jobs:
        if not exe.is_file():
            raise FileNotFoundError(exe)
        asset_records = {name: {"bytes": (assets / name).stat().st_size, "sha256": sha256(assets / name)} for name in asset_names}
        for workload, text in (("short", SHORT), ("paragraph", LONG)):
            samples = []
            for iteration in range(args.repeats + 1):
                output = args.output / f"{engine}-{workload}-{iteration}.wav"
                command = [str(exe), "--assets", str(assets), "--text", text, "--threads", str(args.threads),
                           "--output", str(output), *extra]
                start = time.perf_counter()
                completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
                wall_ms = (time.perf_counter() - start) * 1000
                if completed.returncode:
                    raise RuntimeError(completed.stderr)
                diagnostics = json.loads(output.with_suffix(".diagnostics.json").read_text(encoding="utf-8"))
                with wave.open(str(output)) as audio:
                    seconds = audio.getnframes() / audio.getframerate()
                    sample_rate = audio.getframerate()
                acoustic = diagnostics.get("acoustic_ms")
                vocoder = diagnostics.get("vocoder_ms")
                prosody = diagnostics.get("prosody_ms", 0)
                inference = diagnostics["inference_ms"] if engine == "piper" else prosody + acoustic + vocoder
                sample = {"wall_ms": wall_ms, "inference_ms": inference, "prosody_ms": prosody, "acoustic_ms": acoustic, "vocoder_ms": vocoder,
                          "audio_seconds": seconds, "sample_rate": sample_rate,
                          "inference_rtf": inference / 1000 / seconds, "wall_rtf": wall_ms / 1000 / seconds,
                          "chunks": diagnostics.get("chunks", 1), "frames": diagnostics.get("frames"),
                          "tokens": diagnostics.get("tokens", diagnostics.get("phoneme_tokens")),
                          "missing_model_symbols": diagnostics.get("missing_model_symbols", 0),
                          "fallback_words": diagnostics["fallback_words"], "command": command}
                if iteration: samples.append(sample)
            medians = {key: statistics.median(sample[key] for sample in samples) for key in
                       ("wall_ms", "inference_ms", "audio_seconds", "inference_rtf", "wall_rtf")}
            for key in ("prosody_ms", "acoustic_ms", "vocoder_ms"):
                if samples[0][key] is not None: medians[key] = statistics.median(sample[key] for sample in samples)
            result["cases"].append({"engine": engine, "provenance": provenance, "workload": workload,
                                    "text": text, "characters": len(text), "words": len(text.split()),
                                    "assets": str(assets.resolve()), "asset_files": asset_records,
                                    "binary_sha256": sha256(exe), "samples": samples, "median": medians})
            print(f"{engine}/{workload}: inference={medians['inference_ms']:.2f} ms, wall={medians['wall_ms']:.2f} ms, inference RTF={medians['inference_rtf']:.4f}")
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
