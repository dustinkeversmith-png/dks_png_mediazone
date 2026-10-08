"""Independently recognize generated samples using an existing Whisper checkpoint.

Requires locally installed openai-whisper, torch and numpy. This script never
downloads a recognition model and does not give the recognizer a text prompt.
"""
import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import torch
import whisper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=Path.home() / ".cache/whisper/small.en.pt")
parser.add_argument("--artifacts", type=Path, default=Path("artifacts/explicit_neural"))
args = parser.parse_args()
if not args.checkpoint.is_file():
    parser.error("provide an existing local Whisper checkpoint with --checkpoint")
torch.set_num_threads(4)
model = whisper.load_model(str(args.checkpoint), device="cpu")
rows = []
for emotion in ("neutral", "excited", "somber"):
    path = args.artifacts / (emotion + ".wav")
    # Whisper accepts a 16 kHz array; linear interpolation is sufficient for
    # this independent recognition check and needs no ffmpeg executable.
    with wave.open(str(path)) as audio:
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        sr = audio.getframerate()
        waveform = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    samples = np.interp(np.arange(int(len(waveform) * 16000 / sr)) * sr / 16000,
                        np.arange(len(waveform)), waveform).astype(np.float32)
    transcript = model.transcribe(samples, language="en", fp16=False, temperature=0)["text"].strip()
    normalized = " ".join(transcript.lower().replace("synthesise", "synthesize").strip(". ").split())
    assert normalized == "we synthesize a clear acoustic voice", transcript
    row = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "emotion": emotion, "sample_rate": sr, "seconds": len(waveform) / sr,
           "transcript": transcript}
    rows.append(row)
    print(row, flush=True)
report = {"asr_model": "local Whisper small.en", "checkpoint": str(args.checkpoint),
          "initial_prompt": None, "expected": "We synthesize a clear acoustic voice.",
          "normalization": "case, punctuation and synthesize/synthesise spelling", "results": rows}
(args.artifacts / "intelligibility.json").write_text(json.dumps(report, indent=2) + "\n")
