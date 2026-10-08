"""Analyze existing C++ renders; optional offline ASR, never synthesis/downloads."""
import argparse
import hashlib
import json
import re
import wave
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory", type=Path, default=Path("artifacts/articulation/sweep"))
parser.add_argument("--word-index", type=int, default=3)
parser.add_argument("--recognize", action="store_true")
parser.add_argument("--checkpoint", type=Path, default=Path.home() / ".cache/whisper/small.en.pt")
args = parser.parse_args()
recognizer = None
if args.recognize:
    if not args.checkpoint.is_file(): parser.error("provide an existing local ASR checkpoint")
    import torch
    import whisper
    torch.set_num_threads(4)
    recognizer = whisper.load_model(str(args.checkpoint), device="cpu")

def words(text):
    return re.findall(r"[a-z']+", text.lower().replace("synthesise", "synthesize"))

def error_rate(expected, recognized):
    a, b = words(expected), words(recognized)
    row = list(range(len(b) + 1))
    for i, old in enumerate(a):
        current = [i + 1]
        for j, new in enumerate(b):
            current.append(min(current[-1] + 1, row[j + 1] + 1, row[j] + (old != new)))
        row = current
    return row[-1] / max(1, len(a))

def voiced_pitch(y):
    estimates = []
    for start in range(0, len(y) - 1200, 150):
        z = y[start:start + 1200]; z = z - z.mean()
        if np.sqrt(np.mean(z * z)) < .02: continue
        correlation = np.correlate(z, z, mode="full")[1199:]
        lags = np.arange(60, 344)
        norm = np.array([correlation[k] / max(1e-12, np.sqrt(np.dot(z[:-k], z[:-k]) * np.dot(z[k:], z[k:]))) for k in lags])
        peaks = np.flatnonzero((norm[1:-1] > norm[:-2]) & (norm[1:-1] >= norm[2:])) + 1
        if not len(peaks) or norm[peaks].max() < .7: continue
        chosen = peaks[norm[peaks] >= max(.7, norm[peaks].max() * .95)][0]
        estimates.append(24000 / lags[chosen])
    return float(np.median(estimates)) if estimates else None

rows = []
for path in sorted(args.directory.glob("*.wav")):
    d = json.loads(path.with_suffix(".diagnostics.json").read_text(encoding="utf-8"))
    with wave.open(str(path)) as audio:
        assert audio.getframerate() == 24000 and audio.getsampwidth() == 2 and audio.getnchannels() == 1
        y = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    assert len(y) == d["frames"] * 300 == d["samples"]
    assert len(d["f0_hz"]) == len(d["energy"]) == sum(d["durations"]) == d["frames"]
    span = d["words"][args.word_index]
    begin, end = span["frame_begin"], span["frame_end"]
    row = {"case": path.stem, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "seconds": len(y) / 24000, "rms": float(np.sqrt(np.mean(y * y))),
           "peak": float(np.max(np.abs(y))), "target_word": span["text"],
           "target_frames": end - begin, "target_requested_mean_hz": float(np.mean(d["f0_hz"][begin:end])),
           "target_estimated_voiced_median_hz": voiced_pitch(y[begin * 300:end * 300])}
    if recognizer:
        sampled = np.interp(np.arange(len(y) * 2 // 3) * 1.5, np.arange(len(y)), y).astype(np.float32)
        transcript = recognizer.transcribe(sampled, language="en", fp16=False, temperature=0, initial_prompt=None)["text"].strip()
        row.update(transcript=transcript, word_error_rate=error_rate(d["text"], transcript))
    rows.append(row)
    print(json.dumps(row), flush=True)
report = {"pitch_method": "normalized autocorrelation, 1200 window/150 hop, 70..400 Hz, RMS>.02, confidence>.7; approximate word-local voiced median",
          "asr": "existing local Whisper small.en, no initial prompt" if recognizer else None,
          "rows": rows}
(args.directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
