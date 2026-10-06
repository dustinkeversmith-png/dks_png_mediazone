"""Small real-noise evaluation sets for the caption engines (<= 50 MB each).

Standard library + the repository's ffmpeg only. Writes NNNNNN.wav + NNNNNN.txt
folders that `captions --eval <dir>` reads directly:

  data/audio/eval/noisy_librispeech_other/   LibriSpeech test-other (harder
      speakers) with MUSAN noise pre-mixed at a random 0-20 dB SNR
      (zhaoyang9425/NoisyLibriSpeechDataset-MUSAN); references from
      openslr/librispeech_asr.
  data/audio/eval/musan_noise/{clean,10dB,5dB,0dB}/   synthesized sentences
      with MUSAN noise (noisy-alpaca-test/MUSAN-noise-audio-only)
  data/audio/eval/musan_music/{clean,10dB,5dB,0dB}/   the same with MUSAN
      music (JST-SUPERB/noisy-alpaca-test-MUSAN-music). Where the dataset
      ships Whisper transcriptions they are saved as NNNNNN.whisper-*.txt
      for comparison (`captions --eval <dir> --score-hyp whisper-large-v3`).

Usage (repository root): python src/captions/models/int8_zip/scripts/fetch_noisy_eval.py
Rerunning skips sets that are complete.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
EVAL = os.path.join(REPO, "data", "audio", "eval")
FFMPEG = os.path.join(REPO, "dependencies", "ffmpeg", "bin", "ffmpeg.exe")
if not os.path.exists(FFMPEG):
    FFMPEG = "ffmpeg"
CAP_BYTES = 50 * 1024 * 1024
VIEWER = "https://datasets-server.huggingface.co"


def get(url, binary=False, tries=8):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
                return data if binary else json.loads(data)
        except Exception as e:  # network hiccups and viewer rate limits
            if attempt == tries - 1:
                raise RuntimeError(f"{url}: {e}")
            limited = getattr(e, "code", None) == 429
            time.sleep(30 * (attempt + 1) if limited else 2 + 3 * attempt)


def to_wav(data, out):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(data)
        src = f.name
    try:
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", out], check=True)
    finally:
        os.remove(src)
    return os.path.getsize(out)


def rows(dataset, config, split, offset, length=100):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                "offset": offset, "length": length})
    return get(f"{VIEWER}/rows?{q}")


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)


def noisy_librispeech():
    out = os.path.join(EVAL, "noisy_librispeech_other")
    if os.path.isdir(out) and dir_bytes(out) > CAP_BYTES * 0.9:
        print(f"{out}: present"); return
    os.makedirs(out, exist_ok=True)
    # References for every test-other utterance id (text only, no audio).
    refs = {}
    first = rows("openslr/librispeech_asr", "other", "test", 0, 1)
    for offset in range(0, first["num_rows_total"], 100):
        for r in rows("openslr/librispeech_asr", "other", "test", offset)["rows"]:
            refs[r["row"]["id"]] = r["row"]["text"]
    api = "https://huggingface.co/api/datasets/zhaoyang9425/NoisyLibriSpeechDataset-MUSAN/tree/main/"
    raw = "https://huggingface.co/datasets/zhaoyang9425/NoisyLibriSpeechDataset-MUSAN/resolve/main/"
    root = "LibriNoisySpeech_MUSAN/test-other"
    speakers = [e["path"] for e in get(api + root) if e["type"] == "directory"]
    # Round-robin over speakers, two utterances at a time, until the cap.
    queues = []
    for s in speakers:
        files = []
        for chapter in (e["path"] for e in get(api + s) if e["type"] == "directory"):
            files += [e["path"] for e in get(api + chapter) if e["path"].endswith(".flac")]
        queues.append(sorted(files))
    total, n = dir_bytes(out), 0
    while total < CAP_BYTES and any(queues):
        for q in queues:
            for _ in range(2):
                if not q or total >= CAP_BYTES:
                    break
                path = q.pop(0)
                uid = os.path.basename(path)[:-5]
                if uid not in refs:
                    continue
                stem = os.path.join(out, f"{n:06d}")
                total += to_wav(get(raw + urllib.parse.quote(path), binary=True), stem + ".wav")
                with open(stem + ".txt", "w", encoding="utf-8") as f:
                    f.write(refs[uid])
                n += 1
        print(f"noisy_librispeech_other: {n} clips, {total / 1e6:.1f} MB", flush=True)


def alpaca(name, dataset, count=110):
    out = os.path.join(EVAL, name)
    if os.path.isdir(out) and dir_bytes(out) > CAP_BYTES * 0.8:
        print(f"{out}: present"); return
    columns = {"clean": "clean_audio", "10dB": "noisy_10dB", "5dB": "noisy_5dB", "0dB": "noisy_0dB"}
    for sub in columns:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    total = rows(dataset, "default", "test", 0, 1)["num_rows_total"]
    step = max(100, total // max(1, count // 10))
    n, size = 0, dir_bytes(out)
    for offset in range(0, total, step):
        for r in rows(dataset, "default", "test", offset)["rows"][:10]:
            row = r["row"]
            text = (row.get("speech_input") or "").strip()
            if len(text.split()) < 3 or n >= count or size >= CAP_BYTES:
                continue
            for sub, col in columns.items():
                stem = os.path.join(out, sub, f"{n:06d}")
                size += to_wav(get(row[col][0]["src"], binary=True), stem + ".wav")
                with open(stem + ".txt", "w", encoding="utf-8") as f:
                    f.write(text)
                prefix = "clean_audio" if sub == "clean" else col
                for model in ("whisper-small.en", "whisper-medium.en", "whisper-large-v3"):
                    hyp = row.get(f"{prefix}_transcription_{model}")
                    if hyp is not None:
                        with open(f"{stem}.{model}.txt", "w", encoding="utf-8") as f:
                            f.write(hyp)
            n += 1
        print(f"{name}: {n} sentences x 4 versions, {size / 1e6:.1f} MB", flush=True)
        if n >= count or size >= CAP_BYTES:
            break


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"noisy_librispeech_other", "musan_noise", "musan_music"}
    if "musan_noise" in which:
        alpaca("musan_noise", "noisy-alpaca-test/MUSAN-noise-audio-only")
    if "noisy_librispeech_other" in which:
        noisy_librispeech()
    if "musan_music" in which:
        alpaca("musan_music", "JST-SUPERB/noisy-alpaca-test-MUSAN-music")
