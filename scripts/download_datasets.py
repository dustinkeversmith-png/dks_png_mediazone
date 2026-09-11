#!/usr/bin/env python3
"""
Download three benchmark datasets for speech analysis:

  1. Singular Digits            - Google Speech Commands digits (0-9)
  2. Isolated Words & Phonetics - TIMIT (audio + phoneme/word alignments)
  3. Continuous Sentences       - LibriSpeech test-clean

Everything is streamed to data/audio/<tier>/ with a per-tier size budget
(250 MB by default) and written as 16 kHz mono 16-bit WAV.

Usage:
  python scripts/download_datasets.py                     # all three tiers
  python scripts/download_datasets.py --tiers digits      # one tier
  python scripts/download_datasets.py --max-mb 50         # smaller budget
  python scripts/download_datasets.py --list-sources      # show mirrors used
  python scripts/download_datasets.py --source fsdd       # force a mirror
  python scripts/download_datasets.py --overwrite         # re-download a tier

Downloads resume: re-running tops a tier back up to its budget instead of
starting over. Clips are balanced by default - each digit gets an equal share
of the budget, and TIMIT/LibriSpeech take a few clips per speaker - because
these corpora are stored grouped, so an unbalanced budget-limited run would
collect a single digit or a couple of voices. Use --no-balance to take them in
stream order (faster, less diverse).

Implementation notes (why the mirrors below and not the canonical repos):
  * datasets >= 4.0 removed script-based loading, so `google/speech_commands`
    and `timit_asr` (both script-only on the Hub) can no longer be loaded at
    all. The sources here are parquet mirrors of the same audio.
  * datasets >= 4.0 also decodes audio through torchcodec. We disable decoding
    (Audio(decode=False)) and decode the raw bytes with soundfile instead, so
    no torchcodec/ffmpeg install is required.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

try:
    import numpy as np
except ImportError:
    print("Error: 'numpy' not found. Install with: pip install -r requirements_audio.txt")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("Error: 'soundfile' not found. Install with: pip install -r requirements_audio.txt")
    sys.exit(1)

try:
    from datasets import Audio, load_dataset
except ImportError:
    print("Error: 'datasets' not found. Install with: pip install -r requirements_audio.txt")
    sys.exit(1)

try:
    import librosa
except ImportError:
    librosa = None

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None


TARGET_SR = 16000
DEFAULT_OUT = Path("data/audio")

# These corpora are stored grouped (digits by class, TIMIT/LibriSpeech by
# speaker), so a budget-limited run would otherwise collect one digit or a
# handful of speakers. Digits get an equal byte share each; the speech tiers
# take at most this many clips per speaker before moving on. Override with
# --per-class / --per-speaker (0 = no cap).
PER_SPEAKER_DEFAULT = {"timit": 4, "librispeech": 15}

WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


# --------------------------------------------------------------------------
# Source registry
# --------------------------------------------------------------------------

@dataclass
class Source:
    """One candidate mirror for a tier. Tried in order until one yields data."""
    id: str
    repo: str
    split: str
    config: Optional[str] = None
    audio_col: str = "audio"
    label_col: Optional[str] = None
    text_col: Optional[str] = None
    speaker_col: Optional[str] = None
    note: str = ""


SOURCES: dict[str, list[Source]] = {
    "digits": [
        Source(
            id="speech_commands_digits",
            repo="mazkooleg/0-9up_google_speech_commands_augmented_raw",
            split="test",
            label_col="label",
            note="Speech Commands digit clips, 16 kHz (~112 MB shard, fastest)",
        ),
        Source(
            id="speech_commands_full",
            repo="soerenray/speech_commands_enriched_and_annotated",
            split="train",
            label_col="Annotated Labels",
            speaker_col="speaker_id",
            note="Full Speech Commands v0.02 mirror; scans ~2 GB to reach every digit",
        ),
        Source(
            id="fsdd",
            repo="mteb/free-spoken-digit-dataset",
            split="train",
            label_col="label",
            note="Free Spoken Digit Dataset, 8 kHz upsampled, 6 speakers (fallback)",
        ),
    ],
    "timit": [
        Source(
            id="timit_asr",
            repo="kylelovesllms/timit_asr",
            split="train",
            text_col="text",
            speaker_col="speaker_id",
            note="TIMIT mirror with phonetic_detail + word_detail alignments",
        ),
        Source(
            id="timit_alt",
            repo="nh0znoisung/timit",
            split="train",
            text_col="text",
            speaker_col="speaker_id",
            note="Alternate TIMIT mirror (same alignments, different layout)",
        ),
    ],
    "librispeech": [
        Source(
            id="librispeech_clean",
            repo="openslr/librispeech_asr",
            config="clean",
            split="test",
            text_col="text",
            speaker_col="speaker_id",
            note="LibriSpeech test-clean (FLAC, 16 kHz)",
        ),
        Source(
            id="librispeech_clean_mirror",
            repo="librispeech_asr",
            config="clean",
            split="test",
            text_col="text",
            speaker_col="speaker_id",
            note="Same data under the legacy repo id",
        ),
    ],
}

TIER_ORDER = ["digits", "timit", "librispeech"]

TIER_TITLES = {
    "digits": "TIER 1: Singular Digits (Google Speech Commands, digits 0-9)",
    "timit": "TIER 2: Isolated Words & Phonetics (TIMIT)",
    "librispeech": "TIER 3: Continuous Sentences (LibriSpeech test-clean)",
}


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------

def resample(array: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample to target_sr using whatever backend is available."""
    if sr == target_sr:
        return array
    if librosa is not None:
        return librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    if resample_poly is not None:
        from math import gcd
        g = gcd(int(sr), int(target_sr))
        return resample_poly(array, target_sr // g, sr // g).astype(np.float32)
    # Last resort: linear interpolation.
    return np.interp(
        np.linspace(0, len(array) - 1, int(round(len(array) * target_sr / sr))),
        np.arange(len(array), dtype=np.float64),
        array,
    ).astype(np.float32)


def _read_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    try:
        array, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return array, int(sr)
    except Exception:
        if librosa is None:
            raise
        array, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return np.asarray(array, dtype=np.float32), int(sr)


def decode_audio(field_value: Any) -> tuple[np.ndarray, int]:
    """Turn whatever the `audio` column holds into (float32 mono array, sr).

    Handles undecoded {bytes, path} dicts (our normal path), already-decoded
    {array, sampling_rate} dicts, plain file paths, and torchcodec decoder
    objects, so this keeps working across datasets versions.
    """
    if field_value is None:
        raise ValueError("sample has no audio")

    if isinstance(field_value, dict):
        raw = field_value.get("bytes")
        if raw:
            array, sr = _read_bytes(raw)
        elif field_value.get("array") is not None:
            array = np.asarray(field_value["array"], dtype=np.float32)
            sr = int(field_value.get("sampling_rate") or TARGET_SR)
        else:
            path = field_value.get("path")
            if not path or not os.path.exists(path):
                raise ValueError("audio dict has neither bytes nor a readable path")
            array, sr = sf.read(path, dtype="float32", always_2d=False)
            sr = int(sr)
    elif isinstance(field_value, (str, os.PathLike)) and os.path.exists(field_value):
        array, sr = sf.read(str(field_value), dtype="float32", always_2d=False)
        sr = int(sr)
    elif hasattr(field_value, "get_all_samples"):  # torchcodec AudioDecoder
        samples = field_value.get_all_samples()
        array = np.asarray(samples.data, dtype=np.float32)
        sr = int(samples.sample_rate)
    else:
        raise TypeError(f"unsupported audio field: {type(field_value)!r}")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:  # downmix to mono
        array = array.mean(axis=1 if array.shape[0] > array.shape[1] else 0)
    return array, sr


def prepare(array: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz and keep the waveform inside [-1, 1]."""
    array = resample(array, sr)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    return array


# --------------------------------------------------------------------------
# Tier bookkeeping
# --------------------------------------------------------------------------

class TierWriter:
    """Writes clips + sidecars into a tier directory, enforcing the budget."""

    def __init__(self, out_dir: Path, max_bytes: int, limit: Optional[int],
                 prior_bytes: int = 0, prior_count: int = 0):
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        self.limit = limit
        self.bytes_written = prior_bytes
        self.count = prior_count
        self.new_count = 0
        self.entries: list[dict] = []
        self.quota_hit = False

    @property
    def full(self) -> bool:
        if self.quota_hit:
            return True
        if self.limit is not None and self.count >= self.limit:
            return True
        return self.bytes_written >= self.max_bytes

    @property
    def mb(self) -> float:
        return self.bytes_written / (1024 * 1024)

    def save(self, rel_path: str, array: np.ndarray, meta: dict,
             sidecars: Optional[dict[str, str]] = None) -> bool:
        """Write one clip. Returns False (writing nothing) if it will not fit."""
        if array.size == 0:
            return False

        # 16-bit mono PCM: 2 bytes per sample, plus a 44-byte header.
        projected = array.size * 2 + 44
        if self.bytes_written + projected > self.max_bytes:
            self.quota_hit = True
            return False

        path = self.out_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, array, TARGET_SR, subtype="PCM_16")

        for suffix, text in (sidecars or {}).items():
            path.with_suffix(suffix).write_text(text, encoding="utf-8")

        actual = path.stat().st_size
        self.bytes_written += actual
        self.count += 1
        self.new_count += 1
        self.entries.append({
            "file": rel_path.replace("\\", "/"),
            "bytes": actual,
            "duration_sec": round(len(array) / TARGET_SR, 3),
            "sample_rate": TARGET_SR,
            **meta,
        })
        return True


def load_manifest(out_dir: Path) -> dict:
    path = out_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_manifest(out_dir: Path, tier: str, source: Source, writer: TierWriter,
                   next_index: int, exhausted: bool, previous: dict) -> None:
    entries = list(previous.get("entries", [])) + writer.entries
    manifest = {
        "tier": tier,
        "source_id": source.id,
        "repo": source.repo,
        "config": source.config,
        "split": source.split,
        "sample_rate": TARGET_SR,
        "clips": len(entries),
        "bytes": writer.bytes_written,
        "next_index": next_index,
        # exhausted == the source stream ran out, so there is nothing left to
        # fetch even if the budget is raised later.
        "exhausted": exhausted,
        "budget_bytes": writer.max_bytes,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def existing_state(out_dir: Path, source_id: str) -> tuple[dict, int, int, int]:
    """Return (manifest, prior_bytes, prior_count, resume_index) for a tier."""
    manifest = load_manifest(out_dir)
    if manifest.get("source_id") != source_id:
        return {}, 0, 0, 0
    # Trust the files on disk, not just the manifest.
    kept = [e for e in manifest.get("entries", []) if (out_dir / e["file"]).exists()]
    manifest["entries"] = kept
    prior_bytes = sum(e.get("bytes", 0) for e in kept)
    return manifest, prior_bytes, len(kept), int(manifest.get("next_index", 0))


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------

def open_stream(source: Source, token: Optional[str]):
    """Open a streaming dataset with audio decoding disabled."""
    kwargs: dict[str, Any] = {"path": source.repo, "split": source.split, "streaming": True}
    if source.config:
        kwargs["name"] = source.config
    if token:
        kwargs["token"] = token

    ds = load_dataset(**kwargs)
    features = ds.features or {}
    if source.audio_col in features:
        ds = ds.cast_column(source.audio_col, Audio(decode=False))
    return ds, features


def iter_samples(source: Source, token: Optional[str], start_index: int,
                 retries: int) -> Iterator[tuple[int, dict, dict]]:
    """Yield (index, sample, features), restarting the stream on network errors."""
    index = start_index
    attempt = 0
    while True:
        try:
            ds, features = open_stream(source, token)
            for i, sample in enumerate(ds):
                if i < index:
                    continue
                index = i + 1
                yield i, sample, features
            return
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                raise
            wait = min(30, 5 * attempt)
            print(f"  [WARN] stream error ({type(exc).__name__}: {str(exc)[:120]})")
            print(f"         retrying from sample {index} in {wait}s [{attempt}/{retries}]")
            time.sleep(wait)


# --------------------------------------------------------------------------
# Per-tier sample handlers
# --------------------------------------------------------------------------

def digit_label(value: Any, features: dict, column: str) -> Optional[str]:
    """Map a label cell to '0'..'9', or None if it is not a digit."""
    feature = features.get(column) if features else None
    if isinstance(value, (int, np.integer)) and getattr(feature, "names", None):
        names = feature.names
        if not 0 <= int(value) < len(names):
            return None
        value = names[int(value)]
    name = str(value).strip().lower()
    if name in WORD_TO_DIGIT:
        return WORD_TO_DIGIT[name]
    if len(name) == 1 and name.isdigit():
        return name
    return None


def handle_digits(idx: int, sample: dict, features: dict, source: Source,
                  writer: TierWriter, state: dict) -> bool:
    digit = digit_label(sample.get(source.label_col), features, source.label_col or "")
    if digit is None:
        return False

    counts = state["counts"]
    digit_bytes = state["digit_bytes"]
    per_class = state.get("per_class")
    share = state.get("digit_share")

    if per_class and counts[digit] >= per_class:
        return False
    if share and digit_bytes[digit] >= share:
        # This digit has its slice of the budget; stop once every digit does.
        if all(digit_bytes[d] >= share for d in digit_bytes):
            state["done"] = True
        return False

    array, sr = decode_audio(sample.get(source.audio_col))
    array = prepare(array, sr)

    meta = {"digit": digit, "index": idx}
    if source.speaker_col and sample.get(source.speaker_col) is not None:
        meta["speaker_id"] = str(sample[source.speaker_col])

    if writer.save(f"{digit}/sample_{counts[digit]:05d}.wav", array, meta):
        counts[digit] += 1
        digit_bytes[digit] += writer.entries[-1]["bytes"]
        return True
    return False


def speaker_capped(speaker: str, state: dict) -> bool:
    """True when this speaker already contributed its share of clips."""
    cap = state.get("per_speaker")
    if not cap:
        return False
    return state.setdefault("speaker_counts", {}).get(speaker, 0) >= cap


def normalize_alignment(detail: Any) -> list[dict]:
    """TIMIT alignments arrive as a list of dicts or as a dict of lists."""
    if not detail:
        return []
    if isinstance(detail, dict):
        return [
            {"start": int(a), "stop": int(b), "utterance": str(c)}
            for a, b, c in zip(detail.get("start") or [],
                               detail.get("stop") or [],
                               detail.get("utterance") or [])
        ]
    return [
        {"start": int(item.get("start", 0)), "stop": int(item.get("stop", 0)),
         "utterance": str(item["utterance"])}
        for item in detail if isinstance(item, dict) and "utterance" in item
    ]


def alignment_text(segments: list[dict]) -> str:
    """TIMIT-style '<start> <stop> <label>' lines."""
    return "".join(f"{s['start']} {s['stop']} {s['utterance']}\n" for s in segments)


def handle_timit(idx: int, sample: dict, features: dict, source: Source,
                 writer: TierWriter, state: dict) -> bool:
    speaker = str(sample.get(source.speaker_col) or "unknown")
    if speaker_capped(speaker, state):
        return False

    array, sr = decode_audio(sample.get(source.audio_col))
    scale = TARGET_SR / sr if sr else 1.0
    array = prepare(array, sr)

    phones = normalize_alignment(sample.get("phonetic_detail"))
    words = normalize_alignment(sample.get("word_detail"))
    if scale != 1.0:  # alignments are in source-rate samples; rescale to 16 kHz
        for seg in (*phones, *words):
            seg["start"] = int(round(seg["start"] * scale))
            seg["stop"] = int(round(seg["stop"] * scale))

    text = str(sample.get(source.text_col) or "")
    meta = {
        "index": idx,
        "text": text,
        "speaker_id": speaker,
        "dialect_region": str(sample.get("dialect_region") or ""),
        "sentence_type": str(sample.get("sentence_type") or ""),
        "n_phones": len(phones),
        "n_words": len(words),
    }
    sidecars = {
        ".txt": text + "\n",
        ".phn": alignment_text(phones),
        ".wrd": alignment_text(words),
        ".json": json.dumps({
            "id": str(sample.get("id") or ""),
            "text": text,
            "speaker_id": meta["speaker_id"],
            "speaker_sex": str(sample.get("speaker_sex") or ""),
            "dialect_region": meta["dialect_region"],
            "sentence_type": meta["sentence_type"],
            "sample_rate": TARGET_SR,
            "phonetic_detail": phones,
            "word_detail": words,
        }, indent=2),
    }
    if writer.save(f"sample_{writer.count:06d}.wav", array, meta, sidecars):
        counts = state.setdefault("speaker_counts", {})
        counts[speaker] = counts.get(speaker, 0) + 1
        return True
    return False


def handle_librispeech(idx: int, sample: dict, features: dict, source: Source,
                       writer: TierWriter, state: dict) -> bool:
    speaker = str(sample.get(source.speaker_col) or "unknown")
    if speaker_capped(speaker, state):
        return False

    array, sr = decode_audio(sample.get(source.audio_col))
    array = prepare(array, sr)

    text = str(sample.get(source.text_col) or "")
    meta = {"index": idx, "text": text, "speaker_id": speaker,
            "utterance_id": str(sample.get("id") or "")}

    if writer.save(f"sample_{writer.count:06d}.wav", array, meta, {".txt": text + "\n"}):
        counts = state.setdefault("speaker_counts", {})
        counts[speaker] = counts.get(speaker, 0) + 1
        return True
    return False


HANDLERS: dict[str, Callable[..., bool]] = {
    "digits": handle_digits,
    "timit": handle_timit,
    "librispeech": handle_librispeech,
}

PROGRESS_EVERY = {"digits": 100, "timit": 25, "librispeech": 25}


# --------------------------------------------------------------------------
# Tier driver
# --------------------------------------------------------------------------

def download_tier(tier: str, out_dir: Path, max_mb: float, args) -> dict:
    print()
    print("=" * 70)
    print(TIER_TITLES[tier])
    print("=" * 70)

    candidates = SOURCES[tier]
    if args.source:
        picked = [s for s in candidates if s.id == args.source]
        if picked:
            candidates = picked

    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_mb * 1024 * 1024)
    handler = HANDLERS[tier]
    last_error = None

    for source in candidates:
        manifest, prior_bytes, prior_count, resume_index = existing_state(out_dir, source.id)

        if prior_count and (manifest.get("exhausted") or prior_bytes >= max_bytes):
            print(f"Target:  {out_dir}")
            print(f"[SKIP] Already satisfied: {prior_count} clips, "
                  f"{prior_bytes / (1024 * 1024):.2f} MB (use --overwrite to redo)\n")
            return {"tier": tier, "status": "cached", "clips": prior_count,
                    "mb": prior_bytes / (1024 * 1024), "source": source.id}

        if not manifest and any(out_dir.rglob("*.wav")):
            print(f"[WARN] {out_dir} holds clips from another source or run; "
                  f"new files may overwrite them (use --overwrite for a clean tier).")

        config = f" [{source.config}]" if source.config else ""
        print(f"Source:  {source.repo}{config} ({source.split})")
        print(f"         {source.note}")
        print(f"Target:  {out_dir}")
        print(f"Budget:  {max_mb:.0f} MB")
        if prior_count:
            print(f"Resume:  {prior_count} clips on disk "
                  f"({prior_bytes / (1024 * 1024):.2f} MB), continuing from sample {resume_index}")

        if args.dry_run:
            print("[DRY RUN] Nothing downloaded.\n")
            return {"tier": tier, "status": "dry-run", "clips": prior_count,
                    "mb": prior_bytes / (1024 * 1024), "source": source.id}

        writer = TierWriter(out_dir, max_bytes, args.limit, prior_bytes, prior_count)
        state: dict[str, Any] = {"per_class": args.per_class}
        prior_entries = manifest.get("entries", [])

        if tier == "digits":
            counts = {str(d): 0 for d in range(10)}
            digit_bytes = {str(d): 0 for d in range(10)}
            for entry in prior_entries:
                digit = entry.get("digit")
                if digit in counts:
                    counts[digit] += 1
                    digit_bytes[digit] += entry.get("bytes", 0)
            state["counts"] = counts
            state["digit_bytes"] = digit_bytes
            # Equal byte share per digit unless the user caps clips instead.
            state["digit_share"] = None if (args.per_class or args.no_balance) else max_bytes // 10
            if state["digit_share"]:
                print(f"Balance: ~{state['digit_share'] / (1024 * 1024):.0f} MB per digit")
        else:
            cap = PER_SPEAKER_DEFAULT.get(tier, 0) if args.per_speaker is None else args.per_speaker
            if args.no_balance:
                cap = 0
            state["per_speaker"] = cap
            speaker_counts: dict[str, int] = {}
            for entry in prior_entries:
                speaker = entry.get("speaker_id")
                if speaker:
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            state["speaker_counts"] = speaker_counts
            if cap:
                print(f"Balance: up to {cap} clips per speaker")

        seen = 0
        next_index = resume_index
        every = PROGRESS_EVERY[tier]
        interrupted = False
        exhausted = True  # cleared if we stop early rather than run out of data
        started = time.time()

        try:
            for idx, sample, features in iter_samples(source, args.token, resume_index,
                                                      args.retries):
                next_index = idx + 1
                seen += 1
                try:
                    handler(idx, sample, features, source, writer, state)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    if args.verbose:
                        print(f"  [skip] sample {idx}: {type(exc).__name__}: {str(exc)[:100]}")
                    continue

                if writer.full or state.get("done"):
                    exhausted = False
                    break
                if writer.new_count and writer.new_count % every == 0:
                    rate = writer.new_count / max(time.time() - started, 1e-6)
                    print(f"  {writer.count} clips, {writer.mb:.2f} MB "
                          f"({rate:.1f} clips/s, scanned {seen})")
        except KeyboardInterrupt:
            interrupted = True
            exhausted = False
            print("\n  [INTERRUPTED] Saving what was downloaded so far...")
        except Exception as exc:
            last_error = exc
            exhausted = False
            print(f"  [ERROR] {source.repo}: {type(exc).__name__}: {str(exc)[:200]}")
            if writer.new_count == 0:
                if source is not candidates[-1]:
                    print("  Trying the next mirror...\n")
                continue

        write_manifest(out_dir, tier, source, writer, next_index, exhausted, manifest)

        if writer.count == 0:
            print(f"  [ERROR] No usable clips from {source.repo}.")
            if source is not candidates[-1]:
                print("  Trying the next mirror...\n")
            continue

        status = "interrupted" if interrupted else ("exhausted" if exhausted else "budget")
        print(f"\n[OK] {writer.count} clips, {writer.mb:.2f} MB in {out_dir}")
        if tier == "digits":
            for digit, count in sorted(state.get("counts", {}).items()):
                if count:
                    print(f"     digit '{digit}': {count} clips")
        else:
            speakers = state.get("speaker_counts") or {}
            if speakers:
                print(f"     unique speakers: {len(speakers)}")
        if tier == "timit":
            print("     alignments written as .phn / .wrd / .json sidecars")
        print()
        return {"tier": tier, "status": status, "clips": writer.count,
                "mb": writer.mb, "source": source.id}

    print(f"[ERROR] Could not download tier '{tier}' from any known mirror.")
    if last_error:
        print(f"        Last error: {type(last_error).__name__}: {str(last_error)[:200]}")
    print()
    return {"tier": tier, "status": "failed", "clips": 0, "mb": 0.0, "source": None}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def list_sources() -> None:
    print("\nAvailable sources (tried top-down; --source ID forces one):\n")
    for tier in TIER_ORDER:
        print(f"  {tier}:")
        for source in SOURCES[tier]:
            config = f" [{source.config}]" if source.config else ""
            print(f"    {source.id:<24} {source.repo}{config} ({source.split})")
            print(f"    {'':<24} {source.note}")
        print()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download speech benchmark datasets (digits / TIMIT / LibriSpeech).")
    parser.add_argument("--tiers", nargs="+", choices=TIER_ORDER, default=TIER_ORDER,
                        help="Which tiers to download (default: all).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output root (default: data/audio).")
    parser.add_argument("--max-mb", type=float, default=250.0,
                        help="Size budget per tier in MB (default: 250).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many clips per tier.")
    parser.add_argument("--per-class", type=int, default=None,
                        help="Digits tier: cap clips per digit (default: equal byte share).")
    parser.add_argument("--per-speaker", type=int, default=None,
                        help="TIMIT/LibriSpeech: cap clips per speaker (0 = no cap; "
                             f"defaults {PER_SPEAKER_DEFAULT}).")
    parser.add_argument("--no-balance", action="store_true",
                        help="Take clips in stream order instead of balancing "
                             "across digits/speakers.")
    parser.add_argument("--source", default=None,
                        help="Force a specific source id (see --list-sources).")
    parser.add_argument("--list-sources", action="store_true",
                        help="Print the source registry and exit.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing tier files and re-download.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded, then exit.")
    parser.add_argument("--retries", type=int, default=3,
                        help="Stream restarts on network errors (default: 3).")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token (default: $HF_TOKEN).")
    parser.add_argument("--verbose", action="store_true",
                        help="Report individual skipped samples.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_sources:
        list_sources()
        return 0

    if args.source and args.source not in {s.id for tier in SOURCES.values() for s in tier}:
        print(f"Error: unknown source '{args.source}'. Use --list-sources.")
        return 2

    backend = "librosa" if librosa else ("scipy" if resample_poly else "numpy")
    print("=" * 70)
    print("SPEECH ANALYSIS BENCHMARK DATASET DOWNLOADER")
    print("=" * 70)
    print(f"Tiers:   {', '.join(args.tiers)}")
    print(f"Budget:  {args.max_mb:.0f} MB per tier "
          f"(~{args.max_mb * len(args.tiers):.0f} MB total)")
    print(f"Output:  {args.out.resolve()}")
    print(f"Backend: soundfile decoding, {backend} resampling")

    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for tier in args.tiers:
        out_dir = args.out / tier
        if args.overwrite and out_dir.exists() and not args.dry_run:
            removed = 0
            for path in sorted(out_dir.rglob("*")):
                if path.is_file():
                    path.unlink()
                    removed += 1
            print(f"\n[--overwrite] Removed {removed} existing files from {out_dir}")
        try:
            results.append(download_tier(tier, out_dir, args.max_mb, args))
        except KeyboardInterrupt:
            print("\nAborted by user.")
            break

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for result in results:
        print(f"  {result['tier']:<12} {result['status']:<12} "
              f"{result['clips']:>6} clips  {result['mb']:>8.2f} MB  "
              f"[{result['source'] or 'none'}]")
    print()

    if any(r["status"] == "failed" for r in results):
        print("Some tiers failed. Try:")
        print("  python scripts/download_datasets.py --list-sources")
        print("  python scripts/download_datasets.py --tiers <tier> --source <id>")
        print()
        return 1

    if not args.dry_run and any(r["clips"] for r in results):
        print("Next steps:")
        print("  1. python scripts/test_audio_methods.py")
        print("  2. Review the vowel recognition results")
        print("  3. See data/audio/README.md for the analysis guide")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
