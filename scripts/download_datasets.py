#!/usr/bin/env python3
"""
Download three benchmark datasets for speech analysis:
1. Singular Digits: Google Speech Commands (digits 0-9)
2. Isolated Words & Phonetics: TIMIT (with phonetic alignments)
3. Continuous Sentences: LibriSpeech test-clean

All datasets streamed to data/audio/ with 250 MB limit per tier.
"""

import sys
import json
import soundfile as sf
from pathlib import Path
from datasets import load_dataset

def download_digits_subset(
    output_dir: str = "data/audio/digits",
    max_mb: float = 250.0
):
    """
    Download Google Speech Commands v0.02 (digits 0-9 only).
    Streams audio and halts when max_mb is reached.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("TIER 1: Singular Digits (Google Speech Commands v0.02)")
    print(f"{'='*70}")
    print(f"Target: {output_dir}")
    print(f"Max size: {max_mb} MB")
    
    try:
        # Stream the training set
        ds = load_dataset("google/speech_commands", "v0.02", split="train", streaming=True)
        
        total_bytes = 0
        max_bytes = max_mb * 1024 * 1024
        saved_count = 0
        digit_counts = {str(i): 0 for i in range(10)}

        for idx, sample in enumerate(ds):
            # Only keep digits (0-9)
            label = sample.get("label", "")
            if label not in digit_counts:
                continue

            audio = sample["audio"]
            array = audio["array"]
            sr = audio["sampling_rate"]

            # Resample to 16 kHz if needed
            if sr != 16000:
                import librosa
                array = librosa.resample(array, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Approximate raw PCM payload (16-bit mono = 2 bytes per sample)
            sample_bytes = len(array) * 2
            if total_bytes + sample_bytes > max_bytes:
                print(f"\n✓ Reached quota: {total_bytes / (1024 * 1024):.2f} MB ({saved_count} clips)")
                break

            # Save to disk: data/audio/digits/0/sample_xxxxx.wav, etc.
            digit_dir = out_path / label
            digit_dir.mkdir(exist_ok=True)
            file_name = digit_dir / f"sample_{digit_counts[label]:05d}.wav"
            
            sf.write(file_name, array, sr, subtype="PCM_16")

            total_bytes += sample_bytes
            saved_count += 1
            digit_counts[label] += 1

            if saved_count % 100 == 0:
                print(f"  Progress: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB")

        print(f"\n✓ Download complete:")
        for digit, count in digit_counts.items():
            if count > 0:
                print(f"  Digit '{digit}': {count} samples")
        print(f"  Total: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB\n")

    except Exception as e:
        print(f"✗ Error downloading digits: {e}\n")


def download_timit_subset(
    output_dir: str = "data/audio/timit",
    max_mb: float = 250.0
):
    """
    Download TIMIT Acoustic-Phonetic Continuous Speech Corpus.
    Preserves .wav audio and .phn/.wrd phonetic alignment files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print("TIER 2: Isolated Words & Phonetics (TIMIT)")
    print(f"{'='*70}")
    print(f"Target: {output_dir}")
    print(f"Max size: {max_mb} MB")
    
    try:
        # TIMIT requires authentication; check if available
        ds = load_dataset("timit_asr", split="train", streaming=True)
        
        total_bytes = 0
        max_bytes = max_mb * 1024 * 1024
        saved_count = 0

        for idx, sample in enumerate(ds):
            audio = sample["audio"]
            array = audio["array"]
            sr = audio["sampling_rate"]
            
            # Resample to 16 kHz
            if sr != 16000:
                import librosa
                array = librosa.resample(array, orig_sr=sr, target_sr=16000)
                sr = 16000

            sample_bytes = len(array) * 2
            if total_bytes + sample_bytes > max_bytes:
                print(f"\n✓ Reached quota: {total_bytes / (1024 * 1024):.2f} MB ({saved_count} clips)")
                break

            # Save audio
            file_name = out_path / f"sample_{idx:06d}.wav"
            sf.write(file_name, array, sr, subtype="PCM_16")

            # Save metadata (phonemes, text) alongside audio
            meta_file = out_path / f"sample_{idx:06d}.json"
            metadata = {
                "text": sample.get("text", ""),
                "phonetic_detail": sample.get("phonetic_detail", []),
            }
            with open(meta_file, "w") as f:
                json.dump(metadata, f)

            total_bytes += sample_bytes
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"  Progress: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB")

        print(f"\n✓ Download complete: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB")
        print(f"  Metadata saved as .json files alongside .wav audio\n")

    except Exception as e:
        print(f"✗ TIMIT download failed: {e}")
        print(f"  Note: TIMIT requires institutional access or special authentication.\n")
        print(f"  Alternative: Use 'timit_asr' dataset if you have credentials.\n")


def download_librispeech_subset(
    output_dir: str = "data/audio/librispeech",
    max_mb: float = 250.0
):
    """
    Download LibriSpeech test-clean split (continuous read speech, high quality).
    Streams audio and halts when max_mb is reached.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print("TIER 3: Continuous Sentences (LibriSpeech test-clean)")
    print(f"{'='*70}")
    print(f"Target: {output_dir}")
    print(f"Max size: {max_mb} MB")
    
    try:
        # Stream test-clean split
        ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
        
        total_bytes = 0
        max_bytes = max_mb * 1024 * 1024
        saved_count = 0
        speaker_count = {}

        for idx, sample in enumerate(ds):
            audio = sample["audio"]
            array = audio["array"]
            sr = audio["sampling_rate"]
            text = sample.get("text", "")
            speaker_id = sample.get("speaker_id", "unknown")

            # Resample to 16 kHz
            if sr != 16000:
                import librosa
                array = librosa.resample(array, orig_sr=sr, target_sr=16000)
                sr = 16000

            sample_bytes = len(array) * 2
            if total_bytes + sample_bytes > max_bytes:
                print(f"\n✓ Reached quota: {total_bytes / (1024 * 1024):.2f} MB ({saved_count} clips)")
                break

            # Save audio
            file_name = out_path / f"sample_{idx:06d}.wav"
            sf.write(file_name, array, sr, subtype="PCM_16")

            # Save transcript alongside audio
            trans_file = out_path / f"sample_{idx:06d}.txt"
            with open(trans_file, "w") as f:
                f.write(text)

            total_bytes += sample_bytes
            saved_count += 1
            speaker_count[speaker_id] = speaker_count.get(speaker_id, 0) + 1

            if saved_count % 50 == 0:
                print(f"  Progress: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB")

        print(f"\n✓ Download complete: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB")
        print(f"  Unique speakers: {len(speaker_count)}")
        print(f"  Transcriptions saved as .txt files alongside .wav audio\n")

    except Exception as e:
        print(f"✗ Error downloading LibriSpeech: {e}\n")


def main():
    """Download all three benchmark datasets."""
    print("\n" + "="*70)
    print("SPEECH ANALYSIS BENCHMARK DATASET DOWNLOADER")
    print("="*70)
    print("Downloading 3 tiers: Digits, Phonetics, Continuous Sentences")
    print("Max 250 MB per tier (~750 MB total)\n")

    # Create parent directory
    Path("data/audio").mkdir(parents=True, exist_ok=True)

    # Download each tier
    download_digits_subset("data/audio/digits", max_mb=250.0)
    download_timit_subset("data/audio/timit", max_mb=250.0)
    download_librispeech_subset("data/audio/librispeech", max_mb=250.0)

    print("="*70)
    print("ALL DOWNLOADS COMPLETE")
    print("="*70)
    print("Directory structure:")
    print("  data/audio/digits/        - Isolated digits (0-9)")
    print("  data/audio/timit/         - Phonetically labeled utterances")
    print("  data/audio/librispeech/   - Continuous read speech")
    print()


if __name__ == "__main__":
    main()