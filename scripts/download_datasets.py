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
import numpy as np
import soundfile as sf
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Install with: pip install datasets")
    sys.exit(1)

# Try to import audio processing libraries
try:
    import librosa
except ImportError:
    librosa = None

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    wavfile = None

def download_digits_subset(
    output_dir: str = "data/audio/digits",
    max_mb: float = 250.0
):
    """
    Download Google Speech Commands v0.02 (digits 0-9 only).
    Uses streaming and handles modern dataset library requirements.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("TIER 1: Singular Digits (Google Speech Commands v0.02)")
    print(f"{'='*70}")
    print(f"Target: {output_dir}")
    print(f"Max size: {max_mb} MB")
    
    try:
        # Try loading with trust_remote_code=True for newer versions
        print("Loading Google Speech Commands dataset...")
        ds = load_dataset(
            "google/speech_commands", 
            "v0.02", 
            split="train", 
            streaming=True,
            trust_remote_code=True
        )
        
        total_bytes = 0
        max_bytes = max_mb * 1024 * 1024
        saved_count = 0
        digit_counts = {str(i): 0 for i in range(10)}

        for idx, sample in enumerate(ds):
            # Only keep digits (0-9)
            label = sample.get("label", "")
            if label not in digit_counts:
                continue

            try:
                audio = sample["audio"]
                
                # Handle both dict format and direct array format
                if isinstance(audio, dict):
                    array = np.array(audio["array"], dtype=np.float32)
                    sr = audio.get("sampling_rate", 16000)
                else:
                    array = np.array(audio, dtype=np.float32)
                    sr = 16000

                # Resample to 16 kHz if needed
                if sr != 16000:
                    if librosa is not None:
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
                
                # Normalize to [-1, 1] if needed
                if np.max(np.abs(array)) > 1.0:
                    array = array / np.max(np.abs(array))
                
                sf.write(file_name, array, 16000, subtype="PCM_16")

                total_bytes += sample_bytes
                saved_count += 1
                digit_counts[label] += 1

                if saved_count % 100 == 0:
                    print(f"  Progress: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB")

            except Exception as e:
                # Skip problematic samples
                continue

        print(f"\n[OK] Download complete:")
        for digit, count in digit_counts.items():
            if count > 0:
                print(f"  Digit '{digit}': {count} samples")
        print(f"  Total: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB\n")

    except Exception as e:
        print(f"[ERROR] Error downloading digits: {e}")
        print(f"  Note: Google Speech Commands may require authentication or internet access.\n")


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
        print("Note: TIMIT requires LDC institutional credentials.")
        print("Attempting to load with authentication...")
        
        # Try loading with trust_remote_code=True
        ds = load_dataset("timit_asr", split="train", streaming=True, trust_remote_code=True)
        
        total_bytes = 0
        max_bytes = max_mb * 1024 * 1024
        saved_count = 0

        for idx, sample in enumerate(ds):
            try:
                audio = sample["audio"]
                
                # Handle both dict format and direct array format
                if isinstance(audio, dict):
                    array = np.array(audio["array"], dtype=np.float32)
                    sr = audio.get("sampling_rate", 16000)
                else:
                    array = np.array(audio, dtype=np.float32)
                    sr = 16000

                # Resample to 16 kHz
                if sr != 16000:
                    if librosa is not None:
                        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
                    sr = 16000

                sample_bytes = len(array) * 2
                if total_bytes + sample_bytes > max_bytes:
                    print(f"\n✓ Reached quota: {total_bytes / (1024 * 1024):.2f} MB ({saved_count} clips)")
                    break

                # Save audio
                file_name = out_path / f"sample_{idx:06d}.wav"
                
                # Normalize to [-1, 1] if needed
                if np.max(np.abs(array)) > 1.0:
                    array = array / np.max(np.abs(array))
                
                sf.write(file_name, array, 16000, subtype="PCM_16")

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

            except Exception as e:
                # Skip problematic samples
                continue

        print(f"\n[OK] Download complete: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB")
        print(f"  Metadata saved as .json files alongside .wav audio\n")

    except Exception as e:
        print(f"[ERROR] TIMIT download failed: {e}")
        print(f"  TIMIT requires institutional LDC access (optional - continuing with other tiers).\n")


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
        print("Loading LibriSpeech dataset (this may take a moment)...")
        
        # Try loading with trust_remote_code=True
        ds = load_dataset(
            "librispeech_asr", 
            "clean", 
            split="test", 
            streaming=True,
            trust_remote_code=True
        )
        
        total_bytes = 0
        max_bytes = max_mb * 1024 * 1024
        saved_count = 0
        speaker_count = {}

        for idx, sample in enumerate(ds):
            try:
                audio = sample["audio"]
                
                # Handle both dict format and direct array format
                if isinstance(audio, dict):
                    array = np.array(audio["array"], dtype=np.float32)
                    sr = audio.get("sampling_rate", 16000)
                else:
                    array = np.array(audio, dtype=np.float32)
                    sr = 16000
                
                text = sample.get("text", "")
                speaker_id = sample.get("speaker_id", "unknown")

                # Resample to 16 kHz
                if sr != 16000:
                    if librosa is not None:
                        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
                    sr = 16000

                sample_bytes = len(array) * 2
                if total_bytes + sample_bytes > max_bytes:
                    print(f"\n✓ Reached quota: {total_bytes / (1024 * 1024):.2f} MB ({saved_count} clips)")
                    break

                # Save audio
                file_name = out_path / f"sample_{idx:06d}.wav"
                
                # Normalize to [-1, 1] if needed
                if np.max(np.abs(array)) > 1.0:
                    array = array / np.max(np.abs(array))
                
                sf.write(file_name, array, 16000, subtype="PCM_16")

                # Save transcript alongside audio
                trans_file = out_path / f"sample_{idx:06d}.txt"
                with open(trans_file, "w") as f:
                    f.write(text)

                total_bytes += sample_bytes
                saved_count += 1
                speaker_count[speaker_id] = speaker_count.get(speaker_id, 0) + 1

                if saved_count % 50 == 0:
                    print(f"  Progress: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB")

            except Exception as e:
                # Skip problematic samples (audio codec issues, etc.)
                if "torchcodec" in str(e).lower() or "audio" in str(e).lower():
                    # This is expected for some codec issues
                    pass
                continue

        print(f"\n[OK] Download complete: {saved_count} clips, {total_bytes / (1024 * 1024):.2f} MB")
        print(f"  Unique speakers: {len(speaker_count)}")
        print(f"  Transcriptions saved as .txt files alongside .wav audio\n")

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Error downloading LibriSpeech: {error_msg}")
        
        if "torchcodec" in error_msg.lower():
            print(f"\n  Fix: Install torchcodec with: pip install torchcodec")
            print(f"  Or install librosa: pip install librosa\n")
        else:
            print(f"  LibriSpeech may require internet connectivity or special access.\n")


def main():
    """Download all three benchmark datasets."""
    print("\n" + "="*70)
    print("SPEECH ANALYSIS BENCHMARK DATASET DOWNLOADER")
    print("="*70)
    print("Downloading 3 tiers: Digits, Phonetics, Continuous Sentences")
    print("Max 250 MB per tier (~750 MB total)\n")

    # Check dependencies
    print("Checking dependencies...")
    if librosa is None:
        print("  WARNING: librosa not installed (optional but recommended)")
        print("     Install with: pip install librosa")
    else:
        print("  OK: librosa installed")
    
    print("  OK: soundfile installed")
    print("  OK: datasets installed\n")

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
    print("\nNext steps:")
    print("  1. Run: python scripts/test_audio_methods.py")
    print("  2. Review the vowel recognition results")
    print("  3. See data/audio/README.md for detailed analysis guide\n")


if __name__ == "__main__":
    main()