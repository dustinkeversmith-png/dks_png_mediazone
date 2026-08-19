import os
import io
import json
import soundfile as sf
import librosa
from datasets import load_dataset, Audio

def export_audiomnist_subset(output_dir="./data/audiomnist", max_samples=300):
    """
    Downloads a lightweight subset of AudioMNIST (0-9 spoken digits).
    Target sample rate: 16 kHz Mono WAV.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Downloading & extracting AudioMNIST subset to {output_dir}...")

    # Load dataset in streaming mode and disable built-in audio decoding
    ds = load_dataset("gilkeyio/AudioMNIST", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    
    metadata = []
    count = 0

    for item in ds:
        # Read raw bytes into memory using soundfile
        audio_bytes = item["audio"]["bytes"]
        audio_data, orig_sr = sf.read(io.BytesIO(audio_bytes))

        # Resample to 16kHz mono if needed
        if orig_sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=16000)

        filename = f"digit_{item['digit']}_sample_{count:04d}.wav"
        file_path = os.path.join(output_dir, filename)

        # Write WAV to disk (16-bit PCM Mono)
        sf.write(file_path, audio_data, 16000, subtype='PCM_16')

        metadata.append({
            "file": filename,
            "digit": item.get("digit"),
            "speaker_id": item.get("speaker_id", "unknown")
        })

        count += 1
        if count >= max_samples:
            break

    # Save index file for C++ loading
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✓] Saved {count} AudioMNIST WAV files to {output_dir}.")


def export_timit_subset(output_dir="./data/timit_phonemes", max_samples=300):
    """
    Downloads a lightweight subset of TIMIT phonetic utterances.
    Target sample rate: 16 kHz Mono WAV.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Downloading & extracting TIMIT subset to {output_dir}...")

    # Load dataset in streaming mode and disable built-in audio decoding
    ds = load_dataset("IParraMartin/TIMITPhones", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    
    metadata = []
    count = 0

    for item in ds:
        audio_bytes = item["audio"]["bytes"]
        audio_data, orig_sr = sf.read(io.BytesIO(audio_bytes))

        if orig_sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=16000)

        filename = f"timit_sample_{count:04d}.wav"
        file_path = os.path.join(output_dir, filename)

        sf.write(file_path, audio_data, 16000, subtype='PCM_16')

        metadata.append({
            "file": filename,
            "phoneme": item.get("phoneme", item.get("label", "unknown")),
            "speaker": item.get("speaker", "unknown")
        })

        count += 1
        if count >= max_samples:
            break

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✓] Saved {count} TIMIT WAV files to {output_dir}.")


if __name__ == "__main__":
    export_audiomnist_subset()
    export_timit_subset()