# Quick Start: Audio Dataset & Method Comparison

## One-Command Setup (Windows)

```powershell
scripts\setup_audio_datasets.bat
```

This will:
1. Install Python dependencies (soundfile, librosa, datasets)
2. Build LPC and Fourier methods
3. Download all three dataset tiers (~750 MB total)
4. Run validation tests on both methods
5. Display a benchmark report

## One-Command Setup (Linux/macOS)

```bash
bash scripts/setup_audio_datasets.sh
```

## Manual Setup Steps

### 1. Install Dependencies

```bash
pip install soundfile librosa datasets
```

### 2. Build Methods

```bash
cd build
cmake --build . --target lpc_method --config Release
cmake --build . --target fourier_method --config Release
cd ..
```

### 3. Download Datasets

```bash
python scripts/download_datasets.py
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--tiers digits timit librispeech` | Pick which tiers to fetch |
| `--max-mb 50` | Smaller per-tier budget (default 250 MB) |
| `--limit 200` / `--per-class 100` / `--per-speaker 5` | Cap clip counts |
| `--list-sources` / `--source <id>` | Show or force a specific mirror |
| `--overwrite` | Delete a tier and re-download it |
| `--dry-run` | Show the plan without downloading |

Re-running resumes: each tier keeps a `manifest.json` and tops itself back up
to the budget instead of starting over.

Expected output:
```
======================================================================
STEP 1: Singular Digits (Google Speech Commands v0.02)
  Progress: 100 clips, 50.00 MB
  ✓ Download complete: 1500 clips, 250.00 MB

======================================================================
STEP 2: Isolated Words & Phonetics (TIMIT)
  Progress: 50 clips, 100.00 MB
  ✓ Download complete: 300 clips, 250.00 MB

======================================================================
STEP 3: Continuous Sentences (LibriSpeech test-clean)
  Progress: 100 clips, 150.00 MB
  ✓ Download complete: 1000 clips, 250.00 MB
```

### 4. Validate & Test

```bash
python scripts/test_audio_methods.py
```

Or test specific tier:
```bash
python scripts/test_audio_methods.py digits
python scripts/test_audio_methods.py timit
python scripts/test_audio_methods.py librispeech
```

## Testing Individual Files

### Test with LPC method:
```bash
build/bin/Release/lpc_method.exe data/audio/digits/5/sample_00000.wav
```

### Test with Fourier method:
```bash
build/bin/Release/fourier_method.exe data/audio/digits/5/sample_00000.wav
```

## Expected Output

Both methods output vowel strings:

```
=== LPC METHOD RESULTS ===
Vowel string (all predicted frames): IY EH AA SIL EH IY
Vowel string (collapsed): IY EH AA EH IY

=== FOURIER/SPECTRAL METHOD RESULTS ===
Vowel string (all predicted frames): IY EH AA SIL EH IY
Vowel string (collapsed): IY EH AA EH IY
```

## File Organization

After download, your structure will be:

```
data/audio/
├── README.md                      (full documentation)
├── digits/
│   ├── 0/
│   │   ├── sample_00000.wav      (isolated "zero")
│   │   └── ...
│   ├── 1/
│   │   ├── sample_00000.wav      (isolated "one")
│   │   └── ...
│   └── ... (digits 2-9)
├── timit/
│   ├── sample_000000.wav         (phonetically balanced sentence)
│   ├── sample_000000.json        (phoneme labels & times)
│   └── ...
└── librispeech/
    ├── sample_000000.wav         (continuous multi-word)
    ├── sample_000000.txt         (word transcription)
    └── ...
```

## Troubleshooting

### Python Errors
```
ModuleNotFoundError: No module named 'soundfile'
```
→ Run: `pip install soundfile librosa datasets`

### Build Errors
```
error: target 'lpc_method' not found
```
→ Make sure you ran CMake config: `cd build && cmake ..`

### Download Errors
```
ConnectionError: Unable to connect to dataset
```
→ Check internet connection, then try another mirror:
- `python scripts/download_datasets.py --list-sources` shows the mirrors per tier
- `--source <id>` forces one (the script otherwise falls through them in order)
- TIMIT comes from a public parquet mirror, so no LDC credentials are needed
- No `torchcodec` install is needed: audio is decoded with `soundfile`

### Audio File Errors
```
No audio samples were loaded
```
→ File may be corrupted or not WAV format. Re-download datasets.

## Performance Benchmarks

On modern hardware (~2 sec per 1-sec audio file):

| Dataset | Tier 1 Time | Both Methods |
|---------|-----------|--------------|
| Digits  | ~1-2 min  | Total ~5 min |
| TIMIT   | ~2-3 min  | Total ~8 min |
| LibriSpeech | ~3-5 min | Total ~12 min |

**Total benchmark time**: ~25 minutes for all three tiers

## Next Steps

1. **Review Results**
   - Check `test_audio_methods.py` output
   - Compare LPC vs Fourier vowel strings
   - Identify agreement/disagreement patterns

2. **Analyze Accuracy**
   - For Digits: Compare against known digit (you can infer from file path)
   - For TIMIT: Compare against `.json` phoneme labels
   - For LibriSpeech: Compare against `.txt` transcriptions

3. **Tune Thresholds**
   - Consonant detection (ZCR, spectral centroid)
   - Voicing confidence (pitch prominence)
   - Edit threshold constants in `*_method.cpp`

4. **Extend Methods**
   - Add confidence scoring
   - Implement frame-by-frame debugging
   - Compute formal metrics (precision, recall, F1-score)

## Documentation

See **data/audio/README.md** for:
- Detailed dataset descriptions
- Method comparison table
- Vowel label reference (IPA + formant ranges)
- Calibration guide
- Troubleshooting & known issues

---

**Questions?** Check the comprehensive guide at `data/audio/README.md`
