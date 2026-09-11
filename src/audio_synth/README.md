# Vocal Acoustics

A small C++20 foundation for evaluating **text-to-mel acoustic models**. The
initial harness separates model inference from evaluation and supplies:

- a `vocal::AcousticModel` interface for text/speaker-conditioned inference;
- a LibriTTS/LibriTTS-R utterance scanner;
- mel-cepstral distortion (MCD) with dynamic-time-warping alignment;
- F0 RMSE and voiced/unvoiced (V/UV) error;
- normalized word error rate (WER) for transcripts produced by an external ASR;
- a CLI and dependency-free metric tests.

The neural ASR, pitch tracker, mel-cepstrum extractor, UTMOS/NISQA adapter, and
human MOS/MUSHRA UI are deliberately integration boundaries rather than hidden
dependencies. This keeps metric definitions testable while letting experiments
choose their preferred feature and inference runtimes.

## Build

Requirements: CMake 3.20+, a C++20 compiler, and (for the preset) Ninja.

```sh
cmake --preset default
cmake --build --preset default
ctest --preset default
```

Without Ninja, use a native generator:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Visual Studio generators are multi-configuration. Build and test them with an
explicit configuration:

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

## Fetch an evaluation sample

LibriTTS-R's official `test_clean` archive is roughly 1.2 GB; OpenSLR does not
offer a 250 MB shard. The fetcher streams the official archive and stops after
about 250 MiB of extracted WAV and transcript files. It does **not** retain the
archive or claim that this prefix sample is an official split.

```sh
python scripts/fetch_libritts_r.py --budget-mb 250
```

The default output is `data/libritts-r-sample/`. A `sample_manifest.json`
records the source, license, byte counts, and file counts. It tries the three
certificate-valid mirrors published by OpenSLR; repeat `--url` to provide custom
fallbacks. The corpus is CC BY 4.0; publications should cite the
LibriTTS-R paper and preserve dataset attribution.

Inspect the downloaded subset:

```sh
build/default/vocal-eval scan data/libritts-r-sample
```

## Score precomputed model outputs

MCEP inputs are CSV or whitespace matrices with one frame per line. Column zero
is treated as the energy coefficient and excluded from MCD. F0 files contain
one value per frame, with zero representing unvoiced frames.

```sh
vocal-eval score reference.mcep synthesized.mcep reference.f0 synthesized.f0 \
  "the reference text" "the ASR hypothesis"
```

MCD uses DTW by default, so it is a diagnostic similarity score rather than a
proof of naturalness. Track it alongside F0/VUV and ASR WER, then use blinded
listening tests at milestone checkpoints.

## Render the model comparison

```powershell
build\Debug\vocal-eval.exe render-all artifacts "We synthesize a clear acoustic voice."
```

This writes eight mono 24 kHz WAV files plus `manifest.csv`: formant,
statistical/HMM, articulatory, LPC, sine-wave, procedural unit-selection,
homebrew neural, and an ONNX-style FastSpeech2-shaped baseline. These are
deterministic acoustic sketches intended to expose how each paradigm behaves;
they are not pretrained voices. `OnnxStyleAcousticModel` preserves a clean
`AcousticModel` boundary for a later ONNX Runtime-backed implementation.
It also writes `neural_comparison.json` with aligned log-mel MAE/RMSE and the
predicted-duration ratio. See `docs/MODELS.md` for implementation details and
listening cues.

## Implemented evaluation integrations

The evaluation integrations are available from `vocal-eval`:

```powershell
# PCM16/float32 WAV -> mono 24 kHz -> 80-bin log-mel + 25-coefficient MCEP
build\Debug\vocal-eval.exe features input.wav artifacts/input

# Aggregate a score table with deterministic 95% bootstrap intervals
build\Debug\vocal-eval.exe summarize scores.csv artifacts/summary 2000 42

# Randomized, blinded MOS/MUSHRA trial definitions
build\Debug\vocal-eval.exe study-manifest wavs artifacts/study 42
```

The score CSV schema is `utterance_id,speaker_id,mcd_db,f0_rmse_hz,vuv_error,wer`;
metric cells may be empty. The study builder groups names formatted as
`utterance__system.wav`. Systems named `reference`, `natural`, or `ground_truth`
become hidden references; systems containing `anchor` become anchors. A trial
without both is explicitly marked `mushra_complete: false`.

### Optional ONNX Runtime adapter

`OnnxRuntimeAcousticModel` is a real C++ Runtime session adapter, kept optional
so the core project remains dependency-free. It accepts an int64 `[1, tokens]`
input called `input_ids` and a float `[1, frames, 80]` (or `[frames, 80]`) output
called `mel` by default. Output names are configurable in `OnnxModelConfig`.

```powershell
cmake -S . -B build-ort -DVA_ENABLE_ONNX_RUNTIME=ON `
  -DONNXRUNTIME_ROOT=C:\sdk\onnxruntime
cmake --build build-ort --config Release
build-ort\Release\vocal-eval.exe onnx-infer model.onnx "Text to synthesize" artifacts/model
```

When duration/alignment outputs are configured, diagnostics include inference
time, frames per token, duration/frame disagreement, zero-duration tokens,
alignment monotonicity violations, and token coverage. Tokenization is byte-level
by default; replace it at the application boundary for a phoneme-based export.

ASR remains an external boundary: WER accepts supplied reference/hypothesis text,
but this project intentionally installs or invokes no transcription model.

## Dataset sources

- LibriTTS-R: <https://www.openslr.org/141/>
- Original LibriTTS: <https://www.openslr.org/60/>
