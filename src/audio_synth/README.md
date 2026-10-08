# Explicit neural speech synthesis and Piper reference

The active executables are `explicit-tts` (duration/F0/energy-conditioned acoustic
ONNX + neural vocoder) and `piper-tts` (pretrained VITS reference). Matcha has been
removed. Procedural DSP and untrained neural experiments are preserved under
`archive/` and excluded from the build.

## Build

From the repository root:

```sh
cd src/audio_synth
```

Windows requires CMake 3.25+, Visual Studio 2022 / Build Tools 2022 with Desktop
development with C++ and a Windows SDK. Presets select MSVC explicitly. Commands
below work as single lines in PowerShell and Git Bash; quote paths with spaces.

Fetch Piper assets and the shared ONNX Runtime SDK once, then build both runners
and run all tests:

```sh
python scripts/fetch_piper_voice.py
cmake --workflow --preset msvc-onnx
```

For only the explicit runner and its control/runtime tests:

```sh
cmake --workflow --preset explicit-neural
```

Reuse the configured directory for fast incremental builds:

```sh
cmake --build --preset explicit-neural
cmake --build --preset msvc-onnx --target piper-tts
ctest --preset msvc-onnx
```

Both ONNX presets use `build/onnx/Release/`. MSVC uses `/MP`, `/EHsc`, and `/utf-8`;
build presets allow eight project jobs. There are no archived model targets.
The explicit preset selects only its runner and two C++ test targets.

The SDK defaults to `../../dependencies/onnxruntime/`. With an existing SDK:

```sh
cmake --preset msvc-onnx -DONNXRUNTIME_ROOT="C:/sdk/onnxruntime"
cmake --build --preset msvc-onnx
ctest --preset msvc-onnx
```

`cmake --workflow --preset default` builds framework/prosody libraries and tests
without ONNX Runtime. It does not produce a usable neural speech executable.
`msvc-debug` uses `build/msvc-debug/Debug/`. Linux/macOS use `ninja-release`;
to enable neural runners, configure with `-DVA_ENABLE_ONNX_RUNTIME=ON` and
`-DONNXRUNTIME_ROOT` pointing to a platform SDK. Executables then have no `.exe`.
The SDK downloader supplies Windows x64 only.

## Explicit pipeline and emotion controls

`text -> CMU ARPAbet tokens -> trained prosody predictor -> explicit duration/F0/energy controls -> emotion preset -> scalar sliders -> acoustic ONNX -> 80-bin log-mel -> vocoder ONNX -> 24 kHz PCM16 WAV`

The supplied setup uses **PaddleSpeech FastSpeech2 VCTK + matching HiFi-GAN VCTK**,
both trained, native 24 kHz with hop 300. This is a public pretrained acoustic
model adapted to our explicit five-input interface, not a newly trained custom
network. The acoustic graph honors exact durations and frame-level pitch/energy;
its separate trained predictor supplies sensible controls when no CSVs are given.
See [the exact export contract](models/explicit_neural/README.md).

Prepare once (Python 3.11+; commands work in PowerShell and Git Bash):

```sh
python -m venv build/model-export-env
./build/model-export-env/Scripts/python.exe -m pip install onnx numpy
./build/model-export-env/Scripts/python.exe scripts/fetch_explicit_voice.py
cmake --workflow --preset explicit-neural
```

The fetcher uses public HTTPS releases, sequential cached downloads and pinned
checksums. It reads only the small statistics files from the training ZIP using
standard HTTP ranges. The trained graphs total about 257 MiB after export;
the acoustic/predictor split duplicates encoder weights. This matched vocoder
is about 50 MiB. The separate 3.75 MB candidate below is not compatible with it.
The shared ONNX SDK must already be prepared using the build instructions above.

Stage matching local exports (or publisher HTTPS URLs) outside output artifacts:

```sh
python scripts/prepare_explicit_models.py --acoustic "C:/exports/acoustic_generator.onnx" --vocoder "C:/exports/vocoder_hifigan.onnx" --tokens "C:/exports/tokens.tsv" --dictionary model_assets/piper/cmudict.dict
```

The helper records hashes and refuses to overwrite existing assets. Model
vocabulary, speaker IDs, mel preprocessing, hop length and sample rate must match.
Keep model cards, licenses and any ONNX external-data sidecars with the exports.

Generate real speech:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "We synthesize a clear acoustic voice." --emotion excited --output artifacts/explicit_neural/excited.wav --threads 4
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "We synthesize a clear acoustic voice." --emotion somber --output artifacts/explicit_neural/somber.wav --threads 4
```

For an immediately runnable **test-audio** example:

```sh
./build/onnx/Release/explicit-tts.exe --assets tests/fixtures/explicit --text "hello" --emotion whisper --output artifacts/explicit_fixture/whisper.wav
./build/onnx/Release/explicit-tts.exe --assets tests/fixtures/explicit --text "hello" --emotion somber --output artifacts/explicit_fixture/somber.wav
```

`--emotion` accepts `neutral`, `whisper`, `excited`, `somber`, `calm` (alias for
somber), and `authoritative`. Presets are deterministic prosody heuristics:
whisper lowers energy 40% and flattens voiced pitch; excited raises pitch/range
and speeds cadence; somber lowers pitch/energy variation and slows cadence;
authoritative emphasizes the first annotated stressed vowel and sharpens phrase
boundaries. These transforms do not guarantee natural emotion or physical whisper.
They retain `speaker_id`, rather than creating a new voice identity.

Presets run after control-file loading and before scalar sliders. `--speed`
above 1 is faster; below 1 is slower. `--pitch-scale` scales voiced F0 in Hz.
`--energy-scale` scales energy and `--energy-variance` scales its deviations from
the mean. To preserve exact supplied vectors, use neutral and default sliders:

```sh
./build/onnx/Release/explicit-tts.exe --assets tests/fixtures/explicit --text "hello" --durations examples/prosody/durations.csv --f0 examples/prosody/f0.csv --energy examples/prosody/energy.csv --emotion neutral --output artifacts/explicit_fixture/controlled.wav
```

Durations need one nonnegative integer per emitted token. The VCTK frontend uses
the publisher's stressed ARPAbet IDs without Piper padding or BOS/EOS. F0 and
energy need one value per frame, matching sum(durations). Cadence changes resample
within each token. The VCTK model uses continuous log-F0: zero F0 is mapped to
the training pitch mean, so it cannot force physical unvoiced/whispered speech.
Energy is relative to the training feature mean, not PCM volume or dB.
Defaults come from the trained predictor and `pipeline.properties`; only test
fixtures without a predictor use six frames, 180 Hz and energy 1.
Diagnostics record the selected preset and final controls. CMUdict is staged
locally, with adjacent `shared/`, parent, then `piper/` dictionary fallback.
The model's token map always remains its own. `--speaker-id` selects one of
107 trained VCTK speakers (0..106); names are in `speaker_id_map.txt`.
`--speaker-id`, `--frames-per-token`, `--f0-hz`, `--hop-length`, `--threads` and
`--thread-affinities` are also supported. Use `--help` for syntax.

For ten reusable voice profiles, speaker auditions, copyable CLI commands and
instructions for designing your own settings, see
[the vocal profiles guide](docs/VOCAL_PROFILES.md). Render all ten locally with
`python scripts/render_vocal_profiles.py`, or select one with
`python scripts/render_vocal_profiles.py --profile lower_narrator --text "Hello there."`.
For a C++-focused explanation of profile construction, expressive delivery,
exact contours and adding emotion policies, read
[Making voice profiles and expressive speech](docs/VOICE_PROFILE_DESIGN.md).

## Small vocoder asset

The retained HiFi-GAN v2 export is about 3.75 MB. Fetch only this vocoder with:

```sh
python scripts/fetch_vocoder.py
```

It is stored in `model_assets/vocoders/hifigan_v2.onnx` with a pinned hash. This
export is native **22,050 Hz**, 80 mel bins, hop 256. The current explicit CLI
expects **24,000 Hz**. It needs a compatible native-rate acoustic/vocoder pipeline
before use; renaming the graph or resampling a WAV cannot fix a mel mismatch.
It is not automatically substituted for the explicit pipeline's vocoder.

## Piper reference and voice controls

```sh
./build/onnx/Release/piper-tts.exe --voice en_US-lessac-medium --text "Can you understand this sentence clearly?" --output artifacts/piper/lessac.wav
./build/onnx/Release/piper-tts.exe --voice en_US-hfc_male-medium --text "Can you understand this sentence clearly?" --output artifacts/piper/hfc_male.wav
./build/onnx/Release/piper-tts.exe --voice en_US-lessac-medium --text "A slightly faster voice." --length-scale 0.85 --noise-scale 0.5 --noise-w 0.6 --threads 4
```

Change `--voice` to select a checkpoint. These two voices are single-speaker;
`--speaker-id` is 0. Prepared multi-speaker exports accept valid speaker IDs.
The fetcher supports these voices; `--voice en_US-lessac-medium` fetches just one.
`--length-scale` below 1 speeds up speech; above 1 slows it down. `--noise-scale`
controls generator noise and `--noise-w` duration noise. Defaults come from voice
properties. This runner has no direct F0/energy vectors or emotion preset flag.
It preserves the checkpoint's native sample rate.

## Files, tests and performance report

| Folder | Contents |
| --- | --- |
| `models/piper_onnx/`, `models/explicit_neural/` | Active C++ adapters |
| `model_assets/piper/` | Downloaded voices, dictionaries, vocabularies and provenance |
| `model_assets/explicit_neural/` | Trained FastSpeech2, prosody predictor, matching HiFi-GAN and frontend assets |
| `model_assets/vocoders/` | Separate vocoder candidates |
| `artifacts/` | Generated WAVs, diagnostics and benchmark results |
| `build/` | Binaries, object files, test outputs and caches |
| `../../dependencies/onnxruntime/` | Shared C++ SDK |
| `third_party/downloads/` | Download caches |
| `tests/fixtures/explicit/` | Tiny untrained contract-test graphs |
| `archive/` | Preserved experiments and scratch notes; excluded from CMake |

Synthesis executables read local assets and make no application-level network
requests. Downloads are explicit preparation steps. No downloaded input weights
or dictionaries belong in `artifacts/`. If older downloads exist under
`artifacts/models/`, migrate them to `model_assets/piper/`.

CTest covers framework metrics/features/WAV I/O, phonemizer annotations, duration
expansion, emotion formulas and invariants, CLI controls, tensor layout, and
incompatible ONNX exports. Test graphs do not establish speech quality.

Run the reproducible local benchmark (three measured fresh processes after one
discarded cache-warming process per case):

```sh
python scripts/benchmark_tts.py --explicit-assets model_assets/explicit_neural
```

Use `--include-fixture` only for an explicitly untrained transport comparison.
Inference timing includes prosody prediction, acoustic generation and vocoding.
Results and WAVs go under
`artifacts/benchmarks/`. Read [the architecture, emotion and measured performance
report](docs/TTS_ARCHITECTURE_AND_EMOTION_REPORT.md).

Evaluation libraries and the LibriTTS-R data preparation helper remain available:
`python scripts/fetch_libritts_r.py --budget-mb 250` fetches a streamed prefix of
the roughly 1.2 GB official archive; this prefix is not an official split.
Keep its CC BY 4.0 attribution and manifest. Sources:
[LibriTTS-R](https://www.openslr.org/141/) and [LibriTTS](https://www.openslr.org/60/).
Captions remain a separate project under `../captions`.
