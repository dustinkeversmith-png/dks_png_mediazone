# C++ speech synthesis engine

`explicit-tts` is the primary speech generator: trained FastSpeech2 VCTK with
explicit duration, pitch and energy controls plus matching HiFi-GAN.
`piper-tts` is the separate pretrained VITS reference. Phonemization, control
transforms, ONNX inference and WAV generation all run in C++.

An optional, separate C++ [`articulation-lab`](experiments/articulation/README.md)
tests individual-word and consonant/vowel controls with the existing explicit
model. Build with `cmake --workflow --preset articulation-lab`; it uses its own
build/output paths and is disabled in normal builds. See the
[measured sweeps and vocal-event research](docs/ARTICULATION_EXPERIMENT_AND_VOCAL_EVENTS.md)
for results and approaches to sighs, breaths and mouth sounds.

## Build and run on Windows

Run every command below from `src/audio_synth`. If you are at the repository root:

```sh
cd src/audio_synth
```

Requirements: CMake 3.25+, Visual Studio 2022 / Build Tools 2022 with Desktop
Development with C++ and a Windows SDK. Python 3.11+ is used for one-time model
preparation and optional test/tool scripts; it is not used for speech inference.
Commands are single lines compatible with PowerShell and Git Bash. No environment
activation, Bash line continuations or PowerShell backticks are required.

### Existing local assets: build and speak

```sh
cmake --workflow --preset msvc-onnx
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion excited --text "We synthesize a clear acoustic voice." --output artifacts/explicit_neural/excited.wav --threads 4
./build/onnx/Release/piper-tts.exe --assets model_assets/piper --voice en_US-lessac-medium --text "We synthesize a clear acoustic voice." --output artifacts/piper/reference.wav --threads 4
```

### First-time preparation

These explicit preparation steps download public releases, stage the shared
Windows x64 ONNX Runtime SDK and place model inputs outside output artifacts:

```sh
python -m venv build/model-export-env
./build/model-export-env/Scripts/python.exe -m pip install onnx numpy
./build/model-export-env/Scripts/python.exe scripts/fetch_piper_voice.py
./build/model-export-env/Scripts/python.exe scripts/fetch_explicit_voice.py
cmake --workflow --preset msvc-onnx
```

The explicit fetcher caches sequential HTTPS downloads, checks pinned release
hashes and reads the small training statistics with ordinary HTTP byte ranges.
No synthesis executable downloads assets. The explicit trained graphs total
about 257 MiB after export; splitting the predictor duplicates encoder weights.
The matching vocoder is about 50 MiB. Model provenance is in
`model_assets/explicit_neural/download_manifest.json` and
[the model card](docs/EXPLICIT_VOICE_MODEL_CARD.md).

### Fast incremental builds and other presets

```sh
cmake --workflow --preset explicit-neural
cmake --build --preset explicit-neural
ctest --preset explicit-neural
cmake --build --preset msvc-onnx --target piper-tts
ctest --preset msvc-onnx
cmake --workflow --preset default
cmake --workflow --preset msvc-debug
```

The two ONNX presets share `build/onnx/Release/`. The explicit preset builds its
runner and control/runtime tests; the full preset builds both runners and all
tests. MSVC uses C++20, `/W4`, `/MP`, `/EHsc`, `/utf-8` and eight build jobs.
`default` and `msvc-debug` build framework tests without ONNX inference; their
outputs are `build/msvc-release/Release/` and `build/msvc-debug/Debug/`.
The SDK prefix is `../../dependencies/onnxruntime/`; configure the
`ONNXRUNTIME_ROOT` CMake cache entry if using another installed SDK.
The `ninja-release` preset is for Linux/macOS and is not a Windows command.

## Pipeline and operational modes

```text
Raw Text -> CMU Phonemizer -> Learned Predictor / Base Contours
         -> Emotion & Scalar Sliders -> Explicit Acoustic ONNX
         -> HiFi-GAN Vocoder -> 24 kHz WAV
```

The pretrained acoustic graph is adapted to exactly five inputs: `input_ids`,
`durations`, `f0`, `energy`, `sid`. Output is `[1,80,frames]` mel. The matching
pair uses native 24 kHz, hop 300. This is pretrained FastSpeech2 with our explicit
control layer, not a newly trained custom network. See
[the full tensor contract](models/explicit_neural/README.md).

### Mode 1: automated synthesis

The trained predictor generates duration, pitch and energy for the supplied
text and speaker. Emotion policies and scalar sliders modify those predictions:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 1 --emotion neutral --pitch-scale 0.95 --speed 0.95 --energy-scale 0.95 --energy-variance 0.8 --text "The moon rose above the quiet garden." --output artifacts/explicit_neural/narrator.wav --threads 4
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion somber --text "We synthesize a clear acoustic voice." --output artifacts/explicit_neural/somber.wav --threads 4
```

Speaker IDs 0..106 select trained VCTK identities; the publisher names are in
`speaker_id_map.txt`. The frontend uses the publisher's stressed ARPAbet IDs
without Piper BOS/EOS/padding. CMUdict is resolved locally, then in adjacent
`shared/`, the parent directory and adjacent `piper/`; token maps stay model-specific.

`--emotion` accepts `neutral`, `excited`, `somber`, `calm` (somber alias),
`authoritative` and `whisper`. Presets run before scalar sliders. Speed above 1
is faster; pitch scale multiplies Hz; energy scale and variance change acoustic
features, not exact playback gain. Emotion policies are adjustable heuristics.
The continuous-F0 model cannot guarantee true whisper or unvoiced excitation.

### Mode 2: fully manual curves

Supplying **all three** curve files bypasses the predictor entirely. The
checked-in example is for the exact sentence below, speaker 0 and this model:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion neutral --text "We synthesize a clear acoustic voice." --durations examples/prosody/durations.csv --f0 examples/prosody/f0.csv --energy examples/prosody/energy.csv --output artifacts/explicit_neural/manual.wav --threads 4
```

Durations contain one nonnegative integer per phoneme token; F0 and energy
contain one value per frame, with lengths equal to the positive sum of durations.
F0 is Hz; energy is relative to the training feature mean. Use neutral/default
sliders to preserve the supplied vectors, or deliberately apply another preset.
Partial file overrides retain the predictor for missing curves. `--f0-hz` and
`--frames-per-token` explicitly replace learned defaults. Manual CSVs are
text-specific; see [the manual examples](examples/prosody/README.md).

Each WAV has a `.diagnostics.json` recording final controls, speaker/emotion,
actual prosody source and timings. Fully manual mode reports `manual-curves`
and `prosody_ms: 0`. Input/shape errors still fail explicitly.

## Profiles and expressive delivery

[Ten direct C++ commands](docs/VOCAL_PROFILES.md) and
[the C++ profile design tutorial](docs/VOICE_PROFILE_DESIGN.md) explain identity,
delivery, word emphasis and exact contours. The optional Python batch helper
only launches the same executable; edit the saved recipes in
`examples/vocal_profiles/profiles.json` if desired:

```sh
python scripts/render_vocal_profiles.py --list
python scripts/render_vocal_profiles.py --profile lower_narrator --text "Hello there."
```

Word-level emotion tags and coupled pitch/energy/duration transforms are planned
in [the expressive prosody RFC](docs/ROADMAP_EXPRESSIVE_PROSODY.md). They are not
current CLI flags or supported text markup.

## Piper reference

```sh
./build/onnx/Release/piper-tts.exe --assets model_assets/piper --voice en_US-hfc_male-medium --text "Can you understand this sentence clearly?" --output artifacts/piper/hfc_male.wav --threads 4
./build/onnx/Release/piper-tts.exe --assets model_assets/piper --voice en_US-lessac-medium --text "A slightly faster voice." --length-scale 0.85 --noise-scale 0.5 --noise-w 0.6 --output artifacts/piper/faster.wav --threads 4
```

`--voice` selects a checkpoint. The prepared voices are single-speaker, ID 0.
Piper length scale below 1 speeds speech; above 1 slows it. Noise scale and
noise-w control generator/duration noise. Piper retains its native sample rate
and has no explicit F0/energy curve or emotion-preset interface.

## Repository layout and validation

| Directory | Purpose |
| --- | --- |
| `src/` | Audio I/O, features, metrics, evaluation, reporting, datasets and studies |
| `include/vocal/` | Shared public interfaces, frontend and control structs |
| `models/explicit_neural/` | Primary acoustic generator, predictor, regulator and vocoder |
| `models/piper_onnx/` | Reference VITS adapter, shared frontend and ONNX session tools |
| `app/` | The two C++ runners and CLI/asset helpers |
| `tests/` | Framework, prosody, runtime, CLI and trained-model checks; tiny fixtures |
| `examples/` | Real manual contours, reusable profiles and evaluation example data |
| `docs/` | Current architecture, profile guides, model card and roadmap |
| `model_assets/` | Local model inputs and dictionaries; ignored by Git |
| `artifacts/` | Generated audio, diagnostics and measurements |
| `build/` | Binaries, object files, tool environments, test output and caches |
| `third_party/downloads/` | Cached publisher downloads |
| `archive/` | Legacy prototypes, scratch snapshots, historical docs and inactive candidates |

CMake lists active sources explicitly and has no archive build option.
CTest checks metrics/features/WAV I/O, phonemization, duration expansion, emotion
policies, malformed model contracts, CLI options and both operational modes.
The trained CLI suite is enabled when local predictor assets exist; Python is
optional for the CLI test harness, not for C++ inference. Fixture graphs are
confined to tests and do not generate real speech.

```sh
python scripts/benchmark_tts.py --explicit-assets model_assets/explicit_neural
```

Benchmarks write under `artifacts/benchmarks/` and include predictor, acoustic
and vocoder inference plus separate wall latency. Existing measurements and
independent recognition/export checks are described in
[the technical report](docs/TTS_ARCHITECTURE_AND_EMOTION_REPORT.md).
Evaluation libraries and `scripts/fetch_libritts_r.py` remain available for
explicit data preparation. Captions remain a separate project under `../captions`.
