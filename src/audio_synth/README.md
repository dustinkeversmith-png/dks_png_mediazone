# Audio synthesis runners

Each model setup has its own executable for generating vocal audio artifacts.
The previous `vocal-eval` application has been removed. Framework evaluation
APIs remain available in `include/vocal/`, with unit tests.

## Build

From the repository root, enter the folder containing this README and the presets:

```sh
cd src/audio_synth
```

Windows requires CMake 3.25+ and Visual Studio 2022 / Build Tools 2022 with
Desktop development with C++ and a Windows SDK. These presets select MSVC
explicitly, avoiding the MSYS compiler that could not find C++ standard headers.
All Windows commands below are single lines usable in Git Bash and PowerShell.
Use forward slashes, `./` for executables, and quotes around paths with spaces.

For the procedural and untrained baseline runners:

```sh
cmake --workflow --preset default
```

For all ONNX runners, with the shared ONNX Runtime SDK:

```sh
python scripts/fetch_piper_voice.py
cmake --workflow --preset msvc-onnx
```

The existing voice fetcher puts downloads in `model_assets/piper/` and the SDK in
`../../dependencies/onnxruntime/`. If the SDK is installed elsewhere, configure
with its actual path, then build and test:

```sh
cmake --preset msvc-onnx -DONNXRUNTIME_ROOT="C:/sdk/onnxruntime"
cmake --build --preset msvc-onnx
ctest --preset msvc-onnx
```

For the chosen compact pretrained acoustic/vocoder setup:

```sh
python scripts/fetch_matcha_voice.py
cmake --workflow --preset matcha
```

The `matcha` workflow builds the native Matcha runner and shared contract tests.
The downloader makes sequential ordinary HTTPS requests to public publisher
release assets, totaling about 80 MB plus CMUdict if it is not already available.

For a quick build of the strict explicit-control pipeline and its contract tests:

```sh
cmake --workflow --preset explicit-neural
```

This workflow uses the same `build/onnx/` directory as `msvc-onnx`; it requires
an installed SDK but does not download or require pretrained speech weights
for its tests. To build only one runner after configuring:

```sh
cmake --build --preset msvc-onnx --target matcha-tts
cmake --build --preset msvc-onnx --target explicit-tts
cmake --build --preset msvc-onnx --target piper-tts
cmake --build --preset default --target formant-tts
```

MSVC uses `/MP` for parallel source compilation, `/EHsc` for C++ exceptions and
`/utf-8` for IPA literals. Build presets allow eight project jobs. Reuse build
directories for incremental builds; avoid cleaning or configuring on every edit.
`msvc-debug` builds debug runners under `build/msvc-debug/Debug/`.
Linux/macOS can use `ninja-release` with Ninja and a C++20 compiler; executables
are in `build/ninja-release/` without `.exe`. For ONNX on those platforms,
configure that preset with `-DVA_ENABLE_ONNX_RUNTIME=ON` and the path to a
platform-appropriate SDK; the bundled SDK downloader is Windows x64 only.

## Separate executables

| Executable | Model setup | Downloaded weights required |
| --- | --- | --- |
| `formant-tts` | Formant source-filter sketch | No |
| `hmm-tts` | Statistical/HMM sketch | No |
| `articulatory-tts` | Articulatory sketch | No |
| `lpc-tts` | LPC sketch | No |
| `sine-wave-tts` | Sine-wave sketch | No |
| `unit-selection-tts` | Procedural unit selection | No |
| `homebrew-tts` | Fixed, untrained neural baseline | No |
| `onnx-style-tts` | Untrained FastSpeech2-shaped baseline | No; this does not run ONNX |
| `piper-tts` | Pretrained end-to-end Piper/VITS | Piper checkpoint and frontend assets |
| `matcha-tts` | Three-step pretrained Matcha acoustic model + small HiFi-GAN v2 | `model_assets/matcha/` |
| `explicit-tts` | Explicit acoustic generator followed by a frozen neural vocoder | Compatible acoustic/vocoder exports and vocabulary |

The first eight runners are deterministic acoustic sketches, not trained voices.
Each accepts `--text` and `--output`; with no arguments it renders a default
sentence into its own `artifacts/<model>/speech.wav`. All runners provide `--help`
and write a sibling `.diagnostics.json`. Baseline, Matcha and explicit audio is mono
24 kHz PCM16; Piper preserves its checkpoint's native sample rate.

```sh
./build/msvc-release/Release/formant-tts.exe --text "Hello from the formant model."
./build/msvc-release/Release/homebrew-tts.exe --text "Hello from the neural sketch." --output artifacts/homebrew/example.wav
```

## Piper example and voice controls

After fetching voices and building `msvc-onnx`:

```sh
./build/onnx/Release/piper-tts.exe --voice en_US-lessac-medium --text "Can you understand this sentence clearly?" --output artifacts/piper/lessac.wav
./build/onnx/Release/piper-tts.exe --voice en_US-hfc_male-medium --text "Can you understand this sentence clearly?" --output artifacts/piper/hfc_male.wav
```

Change `--voice` to switch checkpoints. Both bundled voices have one speaker,
so their speaker ID is 0. Prepared multi-speaker voices support `--speaker-id N`
within the model's speaker range. The fetcher currently supports the two English
voices above; `--voice en_US-lessac-medium` fetches just that voice.

```sh
./build/onnx/Release/piper-tts.exe --voice en_US-lessac-medium --text "A slightly faster voice." --length-scale 0.85 --noise-scale 0.5 --noise-w 0.6 --threads 4
```

`--length-scale` below 1 is faster and above 1 is slower. `--noise-scale` controls
generator noise, and `--noise-w` controls duration noise. Without those flags,
the CLI uses the voice's `.properties` defaults. There is no dedicated pitch
control in Piper here. See [Piper's synthesis controls](https://github.com/OHF-Voice/piper1-gpl/blob/main/src/piper/config.py).

## Small pretrained acoustic + vocoder example (Matcha)

The selected [public English Matcha export](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html)
uses a three-step acoustic model and the publisher-supported HiFi-GAN v2 vocoder.
The acoustic archive is 76.7 MB and the vocoder is 3.75 MB; no framework training
packages are needed for C++ inference. Downloads are cached, SHA-256 recorded,
and the acoustic archive is checked against the publisher's pinned release hash.

```sh
python scripts/fetch_matcha_voice.py
cmake --workflow --preset matcha
./build/onnx/Release/matcha-tts.exe --text "Can you understand this sentence clearly?" --output artifacts/matcha/example.wav --threads 4
./build/onnx/Release/matcha-tts.exe --text "Can you understand this sentence clearly?" --output artifacts/matcha/faster.wav --speed 1.2 --noise-scale 0.8 --threads 4
```

The pipeline is `CMUdict/IPA -> native Matcha ONNX -> 80-bin mel -> HiFi-GAN v2 ONNX -> WAV`.
The original graphs are used without rewriting or retraining. The source pair
runs at 22.05 kHz; the runner resamples its waveform to canonical 24 kHz before
PCM16 export. `--speed` above 1 is faster, and `--noise-scale` controls acoustic
sampling noise. It is a single female English voice trained on LJSpeech.

This public export predicts its own durations and exposes speed/noise inputs;
it does **not** expose per-token durations or F0/energy contours. The runner
therefore rejects unsupported pitch/energy flags. The stricter implementation
below remains available for exports that actually support those controls.
The project's CMUdict frontend is reused with the publisher's vocabulary and
blank/boundary convention. Upstream uses eSpeak, so pronunciation/text
normalization can differ, especially for numbers and abbreviations. Use written
English words for initial comparisons; OOV words fall back to letter pronunciations.

## Standalone explicit acoustic + vocoder example

The pipeline is:

`text -> CmuPhonemizer -> token IDs + duration/F0/energy targets -> acoustic ONNX -> 80-bin log-mel -> vocoder ONNX -> PCM16 WAV`

The supplied design specifies a model interface, **not trained checkpoints**.
The downloaded native Matcha pair above uses a different input interface; it
cannot be passed to this stricter runner. Piper VITS checkpoints also cannot be
used here. To generate intelligible
speech, provide trained exports matching the contract in
[models/explicit_neural/README.md](models/explicit_neural/README.md), including
the acoustic model's exact vocabulary and a vocoder trained for its mel convention.

Put matching assets in `model_assets/explicit_neural/`. You can stage local files
or download from your model publisher's HTTPS URLs with the helper below.
Replace these example source paths with existing exports:

```sh
python scripts/prepare_explicit_models.py --acoustic "C:/exports/acoustic_generator.onnx" --vocoder "C:/exports/vocoder_hifigan.onnx" --tokens "C:/exports/tokens.tsv" --dictionary model_assets/piper/cmudict.dict
```

The helper records source paths/URLs and SHA-256 hashes in the assets folder.
It refuses to overwrite existing files; use a new `--output` for a different
model setup. Preserve upstream model cards and licenses alongside your exports.
For exports using ONNX external tensor data, stage their sidecar files manually.

Run the dedicated executable with compatible trained assets:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello from the explicit neural pipeline." --output artifacts/explicit_neural/example.wav --pitch-scale 1.1 --speed 0.95 --threads 4
```

`--speed` is a rate multiplier: below 1 is slower, above 1 faster. Each positive
token duration is divided by speed, rounded to frames and kept at least one;
zero durations remain zero. Pitch scales F0 in Hz and preserves unvoiced zeros.
`--energy-scale` multiplies normalized energy; `--energy-variance` scales deviations
from mean energy, clipped at zero. These inputs condition the exported model;
the model must have been trained to interpret them.

For precise controls, supply CSV/whitespace files:

```sh
./build/onnx/Release/explicit-tts.exe --assets tests/fixtures/explicit --text "hello" --durations examples/prosody/durations.csv --f0 examples/prosody/f0.csv --energy examples/prosody/energy.csv --output artifacts/explicit_neural/controlled.wav
```

Those checked-in files are a runnable control example for the tiny `hello`
fixture vocabulary. For trained exports, supply matching control files. Durations need one
integer per token, including BOS/EOS and padding emitted by the frontend.
F0 and energy need one value per frame, matching the sum of durations. When speed
changes, contours are resampled within each token; pitch/energy sliders are then
applied. Diagnostics record the actual token IDs, targets, timing and sample counts.
Defaults are six frames per token, constant 180 Hz F0 and energy 1, an illustrative
baseline rather than a trained prosody predictor. `--frames-per-token`, `--f0-hz`,
`--speaker-id`, `--hop-length` and `--thread-affinities` are also available.

### Runnable contract example without trained weights

The checked-in tiny fixtures exercise the complete executable and produce
**step-shaped test audio, not speech**:

```sh
./build/onnx/Release/explicit-tts.exe --assets tests/fixtures/explicit --text "hello" --output artifacts/explicit_fixture/hello.wav --pitch-scale 1.1 --speed 0.95
```

They are not downloaded models. To regenerate a separate fixture asset set,
install the Python `onnx` package into an isolated environment, then run:

```sh
python scripts/create_explicit_test_models.py
```

This writes to `model_assets/explicit_fixture/`. The inference executables and
normal CTest runs do not require Python ONNX or download any models.

## Files and outputs

| Folder | Contents |
| --- | --- |
| `models/` | C++ model implementations |
| `model_assets/piper/` | Downloaded Piper voices, dictionary, vocabulary, configs, cards and manifest |
| `model_assets/matcha/` | Public pretrained Matcha acoustic graph, HiFi-GAN v2, CMUdict, vocabulary and manifest |
| `model_assets/explicit_neural/` | Supplied/downloaded trained acoustic generator, vocoder, CMUdict, vocabulary and manifest |
| `model_assets/explicit_fixture/` | Locally generated untrained example graphs, if requested |
| `../../dependencies/onnxruntime/` | Shared C++ SDK |
| `third_party/downloads/` | Cached SDK archives |
| `artifacts/<model>/` | Generated WAVs and execution diagnostics |
| `build/` | Executables, object files, caches and test outputs |
| `tests/fixtures/explicit/` | Tiny versioned contract-test graphs, with no trained weights |

No input weights or dictionaries are stored in `artifacts/`. If you have older
voice downloads in `artifacts/models/`, move them into `model_assets/piper/`.
`model_assets/`, download caches and generated output are ignored by Git.

## Evaluation data and tests

The existing evaluation features, scoring, reporting and dataset libraries remain
available to C++ callers. To fetch the LibriTTS-R prefix sample (about 250 MiB):

```sh
python scripts/fetch_libritts_r.py --budget-mb 250
```

The official test_clean archive is roughly 1.2 GB; this streamed prefix is not an
official split. Preserve CC BY 4.0 attribution and the generated manifest.
Sources: [LibriTTS-R](https://www.openslr.org/141/) and [LibriTTS](https://www.openslr.org/60/).

CTest covers existing metrics, deterministic length expansion, invalid controls,
mel layout, vocoder layout, CLI control overrides and incompatible ONNX exports.
The ONNX tests use only tiny local fixtures. The explicit pipeline uses CPU
sequential execution, one inter-op thread, memory arenas, memory patterns and
no spinning; optional worker affinity follows
[ONNX Runtime's threading configuration](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

Captions remain a separate project under `../captions`; only the shared SDK is
common. See [models/README.md](models/README.md) for the model-family layout.
