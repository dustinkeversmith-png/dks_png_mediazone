// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\README.md ===
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

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\docs\EXPLICIT_VOICE_MODEL_CARD.md ===
# Local explicit English voice

This setup adapts public PaddleSpeech FastSpeech2 VCTK and HiFi-GAN VCTK
ONNX releases. The acoustic model is pretrained FastSpeech2; our contribution
is an explicit inference interface and programmatic controls, not new training.

Publisher resources:

- [Released models](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md)
- [FastSpeech2 implementation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/models/fastspeech2/fastspeech2.py)
- [English frontend](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/frontend/phonectic.py)
- [Publisher Apache 2.0 license](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/LICENSE)

`download_manifest.json` records the exact publisher archive URLs, pinned hashes,
statistics and hashes of generated assets. Dataset and checkpoint terms remain
those of their publishers; this card does not grant additional rights.

The acoustic export retains trained encoder, speaker conditioning, decoder,
postnet and variance embeddings. The predictor export retains trained duration,
pitch and energy heads. We replace internal predicted duration expansion with
an exact Gather expansion and move trained kernel-size-one variance embeddings
to frame positions. Repeated original token contours reproduce the upstream
neutral mel within float rounding; frame contours permit finer control.
The vocoder modification changes tensor layout only.

Native rate: 24000 Hz. Hop: 300. FFT: 2048. Hann window: 1200.
Mel: 80 bins, 80..7600 Hz, original model denormalization preserved.
Vocabulary: publisher stressed ARPAbet IDs, no Piper BOS/EOS/padding.
Speaker map: 107 identities, IDs 0..106; default ID 0 is p225.

F0 is supplied in Hz and transformed to continuous normalized log-F0, mean
5.0610495 and standard deviation 0.35158658. Zero F0 maps to the pitch mean;
there is no independent voicing control. Energy is relative to the training
feature mean 29.903042 with standard deviation 25.684935. It is not waveform dB.
Emotion presets alter duration/pitch/energy; they are not trained emotion labels
and do not guarantee natural emotion or physical whisper. Extreme values can
degrade speech. CMUdict letter fallback is limited for arbitrary names/numbers.

All inference is local. Only explicit preparation scripts download assets.
Weights belong in `model_assets/explicit_neural/`; generated WAVs and validation
reports belong in `artifacts/`. Tiny fixture graphs are exclusively test assets.

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\docs\HISTORICAL_FIXTURE_REPORT.md ===
# Historical snapshot: fixture-only state before trained assets

This report is retained as history. Current results are in TTS_ARCHITECTURE_AND_EMOTION_REPORT.md.

# TTS architecture, emotion controls and local performance report

Measured on October 7, 2026 (America/Los_Angeles); benchmark UTC timestamp:
`2026-10-08T01:40:53.388801+00:00`.

## 1. Result and scope

The active model setups are now `explicit-tts` and `piper-tts`. The classical DSP
sandbox and deterministic neural-shaped baselines are preserved under `archive/`
and excluded from CMake. Matcha remains removed, following the user's later
instruction; the older pasted request to preserve and benchmark it is superseded.

The explicit control system now includes a C++ emotion-preset API and CLI flag.
Its transforms, annotation transport, duration accounting, runtime adapter and
WAV output pass automated tests. This establishes the implementation's control
and tensor contracts. It does **not** establish a trained, natural-sounding
explicit voice: no compatible trained acoustic generator is currently present.

Piper can generate trained speech using existing local checkpoints. The retained
small HiFi-GAN v2 vocoder is a separate 22.05 kHz candidate. It does not complete
the 24 kHz explicit pipeline and is not automatically loaded by that runner.

## 2. Repository cleanup and build organization

| Previous location | Current location / disposition |
| --- | --- |
| `models/dsp_paradigms/` | `archive/dsp_paradigms/` |
| `models/homebrew_neural/` | `archive/homebrew_neural/` |
| `app/demo_main.cpp`, `demo_tts.cpp`, `demo_tts.hpp` | `archive/app/` |
| Toy-model assertions mixed into `tests/metrics_test.cpp` | Preserved in `archive/tests/baseline_checks.cpp.txt`; removed from active tests |
| `DSP.md`, `UNIT_SELECTION.md`, `acoutics.txt`, `include/net/README.md` | `archive/notes/` |
| Combined C++/header scratch snapshots | `archive/scratch/`, preserved without rewriting |
| Matcha runner, adapter, downloader, presets, weights, cache, generated samples | Removed in the preceding cleanup |
| Obsolete baseline executables, project files and intermediates in `build/` | Retired so they cannot be mistaken for active targets |

There were six DSP demo families and two homebrew/neural-shaped demos, eight
runners in total. No inactive implementation was found in `src/`: `audio.cpp`,
`features.cpp`, `metrics.cpp`, `reporting.cpp`, `evaluator.cpp`, `dataset.cpp`,
and `study.cpp` all remain active. Relevant unit/runtime tests and data
preparation helpers remain active. Archive code is reference material; its
historical includes are not an automatically supported build configuration.

`CMakeLists.txt` no longer defines or links any archived model/demo target.
The active library boundaries separate framework code, phonemization, prosody,
Piper inference and explicit inference. Archived code adds no source-compilation
work to either current workflow.

Use `cmake --workflow --preset msvc-onnx` to configure/build/test both neural
runners, or `cmake --workflow --preset explicit-neural` for just the explicit
runner and its tests. After initial configuration, use `cmake --build --preset
explicit-neural`; avoid clean rebuilds and repeated SDK/model preparation.
MSVC `/MP` enables source compilation in parallel; build presets permit eight
project jobs. Release builds remain incremental. A measured no-change build of
all retained ONNX targets took **0.875 seconds** on this computer.
This is an incremental scheduling/link-check measurement, not a clean-build time
or a claimed before/after speedup.

## 3. Local assets, provenance and footprint

| Asset | Current size | Role / compatibility |
| --- | --- | --- |
| `model_assets/piper/en_US-lessac-medium.onnx` | 63,201,294 bytes (60.27 MiB) | Trained single-speaker VITS reference, 22,050 Hz |
| `model_assets/piper/en_US-hfc_male-medium.onnx` | 63,201,294 bytes (60.27 MiB) | Alternate trained single-speaker VITS voice, 22,050 Hz |
| `model_assets/piper/cmudict.dict` | 3,618,488 bytes (3.45 MiB) | English dictionary shared by the frontend |
| Piper voice-specific `*.tokens.tsv`, `*.properties`, `*.onnx.json`, model cards | Small sidecars | Exact model IDs, native rate, speaker count, default noise/rate controls and provenance |
| `model_assets/vocoders/hifigan_v2.onnx` | 3,749,714 bytes (3.58 MiB / 3.75 MB) | Pretrained vocoder only, 80 bins, hop 256, 22,050 Hz |
| `model_assets/explicit_neural/` | No compatible trained set currently staged | Intended location for the supplied acoustic generator, vocoder, exact token map and dictionary |
| `tests/fixtures/explicit/acoustic_generator.onnx` | 1,531 bytes | Untrained arithmetic contract fixture |
| `tests/fixtures/explicit/vocoder_hifigan.onnx` | 489 bytes | Untrained repeat/tanh fixture, not a HiFi-GAN network |

The two Piper ONNX weights occupy 120.55 MiB together; choosing one voice avoids
downloading/loading the other. Both installed voice cards report one speaker.
`--voice` selects the checkpoint; `--speaker-id 0` selects its only speaker.
An arbitrary speaker ID does not create a new voice. Multi-speaker acoustic
exports must have been trained with the requested IDs/embeddings.

The tiny acoustic fixture uses simple arithmetic on F0, energy, token-ID sum,
duration sum, speaker ID and mel-bin index. The fixture vocoder selects mel
channel zero, applies tanh and repeats each frame 256 times. Neither contains
learned speech weights. Their behavior is useful for checking that controls
reach the graph, never for assessing naturalness or trained-model speed.

The preserved HiFi-GAN graph is the original public export:
[HiFi-GAN v2 release asset](https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx).
Its pinned SHA-256 is
`a41d404cce7924493540238da5b30a4bc14b6ddaf1a37f3c79fa4f59548c19f0`.
`scripts/fetch_vocoder.py` fetches only this file with an ordinary HTTPS request,
reuses existing files and verifies the hash. It downloads no acoustic model.

The explicit CLI currently requires 24 kHz output. This 22.05 kHz vocoder requires
a compatible native-rate acoustic/mel pipeline before it can be used. Matching
80-bin tensor shapes is insufficient: sample rate, hop, FFT/window, frequency
limits, log convention and normalization must agree. The upstream HiFi-GAN
pipeline has explicit mel preprocessing/configuration parameters.
See [HiFi-GAN's mel preprocessing](https://github.com/jik876/hifi-gan/blob/master/meldataset.py).
A WAV resampler changes an already generated waveform; it cannot repair wrongly
conditioned mel input. No graph rewriting, fake model or silently substituted
synthesizer was added to bridge this missing trained pair.

### Runtime networking

The synthesis binaries open dictionaries, vocabularies, properties and ONNX
weights from local paths. The C++ inference path constructs local CPU ONNX
sessions and calls synchronous `Run()`. Inspection of active `app/`, `models/`,
`src/` and `include/` found no HTTP/download/socket code in synthesis.
The executables do not invoke fetch scripts, fall back to downloading missing
weights, or request a hosted inference service. Missing assets produce errors.

Network access is confined to separately invoked preparation helpers such as
`fetch_piper_voice.py`, `fetch_vocoder.py` and HTTPS staging in
`prepare_explicit_models.py`. `benchmark_tts.py` performs no downloads. Download
manifests record upstream sources and file hashes; benchmark results also record
actual asset and executable hashes. This is a source-level application-network
finding, rather than a claim based on packet capture of every third-party DLL.

Downloaded inputs live under `model_assets/`; generated WAVs/diagnostics/results
live under `artifacts/`. SDKs remain in `../../dependencies/onnxruntime/`, with
archives under `third_party/downloads/`. Benchmark-only fixture assets are
staged in `build/benchmark-fixture/`, not the output-artifact folder.

## 4. Actual explicit architecture

```text
Raw English text
    |
    v
CMUdict lookup -> ARPAbet -> IPA -> export-specific model token IDs
    |                             + token annotations (consonant/vowel/stress/boundary)
    v
Base ProsodyControls: durations[token], F0[frame], energy[frame], sid
    |  CSV targets or illustrative constant defaults
    v
apply_emotion_preset() -> apply_prosody_sliders()
    |  token-local contour remapping when durations change
    |  validate sum(durations) == len(F0) == len(energy)
    v
Acoustic ONNX: input_ids, durations, f0, energy, sid
    |  single graph owns hidden-state expansion / length regulation
    v
mel [1, 80, frames] -> framework frame-major MelSpectrogram
    v
Neural vocoder ONNX: channel-major mel -> float waveform
    v
24,000 Hz mono PCM16 WAV + final-control/timing diagnostics
```

The independent C++ `length_regulate()` expands `[tokens, hidden_dim]` rows by
exact integer frame counts and skips zero-duration tokens. It is tested and
available to split encoder/decoder designs. **It is not inserted into the current
single-graph inference path**, because that ONNX graph does not expose hidden
states. Its exporter must implement equivalent expansion internally and honor
the supplied targets. Output shape checks alone cannot prove that it does so
semantically. This distinction corrects the pasted diagram's implication that a
C++ length regulator already runs between two exported neural subgraphs.

The acoustic contract is five inputs: int64 `input_ids` and `durations` of
shape `[1,tokens]`, float32 `f0` and `energy` of shape `[1,frames]`, int64 `sid`
of shape `[1]`; output is finite float32 `mel [1,80,frames]`. The vocoder receives
that mel and emits `[1,frames*hop]` or `[1,1,frames*hop]` float32 audio. Default
hop is 256. Both layout transposes are explicit, and local input/output storage
outlives the synchronous ONNX calls.

The current interface is FastSpeech2-style, but a compatible **trained
FastSpeech2 checkpoint has not been loaded**. Architecture names alone do not
supply the required tensor interface. FastSpeech 2 research motivates duration,
pitch and energy conditioning; existing public implementations also expose
scalar control ratios, which are not automatically the exact per-frame export
used here. Sources: [FastSpeech 2 paper](https://arxiv.org/abs/2006.04558) and
[reference implementation](https://github.com/ming024/FastSpeech2).

## 5. Emotion presets: implemented policies and limits

The API is `ProsodyControls apply_emotion_preset(const ProsodyControls& base,
VocalEmotion emotion)`. Presets return a copy and retain speaker identity.
Define muF as the mean of **positive** input F0 values and muE as mean input
energy. Unvoiced F0 zero is never included in muF or converted to voiced pitch.
Voiced values after transformations are floored at 1 Hz to avoid turning a low
positive F0 into an unvoiced zero; this is a mathematical guard, not a physically
recommended pitch. Model-appropriate control ranges remain the caller's job.

| Preset | Voiced F0 transformation | Energy transformation | Cadence / token-specific edits |
| --- | --- | --- | --- |
| Neutral | Identity | Identity | Identity; validates the input |
| Whisper | muF + 0.15 * (F0 - muF) | 0.60 * E | Annotated unvoiced consonants: ceil(duration * 1.10); all other durations unchanged |
| Excited | muF + 35 Hz + 1.60 * (F0 - muF) | max(0, 1.10 * (muE + 1.35 * (E - muE))) | Speed 1.15: round(duration / 1.15), positive durations at least one frame |
| Somber / Calm | muF - 25 Hz + 0.65 * (F0 - muF) | max(0, 0.85 * muE + 0.40 * (E - muE)) | Speed 0.88; final 20% of each annotated phrase gets a gentle downward ramp reaching 10% pitch reduction |
| Authoritative | clamp(muF, 140, 190) + 0.35 * (F0 - muF) | 1.05 * E; first annotated stressed-vowel run per phrase gets another 1.25 multiplier | Annotated boundaries halve duration (min one if positive) and multiply energy by 0.35 |

F0 and energy contours are remapped by nearest-neighbor sampling **within each
token** when cadence changes. Speaker IDs, annotation ordering and zero-duration
tokens are retained. The frame budget remains 15,000 and the token budget 4,096.
Invalid inputs, non-finite output, unknown emotions and malformed annotations
fail rather than silently continuing.

`token_kinds` is optional for C++ callers. It labels unvoiced consonants, vowels,
stressed vowels and boundaries; other tokens remain unknown. The CMU frontend
adds annotations for each emitted ID, including multiple IDs per symbol; padding
and stress/length marks are unknown. Punctuation and EOS mark boundaries. When
annotations are absent, token-specific consonant/stress/boundary edits are
skipped and somber falls at the utterance end. Zero F0 alone cannot identify an
unvoiced consonant versus padding, silence or another token.

The 140..190 Hz authoritative register is a configurable-in-source heuristic,
not a universal speaker-independent definition of authority. Integer rounding
can leave very short token durations unchanged. Authoritative accent targets the first adjacent run of stressed vowel tokens
(padding ends the run), not a full syllable or linguistic stress predictor.

CLI order is: load base CSV/default controls -> attach frontend annotations ->
apply emotion -> apply scalar sliders -> acoustic inference. `calm` aliases
somber. Use `--emotion neutral` and default sliders to retain exact input curves.
Diagnostics contain the selected name and the final durations/F0/energy/speaker
ID. The default six frames/token, constant 180 Hz and energy 1 are illustrative;
they do not include a learned duration/pitch/energy predictor, and may even mark
consonants with positive F0 unless the caller supplies suitable curves.

### What the vocoder and presets can establish

A neural vocoder renders acoustic conditioning into a waveform. Emotional
choices belong upstream in the supplied contours/conditioning and the trained
acoustic generator. HiFi-GAN does not expose an independent emotion or speaker
slider in this adapter. Source: [HiFi-GAN paper](https://arxiv.org/abs/2010.05646).

These presets offer deterministic programmatic target construction without
retraining the vocoder. They do **not** guarantee perceptual excitement, sadness,
authority or true breathy whisper. True whisper changes phonation/spectral
structure and may require matching training/style conditioning; simply lowering
F0 or energy is insufficient. Vibrato is not implemented as a named preset.
Speaker changes depend on trained speaker conditioning; sliders do not create
new speaker identities. Listening tests and a trained acoustic model are needed
to assess emotional naturalness and control response.

## 6. Architecture and control comparison

| Property | Piper runner | Explicit runner | Matcha |
| --- | --- | --- | --- |
| Current status | Trained local VITS reference | Implemented ONNX/control interface; trained pair missing | Removed at user request |
| Acoustic-to-audio path | Integrated VITS graph, raw waveform output | Separate acoustic graph and vocoder graph | No active executable |
| Frontend | Shared CMUdict/IPA frontend mapped to voice vocabulary | Same frontend, export-specific map and annotations | N/A |
| Cadence control | Global `length_scale`; model predicts local durations | Exact supplied per-token frame targets, then optional deterministic transformations | N/A |
| F0 / energy vectors | Not exposed by this runner | Explicit per-frame inputs to compatible model | N/A |
| Identity | Checkpoint plus valid trained speaker ID | Trained sid conditioning, if supported by supplied export | N/A |
| Emotion flag | None | Five preset policies, calm alias | N/A |
| WAV sample rate | Checkpoint-native 22.05 kHz for installed voices | 24 kHz contract | N/A |

Piper controls are `--voice`, `--speaker-id`, `--length-scale`, `--noise-scale`,
and `--noise-w`. Its familiar scalar controls do not imply every VITS model has
only recording-specific emotions; this report describes the interface actually
exposed here. Similarly, having per-frame explicit inputs does not prove full
emotional control of every trained export. The original pasted RTF ranges were
unverified estimates and are not repeated as measured facts.

## 7. Measured CPU latency and real-time factor

### Environment and methodology

CPU: AMD Ryzen 5 5600X, 6 cores / 12 logical processors. Windows x64; 17,102,323,712
bytes reported physical memory (about 15.93 GiB). Release MSVC build; MSBuild
17.14.23. ONNX Runtime SDK version 1.26.0; DLL product version
`1.26.20260508.3.8c546c3`. CPU inference with four intra-op threads and one
inter-op thread. Explicit sessions use sequential execution, full graph
optimization, arenas/memory patterns and disabled spinning. Piper's current
adapter uses sequential/full optimization and memory arenas/patterns with its
own session setup; its worker-spinning default was not changed for this report.

`scripts/benchmark_tts.py --include-fixture` measures two identical texts:
a 73-character / 12-word short passage and a 549-character / 90-word paragraph.
One discarded fresh process per case warms file caches; three further fresh
processes provide reported medians. **Every process creates new ONNX sessions.**
This is not a warm persistent-service benchmark. Tests/builds were completed
before the reported inference run; measurements were executed outside the
sandbox with ordinary local process permissions.

Inference time is the accumulated synchronous ONNX `Run()` time only. Process
wall time includes process/DLL/session startup, dictionary parsing, phonemization,
controls, layout/data copies, inference, WAV output and JSON output. These
categories were not separately profiled, so their individual costs cannot be
inferred from the difference. RTF = elapsed seconds / actual output-audio seconds;
less than 1 means faster than playback at that measured scope.

### Results: observed medians

| Setup | Text | Audio seconds | ONNX inference ms | Full CLI wall ms | Inference RTF | Wall RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Piper Lessac (trained) | short | 4.168 | 140.969 | 2021.2 | 0.03373 | 0.4850 |
| Piper Lessac (trained) | paragraph | 31.330 | 1117.030 | 3222.3 | 0.03544 | 0.1037 |
| Explicit fixtures (not speech) | short | 10.816 | 1.333 | 784.0 | 0.00012 | 0.0725 |
| Explicit fixtures (not speech) | paragraph | 77.504 | 7.081 | 892.0 | 0.00009 | 0.0115 |
| Explicit trained acoustic + trained vocoder | short / paragraph | N/A | N/A | N/A | N/A | N/A |
| Matcha | short / paragraph | N/A | N/A | N/A | N/A | N/A |

The explicit fixture's individual stage medians were:

| Text | Arithmetic acoustic graph ms | Repeat/tanh vocoder graph ms |
| --- | ---: | ---: |
| short | 0.1943 | 1.1207 |
| paragraph | 0.4264 | 6.6547 |

Medians of individual stages need not sum exactly to the median of total time.
These fixture times describe tiny arithmetic graphs and memory/tensor plumbing;
they are **not FastSpeech2/HiFi-GAN performance measurements**. The fixture
uses the existing complete Piper CMU dictionary/token map solely to exercise
identical text workloads with arbitrary IDs; no graph or learned weights are
rewritten. Its audio duration comes from six frames per token, not predicted
speech rhythm, and its output is not intelligible speech. Accordingly its tiny
RTF is not a meaningful speech-synthesis speed comparison or a speedup claim.

Piper's short case used 169 tokens in one chunk; the paragraph used 1,213 tokens
across three chunks, with inserted inter-chunk silence. The paragraph had one
letter-pronunciation fallback word and zero missing model symbols. The explicit
fixture used 169 and 1,211 tokens without chunking. Piper's injected pauses
contribute to audio duration/RTF; token boundaries and rhythm therefore differ
between setups even with identical raw text. Stochastic Piper output also makes
audio duration and RTF vary slightly across repetitions.

Piper's measured inference was faster than playback for both passages. The
fresh short CLI process still took about two seconds, so low kernel RTF alone
should not be treated as instant response. For an application serving repeated
requests, persistent session/dictionary reuse is a useful future optimization;
this report does not claim it has already been implemented in the standalone
CLI. The explicit pipeline's true neural runtime cannot be inferred by adding a
generic published FastSpeech2 and HiFi-GAN estimate. It needs measurements of
the actual compatible trained exports, their chosen hop/mel convention and
real speech durations.

No Matcha benchmark was rerun or restored after removal. Its entry is unavailable
by design. Historical output or quoted ranges would not validate the current
repository state.

### Reproduce and extend

```sh
cmake --workflow --preset msvc-onnx
python scripts/benchmark_tts.py --include-fixture
python scripts/benchmark_tts.py --explicit-assets model_assets/explicit_neural
```

The third command requires compatible trained assets; it is not runnable with
the current unstaged directory. The script accepts `--threads`, `--repeats`,
`--voice`, `--bin-dir`, `--piper-assets` and `--output`. It writes WAVs, per-run
diagnostics, commands, texts, binary/asset SHA-256 hashes and raw measurements
to `artifacts/benchmarks/`; the measured summary above comes directly from
`results.json`. Build timing evidence is `incremental_build.json`. Results are
local execution artifacts and ignored by Git; this report records their summary.

## 8. Verification and outstanding model integration

The retained ONNX workflow passed all four suites:

1. Framework metrics/features/WAV round-trip/resampling and CMU annotation checks.
2. Prosody/length-regulator checks: exact ordering, zero durations, budgets,
   invalid controls, neutral identity, preset formulas, all-unvoiced input,
   missing/invalid annotations, speaker preservation and cadence remapping.
3. Explicit ONNX runtime checks: tensor layouts, frame/sample dimensions and
   invalid exports.
4. CLI integration: every emotion and calm alias, preset/scalar composition,
   speaker/control diagnostics, WAV frame counts and error cases.

The runtime-free `default` workflow passed its two framework/prosody suites.
Source/build configuration checks confirm no archive or Matcha targets remain.
Benchmark WAVs/diagnostics were inspected for matching sample counts, native
sample rates and asset provenance. No perceptual listening/emotion score,
trained-explicit speech quality score, or persistent-session performance result
is claimed.

To obtain the intended explicit voice, the remaining model-integration work is
to supply/export a trained acoustic model accepting these exact duration/F0/
energy/speaker inputs and pair it with a vocoder trained for the same mel
features. The pair must satisfy the CLI's 24 kHz contract, or the pipeline must
be deliberately extended/tested for a compatible native rate. Then measure
short and long passages, validate actual control response, and assess naturalness
through listening. These are model availability/training/export requirements;
the new presets and a small vocoder alone cannot supply them.


// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\docs\TTS_ARCHITECTURE_AND_EMOTION_REPORT.md ===
# Explicit neural speech: architecture, controls and measured results

Validated October 7, 2026 (America/Los_Angeles). Benchmark UTC timestamp:
`2026-10-08T02:15:03.753087+00:00`.

## Current result

`explicit-tts` generates intelligible English using real pretrained PaddleSpeech
FastSpeech2 VCTK and matching HiFi-GAN VCTK. Matcha code, runner, downloader and
build targets are removed. Piper remains a separate reference executable.
Procedural DSP and deterministic neural-shaped experiments remain in `archive/`
and are excluded from active CMake targets.

This is a pretrained FastSpeech2 model adapted to an explicit inference interface.
It is not a newly trained custom acoustic network. The separate trained predictor
supplies useful default prosody; exact durations, frame-level F0 and energy, and
speaker ID remain inputs to the acoustic graph. No test oscillator, arithmetic
fixture or substitute waveform is used in the trained pipeline.

The earlier report is preserved as [historical fixture measurements](HISTORICAL_FIXTURE_REPORT.md).
Its missing-model status and fixture timings describe the earlier state only.

## Assets and preparation

All model inputs live under `model_assets/explicit_neural/`. Downloads are explicit,
sequential, cached HTTPS requests to public publisher releases. The fetcher pins
archive/statistics SHA-256 values. Standard HTTP byte ranges retrieve the small
training statistics without downloading the large training snapshot.

| Export | Bytes | Purpose |
| --- | ---: | --- |
| `acoustic_generator.onnx` | 138,766,536 | Trained encoder/decoder with explicit expansion and frame controls |
| `prosody_predictor.onnx` | 78,864,267 | Trained duration, pitch and energy prediction |
| `vocoder_hifigan.onnx` | 51,969,557 | Matching trained native 24 kHz waveform generator |
| `cmudict.dict` | 3,618,488 | English pronunciation dictionary |
| `tokens.tsv` | 607 | Publisher stressed ARPAbet IDs |

The graph split duplicates trained encoder weights. The matched vocoder is about
50 MiB; the independent 3.75 MB HiFi-GAN v2 candidate is 22.05 kHz and is not used
here. Mel compatibility requires more than matching bin counts.

Prepare from `src/audio_synth`, after installing the shared ONNX Runtime SDK:

```sh
python -m venv build/model-export-env
./build/model-export-env/Scripts/python.exe -m pip install onnx numpy
./build/model-export-env/Scripts/python.exe scripts/fetch_explicit_voice.py
cmake --workflow --preset explicit-neural
```

These commands target Windows PowerShell or Git Bash. Incremental builds use
`cmake --build --preset explicit-neural`. MSVC Release, `/MP` and eight project
jobs are configured in `CMakePresets.json`; the explicit preset builds only its
runner and required tests. Runtime inference makes no application-level network
requests. Generated WAVs, diagnostics and validation reports go under `artifacts/`.

CMUdict resolution tries the asset root, adjacent `shared/`, the parent folder,
then adjacent `piper/`. The trained model's token map stays local and is never
replaced with Piper's incompatible vocabulary. `# frontend=arpabet` selects
literal stressed ARPAbet keys, no padding or BOS/EOS, internal punctuation `sp`,
and trimming of trailing `sp`. OOV words use letter pronunciation fallback.

See [the model card and publisher references](EXPLICIT_VOICE_MODEL_CARD.md).
The publisher's [release list](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md)
is the source of the selected trained releases. Exact URLs, hashes, statistics,
export details and asset hashes are recorded in `download_manifest.json`.

## Explicit contract and trained export

```text
CMUdict -> publisher ARPAbet IDs + token annotations
        -> trained prosody predictor (duration, continuous F0, relative energy)
        -> optional CSV / scalar baseline overrides
        -> emotion preset -> scalar sliders
        -> explicit acoustic ONNX -> native log-mel
        -> matched HiFi-GAN -> 24 kHz mono PCM16 WAV
```

The acoustic graph accepts exactly these five inputs:

| Name | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `input_ids` | int64 | `[1,T]` | Publisher phoneme IDs |
| `durations` | int64 | `[1,T]` | Exact nonnegative token frame allocations |
| `f0` | float32 | `[1,F]` | Hertz, transformed to normalized continuous log-F0 |
| `energy` | float32 | `[1,F]` | Energy relative to the training feature mean |
| `sid` | int64 | `[1]` | Trained speaker ID |
| `mel` (output) | float32 | `[1,80,F]` | Publisher-native denormalized log-mel |

`F = sum(durations)`. Gather indices derived from duration cumulative sums expand
trained hidden states exactly and skip zero-duration tokens. The trained pitch
and energy embeddings are kernel-size-one convolutions; moving them to frame
positions preserves their response to repeated token controls while permitting
within-token variation. The trained decoder and postnet remain unchanged.
The vocoder change only adapts layout `[F,80]` to/from the public interface.

Native settings: 24,000 Hz, hop 300, FFT 2048, Hann window 1200, 80 mel bins,
80..7600 Hz. `pipeline.properties` supplies rate/hop and speaker count.
Speaker IDs 0..106 are valid; ID 0 is p225, and the publisher map is staged.
There is no resampling or peak normalization in final synthesis.

Pitch normalization is `(log(Hz) - 5.0610495) / 0.35158658`.
Zero F0 maps to normalized pitch zero, the continuous training mean. This model
has no separate voicing input: a zero curve cannot force truly unvoiced excitation.
Energy normalization is `(relative_energy * 29.903042 - 29.903042) / 25.684935`.
This feature is neither output PCM amplitude nor dB.

## Export validation

`scripts/validate_explicit_export.py` compares locally cached publisher graphs
and adapted exports. It requires ONNX Runtime, ONNX and NumPy, but downloads nothing.
Measured results are in `artifacts/explicit_neural/export_validation.json`:

| Check | Result |
| --- | --- |
| Retained acoustic constants | 584, byte-identical tensor values |
| Retained predictor constants | 418, byte-identical tensor values |
| Retained vocoder constants | 162, byte-identical tensor values |
| Neutral mel vs publisher graph | Maximum absolute difference 0.00000202656 |
| Vocoder wrapper vs publisher vocoder | Maximum absolute difference 0 |
| Pitch / energy / speaker changes | All change mel; max differences 1.9562 / 0.6249 / 1.7710 |
| Exact durations with a zero token | Correct output frame count, 163 |

These checks distinguish a real control path from an interface whose inputs are
ignored. The neutral equivalence measurement uses the target sentence and speaker
0; it is not a claim of equivalence for every out-of-range control curve.

## Emotion policies and limits

Presets run after base controls/CSV overrides and before scalar sliders. They
preserve speaker ID and remap each token's contour when changing duration.

| Preset | Transform |
| --- | --- |
| neutral | Identity |
| whisper | Energy x0.6; pitch deviations x0.15; unvoiced-consonant durations x1.1, rounded up |
| excited | Pitch mean +35 Hz, deviations x1.6; speed 1.15; boosted energy peaks |
| somber / calm | Pitch mean -25 Hz, deviations x0.65; reduced energy/variation; speed 0.88; falling phrase tails |
| authoritative | Pitch mean clamped to 140..190 Hz, deviations x0.35; emphasized first stressed vowel; shortened boundaries |

`--pitch-scale`, `--speed`, `--energy-scale`, and `--energy-variance` compose after
these policies. Neutral with default sliders preserves supplied control vectors.
`--speaker-id` changes trained identity, independently of emotion. `--f0-hz` and
`--frames-per-token` explicitly replace learned defaults. CSVs specify exact
per-token durations and per-frame contours.

These presets are mathematical heuristics, not trained emotion embeddings.
Whisper does not guarantee physical whisper; extreme pitch/energy values can
reduce intelligibility. Control influence is verified, but arbitrary contours
and all 107 speakers have not been assessed for naturalness by human listeners.

## Requested command and generated speech

The exact command succeeded:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "We synthesize a clear acoustic voice." --emotion excited --output artifacts/explicit_neural/excited.wav --threads 4
```

CMUdict opens from the assets root. All six words are dictionary hits, zero
fallback words. The trained predictor, acoustic graph and vocoder run through
CPU ONNX Runtime. The excited result has 25 tokens, 147 frames and 44,100 samples
at 24 kHz. Diagnostics record the actual controls and stage times.

An independent locally cached Whisper small.en recognized all three generated
variants with no initial text prompt. Transcripts were “We synthesise a clear
acoustic voice.” for neutral/excited and “we synthesize a clear acoustic voice.”
for somber. The spelling difference is recognized as the same spoken word.
The verification script records audio hashes and literal transcriptions in
`artifacts/explicit_neural/intelligibility.json`; it never downloads a recognizer.

| Variant | Seconds | Estimated voiced median Hz | PCM RMS |
| --- | ---: | ---: | ---: |
| neutral | 2.1750 | 181.8 | 0.08389 |
| excited | 1.8375 | 220.2 | 0.09774 |
| somber | 2.5375 | 160.5 | 0.05150 |

Pitch estimates use normalized waveform autocorrelation, 1200-sample windows,
hop 300, 70..400 Hz, RMS >0.02 and periodicity confidence >0.7. They are approximate
voiced medians, not exact tracking of every input frame. They confirm the expected
pitch direction alongside measured cadence and amplitude. Reproduce with
`python scripts/measure_explicit_tone.py` on the three existing samples.

## Performance

Windows 11, AMD64 Family 25 Model 33, 12 logical processors, four inference threads.
Three measured fresh processes per case after one discarded process warms disk
caches. Sessions are recreated per sample. Wall latency includes loading,
frontend, inference and output writing; inference includes the trained predictor,
acoustic model and vocoder. These are machine-specific observations.

```sh
python scripts/benchmark_tts.py --explicit-assets model_assets/explicit_neural
```

| Pipeline | Workload | Audio seconds | Inference ms | Wall ms | Inference RTF |
| --- | --- | ---: | ---: | ---: | ---: |
| Piper Lessac | short | 4.238 | 155.6 | 2106.8 | 0.0367 |
| Piper Lessac | paragraph | 31.783 | 1201.2 | 3262.1 | 0.0379 |
| Explicit VCTK | short | 3.913 | 594.7 | 2087.6 | 0.1520 |
| Explicit VCTK | paragraph | 30.313 | 6117.1 | 7839.2 | 0.2018 |

RTF below 1 means inference is faster than generated playback duration. HiFi-GAN
accounts for most explicit inference time: paragraph medians are about 61 ms
predictor, 429 ms acoustic and 5627 ms vocoder. Persistent sessions would avoid
fresh-process loading, but that optimization is not claimed or benchmarked here.
Raw results, exact texts, binaries and asset hashes are in
`artifacts/benchmarks/results.json`. Fixture timings are excluded from this table.

## Automated coverage

The full ONNX workflow passed all five suites: framework/phonemizer,
prosody/length-regulation, explicit runtime, fixture CLI, and trained CLI.
The explicit-only workflow passed all four selected suites; runtime-free default
passed both selected suites. Trained CLI tests cover multiple texts, speaker
changes, duration/pitch direction, waveform changes and adjacent dictionary fallback.
Tests never download models; the trained suite is enabled only when assets exist.
Contract tests cover malformed graphs, invalid types/layouts, non-finite outputs,
control counts and malformed CLI values. Independent ASR and export equivalence
checks complement those tests; fixture tests alone do not demonstrate speech.

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\docs\VOCAL_PROFILES.md ===
# Vocal profiles for the explicit pipeline

For the C++ profile design tutorial, emotion recipes, exact contour editing and
model limitations, see [Making voice profiles and expressive speech](VOICE_PROFILE_DESIGN.md).

Run these commands from `src/audio_synth`, with the trained assets already prepared.
They work as single lines in PowerShell and Git Bash. Output files and diagnostics
are written under `artifacts/vocal_profiles/`. Models remain under `model_assets/`.

A profile is a saved combination of trained speaker identity and delivery controls.
These are design starting points, not learned character or emotion labels.
Listen to the generated samples and tune them for your text.

## Choose an identity first

`--speaker-id` selects a trained VCTK identity. The staged model accepts 0..106.
The mapping starts p225=0, p226=1, p227=2, p228=3; the full map is
`model_assets/explicit_neural/speaker_id_map.txt`. Audition speakers with the same
sentence, neutral emotion, and default sliders before choosing one. The names in
this guide describe intended delivery, not verified speaker gender or accent.

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 0 --emotion neutral --output artifacts/vocal_profiles/speaker_0.wav --threads 4
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 1 --emotion neutral --output artifacts/vocal_profiles/speaker_1.wav --threads 4
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 2 --emotion neutral --output artifacts/vocal_profiles/speaker_2.wav --threads 4
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 3 --emotion neutral --output artifacts/vocal_profiles/speaker_3.wav --threads 4
```

## What each control changes

| Control | Meaning | Useful starting values |
| --- | --- | --- |
| `--speaker-id` | Trained speaker identity; affects acoustic timbre and predicted prosody | Audition 0, 1, 2, 3, then other valid IDs |
| `--emotion` | Programmatic duration, pitch and energy policy | neutral, excited, somber, authoritative, whisper; calm aliases somber |
| `--pitch-scale` | Multiplies the F0 curve; does not directly change vocal-tract/formant size | 0.9 lower, 1 unchanged, 1.1 higher |
| `--speed` | Divides token durations; above 1 faster, below 1 slower | 0.9 deliberate, 1 normal, 1.1 quicker |
| `--energy-scale` | Scales acoustic energy features; not an exact WAV gain | 0.8 softer, 1 normal, 1.1 stronger |
| `--energy-variance` | Scales deviations around the energy curve's mean | 0.5 even, 1 original, 1.2 more contrast |
| `--f0-hz` | Replaces the entire learned pitch contour with one constant Hz value | 170 for a deliberately flatter/robotic delivery |

Emotion runs first; sliders run afterward. Excited already increases pitch and
uses speed 1.15; somber already uses speed 0.88. For example, excited plus
`--speed 1.05` approximately compounds to 1.2075, with integer-frame rounding.
Start with small changes and adjust one variable at a time. Pitch multiplier
`2 ** (semitones / 12)` gives musical intervals: +2 semitones is about 1.122,
-2 is about 0.891. Pitch changes can alter delivery without creating a new identity.

Whisper is a quieter/flatter heuristic. This trained acoustic model has continuous
F0 and no independent voicing control, so neither whisper nor zero F0 guarantees
physical whisper. These controls cannot guarantee accent, age, gender, breathiness,
rasp or a completely new voice; those qualities depend on training and conditioning.

## Ten ready-to-run profiles

Each command uses the same sentence so differences are easier to compare.
To compare only delivery, change all speaker IDs to one chosen ID.

### Neutral reference

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 0 --emotion neutral --pitch-scale 1.0 --speed 1.0 --energy-scale 1.0 --energy-variance 1.0 --output artifacts/vocal_profiles/neutral_reference.wav --threads 4
```

### Lower, measured narrator

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 1 --emotion neutral --pitch-scale 0.9 --speed 0.93 --energy-scale 0.95 --energy-variance 0.8 --output artifacts/vocal_profiles/lower_narrator.wav --threads 4
```

### Higher, brighter delivery

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 0 --emotion neutral --pitch-scale 1.12 --speed 1.05 --energy-scale 1.05 --energy-variance 1.1 --output artifacts/vocal_profiles/bright_voice.wav --threads 4
```

### Friendly guide

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 2 --emotion neutral --pitch-scale 1.03 --speed 0.98 --energy-scale 1.0 --energy-variance 1.1 --output artifacts/vocal_profiles/friendly_guide.wav --threads 4
```

### Energetic presenter

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 0 --emotion excited --pitch-scale 1.0 --speed 1.0 --energy-scale 0.95 --energy-variance 1.0 --output artifacts/vocal_profiles/energetic_presenter.wav --threads 4
```

### Calm storyteller

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 1 --emotion somber --pitch-scale 1.05 --speed 1.05 --energy-scale 1.05 --energy-variance 1.0 --output artifacts/vocal_profiles/calm_storyteller.wav --threads 4
```

### Soft spoken

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 3 --emotion neutral --pitch-scale 0.98 --speed 0.95 --energy-scale 0.8 --energy-variance 0.7 --output artifacts/vocal_profiles/soft_spoken.wav --threads 4
```

### Firm announcer

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 1 --emotion authoritative --pitch-scale 1.0 --speed 1.0 --energy-scale 0.95 --energy-variance 0.9 --output artifacts/vocal_profiles/firm_announcer.wav --threads 4
```

### Deliberately flatter robot

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 2 --emotion neutral --pitch-scale 1.0 --speed 0.95 --energy-scale 1.0 --energy-variance 0.3 --f0-hz 170 --output artifacts/vocal_profiles/deliberate_robot.wav --threads 4
```

### Urgent messenger

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello there. The little bird sings beside the window." --speaker-id 3 --emotion excited --pitch-scale 0.95 --speed 1.05 --energy-scale 0.9 --energy-variance 0.9 --output artifacts/vocal_profiles/urgent_messenger.wav --threads 4
```

## Save and render your own profile

The ten settings are stored in `examples/vocal_profiles/profiles.json`. Copy an
entry, choose a unique name and change its values. The Python helper translates
JSON settings into ordinary calls to `explicit-tts`; the C++ executable itself
does not accept a `--profile` option. No downloads or Python ML packages are needed.

```sh
python scripts/render_vocal_profiles.py --list
python scripts/render_vocal_profiles.py
python scripts/render_vocal_profiles.py --profile lower_narrator --text "The moon rose above the quiet garden."
python scripts/render_vocal_profiles.py --profile bright_voice --profile soft_spoken --speaker-id 0 --text "Let us compare two different deliveries."
```

No `--profile` selection renders all ten. `--speaker-id` overrides identity for
all selected profiles, useful for comparing delivery alone. Rendering writes
WAVs, diagnostics and a manifest of exact settings/commands under
`artifacts/vocal_profiles/`. Use `--output artifacts/my_voices` for another folder.
Each run overwrites the same named output files in that chosen folder.

For a new JSON entry:

```json
"my_narrator": {
  "speaker_id": 1,
  "emotion": "neutral",
  "pitch_scale": 0.95,
  "speed": 0.95,
  "energy_scale": 0.95,
  "energy_variance": 0.8
}
```

Add it as another member of the JSON object with the appropriate comma, then run:

```sh
python scripts/render_vocal_profiles.py --profile my_narrator --text "This is my new narrator profile."
```

## Design a voice systematically

1. Audition identity at neutral/default settings using a fixed sentence.
2. Keep that speaker and tune pitch slightly, typically 0.9..1.1 to start.
3. Tune pace independently, typically 0.9..1.1.
4. Tune energy and contrast for softness or emphasis.
5. Add an emotion preset only if its coupled changes fit the intended delivery.
6. Save the settings and try questions, short lines and a longer passage.

These ranges are starting experiments, not validated quality limits. Diagnostics
show exact durations, per-frame Hz/energy and inference times. Do not specify
`--frames-per-token` when you want the trained model's natural token durations.
Keep `--hop-length` at the model's native configuration; it is not a voice control.

For detailed phrase shapes, supply `--durations`, `--f0` and `--energy` CSV files.
One duration is required per phoneme ID, and one F0/energy value per expanded
frame. Render the same text/speaker with neutral/default settings first and use
its diagnostics vectors as a starting point. Contours are text-specific: changing
the sentence usually changes token/frame counts. To preserve exact supplied
vectors, use neutral emotion and default sliders. See the main README and
`models/explicit_neural/README.md` for the tensor and contour contracts.

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\docs\VOICE_PROFILE_DESIGN.md ===
# Making voice profiles and expressive speech in C++

This guide describes the current `explicit-tts.exe` pipeline and its C++ APIs.
All synthesis runs locally in C++ through ONNX Runtime. Python is not needed to
use the commands below once the model assets and executable are prepared.

Run commands from `src/audio_synth`. Each command is one line and works in
PowerShell or Git Bash. Input models stay in `model_assets/explicit_neural/`;
generated audio and diagnostics go in `artifacts/`.

## 1. What a voice profile contains

A profile is a reusable choice of **speaker identity plus delivery settings**.
For example, a narrator profile might select speaker 1, lower its predicted
pitch slightly, slow its pace and reduce emphasis contrast. Saving that recipe
does not train another model or create another speaker embedding.

| Layer | What you choose | How it affects the result |
| --- | --- | --- |
| Model setup | The compatible acoustic model, predictor, vocoder and vocabulary | Defines the voices and controls the model can represent |
| Identity | `--speaker-id` | Selects a trained speaker and its predicted prosody |
| Delivery | Emotion, pitch, speed and energy settings | Shapes how that speaker delivers the line |
| Line direction | Text, phrase breaks, exact duration/F0/energy curves | Shapes a particular utterance or emphasized word |

The staged FastSpeech2 VCTK model has 107 mapped speakers, IDs **0..106**.
The map starts p225=0, p226=1, p227=2 and p228=3; see
`model_assets/explicit_neural/speaker_id_map.txt`. Audition speakers to choose
the identity you like. Profile names such as “friendly guide” describe an
intended delivery, not a verified accent, age or gender.

Changing pitch modifies fundamental frequency. It does not directly rescale
formants or vocal-tract size, and cannot alone guarantee a new identity.
Creating an identity outside the trained speaker set requires a model that
supports that identity through suitable training or additional conditioning.
The existing CLI does not train voices, clone voices or accept reference audio.

## 2. How a profile becomes speech

```text
Text + trained speaker ID
    -> CMUdict pronunciation -> publisher ARPAbet token IDs
    -> trained duration / pitch / energy predictions
    -> optional explicit control overrides
    -> emotion preset
    -> pitch / speed / energy sliders
    -> explicit acoustic ONNX -> mel spectrogram
    -> matching HiFi-GAN ONNX -> 24 kHz PCM16 WAV
```

The trained predictor supplies a different starting contour for each line.
A normal profile modifies those predictions rather than replacing every token
with the same duration or pitch. This lets one profile work across many texts.

Exact CSV contours are **line-specific**. A different sentence or pronunciation
usually produces different token and frame counts, so the same CSVs cannot be
assumed to fit another line.

The active acoustic model is pretrained FastSpeech2 adapted to explicit controls;
the profile system and emotion policies are our C++ control layer. The vocoder
turns conditioned mel features into audio. Emotion changes happen upstream of
the vocoder; the CLI has no independent vocoder emotion setting.

## 3. The controls you can use

| CLI control | Practical meaning | Starting experiment |
| --- | --- | --- |
| `--speaker-id` | Trained identity | Compare 0, 1, 2 and 3 using one fixed line |
| `--emotion` | Coordinated prosody policy | Start with `neutral`, then audition presets |
| `--pitch-scale` | Multiplies predicted F0 in Hz | 0.95 slightly lower; 1.05 slightly higher |
| `--speed` | Divides token frame allocations | 0.95 slower; 1.05 faster |
| `--energy-scale` | Scales acoustic energy features | 0.85 softer; 1.05 stronger |
| `--energy-variance` | Scales deviations around mean energy | 0.7 more even; 1.15 more contrast |
| `--f0-hz` | Replaces the predicted pitch contour with a constant | Useful for intentionally flat experiments |
| `--frames-per-token` | Replaces learned token durations with one frame count | Timing experiments, not a normal natural-speech profile |

Default sliders are all 1. Pitch and speed must be positive; energy sliders
must be nonnegative. Values must be finite. Small changes are useful starting
points, not guaranteed naturalness bounds.

Energy is a learned acoustic feature, **not a WAV volume knob or dB value**.
Reducing it can change speech character as well as amplitude. Use a waveform
gain stage if you need exact playback loudness; the current CLI does not expose
such a gain flag.

For musical pitch offsets, `pitch_scale = 2^(semitones/12)`: +2 semitones is
approximately 1.122 and -2 is approximately 0.891. Do not assume the waveform
will follow every requested pitch value exactly.

## 4. Create a profile step by step

### Choose a speaker with neutral controls

Render the same sentence for several speaker IDs. Change only the ID and output
filename during this stage:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 1 --emotion neutral --text "Hello there. The little bird sings beside the window." --output artifacts/profile_design/speaker_1.wav --threads 4
```

### Tune one setting at a time

Keep the speaker fixed. First compare pitch, then pace, then energy/contrast.
For a slightly lower, measured narrator:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 1 --emotion neutral --pitch-scale 0.95 --speed 0.95 --energy-scale 0.95 --energy-variance 0.8 --text "The moon rose above the quiet garden." --output artifacts/profile_design/my_narrator.wav --threads 4
```

The saved profile is the set of model assets, speaker ID and control settings.
The sentence and output filename can change independently. Save the complete
command in your project notes or a shell script, or store the settings in your
own C++ structure. The C++ executable currently accepts individual flags;
it has no native `--profile` or JSON profile-file option.

The existing `examples/vocal_profiles/profiles.json` stores ten recipes for an
optional batch helper. It is not required for direct C++ CLI use.

### Check several kinds of text

Try a short line, a question, multiple sentences and a longer passage. Listen
for clear consonants, acceptable pace, unwanted pitch jumps and clipped peaks.
Use the same comparison text when choosing between two versions of a profile.
Save variants as separate output filenames so you can compare them.

Each WAV has a sibling `.diagnostics.json` containing speaker ID, emotion,
token IDs, final durations, final F0/energy curves and inference timings.
That file describes the controls actually used, after all transforms.

## 5. How the built-in emotion policies work

These are deterministic edits to duration, pitch and energy, not trained
emotion embeddings. `calm` is an alias for `somber`, not another policy.

| Preset | Pitch | Timing | Energy and emphasis |
| --- | --- | --- | --- |
| `neutral` | Keeps the base curve | Keeps durations | Keeps base energy |
| `excited` | Mean +35 Hz; deviations around the original mean multiplied by 1.6 | Speed 1.15 | Energy contrast x1.35, then x1.1, clipped at zero |
| `somber` / `calm` | Mean -25 Hz; deviations x0.65; falling phrase tails | Speed 0.88 | Mean energy x0.85 and deviations x0.4, clipped at zero |
| `authoritative` | Mean clamped to 140..190 Hz; deviations x0.35 | Boundary durations halved, at least one frame | Base energy x1.05; first stressed vowel emphasized; quieter boundaries |
| `whisper` | Deviations around the mean x0.15 | Annotated unvoiced-consonant durations x1.1, rounded up | Energy x0.6 |

Token annotations come from the phonemizer. They identify stressed vowels,
unvoiced consonants and punctuation boundaries so the preset can target those
regions. Authoritative emphasis chooses the first stressed vowel in a phrase;
it does not understand which word is semantically most important.

The presets preserve the selected speaker. They run **before the sliders**:
`excited --speed 1.05` applies the preset's 1.15 speed and then another 1.05,
approximately 1.2075 overall, subject to integer rounding per token. Energy
variance and scale also compose with changes already made by the preset.

The model uses continuous log-F0. Although the control layer preserves zero-F0
entries, this particular export maps them to normalized pitch zero, the training
pitch mean. It has no separate voiced/unvoiced excitation control. The whisper
preset can make quieter, flatter delivery; it cannot guarantee a physical whisper.

## 6. Recipes for emotive lines

These commands are starting recipes. Their names describe the intended delivery;
listen to the result for your chosen speaker and wording. All examples select
speaker 0 so the delivery changes are easier to compare.

### Joy or delighted surprise

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion excited --energy-scale 0.95 --text "Wow. That is wonderful. We did it!" --output artifacts/profile_design/delighted.wav --threads 4
```

### Reassurance or gentle encouragement

Use modest changes rather than assuming somber always sounds reassuring:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion neutral --pitch-scale 0.98 --speed 0.95 --energy-scale 0.85 --energy-variance 0.75 --text "It is all right. Take your time. We can try again." --output artifacts/profile_design/reassuring.wav --threads 4
```

### Sadness or reflective delivery

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion somber --text "Oh. I thought we had more time." --output artifacts/profile_design/reflective.wav --threads 4
```

### Firm direction

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion authoritative --energy-scale 0.95 --text "Please listen carefully. Stay here until I return." --output artifacts/profile_design/firm.wav --threads 4
```

### Urgency

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion excited --pitch-scale 0.95 --speed 1.05 --energy-scale 0.9 --text "Wait. We need to leave now." --output artifacts/profile_design/urgent.wav --threads 4
```

### Hesitation or uncertainty

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion neutral --speed 0.92 --energy-scale 0.9 --energy-variance 0.85 --text "Oh. I am not sure. Could we try another way?" --output artifacts/profile_design/uncertain.wav --threads 4
```

### Quieter, flatter speech

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion whisper --text "Stay close. We should speak quietly." --output artifacts/profile_design/quiet.wav --threads 4
```

There is no `--emotion angry`, `happy`, `fearful` or `laugh` option. Design a
delivery using available controls, implement a new C++ policy, or use a model
with appropriate learned conditioning. A recipe does not make the model an
emotion recognizer or guarantee that listeners will label its output as intended.

## 7. Text, pauses and nonverbal sounds

Spoken interjections such as “Oh” and “Wow” can be rendered as words, then shaped
with the same controls. Writing “ha ha” requests pronounced text; it does not
reliably generate natural laughter. Breath, sigh, gasp, sobbing and real laughter
are not exposed as trained event controls in the current pipeline. For those
events, use recorded audio or a model explicitly trained to generate them.

The frontend maps internal `. , ! ? ; :` punctuation to the **same `sp` token**.
It does not expose separate question, comma or exclamation conditioning.
Repeated exclamation marks do not specify a precise emotion or loudness.
Trailing silence is trimmed by the frontend. Punctuation can introduce an
internal boundary, but it does not guarantee a rising question contour or a
pause of exactly 200 ms. Write natural wording and use explicit contours/timing
when you need precise direction.

Stage directions like `[laughs]`, `[whispers]` or `<sigh>` are not supported
control tags. They may be tokenized as ordinary text; do not use them to request
an event. For changing emotion between clauses, render each clause separately
with the desired settings, or apply segment-specific controls in C++. One
`--emotion` flag applies to the entire invocation.

## 8. Exact pitch shapes and word emphasis

The CLI supports `--durations FILE`, `--f0 FILE` and `--energy FILE` with
CSV/whitespace values. Use neutral emotion and default sliders when you want
the acoustic graph to receive those vectors without preset edits.

| File | Required values |
| --- | --- |
| Durations | One nonnegative integer per emitted phoneme token |
| F0 | One finite, nonnegative Hertz value per expanded frame |
| Energy | One finite, nonnegative relative-energy value per expanded frame |

The contour length must equal the sum of durations. At native 24 kHz with hop
300, one frame spans **12.5 ms**. Twenty-four frames span 300 ms. A 200 ms token
allocation is 16 frames, but allocating frames to a silence token is a model
request, not a guarantee of perfectly silent samples.

Word emphasis is more targeted than increasing energy for the entire sentence:
find the word's token interval, slightly extend its stressed vowel, boost energy
in that interval and add a small pitch movement. Tokens are phonemes, not words;
do not assume token index 3 corresponds to word 3. The current diagnostics have
token IDs and counts, but no explicit word-to-token alignment table.
Changing durations also changes the required contour lengths. Keep the acoustic
frames and both contours aligned; in C++, regenerate/remap them inside each token.

### Editable contour example without Python

First render a neutral baseline for the exact sentence and chosen speaker:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion neutral --text "Could we try another way?" --output artifacts/profile_design/question_base.wav --threads 4
```

The following **PowerShell-only** snippet copies the actual durations and energy
into control files and adds a gentle 12% pitch rise across the final 24 frames.
PowerShell only writes text files; synthesis still runs in the C++ executable.
The rise is an experimental phrase-tail shape, not a question-emotion model.

```powershell
$questionReport = Get-Content -LiteralPath artifacts/profile_design/question_base.diagnostics.json -Raw | ConvertFrom-Json
$questionPitch = @($questionReport.f0_hz)
$questionTailStart = [Math]::Max(0, $questionPitch.Count - 24)
for ($frameIndex = $questionTailStart; $frameIndex -lt $questionPitch.Count; $frameIndex++) {
    if ($questionPitch[$frameIndex] -gt 0) {
        $questionPitch[$frameIndex] *= 1 + 0.12 * ($frameIndex - $questionTailStart + 1) / ($questionPitch.Count - $questionTailStart)
    }
}
$questionReport.durations -join ',' | Set-Content -Encoding ascii artifacts/profile_design/question_durations.csv
($questionPitch | ForEach-Object { ([double]$_).ToString('R', [Globalization.CultureInfo]::InvariantCulture) }) -join ',' | Set-Content -Encoding ascii artifacts/profile_design/question_f0.csv
($questionReport.energy | ForEach-Object { ([double]$_).ToString('R', [Globalization.CultureInfo]::InvariantCulture) }) -join ',' | Set-Content -Encoding ascii artifacts/profile_design/question_energy.csv
```

Then render those exact controls:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion neutral --text "Could we try another way?" --durations artifacts/profile_design/question_durations.csv --f0 artifacts/profile_design/question_f0.csv --energy artifacts/profile_design/question_energy.csv --output artifacts/profile_design/question_rising.wav --threads 4
```

Compare `question_base.wav` and `question_rising.wav`. The acoustic model uses the
altered curve, but the resulting waveform pitch is learned behavior rather than
an exact oscillator following every frame value.
This example was executed successfully: both renders contained 98 frames, and
the final requested F0 value increased by 12% with duration allocations preserved.

## 9. Store profiles directly in C++

The following is an integration fragment using existing project APIs. The
profile structure is an example application type; it is not a new CLI feature.
Include `vocal/control_params.hpp` and the existing predictor header in your
application. `phonemes` is a `PhonemizationResult` from `CmuPhonemizer`, and
`predictor` is a `NeuralProsodyPredictor` loaded from the staged model.

```cpp
struct VoiceProfile {
    std::int64_t speaker_id;
    vocal::VocalEmotion emotion;
    vocal::ProsodySliders sliders;
};

const VoiceProfile narrator{
    1,
    vocal::VocalEmotion::Neutral,
    {0.95F, 0.95F, 0.95F, 0.8F} // pitch, speed, energy scale, energy variance
};

auto controls = predictor.predict(phonemes.token_ids, narrator.speaker_id);
controls.speaker_id = narrator.speaker_id;
controls.token_kinds = phonemes.token_kinds;
controls = vocal::apply_emotion_preset(controls, narrator.emotion);
controls = vocal::apply_prosody_sliders(controls, narrator.sliders);

// acoustic and vocoder are already-loaded project model instances.
const auto mel = acoustic.infer(phonemes.token_ids, controls);
const auto audio = vocoder.synthesize(mel);
vocal::write_wav_pcm16("artifacts/profile_design/cpp_narrator.wav", audio);
```

Keep the matching native vocoder configuration: 24,000 Hz, hop 300. Use the
existing `app/tts_cli.cpp` as the complete integration example, including model
loading, dictionary resolution, validation and error handling.

To add an emotion policy, extend `VocalEmotion`, implement its control transform
in `models/explicit_neural/length_regulator.cpp`, and extend `parse_emotion()` and
help text in `app/tts_cli.cpp`. Keep durations nonnegative, preserve token/frame
alignment, clamp negative energy and validate finite values. Add policy tests
and listen to several texts/speakers. A new preset changes control mathematics;
it does not add a learned emotion embedding to the pretrained model.

For phrase-specific pitch shapes, edit the trained base curve in C++ before
acoustic inference. Smooth, modest changes are a useful first experiment.
When changing timing, remap pitch and energy to the new frame allocations;
the existing global speed slider already performs per-token remapping.

## 10. Troubleshooting and further examples

| Symptom | What to check |
| --- | --- |
| Pitch sounds strained or speech loses clarity | Reduce pitch/energy changes and compare with neutral |
| Pace is much faster than expected | An emotion preset and `--speed` may both change duration |
| Speech sounds mechanically flat | Remove `--f0-hz` or fixed `--frames-per-token` overrides |
| A specific word loses clarity | Inspect CMUdict fallback diagnostics and supplied contours |
| CSV count error | Render the exact text again and match emitted tokens and frame sum |
| Whisper still has voiced pitch | This export has continuous pitch, without independent voicing |
| A question lacks a rising ending | `?` shares the generic pause token; use a tailored F0 curve |
| A sound-event tag is spoken as text | Tags are not supported event controls |

See [ten ready-to-run profiles](VOCAL_PROFILES.md),
[the exact model contract](../models/explicit_neural/README.md), and
[the measured architecture/control report](TTS_ARCHITECTURE_AND_EMOTION_REPORT.md).
The established samples verified pitch/cadence influence and intelligibility;
the new expressive recipes in this guide are starting designs, not independently
validated listener judgments of their intended emotions.

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\examples\prosody\README.md ===
These controls match "hello" with tests/fixtures/explicit. They allocate 32 frames across 15 frontend tokens, including zero-duration BOS/EOS, stress and padding. The three files are accepted by explicit-tts, and the fixture graphs produce test audio rather than speech. For trained models, use targets appropriate to their vocabulary and prosody.

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\README.md ===
# Active synthesis model families

| Directory | Purpose | Executable | Local assets |
| --- | --- | --- | --- |
| `piper_onnx/` | Pretrained VITS reference and shared CMU phonemizer | `piper-tts` | `../model_assets/piper/` |
| [`explicit_neural/`](explicit_neural/README.md) | Duration/F0/energy-conditioned acoustic ONNX plus independent neural vocoder | `explicit-tts` | `../model_assets/explicit_neural/` |

Both runners use the shared ONNX Runtime SDK. The independent `vocal_phonemizer`
and `vocal_prosody` libraries support frontend annotations and deterministic
emotion/control transforms. Model assets are separate from generated artifacts.

The explicit setup stages trained PaddleSpeech FastSpeech2 VCTK, its separate
prosody predictor, and matching native 24 kHz HiFi-GAN using
`scripts/fetch_explicit_voice.py`. The tiny test graphs are untrained.
The separately downloaded HiFi-GAN v2 asset
under `../model_assets/vocoders/` is a 22.05 kHz candidate, not a compatible 24 kHz
pair by itself. Matcha was removed at the user's request.

DSP and deterministic hash-weight neural experiments are preserved under
`../archive/` and excluded from active CMake targets. Framework audio, metrics,
features, evaluation, reporting, datasets and study APIs remain active.

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\explicit_neural\README.md ===
# Explicit acoustic generator and frozen neural vocoder

This is a C++20 ONNX inference pipeline with trained acoustic and vocoder exports.
`scripts/fetch_explicit_voice.py` stages the public PaddleSpeech VCTK releases;
`scripts/export_paddlespeech_explicit.py` adapts their trained graphs without
changing retained weights or retraining. This is FastSpeech2 with explicit
controls, not a custom pretrained acoustic architecture.
FastSpeech2/HiFi-GAN/BigVGAN names describe architecture families, not
interchangeable tensor interfaces. Exports must meet this precise contract.

## Assets

Store inputs under `model_assets/explicit_neural/`, relative to the project README:

```text
model_assets/explicit_neural/
  acoustic_generator.onnx
  vocoder_hifigan.onnx
  prosody_predictor.onnx   # Trained duration/pitch/energy heads
  pipeline.properties     # 24000 Hz, hop 300, 107 speakers
  speaker_id_map.txt
  cmudict.dict
  tokens.tsv
  download_manifest.json   # When staged with prepare_explicit_models.py
```

Vocabulary, acoustic model, speaker IDs and vocoder must be compatible. Keep model
cards and licenses alongside weights. External-data ONNX exports need their
referenced sidecar files staged alongside the graphs. Generated outputs go in
`artifacts/explicit_neural/`, never in the input-assets directory.

## Acoustic export contract

Exactly five inputs, batch size 1:

| Name | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `input_ids` | int64 | `[1, tokens]` | Exact IDs emitted by the CMU/IPA frontend |
| `durations` | int64 | `[1, tokens]` | Nonnegative token frame allocations |
| `f0` | float32 | `[1, frames]` | F0 in Hz, zero means unvoiced |
| `energy` | float32 | `[1, frames]` | Nonnegative normalized energy, not dB |
| `sid` | int64 | `[1]` | Speaker index understood by the trained model |
| `mel` (output) | float32 | `[1, 80, frames]` | Finite log-mel values |

`frames` must equal `sum(durations)`, and both contours must match that count.
At most 4096 tokens and 15000 frames are accepted. Static model dimensions are
checked against the request before inference. Wrong types, missing names,
non-finite outputs and mismatched frame counts fail explicitly.

The single-graph export owns its hidden-state expansion and must actually honor
the durations and contours. C++ verifies output dimensions, which cannot prove
the trained graph's semantic response to controls. The standalone C++
`length_regulate()` utility expands exposed token hidden states deterministically
and is available for split encoder/decoder exports. It is not inserted into this
single-graph contract, because that graph does not expose hidden states.

`length_regulate()` consumes flattened `[1, tokens, hidden_dim]` and repeats
token rows exactly `duration[i]` times. It skips zero-duration tokens and checks
shape, negative durations, overflow and a frame budget before allocation.

## Vocoder export contract

One input `mel`: float32 `[1, 80, frames]`. Output `audio`: finite float32
`[1, frames * hop_length]` or `[1, 1, frames * hop_length]`. The default hop length
is 256; pass `--hop-length` for a matching export using a different hop.
The trained pair must use **24 kHz**, 80 mel bins, the same hop, FFT/window,
frequency range, log base, scaling and normalization. Shape compatibility alone
does not ensure meaningful audio. Waveform samples are copied directly to
`Waveform`, then clipped to the PCM16 range when writing WAV; no synthetic
oscillator or peak normalization is substituted.

The staged VCTK pair uses hop 300, FFT 2048, window 1200, Hann, 80 mel bins,
80..7600 Hz. `pipeline.properties` supplies its native rate/hop automatically.
Its energy input is relative to the training energy mean 29.903042; pitch is
normalized continuous log-F0 (mean 5.0610495, standard deviation 0.35158658).
Zero F0 maps to normalized pitch zero: this particular graph has no explicit
voicing input and cannot guarantee whispered or unvoiced excitation.

The framework `MelSpectrogram` is frame-major `[frames, 80]`; the acoustic
adapter transposes into it, and the vocoder adapter restores channel-major
layout. Local tensor buffers remain alive throughout synchronous `Run()` and
output data is copied before ONNX values are released. RAII owns sessions/options.

## Frontend and controls

`CmuPhonemizer` is exposed by `include/vocal/phonemizer.hpp`. English CMUdict
lookup maps ARPAbet to IPA, with letter-pronunciation fallback for OOV words.
The frontend emits `^`, `$` and `_` padding in addition to IPA symbols; your
trained export and TSV must use these exact semantics. Do not borrow a Piper
vocabulary unless your acoustic model was trained/exported with it. TSV rows are
Unicode codepoints in hexadecimal, a tab, and comma-separated int64 model IDs.
Missing symbols abort the explicit runner rather than silently dropping tokens.
For the staged VCTK graphs, `# frontend=arpabet` switches the TSV to literal
stressed ARPAbet keys and publisher IDs. This path emits no BOS/EOS or padding,
maps internal punctuation to `sp`, and trims trailing `sp` as the publisher does.
Piper's IPA vocabulary must not replace this map. Dictionary resolution tries
the asset root, adjacent shared directory, parent, then adjacent Piper assets.

`ProsodyControls` accepts durations, frame-rate F0 and normalized energy plus
speaker ID. `ProsodySliders` applies pitch, speed, energy scale and energy
variance. Cadence changes resample each token's contour separately with nearest
neighbors; zero F0 stays unvoiced. CSV/whitespace control files allow `#` comments.
The staged `prosody_predictor.onnx` supplies trained duration, pitch and energy
predictions by default. CSV controls and explicit scalar baseline flags override
them. The constant six-frame/180-Hz baseline remains only for fixtures or exports
without a predictor. Speaker IDs 0..106 select trained VCTK identities.

## Programmatic emotion presets

`VocalEmotion` contains `Neutral`, `Whisper`, `Excited`, `Somber`, and
`Authoritative`. Call `apply_emotion_preset(base, emotion)` in C++, or pass
`--emotion` to the CLI (`calm` aliases somber). Presets operate on a copy after
CSV loading, before scalar sliders. They preserve speaker ID, zero-duration
tokens and unvoiced zeros; all resulting contours must match the new frame sum.

Optional `token_kinds` identifies unvoiced consonants, vowels, stressed vowels and
punctuation/EOS boundaries. The shared CMU frontend emits one annotation per
model ID, with unknown annotations for padding and stress/length marks. Without
annotations, token-specific edits are skipped; F0 zero alone cannot identify a
consonant, vowel, padding or pause. Invalid annotation values are rejected.

Whisper reduces energy to 60%, contracts voiced F0 deviations to 15% and extends
annotated unvoiced-consonant duration by 10% (ceil to integer frames). Excited
adds 35 Hz to the voiced mean, expands deviations 1.6x, uses speed 1.15 and
boosts energy peaks. Somber subtracts 25 Hz, contracts pitch/energy deviations,
uses speed 0.88 and adds a 10% falling tail to phrase endings. Authoritative
contracts F0 around a heuristic 140..190 Hz mean, emphasizes the first stressed
vowel token of each phrase, and halves boundary durations with lower energy.

These are adjustable mathematical policies, not learned emotion embeddings.
They neither create a new speaker nor guarantee true whisper or natural emotion.
A trained acoustic model must honor controls within its learned range. Full
formulas, measured performance and limitations are in
[the technical report](../../docs/TTS_ARCHITECTURE_AND_EMOTION_REPORT.md).

## Example

```sh
cmake --workflow --preset explicit-neural
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --text "Hello from a controlled neural voice." --output artifacts/explicit_neural/hello.wav --pitch-scale 1.1 --speed 0.95
```

For a runnable contract-only example without trained weights, use
`--assets tests/fixtures/explicit --text "hello"`. This produces untrained test
audio, not speech. Tiny fixture graphs are checked into the test directory;
the generation script needs Python ONNX, but inference and CTest do not.

ONNX sessions use four intra-op threads by default, one inter-op thread,
sequential execution, full graph optimization, CPU memory arenas, memory patterns
and disabled worker spinning. `--threads N` changes the thread count. Optional
`--thread-affinities` passes ONNX's worker affinity string to both sessions;
leave it unset for portable scheduling. Specify one semicolon-separated group
per worker thread (`N - 1`), using core IDs valid on your machine. See the
[official threading documentation](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\model_assets\explicit_neural\MODEL_CARD.md ===
# Local explicit English voice

This setup adapts public PaddleSpeech FastSpeech2 VCTK and HiFi-GAN VCTK
ONNX releases. The acoustic model is pretrained FastSpeech2; our contribution
is an explicit inference interface and programmatic controls, not new training.

Publisher resources:

- [Released models](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md)
- [FastSpeech2 implementation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/models/fastspeech2/fastspeech2.py)
- [English frontend](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/frontend/phonectic.py)
- [Publisher Apache 2.0 license](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/LICENSE)

`download_manifest.json` records the exact publisher archive URLs, pinned hashes,
statistics and hashes of generated assets. Dataset and checkpoint terms remain
those of their publishers; this card does not grant additional rights.

The acoustic export retains trained encoder, speaker conditioning, decoder,
postnet and variance embeddings. The predictor export retains trained duration,
pitch and energy heads. We replace internal predicted duration expansion with
an exact Gather expansion and move trained kernel-size-one variance embeddings
to frame positions. Repeated original token contours reproduce the upstream
neutral mel within float rounding; frame contours permit finer control.
The vocoder modification changes tensor layout only.

Native rate: 24000 Hz. Hop: 300. FFT: 2048. Hann window: 1200.
Mel: 80 bins, 80..7600 Hz, original model denormalization preserved.
Vocabulary: publisher stressed ARPAbet IDs, no Piper BOS/EOS/padding.
Speaker map: 107 identities, IDs 0..106; default ID 0 is p225.

F0 is supplied in Hz and transformed to continuous normalized log-F0, mean
5.0610495 and standard deviation 0.35158658. Zero F0 maps to the pitch mean;
there is no independent voicing control. Energy is relative to the training
feature mean 29.903042 with standard deviation 25.684935. It is not waveform dB.
Emotion presets alter duration/pitch/energy; they are not trained emotion labels
and do not guarantee natural emotion or physical whisper. Extreme values can
degrade speech. CMUdict letter fallback is limited for arbitrary names/numbers.

All inference is local. Only explicit preparation scripts download assets.
Weights belong in `model_assets/explicit_neural/`; generated WAVs and validation
reports belong in `artifacts/`. Tiny fixture graphs are exclusively test assets.

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\tests\fixtures\explicit\README.md ===
These tiny ONNX graphs are **untrained contract fixtures**, generated by
`scripts/create_explicit_test_models.py --output tests/fixtures/explicit --negative-fixtures`.
They test tensor layout, frame/sample counts, conditioning inputs and error handling.
Their WAV output is step-shaped test data, not intelligible speech.

`acoustic_generator.onnx` computes a simple expression of F0, energy, token IDs,
duration sum, speaker ID and mel-bin index. `vocoder_hifigan.onnx` takes channel
zero, applies tanh, and repeats each value 256 times. The remaining graphs
deliberately violate the runtime contract. They contain no downloaded weights.

