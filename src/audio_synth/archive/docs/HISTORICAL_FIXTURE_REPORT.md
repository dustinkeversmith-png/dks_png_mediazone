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

