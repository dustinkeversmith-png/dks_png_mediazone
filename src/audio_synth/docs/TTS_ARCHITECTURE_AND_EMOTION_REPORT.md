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
