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
Supplying all three duration/F0/energy CSVs selects fully manual mode and bypasses
the predictor even when it is not installed. Partial overrides still use predicted
defaults. Neutral/default sliders preserve the supplied vectors. Real manual
examples are provided in `../../examples/prosody/`.

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
