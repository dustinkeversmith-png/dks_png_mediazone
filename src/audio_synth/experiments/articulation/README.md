# Articulation lab: isolated C++ experiment

`articulation-lab` reuses the existing trained predictor, explicit acoustic ONNX
and matching HiFi-GAN. It changes word/phoneme duration, pitch and energy controls
between prediction and acoustic inference. It does not edit model weights or
production C++ modules. The CMake option `VA_BUILD_ARTICULATION_EXPERIMENT` defaults
to OFF; the experiment preset enables it in its own build directory.

```text
Text -> existing CMU phonemizer -> verified word/token alignment
     -> existing trained prosody predictor
     -> selected word/phoneme edits + optional cross-term coupling
     -> existing explicit acoustic ONNX -> existing HiFi-GAN -> 24 kHz PCM16 WAV
```

Run commands from `src/audio_synth`. All synthesis commands below are single-line
commands for both PowerShell and Git Bash on Windows. Existing assets and the
ONNX Runtime SDK from the root README are used; this executable downloads nothing.
Synthesis, alignment, control editing and diagnostics all run in C++.

## Build and run

```sh
cmake --workflow --preset articulation-lab
./build/articulation-lab/Release/articulation-lab.exe --help
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --text "We synthesize a clear acoustic voice." --speaker-id 0 --output artifacts/articulation/baseline.wav --threads 4
```

Incremental build and repeat the two experiment tests:

```sh
cmake --build --preset articulation-lab
ctest --preset articulation-lab
```

No manual curve files are needed: every experiment starts from the trained
predictor. The existing `explicit-tts` manual-curve mode remains available.

## Public C++ API: articulation.hpp

Include `articulation.hpp` and link `vocal_articulation_experiment`. The public
`vocal::experiment::Articulation` class owns the aligned text, original prediction
and edited controls. Its implementation remains in `articulation.cpp`; the header
declares the reusable interface. No CLI, ONNX session or Python runtime is needed
to use the control class itself.

```cpp
#include "articulation.hpp"

// frontend, vocabulary and predictor are the existing pipeline components.
auto aligned = vocal::experiment::align_text(frontend, text, vocabulary);
auto predicted = predictor.predict(aligned.phonemes.token_ids, speaker_id);
predicted.speaker_id = speaker_id;
predicted.token_kinds = aligned.phonemes.token_kinds;

vocal::experiment::Articulation voice(aligned, predicted);
voice.select_word(3).select_vowels()
     .scale_duration(1.5F).scale_pitch(1.12F).scale_energy(1.25F)
     .ramp_pitch(25.F).couple_energy_to_pitch(.08F)
     .couple_pitch_to_vowel_duration(.5F).apply();

auto mel = acoustic.infer(voice.text().phonemes.token_ids, voice.controls());
auto audio = vocoder.synthesize(mel);
```

| Public methods | Purpose |
| --- | --- |
| `select_word(index)`, `select_all_words()` | Choose a word or every word |
| `select_phone(selector)` | Choose all phones, a category, or an ARPAbet symbol |
| `select_vowels()`, `select_stressed_vowels()` | Choose vowel categories |
| `select_consonants()`, `select_unvoiced_consonants()` | Choose consonant categories |
| `scale_duration(value)`, `scale_pitch(value)`, `scale_energy(value)` | Set independent control scales |
| `ramp_pitch(hz)` | Set a word-relative pitch ramp |
| `couple_energy_to_pitch(gain)` | Configure vowel effort/pitch coupling |
| `couple_pitch_to_vowel_duration(gain)` | Configure nonlinear vowel elongation |
| `apply()` | Commit all pending settings together |
| `apply(edit)`, `apply(span_of_edits)` | Apply explicit policies; batches are atomic |
| `apply_file(path)` | Read and apply the same TSV tables as the CLI |
| `clear_pending()` | Discard uncommitted settings |
| `reset()` | Restore original prediction and discard pending settings |
| `text()`, `baseline()`, `controls()`, `pending_edit()` | Inspect immutable alignment/controls/pending settings |
| `Articulation::expressive_sweep(word)` | Return all 21 named experiment cases |

Setters configure a single pending policy; they do not immediately modify curves.
Calling a setter twice replaces that pending value. `apply()` commits the combined
policy and resets the pending selection and values to defaults. Later applications
operate on the current curves, so intentional successive edits compound. If an
application or any row of a batch fails, current curves remain unchanged. Failed
applications leave pending settings available for correction or `clear_pending()`.

```cpp
voice.reset().select_word(1).select_phone("S").scale_duration(2.F).apply();
voice.apply_file("experiments/articulation/example_edits.tsv");
for (const auto& experiment : vocal::experiment::Articulation::expressive_sweep(3)) {
    voice.reset().apply(experiment.edits); // Each case starts from prediction.
    // Generate with acoustic.infer(..., voice.controls()), then the vocoder.
}
```

The CLI uses this class for its edit batches and the same public sweep presets.
Existing free functions remain available for callers using the stateless API.
Speaker identity is supplied with the predicted baseline; choose the speaker
before running the predictor. Asset paths, runtime threads, WAV saving and
diagnostics remain responsibilities of the synthesis runner.

## Select words and consonants

Word indices start at zero. In the example sentence, `We=0`, `synthesize=1`,
`a=2`, `clear=3`, `acoustic=4`, `voice=5`. Punctuation is not a word. Repeated
words have separate indices; apostrophes inside words remain part of the word.
The supported frontend is the existing English ASCII/ARPAbet frontend.

Lengthen only the S consonants in "synthesize":

```sh
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --text "We synthesize a clear acoustic voice." --word-index 1 --phone S --duration-scale 2 --output artifacts/articulation/synthesize_s.wav --threads 4
```

Emphasize the vowels in "clear" with a rising pitch contour:

```sh
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --text "We synthesize a clear acoustic voice." --word-index 3 --phone vowels --duration-scale 1.5 --pitch-scale 1.12 --energy-scale 1.25 --pitch-rise-hz 25 --output artifacts/articulation/clear_emphasis.wav --threads 4
```

Shorten consonants throughout the sentence:

```sh
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --text "We synthesize a clear acoustic voice." --word-index -1 --phone consonants --duration-scale 0.8 --output artifacts/articulation/crisper.wav --threads 4
```

`--word-index -1` selects every word and is the default. `--phone` defaults to
`all`. Other selectors are `vowels`, `stressed`, `consonants`, `unvoiced`, or an
exact ARPAbet phone such as `HH`, `S`, `M`, `IY1`. A stressless vowel selector
such as `IY` matches its stress variants. Unknown phones and selections matching
no tokens are errors. Word boundaries are preserved and never selected.

These edits change the temporal/prosodic realization of existing phones. They
do not add a tongue-position, lip-pressure or glottal-source control. Lengthening
a stop consonant is not guaranteed to create a longer release burst: the model
learned that phone's realization from speech training data.

## Scalar ranges and coupling

| Option | Default | Accepted range | Effect on selected phones |
| --- | ---: | --- | --- |
| `--duration-scale` | 1 | 0.25–4 | Multiply token frame counts, rounding to integer frames |
| `--pitch-scale` | 1 | 0.45–1.8 | Multiply requested F0 |
| `--energy-scale` | 1 | 0.1–2.5 | Multiply acoustic energy conditioning, not final WAV gain |
| `--pitch-rise-hz` | 0 | -100–100 | Add a ramp from zero at word start to this value at word end |
| `--energy-to-pitch` | 0 | 0–0.2 | Couple energy changes to vowel F0 |
| `--vowel-peak-lengthening` | 0 | 0–2 | Lengthen vowels nonlinearly when their requested pitch rises |

Each frame is 12.5 ms (300 samples at 24 kHz). A positive-duration token keeps
at least one frame; zero-duration tokens remain zero. Modified positive F0 is
bounded to 40–500 Hz. Identity edits preserve the original predictor values.
Curves are resampled within each token when its duration changes, keeping token
boundaries aligned. The shared control validator enforces the frame budget.

For selected vowels, the effort multiplier is
`exp(g * clamp(log((new_energy + 0.01)/(old_energy + 0.01)), -1, 1))`.
With `p` the token's maximum positive `log(new_F0/old_F0)`, the extra vowel
duration multiplier is `1 + k * min(p, 0.7)^2`. These are explicit experimental
policies, not physiological models of vocal effort.

```sh
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --text "We synthesize a clear acoustic voice." --word-index 3 --phone vowels --pitch-scale 1.3 --energy-scale 1.2 --pitch-rise-hz 30 --energy-to-pitch 0.12 --vowel-peak-lengthening 2 --output artifacts/articulation/coupled.wav --threads 4
```

Zero F0 is preserved by the edit layer, but the staged acoustic model normalizes
zero to its training pitch mean. It is not a reliable switch for unvoiced speech,
whispering or breath. Requested F0 and measured output pitch can differ, especially
at extreme settings. No final peak normalization hides clipping in diagnostics.

## Edit multiple words in one generation

Use [example_edits.tsv](example_edits.tsv) with the example sentence:

```sh
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --text "We synthesize a clear acoustic voice." --edits experiments/articulation/example_edits.tsv --output artifacts/articulation/multiword.wav --threads 4
```

Each non-comment line has exactly eight whitespace-separated fields:

```text
# word  phone   duration pitch energy rise_Hz effort vowel_peak
1       S       2        1     1.1    0       0      0
3       vowels  1.5      1.12  1.25   25      0.08   0.5
5       all     0.9      0.95  0.85  -20      0      0
```

Rows apply sequentially; overlapping rows compound their edits. Tables accept
up to 128 edits and cannot be mixed with scalar edit flags or `--word-index`.
There is no inline `[happy]` parser in this experiment. A future emotion map can
resolve named policies into these word-index edits.

## Sweep expressive possibilities

The built-in sweep generates 21 WAVs, including unchanged baseline, word timing,
vowels, stressed vowels, consonants, unvoiced consonants, S sibilants, pitch,
energy, rising/falling contours, coupled controls and two extreme stress cases:

```sh
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --speaker-id 0 --text "We synthesize a clear acoustic voice." --word-index 3 --sweep artifacts/articulation/sweep --threads 4
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --speaker-id 1 --text "The little bird sings beside the window." --word-index 3 --sweep artifacts/articulation/speaker1_sweep --threads 4
```

Word-specific cases use the chosen word; consonant/unvoiced/S cases select those
phones across all words. A sweep requires a valid target word with vowels and a
sentence containing the tested phone classes, including `S`. It prevalidates
all cases before generating samples. Use custom single edits for other texts.
Sweep mode cannot be combined with `--edits`, `--output` or scalar edit flags.
Reusing a sweep path overwrites its named outputs; use a new path to retain runs.

Every WAV has a `.diagnostics.json` containing word/token/frame intervals,
phone symbols, input IDs, base/final durations, final F0/energy arrays, edit
parameters, waveform peak/RMS/clipping and inference timings. A sweep also writes
`measurements.csv`. Inspect this JSON to locate a specific word or consonant.
Outputs must be WAVs outside the model-asset directory.

The optional offline analysis script measures WAVs and estimates word-local
voiced pitch. It is not used by the executable:

```sh
python experiments/articulation/analyze_sweep.py --directory artifacts/articulation/sweep --word-index 3
```

That analysis requires NumPy. `--recognize --checkpoint PATH` additionally uses
an existing local Whisper `small.en` checkpoint and its installed Python runtime;
it never downloads a checkpoint. ASR tests intelligibility, not naturalness or
emotion. See [the measured results and vocal-event research](../../docs/ARTICULATION_EXPERIMENT_AND_VOCAL_EVENTS.md).

## Interjection probes

These are trained speech-phone approximations, not validated sigh/laughter models:

```sh
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --text "Ah." --word-index 0 --phone vowels --duration-scale 2 --pitch-scale 0.85 --energy-scale 0.6 --pitch-rise-hz -30 --output artifacts/articulation/probes/ah.wav --threads 4
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --text "Huh." --word-index 0 --duration-scale 2 --pitch-scale 0.85 --energy-scale 0.6 --output artifacts/articulation/probes/huh.wav --threads 4
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --text "Hmm." --word-index 0 --duration-scale 2 --energy-scale 0.7 --output artifacts/articulation/probes/hmm.wav --threads 4
./build/articulation-lab/Release/articulation-lab.exe --assets model_assets/explicit_neural --text "Ha ha." --duration-scale 0.75 --pitch-scale 1.1 --output artifacts/articulation/probes/ha_ha.wav --threads 4
```
