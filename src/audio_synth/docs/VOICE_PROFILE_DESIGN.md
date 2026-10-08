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
