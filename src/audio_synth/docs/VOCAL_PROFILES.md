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
