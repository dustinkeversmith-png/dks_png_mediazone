# Real-model manual contours

The root CSVs match exactly `We synthesize a clear acoustic voice.` with the
staged FastSpeech2 VCTK frontend and speaker 0: 25 tokens, 174 frames, 24 kHz,
hop 300. They come from neutral trained predictions rounded in diagnostics.
`reference.json` records the text, token IDs and token-map hash.

From `src/audio_synth`, run in PowerShell or Git Bash:

```sh
./build/onnx/Release/explicit-tts.exe --assets model_assets/explicit_neural --speaker-id 0 --emotion neutral --text "We synthesize a clear acoustic voice." --durations examples/prosody/durations.csv --f0 examples/prosody/f0.csv --energy examples/prosody/energy.csv --output artifacts/explicit_neural/manual.wav --threads 4
```

All three overrides bypass prediction. Edit F0/energy values without changing
counts to alter delivery. If durations change, regenerate both per-frame contours
so their counts equal sum(durations). Changing text requires new aligned curves.

`question/` contains a second real example, for `Could we try another way?`,
speaker 0, 98 frames. Its `f0_rising.csv` gently raises the final 24 frame positions
up to 12% relative to the baseline. See the profile design guide for both commands.
Tiny fixture controls for `hello` now live in `tests/fixtures/explicit/controls/`;
they are test-only data and cannot be substituted for these trained-model curves.
