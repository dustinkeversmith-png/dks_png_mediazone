# Implemented synthesis comparisons

The render suite turns the paradigms described in `DSP.md`, `acoutics.txt`,
`UNIT_SELECTION.md`, and `include/net/README.md` into small, deterministic
listening artifacts. They share text normalization, a compact character-to-
acoustic-control mapping, 24 kHz output, and peak normalization so comparisons
are repeatable.

| Output | Implementation | Listen for |
| --- | --- | --- |
| `formant.wav` | Harmonic glottal source weighted around three vowel formants | Robotic but stable source-filter speech |
| `statistical_hmm.wav` | Smoothed duration, F0, energy, and formant means | Averaged, muffled trajectories |
| `articulatory.wav` | Vocal-tract length/opening constraints applied to resonances | Tightly coupled moving resonances |
| `lpc.wav` | Narrow all-pole-like resonant envelope plus excitation noise | Buzzy, coded-speech character |
| `sine_wave.wav` | Three dynamic sinusoids following F1–F3 | Whistling spectral landmarks |
| `unit_selection.wav` | Deterministic per-unit snippets with tapered boundaries | Repeated timbre and changing joins |
| `homebrew_neural.wav` | Tiny fixed-weight encoder producing mel/control variation | The transparent project baseline |
| `onnx_style.wav` | FastSpeech2-shaped encoder, duration expansion, variance conditioning, and decoder projection | Longer vowel timing and smoother variance controls |

The neural implementations are deliberately dependency-free and untrained.
`OnnxStyleAcousticModel` describes the compute shape and C++ boundary expected
from an ONNX deployment, but it does not load a `.onnx` file or HiFi-GAN weights.
That distinction matters: these WAVs validate orchestration, features, metrics,
and artifact generation—not neural voice quality.

`neural_comparison.json` reports frame counts, duration ratio, and normalized-
time log-mel MAE/RMSE. Once both systems use trained weights, add MCD, F0/VUV,
ASR-WER, and listening scores against held-out LibriTTS-R references.

## Next production step

Implement an optional `OnnxRuntimeAcousticModel` target behind a CMake option.
Its inputs should be token IDs, speaker/style ID, duration scale, pitch scale,
and energy scale; its outputs should be a `[frames, mel_bins]` log-mel tensor
plus optional duration/F0/energy tensors. Keep HiFi-GAN as a separate vocoder
interface so acoustic-model and vocoder regressions can be isolated.
