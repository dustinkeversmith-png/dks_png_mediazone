# Synthesis model families

Each setup has a dedicated audio-generation executable; there is no shared
`vocal-eval` application. Source code lives here, downloaded assets under
`../model_assets/`, and generated audio/reports under `../artifacts/`.

| Directory | Family | Executables | Assets |
| --- | --- | --- | --- |
| `dsp_paradigms/` | Six deterministic source-filter sketches | `formant-tts`, `hmm-tts`, `articulatory-tts`, `lpc-tts`, `sine-wave-tts`, `unit-selection-tts` | None |
| `homebrew_neural/` | Fixed, untrained neural architecture baselines | `homebrew-tts`, `onnx-style-tts` | None |
| `piper_onnx/` | Pretrained end-to-end VITS via ONNX Runtime | `piper-tts` | `../model_assets/piper/` |
| [`matcha_onnx/`](matcha_onnx/README.md) | Compact pretrained Matcha acoustic graph + HiFi-GAN v2 | `matcha-tts` | `../model_assets/matcha/` |
| [`explicit_neural/`](explicit_neural/README.md) | Explicit duration/F0/energy-conditioned acoustic export and frozen vocoder | `explicit-tts` | `../model_assets/explicit_neural/` |

All ONNX families use `VA_ENABLE_ONNX_RUNTIME=ON` and the shared
`../../../dependencies/onnxruntime` SDK. Separate libraries keep ONNX dependencies
out of procedural runners. The CMU frontend is public through
`include/vocal/phonemizer.hpp` and built independently as `vocal_phonemizer`.

The sketches expose their paradigm's behavior but are not trained voices.
`OnnxStyleAcousticModel` is a dependency-free architecture sketch, not an ONNX
checkpoint. Piper and the native Matcha/HiFi-GAN pair are provisioned pretrained models.
The strict explicit pipeline needs matching trained exports; its checked-in fixtures
only validate the contract and produce test audio.

Framework audio, features, metrics, dataset, reporting and evaluation APIs remain
in `include/vocal/` and `src/`. Captions live separately in `../../captions`.
