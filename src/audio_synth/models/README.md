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
