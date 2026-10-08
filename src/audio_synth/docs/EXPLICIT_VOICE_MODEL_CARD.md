# Local explicit English voice

This setup adapts public PaddleSpeech FastSpeech2 VCTK and HiFi-GAN VCTK
ONNX releases. The acoustic model is pretrained FastSpeech2; our contribution
is an explicit inference interface and programmatic controls, not new training.

Publisher resources:

- [Released models](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md)
- [FastSpeech2 implementation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/models/fastspeech2/fastspeech2.py)
- [English frontend](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/frontend/phonectic.py)
- [Publisher Apache 2.0 license](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/LICENSE)

`download_manifest.json` records the exact publisher archive URLs, pinned hashes,
statistics and hashes of generated assets. Dataset and checkpoint terms remain
those of their publishers; this card does not grant additional rights.

The acoustic export retains trained encoder, speaker conditioning, decoder,
postnet and variance embeddings. The predictor export retains trained duration,
pitch and energy heads. We replace internal predicted duration expansion with
an exact Gather expansion and move trained kernel-size-one variance embeddings
to frame positions. Repeated original token contours reproduce the upstream
neutral mel within float rounding; frame contours permit finer control.
The vocoder modification changes tensor layout only.

Native rate: 24000 Hz. Hop: 300. FFT: 2048. Hann window: 1200.
Mel: 80 bins, 80..7600 Hz, original model denormalization preserved.
Vocabulary: publisher stressed ARPAbet IDs, no Piper BOS/EOS/padding.
Speaker map: 107 identities, IDs 0..106; default ID 0 is p225.

F0 is supplied in Hz and transformed to continuous normalized log-F0, mean
5.0610495 and standard deviation 0.35158658. Zero F0 maps to the pitch mean;
there is no independent voicing control. Energy is relative to the training
feature mean 29.903042 with standard deviation 25.684935. It is not waveform dB.
Emotion presets alter duration/pitch/energy; they are not trained emotion labels
and do not guarantee natural emotion or physical whisper. Extreme values can
degrade speech. CMUdict letter fallback is limited for arbitrary names/numbers.

All inference is local. Only explicit preparation scripts download assets.
Weights belong in `model_assets/explicit_neural/`; generated WAVs and validation
reports belong in `artifacts/`. Tiny fixture graphs are exclusively test assets.
