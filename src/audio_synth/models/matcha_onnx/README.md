# Compact pretrained Matcha-TTS + HiFi-GAN v2

`matcha-tts` is the ready-to-use pretrained acoustic/vocoder runner. The download
helper uses public release URLs documented by the publisher and sequential HTTPS
downloads, about 80 MB total. Assets live in `model_assets/matcha/`; WAV output
and diagnostics live in `artifacts/matcha/`.

```sh
python scripts/fetch_matcha_voice.py
cmake --workflow --preset matcha
./build/onnx/Release/matcha-tts.exe --text "A clear English sentence." --output artifacts/matcha/example.wav --speed 1.0 --threads 4
```

The original three-step `matcha-icefall-en_US-ljspeech` ONNX graph predicts mel
frames. The small HiFi-GAN v2 graph converts those frames to waveform audio.
Neither graph is rewritten. The publisher's native 22.05 kHz output is linearly
resampled to 24 kHz for canonical PCM16 output. This is one female English voice.

The native acoustic inputs are `x` int64 `[1, tokens]`, `x_length` int64 `[1]`,
`noise_scale` float32 `[1]`, and `length_scale` float32 `[1]`; output is `mel`
float32 `[1, 80, frames]`. `--speed` maps to `length_scale = 1 / speed`.
The vocoder takes `mel` and returns `audio` float32 `[1, frames * 256]`.
The runner accepts speed 0.25..4, noise 0..2 and at most 400 frontend tokens.
Split longer text into separate executions. Model metadata, vocabulary and
output type/shape/finiteness are checked before exporting audio.

The CMU phonemizer uses the publisher's vocabulary, with Matcha's leading and
trailing blanks added around its existing interspersed padding and BOS/EOS.
Upstream uses eSpeak; CMUdict pronunciations and spelling fallback are an
intentional integration with the existing project frontend, so number expansion
and abbreviation handling differ. Start with written English words.

This export does not accept per-token duration, F0 or energy targets. Those
controls remain in `explicit-tts` for truly compatible trained exports;
changing model input names would not make Matcha support them.

Sources and provenance:

- [Publisher's model and supported vocoders](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html)
- [Acoustic export definition](https://github.com/k2-fsa/icefall/blob/master/egs/ljspeech/TTS/matcha/export_onnx.py)
- [Vocoder export definition](https://github.com/k2-fsa/icefall/blob/master/egs/ljspeech/TTS/matcha/export_onnx_hifigan.py)
- [HiFi-GAN repository](https://github.com/jik876/hifi-gan)

The fetcher preserves the archive README and writes source URLs plus hashes to
`download_manifest.json`. The cached acoustic archive is checked against the
publisher's release SHA-256. Follow upstream model/dataset licenses when using
or redistributing assets.
