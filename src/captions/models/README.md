# Caption models

Speech in, text out. Speech synthesis lives in `../../audio_synth`, and no
caption target links against it.

| Directory | Engine | Status |
| --- | --- | --- |
| [`int8_zip/`](int8_zip/) | Parakeet TDT 110M + Silero VAD (batch); NeMo cache-aware streaming FastConformer (live); domain-term biasing in both. INT8, ONNX Runtime CPU | Production: `captions` CLI and `CaptionEngine` API |

| Mode | Accuracy (WER) | Speed (6-core CPU) |
| --- | --- | --- |
| batch (files, video) | 2.68% LibriSpeech · 11.0% GigaSpeech · 14.6% AMI · 33.6% AMI distant mic | ~140× real time; 1 h in ~28 s |
| live (microphone) | 2.86% LibriSpeech · 13.7% GigaSpeech · 29.3% AMI · 62.1% AMI distant mic | ~17× real time per stream, ~0.6 s to first words |

Start with [int8_zip/README.md](int8_zip/README.md) for setup, the full noisy
and out-of-domain results, domain terms, the CLI and the API. The retired
GMM-HMM and DSP experiments are kept under `../archive/` and are not built.
