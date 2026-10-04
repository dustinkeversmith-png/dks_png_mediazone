# Caption models

Speech in, text out. Speech synthesis lives in `../../audio_synth`, and no
caption target links against it.

| Directory | Engine | Status |
| --- | --- | --- |
| [`int8_zip/`](int8_zip/) | Streaming Zipformer2 transducer, INT8, ONNX Runtime CPU | Production: `captions` CLI and `CaptionEngine` API |

| Mode | LibriSpeech test WER | Speed (6-core CPU) |
| --- | --- | --- |
| batch (files, video), beam 4 | 3.41% (450 utterances / 9,650 words) | ~130x real time |
| live (microphone), greedy | 2.26% on the 40-file dev slice | ~15x real time, ~0.5 s to first words |

Start with [int8_zip/README.md](int8_zip/README.md) for the build, the CLI and
the API. The retired GMM-HMM and DSP experiments are kept under `../archive/`
for reference and are not built.
