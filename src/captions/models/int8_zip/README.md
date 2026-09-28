# `int8_zip` — INT8 Zipformer captions (live and batch)

Pretrained streaming Zipformer transducer (20M parameters, INT8-quantized) run
through three native ONNX Runtime C++ sessions. This is the most accurate
caption engine in the repository. The same graphs power two front ends:

| | `caption-streaming` (live) | `caption-batch` (files, video) |
| --- | --- | --- |
| Test WER | 4.15% | 3.96% (beam 4); 3.41% with `models/librispeech` |
| Speed | ~32x real time, one thread pair | ~265x real time, 6-core CPU (1 h of audio in ~14 s) |
| Latency | ~400 ms initial buffering; does **not** meet a strict 200 ms target | offline |
| Docs | [STREAMING_ASR.md](STREAMING_ASR.md) | [BATCH_ASR.md](BATCH_ASR.md) |

Run live captions via [`scripts/caption.ps1`](scripts/caption.ps1), and caption
files or video with `build/bin/Release/caption-batch.exe --input <media>`.

## Contents

| Path | What it is |
| --- | --- |
| `include/captions/streaming_asr.hpp` | Public API: `captions::StreamingOnnxAsr` |
| `include/captions/batch_asr.hpp` | Public API: `captions::BatchOnnxAsr`, `segment_audio` |
| `streaming_asr.cpp` | Streaming implementation (energy gate, cached encoder, greedy search) |
| `batch_asr.cpp` | Batch implementation (segmenter, lockstep batched encoder, worker pool, greedy/beam search) |
| `ort_graph.hpp`, `ort_session_options.hpp` | Shared ONNX session wrapper, fbank settings, CPU tuning |
| `models/compact/` | Default weights: 20M streaming Zipformer + tokens |
| `models/librispeech/` | Larger chunk-16/left-64 Zipformer2 variant (more accurate, ~2x slower) |
| `scripts/` | `fetch_streaming_asr.py` (pinned, checksummed download), `verify_streaming_asr_report.py`, `caption.ps1` |
| `artifacts/` | Measured runs, reports, and the repair write-up |
| `third_party/` | kaldi-native-fbank (pinned), vendored python tooling |

ONNX Runtime itself is **not** vendored here: it is shared with the speech
synthesis project and lives in `dependencies/onnxruntime`.
