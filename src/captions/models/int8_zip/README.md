# `int8_zip` — INT8 Zipformer streaming captions

Pretrained streaming Zipformer transducer (20M parameters, INT8-quantized) run
through three native ONNX Runtime C++ sessions. This is the most accurate
caption engine in the repository.

| Measure | Value |
| --- | --- |
| Test WER | 4.15% (400 errors / 9,650 words, 450 utterances) |
| Compute RTF | 0.031 (~32x real time), single CPU thread pair |
| Caption latency | ~400 ms initial buffering; does **not** meet a strict 200 ms target |

Start with [STREAMING_ASR.md](STREAMING_ASR.md) for setup, the streaming API
contract, and the latency analysis. Run it via
[`scripts/caption.ps1`](scripts/caption.ps1).

## Contents

| Path | What it is |
| --- | --- |
| `include/captions/streaming_asr.hpp` | Public API: `captions::StreamingOnnxAsr` |
| `streaming_asr.cpp` | Implementation (ONNX sessions, fbank, energy gate, transducer search) |
| `ort_session_options.hpp` | Shared CPU session tuning for ONNX Runtime |
| `models/compact/` | Default weights: 20M streaming Zipformer + tokens |
| `models/librispeech/` | Larger chunk-16/left-64 variant |
| `scripts/` | `fetch_streaming_asr.py` (pinned, checksummed download), `verify_streaming_asr_report.py`, `caption.ps1` |
| `artifacts/` | Measured runs, reports, and the repair write-up |
| `third_party/` | kaldi-native-fbank (pinned), vendored python tooling |

ONNX Runtime itself is **not** vendored here: it is shared with the speech
synthesis project and lives in `dependencies/onnxruntime`.
