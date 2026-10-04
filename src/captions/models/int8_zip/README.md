# `int8_zip` — Zipformer2 captions

Speech to text on the CPU, offline. The engine uses a single model: the
streaming Zipformer2 transducer in `models/librispeech/`, INT8-quantized and
run with ONNX Runtime. Recorded audio and video are decoded in batches with
4-way modified beam search. The microphone path is a low-latency streaming
decoder. No Python, GPU or network access is needed at run time.

| Mode | WER (LibriSpeech test, 450 utts) | Speed (Ryzen 5 5600X, 6 cores) |
| --- | --- | --- |
| batch: files, video | **3.41%** | ~130x real time: 1 h of audio in ~30 s |
| live: microphone, streams | 2.26% on the 40-file dev slice, greedy | ~15x real time on 2 threads; first words after ~460 ms |

## Build

From the repository root (ONNX Runtime lives in `dependencies/onnxruntime`):

```powershell
cmake -S . -B build
cmake --build build --config Release --target captions
ctest --test-dir build -C Release -R captions-self-test
```

## Command line

```powershell
$C = "build/bin/Release/captions.exe"
& $C --input lecture.mp4                            # -> lecture.srt next to the video
& $C --input talk.mkv --format srt,vtt,json --out subs/talk
& $C --input podcast.mp3 --format txt --out -       # transcript to stdout
& $C --mic                                          # live captions until Ctrl+C
& $C --mode live --input talk.wav --realtime        # live engine fed at real-time pace
& $C --help
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--input <path>` | — | Any audio/video ffmpeg reads; PCM WAV is read directly |
| `--format <list>` | `srt` | `srt`, `vtt`, `txt`, `json`; comma-separated for several |
| `--out <path>` | input path | Output base path. `talk.srt` → `talk.vtt` etc. per format; `-` = stdout |
| `--mode batch\|live` | `batch` | Batch for files; live = streaming engine (`--mic` implies it) |
| `--mic`, `--seconds N` | — | Live microphone capture, optionally time-limited |
| `--realtime` | off | Live mode on a file: pace input like a live source |
| `--upper` | off | Keep raw uppercase instead of sentence case |
| `--line-chars N` | 42 | Caption line width (cues are at most two lines, 6 s) |
| `--workers`, `--threads`, `--batch` | 6, 2, 16 | Batch parallelism (see [BATCH_ASR.md](BATCH_ASR.md)) |
| `--beam K` | 4 | Beam width in batch mode; 1 = greedy |
| `--max-seg S`, `--gate-rms R` | 20, 0.0003 | Long-form segmentation |
| `--packet-ms N`, `--no-gate` | 100, gate on | Live packetizer and silence gate |
| `--ffmpeg <path>` | `dependencies/ffmpeg/bin/ffmpeg.exe`, else PATH | Decoder for non-WAV input |
| `--models <dir>` | built-in path | Relocated copy of the Zipformer2 checkpoint |
| `--quiet` | off | No progress or statistics on stderr |

Evaluation, using the same frozen split and scorer as all published numbers:

```powershell
& $C --benchmark data/audio/librispeech [--dev] [--limit N] [--report out.json]
& $C --benchmark data/audio/librispeech --mode live --dev --limit 40
& $C --benchmark data/audio/librispeech --longform [--repeat 3]   # test split as one long recording
& $C --self-test data/audio/librispeech                          # what ctest runs
```

## C++ API

```cpp
#include <captions/caption_engine.hpp>

captions::CaptionEngine engine;                        // Zipformer2, beam 4, 6x2 workers
auto segments = engine.transcribe_file("talk.mp4");   // or transcribe_pcm(samples_16k_mono)
auto cues = captions::build_cues(segments);           // sentence case, <= 2x42 chars, <= 6 s
std::string srt = captions::render(captions::CaptionFormat::srt, segments, cues);
```

Each `SegmentResult` has `start`/`end` seconds, `text`, and per-word
`words[i].start/end` at 40 ms resolution. For live input, use
`captions::StreamingOnnxAsr` (`streaming_asr.hpp`): call `accept()` with each
packet, read the growing transcript from `text()`, and call `finish()` at the
end. Link the `captions_streaming_asr` CMake target.

## Contents

| Path | What it is |
| --- | --- |
| `include/captions/caption_engine.hpp` | `CaptionEngine`, `load_audio`, cue building, SRT/VTT/TXT/JSON rendering |
| `include/captions/batch_asr.hpp` | Batch decoder: segmenter, lockstep batching, worker pool, beam search |
| `include/captions/streaming_asr.hpp` | Live decoder: energy gate, cached encoder, greedy search |
| `caption_engine.cpp`, `batch_asr.cpp`, `streaming_asr.cpp` | Implementations |
| `ort_graph.hpp`, `ort_session_options.hpp` | ONNX session wrapper, fbank settings, CPU tuning |
| `../../app/captions.cpp` | The `captions` command line |
| `models/librispeech/` | The Zipformer2 checkpoint (fetched, not versioned) |
| `scripts/` | `fetch_streaming_asr.py` (one-time pinned download), `caption.ps1` launcher |
| `third_party/` | kaldi-native-fbank (pinned) |

Design notes and tuning measurements: [BATCH_ASR.md](BATCH_ASR.md) covers the
batch pipeline, and [STREAMING_ASR.md](STREAMING_ASR.md) covers live latency.
