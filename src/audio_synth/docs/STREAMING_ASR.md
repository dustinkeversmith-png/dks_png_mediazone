# Streaming ONNX captions

The neural caption entry point is `scripts/caption.ps1`. It runs the compact
LibriSpeech-trained Zipformer transducer through three native ONNX Runtime C++
sessions. No Whisper model, server, Python inference, external LM, or reference
transcript is used by the recognizer. The earlier GMM-HMM executable remains a
historical research baseline.

## Build and run (from `src/audio_synth`)

```powershell
python scripts/fetch_streaming_asr.py --runtime
cmake -S . -B build/asr -G "Visual Studio 17 2022" -A x64 -DVA_ENABLE_ONNX_RUNTIME=ON -DVA_ENABLE_STREAMING_ASR=ON -DONNXRUNTIME_ROOT="$PWD/third_party/onnxruntime-win-x64-1.23.2"
cmake --build build/asr --config Release --target caption-streaming vocal-acoustics-tests
./scripts/caption.ps1 --file recording.wav
./scripts/caption.ps1 --mic
# Optional duration limit for microphone capture:
./scripts/caption.ps1 --mic --seconds 30
```

The model download uses pinned revisions and checksums. Runtime inference is
offline. The build uses the repository's existing miniaudio header, the pinned
kaldi-native-fbank source, and its hash-pinned KissFFT dependency. Python is only
needed for setup, not captioning. All three model graphs are INT8-quantized;
cache states and operators not covered by dynamic quantization remain float32.

## Streaming contract

`vocal::StreamingOnnxAsr` accepts normalized 16 kHz mono float samples through
`accept(span)`, exposes accumulating partial text, and flushes on `finish()`.
`reset()` starts an independent utterance without reloading the sessions.
One caller owns each stream. Network/device packets can have arbitrary lengths;
the internal packetizer uses 100 ms by default and supports 10–160 ms.

The filterbank retains overlap only. Every encoder call carries the previous
ONNX cache tensors forward. Feature tensors are contiguous and reused, while
ONNX Runtime uses CPU arenas and memory-pattern reuse. Execution is sequential,
with two encoder threads and one thread for each tiny decoder/joiner session;
the sessions do not execute concurrently. `--threads` caps encoder workers at
four. The original and synthesis adapters share session configuration code.

The energy gate uses RMS 0.0003, 200 ms pre-roll, and a 1,000 ms hangover. It
avoids decoding idle low-energy audio and preserves short pauses. It is not a
speech/noise classifier: loud background noise can still trigger recognition,
and unusually quiet speech can fall below its threshold. `--no-gate` disables
it. At an endpoint, right-context zeros flush the model, the transcript is
committed, and caches reset. The microphone callback only writes to a bounded
single-producer/single-consumer queue; it never runs ONNX. Overflow is reported
as an error instead of silently claiming uninterrupted captions.

## Latency limitation

**The current checkpoint does not satisfy under-200-ms caption latency.**
It consumes a native 320 ms advance and needs 397.5 ms of samples for its first
feature window. With 100 ms packets, the earliest initial encoder invocation
is at 400 ms, plus computation. Token emission may lag further because the
transducer chooses when to emit. Fast per-packet computation is not equivalent
to audio-to-text latency. A different lower-lookahead checkpoint/export is
required to meet that separate target; changing input packet size cannot do it.

## Reproduce validation

```powershell
ctest --test-dir build/asr -C Release --output-on-failure
./scripts/caption.ps1 --self-test ../../data/audio/librispeech/sample_000000.wav
./scripts/caption.ps1 --benchmark ../../data/audio/librispeech --dev --limit 40 --report artifacts/asr_streaming_compact_fixed_dev.json
./scripts/caption.ps1 --benchmark ../../data/audio/librispeech --report artifacts/asr_streaming_test.json
```

The harness requires the same frozen 600-WAV corpus as the old benchmark, sorts
the paths, and evaluates the last 450 files for test. It reuses the old tokenizer
and S/D/I scorer. Missing audio/transcripts cause failure, not silent exclusion.
The references are read only by scoring code, never supplied to the model.
The summary and per-utterance JSONL contain transcripts, counts, and timing.

RTF includes filterbanks, gate, all three inference graphs, greedy search, and
endpoint flushing. `pipeline_rtf` also includes file loading and stream reset.
Model loading is recorded separately. No real-time sleeps, GPU, batching across
utterances, or reference-aware decoding are used in the benchmark. Encoder and
packet p95/p99 timings describe computation only. Microphone hardware latency
requires a separate loopback measurement and has not been inferred from file
benchmarks. See [the experiment results](../artifacts/ASR_REPAIR_RESULTS.md).
