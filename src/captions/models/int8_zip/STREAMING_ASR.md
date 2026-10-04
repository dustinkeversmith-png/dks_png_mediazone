# Live mode: streaming Zipformer2

`captions --mic` (or `--mode live --input <media>`) runs the Zipformer2
checkpoint through `captions::StreamingOnnxAsr`. Text appears as audio
arrives and grows by appending only, so the terminal shows it as a running
transcript. Live mode uses greedy search; batch mode adds beam 4 for
recorded files.

```powershell
$C = "build/bin/Release/captions.exe"
& $C --mic                                   # until Ctrl+C
& $C --mic --seconds 30 --out meeting        # also saves meeting.txt
& $C --mode live --input talk.wav --realtime # file fed at real-time pace
```

`scripts/caption.ps1` forwards every argument to `captions.exe`.

## Setup

Fetch the model once (pinned revisions and checksums). The `--runtime` option
also installs the ONNX Runtime SDK into `dependencies/onnxruntime`:

```powershell
python src/captions/models/int8_zip/scripts/fetch_streaming_asr.py --runtime
```

Python is used only for that one-time download. Captioning itself is offline
native code: ONNX Runtime, the pinned kaldi-native-fbank source with its
hash-pinned KissFFT, and the repository's miniaudio header for capture. All
three graphs are INT8-quantized. Cache states and operators not covered by
dynamic quantization stay float32.

## Streaming contract

- `accept(span)` takes normalized 16 kHz mono float samples, `text()` returns
  the accumulated transcript, and `finish()` flushes the end of the audio.
  `reset()` starts an independent utterance without reloading the sessions.
- One caller owns each stream. Packets may have any length; the internal
  packetizer uses 100 ms (`--packet-ms`, 10–160).
- The filterbank keeps only the overlap it needs. Every encoder call carries
  the ONNX cache tensors forward.
- The encoder uses two threads (`--threads`, at most 4); the decoder and joiner
  use one each.

The energy gate:

- Uses RMS 0.0003 with 200 ms of pre-roll and a 1,000 ms hangover. It skips
  idle audio and keeps short pauses.
- Is not a speech/noise classifier. Loud background noise still triggers
  decoding, and very quiet speech can fall below the threshold.
- Can be turned off with `--no-gate`.
- At an endpoint, right-context zeros flush the model, the text is committed,
  and the caches reset.

The microphone callback only writes into a bounded single-producer,
single-consumer ring and never runs ONNX. Overflow is reported as an error, not
hidden.

## Latency

**Captions do not meet a strict 200 ms latency target.** The checkpoint
advances 320 ms per encoder call and needs 457.5 ms of audio for its first
feature window. With 100 ms packets, the first encoder call happens at
~500 ms plus compute. Token emission can lag further, because the transducer
chooses when to emit. Fast per-packet compute (~15x real time on two threads)
is not the same as audio-to-text latency. Smaller packets cannot remove the
model's lookahead.

## Validation

```powershell
ctest --test-dir build -C Release --output-on-failure -R captions-self-test
& $C --benchmark data/audio/librispeech --mode live --dev --limit 40   # 2.26% WER
& $C --benchmark data/audio/librispeech --mode live --report live_test.json
```

The benchmark uses the frozen 600-WAV corpus:

- The paths are sorted; the first quarter is dev and the last 450 files are
  test.
- Scoring uses the shared tokenizer and S/D/I scorer.
- Missing audio or transcripts are a failure, never silently skipped.
- References are read only by the scorer, never given to the model.
- Speed covers filterbanks, gate, all three graphs, greedy search and endpoint
  flushing. Model loading is excluded.
- Microphone hardware latency needs a separate loopback measurement and is not
  inferred from these file runs.
