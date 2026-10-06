# Live mode: cache-aware streaming FastConformer

`captions --mic` (or `--mode live --input <media>`) runs
`captions::StreamingOnnxAsr` with NVIDIA's
`stt_en_fastconformer_hybrid_large_streaming_480ms` (INT8,
`models/nemo-streaming-480ms`). The model was trained on NVIDIA's large
multi-domain English set, and decoding is greedy RNN-T. Text appears as audio
arrives and grows by appending only, so the terminal shows a running
transcript.

```powershell
$C = "build/bin/Release/captions.exe"
& $C --mic                                         # until Ctrl+C
& $C --mic --seconds 30 --out meeting              # also saves meeting.txt
& $C --mic --hotwords terms.txt                    # favour domain terms
& $C --mode live --input talk.wav --realtime       # file fed at real-time pace
& $C --mode live --models src/captions/models/int8_zip/models/librispeech   # older Zipformer2
```

## How it decodes

- The encoder sees 65 feature frames per call and advances 56 (560 ms). The
  9-frame overlap is the model's pre-encode cache.
- Attention caches (17 layers × 70 frames) and convolution caches
  (17 × 8 frames) carry the left context, so each frame is encoded once.
- Features are the NeMo mel front end without per-utterance normalization;
  the streaming export is trained that way.
- Each new segment starts from zero caches, with silence in the first window's
  left-context frames.
- RNN-T greedy search emits at most 5 symbols per 80 ms encoder frame. The
  LSTM prediction network carries its state across chunks.
- Hotwords bias the joiner logits, using the same prefix tree as batch mode.

## Gate and endpointing (shared with the Zipformer2 path)

- An RMS gate (0.0003) with 200 ms of pre-roll and a 1 s hangover keeps idle
  audio away from the model.
- At an endpoint, right-context zeros flush the model, the text is committed,
  and caches reset. `--no-gate` disables this.
- The microphone callback only writes to a bounded lock-free ring and never
  runs ONNX. Overflow is reported as an error, not hidden.

## Latency

- The first words of a segment appear roughly **0.6 s** after speech starts:
  one 560 ms chunk, plus compute, plus the transducer's emission delay.
- A 1040 ms variant (more context) and an 80 ms variant (lower latency) of the
  same model exist as sherpa-onnx exports. Point `--models` at either; they
  were not benchmarked here.
- Speed is ~17× real time for one stream on 2 threads, ~8–10× each with six
  concurrent streams.

## Accuracy

See the live columns in [README.md](README.md#accuracy-ryzen-5-5600x-6-cores--12-threads).
The new model beats the old LibriSpeech Zipformer2 on every out-of-domain set:

- AMI distant mic: 86.9% → 62.1% WER
- Earnings-22: 50.6% → 23.5%
- GigaSpeech: 23.2% → 13.7%

On clean LibriSpeech it scores 2.86% against 3.54%. The one exception is
0 dB synthetic noise: 12.4% against 11.0%.

## Validation

```powershell
ctest --test-dir build -C Release --output-on-failure -R captions-self-test
& $C --benchmark data/audio/librispeech --mode live --dev --limit 40
& $C --eval data/audio/eval/ami_sdm --mode live --streams 6
```

The self-test covers these live-engine contracts:

- Five seconds of silence never reaches the encoder.
- `accept()` after `finish()` is rejected.
- 137-, 1600- and 2560-sample packets give identical transcripts.
- `reset()` clears state.
