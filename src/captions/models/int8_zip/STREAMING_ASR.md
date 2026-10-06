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

## Automatic gain control

The streaming model has no feature normalization, so it is level-sensitive.
Distant microphones arrive 15–20 dB below close-talk speech (AMI table mic:
about −40 dBFS mean), and the model then emits blanks.

- The live engine tracks the level of active audio (packets above −60 dBFS):
  with a 0.3 s time constant for the first second, then 3 s.
- When that level is below −30 dBFS (`--agc-quiet`), it ramps a gain within
  each packet to bring the level to −23 dBFS (`--agc-target`), up to +32 dB.
  Normal-level audio is never touched or attenuated, since the model was
  trained on it.
- The gate still judges the raw signal; only the model sees the corrected
  one.
- `--no-agc` disables it.

| Live WER | No AGC | Always-on AGC | **Boost-only AGC (default)** | Whole-clip oracle gain |
| --- | --- | --- | --- | --- |
| AMI distant mic (819 utts) | 62.06% | 44.19% | **44.26%** | 43.88% |
| AMI headset (300) | 29.29% | 22.28% | **22.20%** | 22.73% |
| Common Voice (300) | 21.39% | 21.80% | **20.81%** | — |
| LibriSpeech test-clean | 2.86% | 2.75% | **2.86%** | — |
| LibriSpeech + 10 dB noise | 3.61% | 3.79% | **3.89%** | — |
| LibriSpeech test-other (300) | 5.81% | 6.04% | **6.26%** | — |
| LibriSpeech + 0 dB noise | 12.42% | 12.61% | **12.42%** | — |

The causal AGC reaches the oracle, which set each clip's level using the
whole file in advance. Targets of −26, −23 and −20 dBFS are within 0.8
points of each other. A quiet threshold of −35 dBFS instead of −30 traded
AMI for Common Voice and was no better overall.

The remaining 0.2–0.45-point costs on test-other and 10 dB noise come from
clips whose first second is quiet. Each benchmark clip starts a fresh AGC, so
a short file pays this every time. A live microphone session pays it once.

## What did not help (measured, not adopted)

| Idea | Result |
| --- | --- |
| Wiener noise suppression before the model (gain floor −20 / −12 / −6 dB) | Worse everywhere: 0 dB 12.6% → 16.2 / 14.5 / 13.0%; clean 2.75% → ~3.0%. Its artifacts cost more than the noise it removes, the usual result with noise-trained models. |
| Blank penalty 0.5 / 1 / 2 at 0 dB | 12.61% → 12.59 / 12.87 / 14.23%: deletions become substitutions and insertions |
| Turning the energy gate off on AMI distant mic | No change: the gate was not dropping speech, the level was the problem |
| 1040 ms lookahead at 0 dB | 12.66%, no gain; it is better elsewhere (below) |

At 0 dB SNR the new live model still trails the old LibriSpeech Zipformer2
(12.4% vs 11.0%). That is synthetic noise at equal loudness to the speech; on
every real recording set the new model is far ahead. For heavy constant
noise, batch mode is the better tool (9.4% at 0 dB).

## Lookahead variants (`--live-model`)

| | 480ms (default) | 1040ms |
| --- | --- | --- |
| First words after | ~0.6 s | ~1.1 s |
| LibriSpeech test-clean | 2.86% | **2.72%** |
| AMI distant mic | 44.26% | **39.86%** |
| GigaSpeech | 13.87% | **12.18%*** |
| Speed per stream (6 concurrent) | ~9× | ~14× |

\* 1040 ms GigaSpeech and clean were measured before the boost-only AGC. At
normal levels the AGC leaves audio untouched.

`fetch_models.ps1 -Live1040` downloads the 1040 ms variant.

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
