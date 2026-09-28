# Batch captions: hours of audio in seconds

`caption-batch` captions recorded audio and video at ~270x real time on a
6-core desktop CPU: one hour of speech in about 14 seconds, three hours in
about 45. It uses the same INT8 Zipformer graphs, ONNX Runtime and
kaldi-native-fbank as `caption-streaming`. There is no GPU, CUDA, Python,
or new model, and nothing leaves the machine.

```powershell
cmake --build build --config Release --target caption-batch
./build/bin/Release/caption-batch.exe --input lecture.mp4                    # -> lecture.srt, lecture.txt
./build/bin/Release/caption-batch.exe --input talk.mkv --formats srt,vtt,json --out captions/talk
# Most accurate (larger Zipformer2 checkpoint, ~2x slower):
./build/bin/Release/caption-batch.exe --input talk.wav --models src/captions/models/int8_zip/models/librispeech
```

WAV files are read directly. Any other container or codec is decoded to 16 kHz
mono by the repository's native `dependencies/ffmpeg/bin/ffmpeg.exe` through
a pipe (`--ffmpeg <path>` to override). The audio of a one-hour AAC file decodes
in under a second.

## Measured (Ryzen 5 5600X, 6 cores / 12 threads)

Frozen LibriSpeech test split (450 utterances, 9,650 words, the same split
and S/D/I scorer as every other row in `../RESULTS.md`):

| Engine | Search | Test WER | Speed |
| --- | --- | --- | --- |
| `caption-streaming` (before) | greedy | 4.15% | 32x real time |
| `caption-batch`, compact | greedy | 4.18% | 279x |
| `caption-batch`, compact (default) | beam 4 | **3.96%** | **263x** |
| `caption-batch`, zipformer2 (`models/librispeech`) | beam 4 | **3.41%** | 135x |

Long-form: the 450 test utterances joined into one recording with 0.5 s gaps,
segmented automatically and scored as a single word sequence:

| Input | Model | WER | Wall time | Speed |
| --- | --- | --- | --- | --- |
| 1.08 h WAV | compact, beam 4 | 3.96% | 14.6 s | 266x |
| 3.24 h WAV (`--repeat 3`) | compact, beam 4 | 3.91% | 43.6 s | 268x |
| 1.08 h WAV | zipformer2, beam 4 | 3.42% | 30.8 s | 126x |
| 1.02 h AAC `.m4a`, end to end incl. model load + ffmpeg | compact, beam 4 | — | 15.2 s | 242x |

Wall time covers segmentation, filterbanks, all three graphs and search.
Model load (~0.6 s compact, ~2 s zipformer2) is reported separately except in
the end-to-end row. Run-to-run variation on a desktop is roughly ±10%.

## Why it is 8x faster than streaming with the same model

1. **Lockstep batching.** Every encoder cache tensor has exactly one dynamic
   axis, the batch axis. The engine sorts segments by length and stacks up to
   16 of them per encoder call. The INT8 matrix multiplies then run on 16x
   more rows, so a single worker goes from 38x to 90x real time. Rows whose
   segment has ended ride along on zero features and their output is discarded;
   length sorting keeps that waste small.
2. **Worker threads with private sessions.** Six workers (2 intra-op threads
   each) pull batches from a shared queue, which makes 3x more. Each worker
   owns its sessions, so the ONNX thread pools don't contend. The batch count
   is rounded up to a multiple of the worker count so no worker idles at the
   end.
3. **Batched search.** Joiner and decoder calls cover every active row (and
   every beam hypothesis) at once. Search is ~2% of the time greedy and ~8%
   with beam 4. The encoder is the remaining ~85%.

Sweeps behind the defaults (1.08 h long-form, compact):

| Layout | Speed |
| --- | --- |
| 1 worker x 2 threads, batch 1 / 4 / 8 / 16 / 32 / 64 | 38x / 72x / 81x / 90x / 88x / 86x |
| 6 x 2, batch 16 (**default**) | 266–297x |
| 12 x 1 | 286x |
| 3 x 4 / 2 x 6 | 230x / 141x |
| ONNX thread spinning on (`--spin`) | slower (237–256x) |

Beam width 2 / 4 / 8 gives 4.06% / 3.96% / 3.97% WER. Maximum segment length
of 10 / 20 / 30 / 60 s moves WER by less than 0.2 points, which is within noise.

## Long-form segmentation

`segment_audio()` uses the streaming engine's RMS gate (0.0003):

- Quiet stretches longer than 1 s are dropped.
- Shorter pauses stay inside a segment.
- 200 ms of padding is kept on each side.
- Regions longer than 20 s are cut at the quietest 200 ms window between
  8 s and 20 s, so cuts fall in pauses rather than inside words.

Each segment starts from zero caches, like a gated segment in the streaming
engine, and ends with the same right-context flush.

## Captions

The model emits uppercase words with no punctuation. Word timestamps come
from the transducer's emission frames at 40 ms resolution. Emission can lag
the start of the word a little. Cues are built as follows:

- at most two lines of 42 characters (`--line-chars`) and at most 6 s;
- a new cue after any pause over 0.8 s;
- sentence case with "I" restored (`--upper` keeps raw output);
- each cue is held for 0.4 s after its last word, never overlapping the next.

`--formats` chooses any of `srt,vtt,txt,json`. JSON has per-word start and
end times.

## Correctness checks

- `ctest -R batch-asr-parity`: at batch 1 with greedy search, the batch engine
  must reproduce `caption-streaming --no-gate` transcripts exactly (10 dev
  clips, compact checkpoint in ctest; zipformer2 also verified by hand with
  `--parity <dir> --models models/librispeech`).
- Batching changes results slightly (±0.05 WER points). The graphs use
  dynamic INT8 quantization, whose activation scale is computed over the whole
  batch tensor. This is a property of the quantized export, not a decoding
  bug.

## Limits

- CPU-bound. The encoder is about 85% of the time, and ~270x is close to what
  this model reaches on 6 cores. Scaling is roughly linear in physical cores.
- Audio is held in memory: 16 kHz float is about 230 MB per hour.
- The gate is energy-only. Music and background noise are transcribed like
  speech, which costs time and can produce spurious words.
- Accuracy numbers are for read English speech (LibriSpeech). Spontaneous,
  noisy or accented speech will score worse with both checkpoints.
