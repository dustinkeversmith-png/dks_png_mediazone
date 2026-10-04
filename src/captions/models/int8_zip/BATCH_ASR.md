# Batch pipeline: hours of audio in minutes on a CPU

`captions --input <media>` (or `CaptionEngine::transcribe_file`) runs the
Zipformer2 checkpoint over a whole recording at ~130x real time on a 6-core
desktop CPU. One hour of speech takes about 30 seconds and three hours about
90 seconds, at 3.41% WER on LibriSpeech test.

## Measured (Ryzen 5 5600X, 6 cores / 12 threads)

| Test | WER | Wall time | Speed |
| --- | --- | --- | --- |
| Frozen LibriSpeech test split, 450 utterances / 9,650 words | 3.41% (S 266 D 25 I 38) | 30 s | 120–135x |
| Same split as one 1.08 h recording, auto-segmented (`--longform`) | 3.42% | 30 s | 126–130x |

Wall time covers segmentation, filterbanks, all three graphs and beam search.
Model load (~2 s) is excluded. Run-to-run variation on a desktop is about ±10%.
Decoding an hour of AAC audio through ffmpeg takes under a second.

## How it works

1. **Decode.** PCM WAV is read natively. Every other container or codec goes
   through the native `ffmpeg.exe` on a pipe, as 16 kHz mono float.
2. **Segment** (`segment_audio`).
   - An RMS gate at 0.0003 marks speech. Quiet stretches longer than 1 s are
     dropped; shorter pauses stay inside a segment.
   - 200 ms of padding is kept on each side.
   - Regions longer than 20 s are cut at the quietest 200 ms window between
     8 s and 20 s, so cuts fall in pauses and not inside words.
3. **Batch.**
   - Every encoder cache tensor of the Zipformer2 export has exactly one
     dynamic axis, the batch axis. Segments are sorted by length and stacked
     16 per encoder call, so the INT8 matrix multiplies run on 16x more rows.
   - Rows whose segment has ended ride along on zero features, and their output
     is discarded. Length sorting keeps that waste small.
   - The batch count is rounded up to a multiple of the worker count, so no
     worker idles at the end.
4. **Parallelize.** Six workers with two intra-op threads each pull batches
   from a shared queue. Each worker owns private ONNX sessions, and workers
   load lazily, so a short clip only pays for the workers it uses.
5. **Search.** Modified beam search, beam 4: at most one symbol per frame,
   top-4 (hypothesis, token) pairs survive, and equal sequences merge by
   log-add. One joiner call per frame covers every hypothesis of every active
   segment, and one decoder call covers every hypothesis that grew.
6. **Timestamps.** Each word's time comes from the output frame where its first
   token was emitted (40 ms frames). Emission can lag the spoken word slightly.

The encoder accounts for about 85% of the time. Throughput scales roughly with
physical cores.

## Tuning

These defaults won the sweeps:

| Setting | Finding |
| --- | --- |
| Batch size | One worker speeds up from batch 1 to batch 16, then plateaus; 32 and 64 are no faster |
| Layout | 6 workers × 2 threads is best on 6 cores / 12 threads; 12 × 1 is close, 3 × 4 and 2 × 6 are slower |
| Thread spinning | Slower: workers already saturate the cores |
| Beam | 2 / 4 / 8: 4 is the knee; 8 costs search time for no accuracy gain |
| Segment length | 10–60 s caps move WER by less than 0.2 points; 20 s is the default |

## Captions

- Cues are at most two lines of 42 characters (`--line-chars`) and at most 6 s.
- A pause over 0.8 s starts a new cue, and its first word is capitalized.
- Text is in sentence case with "I" restored; `--upper` keeps the model's raw
  output.
- Each cue is held for 0.4 s after its last word, never overlapping the next
  cue.
- `json` has per-word start and end times plus the cues. `txt` has one cue per
  line.

## Correctness checks (`captions --self-test`, ctest `captions-self-test`)

- Live engine contracts:
  - Silence is gated and never decoded.
  - `accept()` after `finish()` is rejected.
  - 137-, 1600- and 2560-sample packets give identical transcripts.
  - `reset()` clears state.
- Batch equals live exactly with greedy search at batch 1, on 10 dev clips.
- Beam-4 end-to-end run:
  - Cues are non-empty and never overlap.
  - SRT and VTT layouts are correct.
  - Silence yields no segments.

Batching can move individual words slightly (±0.05 WER points). The export
uses dynamic INT8 quantization, which computes its activation scale over the
whole batch tensor.

## Limits

- Audio is held in memory: 16 kHz float is about 230 MB per hour.
- The gate is energy-only. Music and steady background noise are decoded like
  speech, which costs time and can produce spurious words.
- Accuracy figures are for read English speech. Spontaneous, noisy or accented
  speech scores worse.
- The model has no punctuation. Sentence case is a heuristic based on pauses.
