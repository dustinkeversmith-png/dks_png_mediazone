# Batch pipeline design

`captions --input <media>` (`CaptionEngine::transcribe_file`) runs these
stages. Accuracy tables are in [README.md](README.md).

1. **Decode.** PCM WAV is read natively. Any other container or codec goes
   through the native `ffmpeg.exe` on a pipe as 16 kHz mono float. One hour of
   AAC audio decodes in under a second.
2. **VAD** (`vad.cpp`). Silero VAD v5 gives a speech probability per 32 ms.
   The model is recurrent, so the recording is cut into up to 256 stripes of at
   least 2 s, each warmed up on the 0.5 s before it, and all stripes advance
   together along the batch axis. That is a few hundred calls per hour
   instead of ~112,000.
3. **Segment** (`segment_audio`).
   - Speech opens at probability ≥ 0.02 and closes below 0.01.
   - Pauses shorter than 1 s stay inside a segment, with 200 ms of padding on
     each side.
   - Regions over 20 s are cut at the least speech-like 200 ms window between
     8 s and 20 s.
   - For short recordings, the cap shrinks so every worker gets about two
     segments, with a 4 s minimum.
4. **Decode segments** (`batch_asr.cpp`).
   - One shared copy of the model serves one worker per hardware thread; each
     worker runs whole encoder calls on its own thread. ONNX Runtime sessions
     are safe to call concurrently.
   - Segments are sorted longest first.
   - NeMo FastConformer (Parakeet):
     - Features are 80 Slaney mel bins with 0.97 pre-emphasis and per-feature
       normalization.
     - The encoder sees each whole segment in one call. Batch size 1 is best,
       since padding only costs time and memory and changes the INT8 dynamic
       quantization scales.
     - Greedy TDT decoding: each joiner step predicts a token and how many
       80 ms frames to skip. The joiner is batched across active segments; the
       LSTM prediction network runs per segment, because its exported cell
       state has batch 1.
     - CTC exports (`model.int8.onnx`) are also supported: 172× real time,
       3.07% WER on LibriSpeech test-clean.
   - Streaming Zipformer2 (live model; also usable for batch with `--models`):
     segments advance in lockstep, 16 per encoder call, with batched modified
     beam search.
5. **Filter** (`keep_segment`): drop a segment only when its mean word
   confidence is < 0.6 AND its mean VAD probability is < 0.3.
6. **Cues** (`build_cues`).
   - At most two lines of 42 characters and at most 6 s per cue.
   - A pause over 0.8 s starts a new cue.
   - Each cue is held for 0.4 s after its last word, never into the next cue.
   - Model casing and punctuation are kept as-is; uppercase-only models are
     sentence-cased.
7. **Load time.** The first session load saves ONNX Runtime's optimized
   graph in `<model dir>/.ort-cache/`, keyed by runtime version, model size
   and modification time, and CPU model. Later loads skip optimization, so
   setup for a short file drops from ~1.3 s to ~0.8 s. Deleting the folder is
   always safe.

## Measured choices (Ryzen 5 5600X, LibriSpeech test-clean)

| Choice | Result |
| --- | --- |
| Parakeet batch size 1 / 2 / 4 / 8 | 148 / 131 / 125 / 98× real time; WER best at 1 |
| Shared model, 12 workers × 1 thread vs private 6 × 2 | Same or faster; one load instead of six |
| Parakeet TDT vs CTC head | 2.60% vs 3.07% WER; 170× vs 172× |
| VAD threshold 0.5 (Silero default) | Dropped about 70% of the words of a music-backed vocal track; 0.02 keeps them |
| Confidence filter alone (no VAD condition) | Also deleted real one-word utterances ("Venice."): rejected |

## Limits

- Audio is held in memory: about 230 MB per hour at 16 kHz float.
- English only. Accented and spontaneous speech still has 10–18% WER on the
  hardest sets (Common Voice, Earnings-22, AMI).
- Live mode keeps the LibriSpeech Zipformer2, so it is not as robust as batch
  mode.
- Word timestamps have 80 ms resolution (Parakeet) and come from emission
  frames.
