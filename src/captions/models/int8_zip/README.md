# `int8_zip` — captions on the CPU

Speech to text on the CPU, offline, with INT8 ONNX models. No Python, GPU or
network access is needed at run time. Both modes use NVIDIA FastConformer
models trained on tens of thousands of hours of multi-domain English.

| Mode | Model | Use |
| --- | --- | --- |
| batch (default) | **Parakeet TDT 110M** (~36k h, punctuation and casing), with **Silero VAD v5** for segmentation and hallucination filtering | Files, video, hours of audio |
| live | **FastConformer hybrid large, cache-aware streaming** (480 ms lookahead, 560 ms chunks), greedy RNN-T | Microphone; first words after ~0.6 s |

Both modes accept a **domain-term list** (`--hotwords`) that biases decoding
toward names and jargon.

## Accuracy (Ryzen 5 5600X, 6 cores / 12 threads)

WER after the shared normalizer in `benchmarks/scoring.hpp`. "Before" means
the previous engines, the LibriSpeech-trained Zipformer2 in both modes.

| Test set | Batch before | **Batch now** | Live before | **Live now** |
| --- | --- | --- | --- | --- |
| LibriSpeech test-clean (450 utts) | 3.41% | **2.68%** | 3.54% | **2.86%** |
| + DEMAND noise, 10 dB SNR | 4.29% | **3.49%** | — | **3.61%** |
| + DEMAND noise, 5 dB SNR | 5.96% | **4.62%** | 5.92% | **5.67%** |
| + DEMAND noise, 0 dB SNR | 11.20% | **9.39%** | **11.04%** | 12.42% |
| **AMI distant microphone** (819 utts, 100 MB of real far-field room noise and reverb) | 86.27% | **33.60%** | 86.86% | **62.06%** |
| AMI headset mics (300) | 53.30% | **14.57%** | 54.44% | **29.29%** |
| GigaSpeech, podcasts / YouTube (300) | 22.88% | **10.99%** | 23.19% | **13.65%** |
| Earnings-22, company calls (274) | 50.15% | **14.70%** | 50.56% | **23.48%** |
| Common Voice, accented read speech (300) | 36.65% | **17.45%** | 37.65% | **21.39%** |
| VoxPopuli, parliament (190) | 21.61% | **8.79%** | 21.51% | **8.89%** |
| LibriSpeech test-other (300) | 8.09% | **5.10%** | 7.97% | **5.81%** |
| Pure noise, 15 min (DEMAND living room + park) | 36 words invented* | **0 words** | | |

\* Batch, measured without the VAD and filter. On the speech-only sets above,
the VAD and filter change WER by less than 0.2 points; their job is silence,
music and noise. One row is missing: the old live engine failed on the 10 dB
run (see Known issues). The new live model is worse than the old one only at
0 dB, where it deletes more words.

### Domain terms (`--hotwords`)

The term list is 58 names, products and acronyms from the Earnings-22
references ("Cyberpunk", "EBITDA", "KGHM"…). Recall counts how many of the 91
occurrences of those terms come out right.

| Earnings-22 | Term recall | WER |
| --- | --- | --- |
| batch, no list | 53.8% | 14.70% |
| batch, list (boost 2, start 0.25) | **62.6%** | **14.64%** |
| live, no list | 34.1% | 23.48% |
| live, list | **42.9%** | **23.37%** |
| batch, same list on unrelated audio (LibriSpeech / GigaSpeech) | — | +0.06 / +0.03 points |

Example: "the team was working on updating Cyberpack" becomes "…updating
Cyberpunk".

### Speed

| | |
| --- | --- |
| 1.08 h recording, batch (`--longform`) | 27.5 s, **141× real time** |
| 3.6 min music-backed vocals, end to end incl. model load | **2.4 s** (was 6.4 s) |
| Live, one stream on 2 threads | ~17× real time |

## Setup and build

```powershell
./src/captions/models/int8_zip/scripts/fetch_models.ps1    # Parakeet, live FastConformer, Silero VAD (pinned, SHA-256 checked)
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
& $C --input call.m4a --hotwords terms.txt          # favour your names and jargon
& $C --mic --hotword "Kubernetes,PostgreSQL"        # same, inline, live
& $C --help                                         # every option
```

A hotwords file has one term per line, with an optional per-term boost
(`ACME Corp :3`) and `#` comments. Terms are spelled with the model's own
subword pieces. A term the vocabulary can't spell is reported and skipped.

| Option | Default | Meaning |
| --- | --- | --- |
| `--input <path>` | — | Any audio or video ffmpeg reads; PCM WAV is read directly |
| `--format <list>` | `srt` | `srt`, `vtt`, `txt`, `json` (per-word times and confidence); comma-separated for several |
| `--out <path>` | input path | Output base path; `-` = stdout |
| `--mode batch\|live`, `--mic` | batch | Live = streaming engine |
| `--hotwords <file>`, `--hotword "a,b"` | none | Domain terms to favour (batch and live) |
| `--hotword-boost B`, `--hotword-start F` | 2, 0.25 | Score bonus per matching token; fraction of it given to a term's first token |
| `--vad-threshold P` | 0.02 | Speech probability that opens a segment. Kept low on purpose: Silero scores sung or music-backed vocals at 0.05–0.3 |
| `--min-confidence C`, `--speech-floor P` | 0.6, 0.3 | Drop a segment only if it is BOTH unsure and unlike speech (see below) |
| `--no-vad` | off | Use the RMS energy gate instead |
| `--workers N`, `--threads T` | hardware threads, 1 | Batch parallelism: one shared model copy, one encoder call per worker thread |
| `--models <dir>` | batch `models/parakeet-tdt-110m`, live `models/nemo-streaming-480ms` | Any NeMo transducer/CTC or Zipformer2 export |
| `--verbose` | off | Time breakdown |

## Robustness: what does what

- **Parakeet TDT 110M** gives the large out-of-domain gains: meetings,
  calls, podcasts, accents and noise. It was trained on ~36k hours across
  domains, against 960 hours of audiobooks for the old model. It is also
  faster, because its encoder runs at 80 ms per frame over a whole segment at
  once.
- **Silero VAD** segments by speech likelihood rather than loudness, so
  steady noise no longer becomes one endless "speech" region, and cuts land
  where speech is least likely. An hour is scored in a few hundred batched
  calls: the recording is cut into up to 256 stripes, each with 0.5 s of
  warm-up.
- **Hallucination filter** (`keep_segment`). Each word carries a confidence:
  the lowest token posterior inside the word. A segment is dropped only when
  its mean confidence is below 0.6 AND its mean VAD probability is below 0.3.
  Here is why both conditions are needed:

  | Case | Confidence | Speech probability | Kept? |
  | --- | --- | --- | --- |
  | Words invented from noise | low | low | no |
  | Isolated real words ("Venice.") | low | high | yes |
  | Sung vocals | high | low | yes |

- **Live FastConformer** is NVIDIA's cache-aware streaming model.
  - Each 560 ms chunk is encoded once, and attention and convolution caches
    carry context forward. Nothing is recomputed.
  - Unlike the batch model, it uses unnormalized features, because
    per-utterance statistics aren't available live.
  - The first window of each segment starts with silence where the model's
    left-context frames would be.
- **Hotwords** (`hotwords.hpp`).
  - Each term is spelled with the model's own subword pieces and stored in a
    prefix tree.
  - At every decoding step, a token that continues a partial match gets the
    full boost added to its logit, and a token that starts a term gets 25% of
    it. Starting tokens are often common pieces, and boosting them fully
    caused insertions (boost 4 at full strength: WER 14.70% → 16.25%).
  - Any other token ends the match at no cost.
  - The confidence a word reports is the model's unbiased probability, so the
    hallucination filter isn't fooled by the boost.
  - Supported by greedy and TDT decoding, which covers both defaults. The
    optional Zipformer2 beam search ignores the list.
- **Scoring normalizer** (`benchmarks/scoring.hpp`), so conventions don't
  count as errors:
  - Numbers become digits, so "nineteen seventy five", "one thousand nine
    hundred and seventy five" and "1975" all match.
  - "Mr" and "mister" match, "$5" matches "five dollars", and "20%" matches
    "twenty percent".
  - Hyphens are split, fillers are dropped, and casing and punctuation are
    ignored.
  - The self-test checks these equivalences.

## Evaluation data

`scripts/fetch_eval_data.ps1` (curl + ffmpeg) builds the test sets:

- `data/audio/eval/<set>/`: about 300 clips of at most 100 MB per set, from
  the Open ASR leaderboard sets and AMI. `ami_sdm` has 819 clips (100 MB) from
  the same meetings recorded by one distant table microphone, with real room
  noise, reverberation and overlapping talkers.
- `data/audio/noise/`: one channel from each of three DEMAND environments.

```powershell
& $C --eval data/audio/eval/gigaspeech
& $C --benchmark data/audio/librispeech --noise data/audio/noise --snr 5
& $C --eval data/audio/eval/ami_sdm --mode live --streams 6
& $C --eval data/audio/eval/earnings22 --hotwords data/audio/eval/earnings22.hotwords.txt   # adds term recall
```

## C++ API

```cpp
#include <captions/caption_engine.hpp>

captions::CaptionEngine engine;                        // Parakeet + VAD + filter, all cores
auto segments = engine.transcribe_file("talk.mp4");   // or transcribe_pcm(samples) / transcribe_batch(many)
auto cues = captions::build_cues(segments);           // <= 2x42 chars, <= 6 s, model casing kept
std::string srt = captions::render(captions::CaptionFormat::srt, segments, cues);
```

## Contents

| Path | What it is |
| --- | --- |
| `include/captions/caption_engine.hpp`, `caption_engine.cpp` | `CaptionEngine`, model locations, hallucination filter, cues, SRT/VTT/TXT/JSON |
| `include/captions/batch_asr.hpp`, `batch_asr.cpp` | Segmenter; worker pool; NeMo TDT/CTC and Zipformer2 decoders |
| `include/captions/vad.hpp`, `vad.cpp` | Striped, batched Silero VAD |
| `include/captions/streaming_asr.hpp`, `streaming_asr.cpp` | Live decoder: NeMo cache-aware FastConformer (default) or Zipformer2 |
| `include/captions/hotwords.hpp`, `hotwords.cpp` | Domain-term prefix tree and logit biasing |
| `ort_graph.hpp`, `ort_session_options.hpp` | ONNX sessions, optimized-graph cache (`<model>/.ort-cache`), NeMo/Kaldi fbank settings |
| `scripts/` | `fetch_models.ps1` (all models), `fetch_eval_data.ps1` (test sets + noise), `caption.ps1`; `fetch_streaming_asr.py` only for the optional Zipformer2 |

## Known issues

- At 0 dB SNR, live mode deletes more words than the old Zipformer2 did
  (12.4% vs 11.0% WER). Batch mode is better at every noise level.
- During the evaluation, the old Zipformer2 live engine failed once with a
  cache-shape error while six engines loaded at once. It did not reproduce in
  ten cold-cache reruns. Cache creation is now serialized within a process,
  and the error names the tensor and shape if it ever recurs.
| `models/` | Weights (fetched, not versioned) |

Design notes: [BATCH_ASR.md](BATCH_ASR.md) covers the batch pipeline, and
[STREAMING_ASR.md](STREAMING_ASR.md) covers live mode.
