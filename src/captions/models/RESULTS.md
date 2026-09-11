> **2026-09-10 update:** The repaired non-neural system measures **39.84% dev WER** (40 utterances) and **46.88% held-out WER** (450 utterances, 9,650 words), at **9.69x real time**. The freshly measured pre-retraining test baseline with the silence-history fix was 63.30%. The requested 6�7% target remains unmet.
>
> See [the complete repair experiment](int8_zip/artifacts/ASR_REPAIR_RESULTS.md) for configurations, counts, ablations, provenance, and reproduction commands. The tested model is in `data/models`; its manifest is `asr_run.json`. The older results below are retained as historical measurements and do not describe the current defaults.

# Networkless Captioning — Implementation & Results

Implementation of the architecture specified in [README.md](README.md), evaluated on the
three downloaded tiers. Everything is deterministic DSP + statistics: no neural network,
no network access at run time, no GPU. Single CPU thread throughout.

Reproduce with:

```bash
python scripts/download_datasets.py          # digits / TIMIT / LibriSpeech
cmake --build build --target train_models benchmark_models caption --config Release
./build/bin/Release/train_models.exe --mixtures 8
./build/bin/Release/benchmark_models.exe --acoustic-scale 0.2 --word-beam 6 \
    --word-penalty -6 --max-active 12000
```

## 1. What was built

| File | Role |
|------|------|
| `mfcc.hpp` | 26-channel log-Mel filterbank → 13 MFCC + Δ + ΔΔ (39-dim), per-utterance CMVN |
| `dtw.hpp` | Sakoe-Chiba banded DTW, early abandoning, LB_Kim/LB_Keogh bounds, 1-NN/k-NN recogniser, endpointing |
| `phone_set.hpp` | TIMIT-61 → ARPAbet-39 + SIL folding (the bridge between TIMIT training and CMUDict decoding) |
| `acoustic_model.hpp` | Monophone 3-state HMM, diagonal-covariance GMM (1–16 components), EM training, binary I/O |
| `lexicon.hpp` | CMUDict loader → phone-id sequences |
| `ngram_lm.hpp` | Word/phone bigram with absolute discounting + backoff, history-indexed for fast successor scans |
| `viterbi_decoder.hpp` | Single-pass token-passing beam search with word-link backtrace |
| `scoring.hpp` | Levenshtein S/D/I counting, WER/PER, real-time-factor timing |
| `train_models.cpp` | Trains the acoustic model and both LMs |
| `benchmark_models.cpp` | Three-tier evaluation incl. the two legacy front-ends |
| `caption.cpp` | `caption file.wav` → text |

Training data is kept strictly disjoint from evaluation data:

* **Acoustic model** — TIMIT, 1236 utterances / 62.8 min / 377k frames from 309 speakers.
  The 54 held-out speakers never appear in training.
* **Language model** — 28,539 LibriSpeech **train-clean-100** transcripts (~1M words).
  Speaker- and book-disjoint from test-clean. Only the `text` column was downloaded
  (parquet column projection), so this cost ~5 MB rather than 30 GB of audio.
* **Lexicon** — CMUDict, restricted to the 20k LM vocabulary → 20,356 pronunciations.

Total model footprint: 0.29 MB acoustic model + 3.13 MB word bigram + 0.01 MB phone bigram,
plus the 3.45 MB CMUDict text file. Training takes **2.1 s** end to end.

## 2. Results

### Tier 1 — isolated digits (Speech Commands, 500 templates, 300 test clips)

| System | Accuracy | Speed | Cost per clip |
|--------|----------|-------|---------------|
| **MFCC + banded DTW (1-NN)** | **85.3 %** | 54× real time | 18.4 ms vs 500 templates |
| the same, with energy endpointing | 84.0 % | 71× real time | 14.1 ms vs 500 templates |
| LPC formants + Bark lookup (legacy) | 41.3 % | 296× real time | — |
| FFT envelope + Bark lookup (legacy) | 22.0 % | 15× real time | — |

The legacy rows are the existing `lpc_method`/`fourier_method` front-ends scored on the
same split, with their collapsed vowel strings classified by nearest-neighbour edit
distance — the most favourable classifier those features admit. Replacing formant
coordinates with a dense filterbank and replacing run-length collapse with time warping
is worth **+44.0 points** of accuracy.

Endpointing is a speed/accuracy trade, not a free win: trimming leading and trailing
silence makes each comparison 1.3× faster (the early-abandon rate rises from 91 % to 98 %)
but costs 1.3 points, because the silence margins carry usable onset/offset information.

Per-digit accuracy is even (22–28 of 30 correct for every digit); no digit collapses.

### Tier 2 — phone recognition (TIMIT, 216 utterances from 54 unseen speakers)

| Metric | Value |
|--------|-------|
| Phone error rate | **39.3 %** (S 1595 / D 778 / I 108 over 6321 phones) |
| Frame accuracy (40-way) | 49.9 % (32,973 / 66,071 frames) |
| Speed | 425× real time |

### Tier 3 — continuous captioning (LibriSpeech test-clean, 450 utterances, 61.2 min)

| Metric | Value |
|--------|-------|
| Word error rate | **77.2 %** (S 5567 / D 1719 / I 164 over 9650 words) |
| — dev slice (40 utterances, used for all tuning) | 70.2 % |
| OOV rate | 4.1 % of reference tokens outside the 20k vocabulary |
| Decode speed | 15.1× real time (RTF 0.066) |
| Front-end speed | 6595× real time (RTF 0.00015) |
| Search network | 377,295 HMM states, built in 0.01 s |

Legacy baselines cannot be scored here at all: they emit a frame-level vowel string
(`IY EH AA …`), not words, so their WER is 100 % by construction. That gap *is* the
result — a lexicon and an LM are what convert phone hypotheses into text.

Sample output (test utterance 0):

```
reference : THE ANALYSIS OF KNOWLEDGE WILL OCCUPY US UNTIL THE END OF THE THIRTEENTH LECTURE
hypothesis: THE NOSES OF KNOWLEDGE LOCK PRESENTLY AND ENOUGH FOR HE PLEASURE AND AS THE
```

The decoder tracks the utterance (`THE … OF KNOWLEDGE …`) but substitutes acoustically
similar words. Substitutions are 74.7 % of all errors, deletions 23.1 %, insertions 2.2 % —
the engine is hearing roughly the right sound shapes and picking the wrong word, which is
an acoustic-resolution problem rather than a search or insertion-penalty problem.

## 3. What each design decision bought

Measured on the 40-utterance dev slice unless noted.

| Change | Effect |
|--------|--------|
| Single Gaussian → 8-component GMM | TIMIT PER 51.3 % → 41.8 % (30 utts); dev WER 84.5 % → 80.1 % (10 utts) |
| 8 → 16 components | PER 41.8 % → 42.1 %; dev WER 80.1 % → 83.3 %: over-fits 1 h of training data |
| Fixing silence as a trap state¹ | dev WER 78.7 % → 70.2 %, test 88.1 % → 77.2 % |
| Word-entry beam (separate from state beam) | word entries per utterance 16.1 M → 7.9 M, no WER change |
| Endpointing digit clips | 1.3× faster per clip, but −1.3 points of accuracy (85.3 % → 84.0 %) |
| 20 → 50 templates per digit | accuracy 74.0 % → 84.0 %, cost 6.5 ms → 14.1 ms per clip (endpointed) |

¹ The optional inter-word silence model originally had no exit path back into the word
network, so any token that entered a pause was lost. Silence now re-enters the entry pool
with an `<unk>` history, which backs off to unigram — the correct behaviour after a pause.

### Beam width vs accuracy vs speed (Tier 3, dev)

| max-active | dev WER | Speed | State visits / utt |
|-----------:|--------:|------:|-------------------:|
| 2,000 | 73.0 % | 26.5× RT | 10.8 M |
| 4,000 | 70.6 % | 23.1× RT | 12.9 M |
| **12,000** | **70.2 %** | 15.1× RT | 20.9 M |
| 24,000 | 70.8 % | 10.0× RT | 32.9 M |

The search saturates at ~12k active states: beyond that, extra compute buys nothing, and
the remaining errors are model errors, not search errors. Going the other way, 2,000
active states costs 2.8 points of WER for a 1.8× speedup.

### Where the time goes

The front-end is free (6595× real time); decoding is 99.98 % of the cost. Within decoding,
acoustic scoring is bounded — 120 states × 8 components × 39 dims per frame regardless of
beam width, because all state likelihoods are computed once per frame into a flat table.
The cost is token propagation and word entry, which is why the beam controls speed almost
linearly.

### DTW pruning: a negative result worth recording

`dtw.hpp` implements two admissible lower bounds. On this data **both prune 0 %** of
templates: after CMVN, 39-dimensional frames spread out enough that a ±12-frame envelope
is almost never violated, so LB_Keogh — the standard tool for 1-D time series — is
useless here and costs more than it saves (14.1 ms → 22.8 ms per clip when enabled). It is
left in, off by default, with the reasoning documented.

What actually works is **early abandoning inside the alignment**: 91.1 % of comparisons are
proven hopeless mid-way and dropped, because templates are visited in LB_Kim order so a
tight incumbent appears early.

## 4. Honest assessment vs the baselines named in README.md

| System | WER on read speech | Notes |
|--------|-------------------|-------|
| This engine | 77.2 % | monophone GMM-HMM, 1 h training audio, 20k bigram |
| PocketSphinx / Kaldi GMM-HMM (published) | 15–25 % | triphone GMMs, ~100 h training, 3-gram LM |
| whisper.cpp `tiny.en` / `base.en` (published) | 5–8 % | neural, ~680 kh training |

Neither external baseline was run here — PocketSphinx and whisper.cpp are not installed in
this project, and the figures above are published numbers, not measurements. Treat them as
context, not as a controlled comparison.

The gap is dominated by **acoustic modelling**, not by search or by the language model:

1. **Monophone, not triphone.** `/t/` in "stop" and in "water" share one Gaussian mixture.
   Context-dependent triphones with decision-tree state tying are the single largest known
   win (typically 30–40 % relative WER reduction) and need no new data.
2. **62 minutes of training audio, from a different corpus.** TIMIT is read sentences
   recorded in 1986 at close mic; LibriSpeech is modern audiobook audio. CMVN removes the
   channel offset but not the mismatch. The model has never heard a LibriSpeech speaker.
3. **Uniform state alignment.** States are assigned by splitting each phone segment in
   three equal parts. A few passes of Viterbi re-alignment (embedded training) would sharpen
   boundaries without any new labels.
4. **Bigram, not trigram.** Cheapest remaining win: the corpus is already downloaded.

In exchange, the properties the architecture was chosen for do hold: the whole system is
3.2 MB of model data, trains in 2.1 seconds, runs at 15× real time on one thread with no
network and no accelerator, and every output is deterministic and traceable to a Gaussian,
a lexicon entry and an n-gram.
