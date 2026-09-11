# Caption models

One directory per model family. Everything here is speech in, text out; speech
synthesis lives in `../../audio_synth` and no caption target links against it.

| Directory | Engine | Needs | Status |
| --- | --- | --- | --- |
| [`naive/`](naive/) | Frame-level vowel lookup: LPC formants or FFT spectral shape, nearest neighbour in Bark space | FFTW, Eigen | Baseline; cannot transcribe words |
| [`toy_pruned_hmm/`](toy_pruned_hmm/) | MFCC → tied-state triphone GMM-HMM → lexicon → bigram Viterbi beam search | FFTW | Trained in-repo; see [RESULTS.md](RESULTS.md) |
| [`int8_zip/`](int8_zip/) | INT8 Zipformer transducer (streaming, pretrained) via ONNX Runtime | ONNX Runtime, kaldi-native-fbank | Best accuracy; see [STREAMING_ASR.md](int8_zip/STREAMING_ASR.md) |
| `optimized_gmm/` | — | — | Placeholder |
| `com_grammar/` | — | — | Placeholder |

## Measured results

| Model | Test WER | Speed | Corpus |
| --- | --- | --- | --- |
| `naive/` | not transcribable | — | — |
| `toy_pruned_hmm/` | 46.88% | 9.7x real time | LibriSpeech test-clean, 450 utterances / 9,650 words |
| `int8_zip/` | 4.15% | ~32x real time (RTF 0.031) | Same 450-utterance split |

Both rows use the same frozen split and scorer, so they are directly
comparable. `int8_zip/scripts/verify_streaming_asr_report.py` re-derives the
4.15% figure from the per-utterance records without trusting the summary.

## Layout conventions

Each family owns its headers, sources, weights and fetch scripts, so a family
can be read - or deleted - on its own:

```
int8_zip/
  STREAMING_ASR.md      setup, streaming API, latency behaviour
  include/captions/     public header (streaming_asr.hpp)
  streaming_asr.cpp     implementation
  models/               INT8 weights + tokens (compact, librispeech)
  scripts/              fetch, verify, and the caption.ps1 entry point
  artifacts/            measured runs and reports
  third_party/          kaldi-native-fbank, vendored python tooling
```

Shared DSP (framing, pre-emphasis, MFCC, FFT, WAV reading) stays one level up
in `../filter`, `../framing`, `../input`, `../power_spectrum`, `../spectral`
and `../formants`, because more than one family uses it.

## Building

`naive/` and `toy_pruned_hmm/` build by default from the repository root:

```bash
cmake -S . -B build && cmake --build build --config Release
```

`int8_zip/` additionally needs ONNX Runtime (shared with the TTS project in
`dependencies/onnxruntime`):

```bash
cmake -S . -B build -DCAPTIONS_ENABLE_STREAMING_ASR=ON
cmake --build build --config Release --target caption-streaming
```
