# Word articulation experiment and nonverbal vocal events

The separate C++ `articulation-lab` executable probes how far the existing trained
explicit voice can be directed through deterministic controls. Its implementation
lives in `experiments/articulation/`; production source and model weights remain
unchanged. [Build instructions, commands and formulas](../experiments/articulation/README.md)
describe individual-word and individual-phone editing.

## What was tested locally

On 2026-10-08, two speakers and two English sentences were rendered through 21
cases each: 42 sweep WAVs, plus one multiword edit and four interjection probes.
All outputs are native 24 kHz mono PCM16. All 42 sweep cases had zero clipped
samples before PCM conversion. This is a bounded exploration of the exposed
controls, not coverage of every possible voice expression.

Speaker 0: "We synthesize a clear acoustic voice.", target word "clear".

| Case | Total duration | Target word frames | Observation |
| --- | ---: | ---: | --- |
| Baseline | 2.175 s | 21 | Reference unchanged prediction |
| Word short | 2.0625 s | 12 | Selected word duration scaled by 0.6 |
| Word long | 2.3875 s | 38 | Selected word duration scaled by 1.8 |
| Consonants short | 1.6375 s | See diagnostics | Consonants throughout sentence scaled by 0.6 |
| Consonants long | 3.2375 s | See diagnostics | Consonants throughout sentence scaled by 1.8 |
| Pitch high | 2.175 s | 21 | Requested word mean 261 Hz; approximate voiced median 267 Hz |
| Pitch low | 2.175 s | 21 | Requested word mean 131 Hz; approximate voiced median 86 Hz |

The baseline requested word mean was 186 Hz and its approximate measured voiced
median was 187 Hz. Requested means include the complete word's conditioning
frames; measured medians use only sufficiently periodic audio windows. These
are different statistics, and autocorrelation can make octave errors. The low
pitch result demonstrates why a slider is not a guarantee of exact output pitch.

An existing local Whisper small.en checkpoint recognized all six words in 20 of
the 21 speaker-0 cases, ignoring punctuation/case and equating synthesize/synthesise.
The extreme low case (word duration 0.35, pitch 0.45, energy 0.15) omitted "clear"
in its transcript: WER 1/6. This is an intelligibility probe on one sentence,
not a human naturalness, speaker-identity or emotional-expression evaluation.
The script supplies no reference-text prompt to the recognizer.

Speaker 1, "The little bird sings beside the window.", also had 20/21 zero-WER
cases. Its extreme low case transcribed "the little birds beside the window."
(WER 2/7). Together, 40/42 cases passed this transcription check; both failures
were the extreme low target-word setting. Moderate settings are a better starting
point than the accepted parameter limits.

Machine-readable results are in `artifacts/articulation/sweep/measurements.csv`
and `analysis.json`; the second speaker has its own `speaker1_sweep/` directory.
Interjection probes are in `artifacts/articulation/probes/`. Output artifacts are
ignored by Git, so rerun the documented commands on a fresh checkout.

The C++ unit tests verify word alignment, repeated words, exact consonant selection,
unchanged controls outside the selected intervals, identity edits, coupled
changes and invalid requests. The real-model integration test checks deterministic
repeat WAVs, local pitch edits, exact S duration edits and valid sample counts.
The production baseline WAV rendered before and after this experiment was
byte-identical, and matched the experimental identity baseline. Model hashes
were checked against the existing download manifest.

## What makes a sigh or a mouth sound different?

A sigh needs an airflow/noise envelope and potentially a voiced component;
mouth clicks and lip smacks need short transient events. Our current acoustic
interface exposes text IDs, durations, F0, energy and speaker identity. It has
no event tokens or independent controls for aspiration, lip seal, vocal-tract
geometry or glottal opening. The experiment can lengthen `HH`, vowels or `M`
and reshape their contours, but cannot promise a real sigh, gasp or click.

Research on articulatory laughter synthesis explicitly discusses breathing noise
and physical articulation beyond ordinary speech. That supports considering a
different source/articulation model for such sounds; it does not establish that
our FastSpeech2 controls expose those mechanisms.
[Lasarcyk and Trouvain, 2007](https://www.isca-archive.org/lw_2007/lasarcyk07_lw.html).

HiFi-GAN converts a Mel-spectrogram into a waveform. To vocode a sigh, the upstream
conditioning must already represent the event's acoustic structure; a vocoder
does not infer the semantic instruction "sigh" from scalar F0/energy controls.
[HiFi-GAN paper](https://arxiv.org/abs/2010.05646).

## Practical routes for future isolated experiments

1. **Recorded event lane.** Use licensed or user-recorded, speaker-compatible
   sigh/breath/click clips. Convert to 24 kHz mono; schedule them between word
   intervals with explicit event timing, gain and short crossfades. Record clip
   provenance and timeline offsets in diagnostics. This offers direct deterministic
   timing without changing the acoustic model. It needs real recordings; no
   sample lane or fabricated breath generator is implemented here.
2. **Matched-Mel resynthesis.** Extract the vocoder's exact training Mel convention
   from a recorded event and compare original versus vocoded audio. Matching
   Mel-bin count alone is insufficient: sample rate, hop, FFT/window, Mel range,
   amplitude/log floors and normalization must match the staged publisher model.
   First validate reconstruction on ordinary speech, then on breaths/transients.
   Existing generic feature extraction is not proof of this compatibility.
3. **Learned event-conditioned generator.** Use a separately trained model or
   fine-tune with speaker-labelled vocal events, explicit event tags and aligned
   temporal intervals. Keep its runner/assets separate and evaluate controllability,
   identity and nonverbal quality independently from text WER.

The third route has public precedents. Bark documents non-speech prompt tags for
laughter, sighs, gasps and throat clearing. Its generative audio architecture
differs from our explicit acoustic/HiFi-GAN interface and does not imply the same
deterministic timing contract. [Official Bark repository](https://github.com/suno-ai/bark).

CosyVoice's official examples demonstrate version-specific laughter and breath
tokens, including a breath example for its newer model. Those tokens only work
with the corresponding model/frontend; passing them to our CMU phonemizer would
not add learned vocal events.
[Official CosyVoice examples](https://github.com/QwenAudio/CosyVoice/blob/main/example.py).

A nonverbal-speech research pipeline describes a 38,718-sample dataset covering
ten event categories and temporal/semantic alignment for generation and
understanding. It illustrates the need for labelled event data rather than
assuming ordinary speech training supplies these controls.
[NonVerbalSpeech38K paper](https://arxiv.org/abs/2508.05385).

These sources were checked online on 2026-10-08. No alternative speech model,
event corpus or checkpoint was downloaded or integrated for this experiment.
For the next deterministic extension, a recorded event lane is the smallest
change; a trained event model is the route to generating new nonverbal sounds.
