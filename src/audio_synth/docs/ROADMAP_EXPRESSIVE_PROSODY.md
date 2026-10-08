# RFC: word emotion maps and coupled expressive prosody

Status: **proposed architecture, not implemented CLI behavior**. The current
engine supports sentence-wide emotion presets and scalar sliders, or fully
manual duration/F0/energy curves. Bracket markup and the APIs described here
are future work. No new speech model is implied by this RFC.

The isolated [articulation lab](../experiments/articulation/README.md) now prototypes
verified word/token alignment and bounded energy/pitch/vowel-timing coupling.
This does not add emotion markup or word maps to the production runner; those
remain proposed below.

## Goal and existing foundation

Direct individual words and phrase segments while keeping a stable speaker
identity, natural learned base contours and the existing five-input acoustic
contract. Couple energy, pitch and vowel timing with bounded, explicit formulas
instead of independent global scaling alone.

```text
Raw Text + optional word metadata
    -> parse metadata and remove tags from spoken text
    -> CMU phonemizer with word-to-token intervals
    -> trained predictor or externally supplied base contours
    -> resolved segment emotion policies
    -> bounded energy/pitch cross terms and vowel timing edits
    -> duration projection + per-token contour remapping
    -> scalar sliders -> validation -> Explicit Acoustic ONNX
    -> matching HiFi-GAN -> 24 kHz WAV
```

The implementation belongs in C++. Model weights and ONNX tensor names stay
unchanged. Word maps are control metadata; they are not learned emotion labels.
Existing untagged requests must retain their current behavior when mapping and
coupling are disabled.

## 1. Word-level emotion maps

### Input representations and explicit semantics

Support structured metadata first, then optional inline shorthand. A simple
lexical dictionary applies to matching normalized words; occurrence metadata
distinguishes repetitions:

```json
{
  "default_emotion": "neutral",
  "dictionary": {"hello": "happy", "world": "somber"},
  "words": [{"word_index": 1, "emotion": "somber", "strength": 0.7}]
}
```

`word_index` is zero-based in the tag-free phonemizer's word sequence. An
explicit range representation can add half-open `[word_begin, word_end)`
intervals for phrases. Metadata must refer to the same text as the request.
Reject out-of-range indices, overlapping explicit ranges, unknown labels,
non-finite strengths and malformed objects before model inference.

Define the user's suggested inline form as **postfix word tags**:

```text
Hello [happy] world [somber]
```

This means Hello=happy and world=somber; neither tag persists onto following
untagged words. The parser produces spoken text `Hello world` and two word
annotations. A tag without a preceding word is an error. Escaped brackets are
literal text, not annotations. Nesting and implicit persistent scopes are
excluded from the first version. Phrase spans use explicit range metadata.

Proposed precedence: explicit indexed/range metadata, then inline tags, then
lexical dictionary, then the request's global/default emotion. Conflicting
entries at the same precedence are errors. Resolve one final policy per word;
do not apply the global preset and then apply a second preset to the same word.
This prevents repeated speed/pitch boosts.

Initially `happy` can be a documented alias for the existing `excited` policy.
Other accepted labels map only to implemented policies. Aliases do not establish
new learned conditioning or subjective emotional accuracy. Proposed strength
is in `[0,1]`, with zero meaning identity and one meaning full policy.

### Preserve alignment through the phonemizer

Extend `PhonemizationResult` with word spans and optional token ownership:

```cpp
struct WordTokenSpan {
    std::size_t word_index;
    std::size_t text_byte_begin; // Offset into tag-free input text
    std::size_t text_byte_end;   // Exclusive
    std::size_t token_begin;
    std::size_t token_end;       // Exclusive
};

struct WordEmotion {
    std::size_t word_index;
    VocalEmotion emotion;
    float strength;
};
```

Record the start/end token indices inside `flush_word()`, where the CMU
pronunciation actually emits IDs. Include all IDs emitted for the word, whether
from a dictionary pronunciation, multi-ID IPA mapping or letter fallback.
Do not reconstruct alignment by counting letters or assume one token per word.
Current ASCII-English word normalization remains the source of truth for the
first version; byte offsets must point to valid UTF-8 boundaries if extended.
Keep original-to-clean text offset mapping in the markup parser for diagnostics.

Punctuation `sp` tokens have no lexical word owner. Assign them separately to
phrase boundaries, using a documented boundary policy. A tag on a word must
not silently consume the next pause. Preserve existing boundary/stress/consonant
annotations; ownership and phonetic kind answer different questions.

After base durations are known, prefix sums project `[token_begin,token_end)`
to `[frame_begin,frame_end)`. Zero-duration tokens occupy empty frame ranges.
Token indices remain stable after duration edits; recompute frame ranges from
the new duration prefix sums. Keep both original and final intervals for tracing.

### Segment transforms and transitions

Resolve adjacent words with the same emotion/strength into segments, split at
punctuation boundaries. Compute policy pitch/energy statistics inside each
segment rather than over the entire utterance. Annotated stressed vowels and
phrase tails are selected within that segment, without leaking into neighbors.

Build continuous candidate curves and real-valued duration targets first.
Blend strength in log-Hz for positive pitch and in the defined energy-feature
space; strength zero must reproduce the base controls. Preserve zero-duration
tokens. Project positive duration targets to at least one integer frame.

Smooth pitch/energy transitions inside adjacent speech regions with short
ramps. Do not smooth across silence boundaries or erase a requested pause.
The first experiment can use up to four frames (50 ms at hop 300), shortened
when a segment is smaller. This is a tuning parameter, not an established
perceptual optimum. Duration edits require per-token contour remapping, not
a resample of the entire sentence.

## 2. Cross-term modulation

### Energy change influencing pitch

Use a bounded relative-energy change to express a vocal-effort-like control.
It is an engineering heuristic; normalized acoustic energy is not measured
vocal effort or an exact waveform loudness quantity.

Let `E0[f]` be base energy and `E1[f]` energy after the segment policy. Define:

$$q_f = \operatorname{clip}\left(\log\frac{E_1[f]+\epsilon}{E_0[f]+\epsilon},-q_{max},q_{max}\right).$$

For positive pitch, couple in log-Hz:

$$\log F_{2}[f] = \log F_{1}[f] + a\,w_f\,\operatorname{smooth}(q_f).$$

`a` is a configurable coupling gain; `w` gates the effect to annotated speech
regions, initially voiced/stressed vowels, and tapers at segment edges.
Positive energy changes can raise pitch. For small changes this approximates
the requested relationship `delta F0 proportional to delta Energy`, with better
scale behavior across speaker pitch ranges. A simpler additive-Hz prototype
may be useful for comparison, but must expose its Hz-per-energy-unit convention.

Use an epsilon and clip relative changes so low-energy boundaries cannot
produce enormous ratios. Zero F0 entries stay zero in control space; exclude
them from logarithms. The current acoustic export still maps zero pitch to
the continuous training mean, so this rule does not add physical voicing control.

Evaluate coupling once from an immutable base/candidate pair. Do not feed the
new pitch back into energy and iterate: that could amplify the two dimensions
without bounds. Apply configurable pitch limits and a maximum semitone offset.
All coefficients and limits are experimental and require speaker/text calibration.

### Nonlinear pitch peaks influencing vowel elongation

For each eligible vowel token, measure its positive log-pitch rise relative to
the base over that token's original frame interval:

$$A_i=\max_{f\in i}\max\left(0,\log\frac{F_2[f]}{F_0[f]}\right).$$

Skip invalid/zero-pitch positions and tokens with no frames. Smooth before
measuring a peak, or use a robust percentile, to avoid elongating a vowel due
to a single frame spike. Let `d_policy[i]` already include segment timing:

$$d_{target}[i]=d_{policy}[i]\left(1+b_i\,\min(A_i,A_{max})^\gamma\right),\qquad\gamma>1.$$

`b_i` is larger for stressed vowels and zero for consonants, boundary tokens
and unknown annotations in the first implementation. Bound elongation per
token and total frame count; the existing budget is 15,000 frames. Pitch peaks
should not automatically turn every vowel into a long stressed vowel.

Project duration targets once, preserving zero allocations and requiring at
least one frame for positive allocations. Remap coupled pitch/energy curves
inside each token to the final duration. Recompute all segment frame spans.
Then apply the existing global sliders and validate. Disabling coupling must
recover the independent segment-policy behavior exactly.

### Proposed API and composition order

```cpp
struct CrossTermControls {
    float energy_to_log_pitch_gain{0.0F}; // Zero disables this term
    float pitch_peak_to_vowel_duration_gain{0.0F};
    float duration_exponent{2.0F};
    // Add explicit ratio, pitch-offset and elongation bounds.
};
```

Add a pure control-layer function in `models/explicit_neural/`, accepting base
controls, token kinds, word spans, resolved segment policies and coupling
parameters. Return new controls plus updated alignment diagnostics. It must
not load model weights or perform acoustic inference.

Keep default gains zero. Proposed new CLI options should be introduced only
after the C++ control API and schema are stable; do not document them as runnable
commands before implementation. Fully manual mode should bypass word mapping
and coupling by default, unless the caller explicitly requests those transforms.
Global sliders remain the final user adjustment, so their energy changes do not
silently feed back into coupling. Document that order as part of the API contract.

## 3. Implementation milestones and acceptance checks

1. **Alignment:** add word spans/ownership to the existing phonemizer and
   diagnostics. Verify repeated words, apostrophes, punctuation, OOV spelling,
   multi-ID mappings, zero durations and unchanged untagged token sequences.
2. **Metadata:** implement deterministic lexical/indexed/range resolution and
   a separate markup parser. Verify tags never reach spoken input, precedence,
   malformed markup, unknown labels and original/clean offset mapping.
3. **Segment policies:** refactor reusable emotion math without changing the
   current sentence-wide behavior. Verify strength-zero identity, unchanged
   speaker ID, outside-segment isolation, stress targeting and smooth transitions.
4. **Coupling:** implement single-pass bounded cross terms and vowel duration
   projection. Test zero-gain equivalence, energy/pitch direction, low-energy
   ratio bounds, zero-F0 handling, frame budgets and per-token remapping.
5. **Integration and assessment:** expose options/schema, render fixed texts
   with several speakers, compare intelligibility and waveform pitch/cadence,
   and assess intended emotion by listening. Record coefficients, exact controls,
   model hashes, timing and samples under `artifacts/`.

Numerical control changes establish influence, not natural emotion. Acceptance
must retain readable speech and stable identity across segment transitions.
Independent recognition can check words; human listening is needed to judge
emotional interpretation. No acoustic-model retraining is required for the first
control-layer prototype. Natural laughter, sighs, crying or true whisper remain
outside the model's explicit event interface.

## Source touchpoints

| Module | Planned responsibility |
| --- | --- |
| `include/vocal/phonemizer.hpp` | Public word/token alignment result |
| `models/piper_onnx/phonemizer.cpp` | Capture actual emitted word intervals |
| `include/vocal/control_params.hpp` | Segment/coupling settings and validation API |
| `models/explicit_neural/` | Pure segment and cross-term transforms; duration remapping |
| `app/tts_cli.cpp` | Parse requests, compose transforms, trace resolved intervals |
| `tests/` | Alignment, parser, control invariants and trained-model integration |

The exact acoustic contract remains in
[the model documentation](../models/explicit_neural/README.md).
Current runnable commands are in [the README](../README.md) and
[the voice design guide](VOICE_PROFILE_DESIGN.md).
