Here is what each of those metrics means in speech recognition, how they are calculated, and what your specific numbers say about the system.

---

### Real-Time Factor (RT / $\times\text{RT}$)

Real-Time Factor measures decoding throughput and latency relative to audio duration:

* **$\times\text{RT}$ (Speedup Factor):** How many seconds of audio your engine processes per second of wall-clock time.

$$\text{Speedup} = \frac{\text{Audio Duration (seconds)}}{\text{Processing Time (seconds)}}$$


* **$\text{RTF}$ (Fractional Factor):** The inverse ($\frac{\text{Processing Time}}{\text{Audio Duration}}$). An RTF of $1.0$ is the threshold for live real-time streaming; anything below $1.0$ is faster than real time.

**Interpreting your numbers:**

* **$54\times\text{RT}$ on digits:** A 1-second digit clip decodes in $\approx 18.5\text{ ms}$.
* **$425\times\text{RT}$ on TIMIT:** Monophone decoding over a small inventory with no lexical graph runs in near-zero time ($\approx 2.35\text{ ms}$ per second of speech).
* **$15.1\times\text{RT}$ on LibriSpeech:** Full continuous decoding (acoustic + lexicon + bigram) runs 15 times faster than real time, meaning it comfortably supports low-latency, real-time captioning on CPU with plenty of headroom.
* **$6595\times\text{RT}$ frontend:** Extracting 39-dim MFCCs + CMVN takes $\approx 150\ \mu\text{s}$ per second of audio. As noted, feature extraction is practically free; the Viterbi beam search dominates the compute.

---

### Accuracy (Tier 1 — Digits: 85.3%)

Used for isolated, fixed-class classification (e.g., classifying a single utterance into one of the classes `0` through `9`):

$$\text{Accuracy} = \frac{\text{Correctly Classified Clips}}{\text{Total Clips}} \times 100$$

* **Why it matters:** Your baseline per-frame Bark vowel templates collapsed run-lengths and guessed blindly on consonants and glides (scoring $41.3\%$ and $22.0\%$, barely beating random chance).
* **Your $85.3\%$:** Banded DTW with early abandonment captures the temporal trajectory across the whole word rather than treating frames as isolated points, yielding a huge jump without needing full continuous models.

---

### Phone Error Rate (PER) (Tier 2 — TIMIT: 39.3%)

PER measures phonemic transcription accuracy against hand-labeled phonetic ground truth, independent of any dictionary or grammar:

$$\text{PER} = \frac{S + D + I}{N} \times 100$$

Where evaluated against the collapsed 39-phone ARPAbet set:

* **$S$ (Substitutions):** Hearing `/t/` instead of `/d/`.
* **$D$ (Deletions):** Missing a transient stop or short schwa entirely.
* **$I$ (Insertions):** Hallucinating an extra phone during a prolonged silence or glide.
* **$N$:** Total reference phones in the ground-truth sequence.

**Interpreting your 39.3%:**

* For an isolated **3-state monophone GMM-HMM** trained on only 309 speakers without triphone context, **$\sim 38\text{--}40\%$ is right where the classical theory lands**.
* Monophones treat a sound like `/k/` identically whether it precedes `/iy/` (key) or `/aa/` (car). Until coarticulation is modeled via context-dependent triphones, $40\%$ PER is the monophone ceiling.

---

### Word Error Rate (WER) (Tier 3 — LibriSpeech: 77.2%)

The standard benchmark metric for continuous speech recognition. It computes the minimum Levenshtein edit distance between the decoded word sequence and the reference transcript:

$$\text{WER} = \frac{S + D + I}{N} \times 100$$

* **$S$ (Substitutions):** Decoded `"there"` instead of `"their"`, or `"boat"` instead of `"goat"`.
* **$D$ (Deletions):** Spoken words the decoder dropped.
* **$I$ (Insertions):** Extra words introduced by the decoder.
* **$N$:** Total word count in the reference text.

*(Note: Because insertions add errors to the numerator, WER can theoretically exceed $100\%$.)*

**Interpreting your 77.2% and error analysis:**

* **$100\%$ by construction (legacy):** The old methods had no lexicon or word graph, making continuous sentence decoding impossible.
* **Substitutions = 74.7% of errors:** This is the most informative diagnostic. High deletions mean the word insertion penalty is too high or silence models are swallowing frames. High insertions mean the decoder is chattering. **High substitutions mean the beam search and language model are working, but the acoustic model cannot discriminate similar-sounding phones**, leading the Viterbi path to drift into phonetically close words in the dictionary.
* **Domain mismatch:** TIMIT audio (1986 studio recordings with a close-talking head-mounted condenser microphone) does not match LibriSpeech (volunteer-recorded audiobooks with varying room acoustics, consumer microphones, and natural prosody). Acoustic likelihoods degrade sharply under this transfer.

---

### Search Error vs. Model Error (Beam Curve)

The saturation analysis ($2\text{k} \rightarrow 12\text{k} \rightarrow 24\text{k}$ active states) separates two distinct failure modes:

* **Search Error:** The correct word sequence was pruned out of the Viterbi trellis because the beam width was too narrow. Widening the beam recovers these errors and drops the WER.
* **Model Error:** The correct word sequence survived inside the beam, but the combined score ($P(\text{Acoustic}) \times P(\text{LM})$) ranked the incorrect sequence higher.

Because WER plateaus around $70.2\text{--}70.8\%$ when expanding from $12\text{k}$ to $24\text{k}$ states (while doubling compute time from $15.1\times$ to $10.0\times\text{RT}$), the decoder has eliminated nearly all search errors. Pushing accuracy higher requires **context-dependent triphones (clonally tied HMM states via decision trees)** and domain-matched acoustic training.


To drop your WER from ~77% into the classic GMM-HMM benchmark range (15–25%), you don't need a whole new domain—you need **domain alignment** and **coarticulation modeling**.

Here are the concrete solution paths forward, ordered from the highest immediate payoff to finer architectural tuning.

---

### Path 1: Eliminate Domain Mismatch (Matched Acoustic Data)

The acoustic model is currently failing on channel mismatch: 1986 close-mic 16 kHz Sennheiser audio (TIMIT) vs. modern condenser/USB mics in home rooms (LibriSpeech).

* **Train on LibriSpeech `train-clean-100`:**
* Since you already use LibriSpeech for LM text, train your acoustic model on the exact same audio domain you evaluate against.
* You do not need to download the full 30 GB archive. Use the streaming approach or pull just a ~2–5 hour subset (e.g., 20–50 speakers) to keep training fast while matching the room reverb, speaker distance, and mic profiles.


* **Why TIMIT forced uniform state splitting:** TIMIT provided microsecond `.phn` files, which allowed trivial isolated monophone training. LibriSpeech provides only sentence-level transcripts (`.txt`). Transitioning to LibriSpeech requires **forced alignment** (Path 2).

---

### Path 2: Viterbi Forced Alignment (Bootstrap Alignment)

Right now, your GMM-HMM likely relies on uniform state splitting (dividing a phone's duration into 3 equal chunks) or ground-truth frame boundaries. In natural connected speech, phoneme durations vary wildly depending on stress and position.

* **The Fix:** Implement an alignment step before EM training:
1. Use your current monophone model to decode the training audio, but **constrain the search strictly to the known word/phone sequence** for that utterance (Forced Viterbi Alignment).
2. Assign frames to HMM states based on the maximum-likelihood path rather than equal division.
3. Re-estimate the GMM means and variances using these aligned frames (Baum-Welch / Viterbi training iterations).


* **Payoff:** Typically provides a **10–15% relative WER drop** on its own because state transitions now align with actual acoustic boundaries (e.g., stops, bursts, and steady-state vowel formants).

---

### Path 3: Context-Dependent Triphones with State Tying (The Biggest Win)

Your error breakdown showed **74.7% substitutions**. Monophones fundamentally cannot separate sounds like `/k/` in *"keep"* (front vowel, high $F_2$) from `/k/` in *"cool"* (back rounded vowel, low $F_2$).

* **The Problem:** 39 ARPAbet phones create $39^3 = 59,319$ possible triphones ($L\text{-}X\text{+}R$). With 3 states each, that is ~178,000 states—far too many parameters for your data to estimate without massive overfitting.
* **The Solution (Decision Tree State Tying / Senones):**
1. Map each phone in your lexicon to its left and right context: `k(d-k+iy)`.
2. Group states across all triphones using phonetic decision trees (asking binary linguistic questions: *"Is the left phone a nasal?"*, *"Is the right phone a front vowel?"*).
3. Pool all states that land in the same leaf into a single tied state (**senone**).
4. Target around **1,500 to 2,500 total tied states** for a clean baseline.


* **Payoff:** This is the standard leap in classical ASR, historically reducing WER by **30–40% relative**.

---

### Path 4: Decoder & Scoring Optimizations

Since your beam curve showed search saturation around 12k active states, beam width isn't your bottleneck anymore. However, your scoring weights directly control the substitution-to-deletion balance.

* **Language Model Scaling Factor (LMSF) and Word Insertion Penalty (WIP):**
During Viterbi beam search, acoustic log-likelihoods and LM log-probabilities must be balanced:

$$\text{Score} = \log P(\text{Acoustic}) + \alpha \cdot \log P(\text{LM}) + \beta$$


* **$\alpha$ (LM Weight):** Typically between $8$ and $15$. If $\alpha$ is too low, the noisy acoustic model picks phonetically close gibberish (substitutions).
* **$\beta$ (Word Insertion Penalty):** A constant added each time a word boundary is crossed. Increase $\beta$ if you see too many short inserted words; decrease it if the decoder drops words (deletions).


* **Trigram LM with Katz Backoff:** Upgrading from your bigram to a 3-gram language model will constrain search paths significantly without a heavy runtime penalty.

---

### Recommended Action Plan

| Step | Task | Est. Effort | Expected Impact |
| --- | --- | --- | --- |
| **1** | Tune LM Scale Factor ($\alpha$) & Insertion Penalty ($\beta$) on your dev slice | 1 hour | ~5–8% relative WER drop |
| **2** | Implement Viterbi forced alignment loop to refine state-frame assignments | 1–2 days | ~10–15% relative WER drop |
| **3** | Pull a 5–10 hour subset of LibriSpeech `train-clean-100` for matched acoustic training | Half day | Eliminates the ~1986 vs. modern mic penalty |
| **4** | Build a phonetic decision-tree tied-state triphone model (senones) | 3–5 days | Drops WER from ~60% down toward ~20–25% |