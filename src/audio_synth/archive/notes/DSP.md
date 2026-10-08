Before deep learning dominated speech synthesis, the field relied on several distinct non-neural paradigms rooted in acoustics, digital signal processing (DSP), and statistical modeling:

---

### 1. Formant Synthesis (Rule-Based Acoustic Synthesis)

Instead of using human audio samples, formant synthesis generates sound purely with oscillators, resonators, and noise generators based on the acoustic **source-filter model**.

* **How it works:** An excitation source (a periodic impulse for voiced sounds or white noise for unvoiced fricatives) passes through a series of tunable digital resonators configured to mimic human vocal tract resonances (formants $F_1, F_2, F_3$).
* **Famous Implementations:** The **Klatt synthesizer** (which powered DECtalk and Stephen Hawking’s famous voice) and **eSpeak / eSpeak-NG**.
* **Key Characteristic:** Sounds distinctly robotic, but has a tiny memory footprint (kilobytes), zero glitching at transitions, and remains legible at high playback speeds (often favored by blind screen-reader users).

### 2. Statistical Parametric Synthesis (HMM / GMM-based)

Statistical Parametric Speech Synthesis (SPSS) represented the bridge between traditional DSP and modern machine learning.

* **How it works:** Instead of storing audio clips, the system trains **Hidden Markov Models (HMMs)** or **Gaussian Mixture Models (GMMs)** on recorded speech to learn the statistical distributions of acoustic parameters (fundamental frequency $F_0$, spectral envelopes, duration, and aperiodicity). A classic vocoder (like STRAIGHT or WORLD) then reconstructs the audio from these predicted parameters.
* **Famous Implementations:** **HTS (HMM-based Speech Synthesis System)** and **Festival HTS**.
* **Key Characteristic:** Extremely smooth and adaptable to new speaker characteristics or emotional styles with very little data, but prone to a characteristic "muffled" or "buzzy" quality due to statistical averaging.

### 3. Articulatory Synthesis (Biomechanical Physical Modeling)

Articulatory synthesis attempts to model human speech from physical first principles.

* **How it works:** It uses differential equations to simulate the physical anatomy of the vocal tract: the lungs creating airflow, vocal cords vibrating, and the positioning of the tongue, jaw, lips, and velum changing the tube geometry over time.
* **Famous Implementations:** **Praat** (articulatory synthesis module) and **Vocaloid/BBS-based research engines**.
* **Key Characteristic:** Highly valuable for linguistic and anatomical research, but mathematically complex to calibrate for general-purpose, fluent text-to-speech.

### 4. Linear Predictive Coding (LPC) Synthesis

LPC represents the human vocal tract as an all-pole digital filter where each sample is predicted as a linear combination of previous samples.

* **How it works:** It separates the spectral envelope (the shape of the vocal tract) from the pitch and excitation source. By transmitting or storing just a few filter coefficients rather than raw waveforms, speech can be generated with minimal computational overhead.
* **Famous Implementations:** The **Texas Instruments TMS5220** chip used in the 1978 *Speak & Spell* toy and 1980s arcade cabinets.

### 5. Sine-Wave Synthesis

* **How it works:** A minimal synthesis technique that replaces complex speech formants with a combination of just 3–4 dynamic, time-varying sine waves tracking the center frequencies of the major formants.
* **Key Characteristic:** When heard in isolation, it sounds like alien electronic whistles; however, the human auditory system can still perceive the linguistic phonetic cues and understand the words, making it a foundational tool in psychoacoustics research.

---

### Paradigm Comparison

| Method | Source Data | Storage / Footprint | Naturalness | Flexibility |
| --- | --- | --- | --- | --- |
| **Formant (Klatt)** | Mathematical rules & resonators | < 2 MB | Low (Robotic) | High (Pure parameter control) |
| **Concatenative** | Sliced human audio recordings | 50 MB – 2 GB+ | High timbre / Choppy cadence | Very Low (Fixed to database) |
| **Statistical (HMM)** | Statistical feature distributions | 5 – 20 MB | Medium (Buzzy/muffled) | High (Smooth style adaptation) |
| **Articulatory** | Biomechanical tract geometry | Very small code footprint | Experimental / Variable | High (Anatomically bounded) |