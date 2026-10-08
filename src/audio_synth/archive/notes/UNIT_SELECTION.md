That describes **Concatenative Text-to-Speech Synthesis**, specifically the subclass known as **Unit Selection Synthesis** (or **Corpus-Based TTS**).

Before deep learning took over, this was the industry standard (powering early Siri, GPS navigation units, and telecom automated systems).

---

### How the System Works

A concatenative pipeline maps text to real audio fragments and joins them using optimization algorithms:

**1. The Database (The "Map")**
A voice actor records hundreds of hours of phonetically rich text. Engineers slice and index this audio database into tiny linguistic units:

* **Diphones:** Transitions from the middle of one phoneme to the middle of the next (e.g., *s-p*, *p-e*, *e-k* in "speak"). Slicing at stable mid-points rather than transitional boundaries reduces abrupt glitching.
* **Non-uniform units:** Variable-length chunks ranging from single phonemes to whole words and frequent phrases.

**2. Unit Selection (The Mapping Algorithm)**
When text is converted into a target phonetic sequence, an algorithm (often a **Viterbi search**) searches the database to find the best candidate audio clips by balancing two mathematical costs:

* **Target Cost:** How well a candidate clip matches the intended pitch, stress, duration, and phonetic environment.
* **Join Cost (Concatenation Cost):** How seamlessly two adjacent candidate slices will fit together without a noticeable click or pitch jump.

**3. Stitching & Blending (Smoothing the Seams)**
Once the optimal sequence of units is chosen, the engine splices them together and applies digital signal processing—most notably **PSOLA (Pitch-Synchronous Overlap and Add)**:

* **Pitch & Time Alignment:** Stretches or compresses the waveforms and shifts pitch markers so adjacent units match in frequency.
* **Windowing & Cross-Fading:** Applies overlapping taper windows across splice points to eliminate clicks, phase cancellation, and amplitude discontinuities.

---

### Varieties of Concatenative Systems

* **Fixed-Inventory / Diphone Synthesis:** Stores exactly one or two recordings per diphone pair. Highly consistent and requires minimal storage, but sounds monotonic and artificial.
* **Unit Selection Synthesis:** Stores thousands of variations for each sound across different pitches, volumes, and contexts. When a long phrase matches an existing recording, it sounds nearly indistinguishable from natural human speech.
* **Domain-Specific / Carrier Phrase:** Used in automated subway announcements or banking lines where fixed templates (*"The next train to [Destination] will arrive in [Number] minutes"*) are stitched with whole recorded words.

---

### Why the Industry Moved to Neural TTS

While individual slices have pristine human timbre, concatenative engines suffer from two core limitations:

* **Boundary Artifacts:** Even with PSOLA smoothing, rapid spectral mismatches create audible glitching or unnatural robotic jumps at slice boundaries.
* **Lack of Expressive Control:** Changing the mood, speaking style, or emotion requires re-recording and indexing an entirely new multi-gigabyte audio corpus.

# Examples

The golden era of **unit-selection / corpus-based TTS** spanned from the late 1990s through the mid-2010s, producing several milestone systems:

### Commercial Production Engines

* **AT&T Natural Voices (formerly AT&T Labs TTS):** Widely considered the gold standard of early 2000s commercial unit selection. Voices like *"Crystal"* and *"Mike"* set the baseline for telecom IVR systems and interactive apps due to vast, meticulously labeled phoneme databases.
* **Nuance Vocalizer (Scansoft / Nuance):** Powered early in-car GPS systems, airline flight alerts, and the **original 2011 Apple Siri** (voiced by Susan Bennett as *"Samantha"*). It used dynamic multi-unit selection, allowing it to seamlessly switch between full words, syllables, and diphones depending on database availability.
* **Ivona TTS:** Renowned for exceptionally smooth transitions and natural human warmth (notably the *"Amy"* and *"Brian"* voices). Amazon acquired Ivona in 2013, making its unit-selection engine the initial backbone of **Amazon Alexa** and Kindle text-to-speech.
* **CereProc:** Specialized in adding distinct personality, humor, and regional accents (Scottish, Irish, Northern English) into unit selection. Unlike many neutral systems, their corpus curation was designed to retain vocal idiosyncrasies and breathing cues.
* **Acapela Group:** Built extensive multilingual corpora, famous for accessibility software, public transit transit feeds, and custom synthetic voices for screen readers.

---

### Pioneering Research & Open-Source Systems

* **ATR CHATR (1996):** Developed in Japan by Andrew Hunt and Alan Black, **CHATR was the breakthrough paper/system** that proved corpus-based unit selection could outperform rule-based systems by framing unit choice as a Viterbi search over target and join costs.
* **Festival Speech Synthesis System / Multisyn:** Built at the University of Edinburgh (CSTR) and Carnegie Mellon University, Multisyn provided the open-source reference implementation of unit selection that educated a generation of speech scientists.

---

### Real-World Fixed/Hybrid Domain Highlights

| Application | Engine/Approach | Why It Worked Well |
| --- | --- | --- |
| **London Underground Announcements** | Carrier-phrase concatenation | Fixed sentence structures allowed whole recorded names and phrases to stitch with zero boundary distortion. |
| **Early Garmin / TomTom GPS** | Micro-corpus unit selection | Limited spatial vocabulary (e.g., *"In 500 feet, turn left onto..."*) meant target costs were nearly zero, producing natural output on low-power automotive chips. |
| **Half-Life 2 (Vortigaunt / Overwatch Voice)** | Hybrid corpus processing | Game audio design combined unit-selected spoken segments with heavily layered DSP filters to mask splice points while keeping authentic vocal resonance. |