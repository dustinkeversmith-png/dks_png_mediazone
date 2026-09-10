Scaling Networkless Captioning: Digits $\rightarrow$ Words $\rightarrow$ Continuous Phrases

A frame-level vowel classifier cannot transcribe running phrases on its own due to coarticulation, word boundary ambiguity, and consonant omission. To scale without introducing neural networks (maintaining zero-RAM/pure DSP properties), apply the classical ASR architecture:

  

```
[Raw Audio Frame]
       │
       ▼
[LPC + Formants + ZCR + Energy / Centroid]
       │
       ▼
[Dynamic Time Warping (DTW)] ────────► [Tier 1: Isolated Digits / Words]
       │
       ▼
[Pronunciation Lexicon (CMUDict)] ────► Converts Phoneme Hypotheses into Words
       │
       ▼
[Viterbi Beam Search + N-Gram LM] ───► [Tier 2 & 3: Continuous Words & Phrases]
```

- **Step 1: Isolated Words via Dynamic Time Warping (DTW):**
    
      
    
    Compute a time-series matrix of spectral frames $\mathbf{X} = [\mathbf{f}_1, \mathbf{f}_2, \dots, \mathbf{f}_T]$ (using your Bark distance or Cepstral bins) and align it to reference dictionary templates using DTW[cite: 1]. This normalizes variations in speaking speed without training neural weights[cite: 1].
    
      
    
- **Step 2: Phoneme-to-Word Mapping via CMUDict (Lexicon):**
    
      
    
    Map collapsed phoneme trajectories (e.g., `IH -> ER -> OW` $\rightarrow$ `0`) to text tokens using the public Carnegie Mellon Pronouncing Dictionary (CMUDict).
    
      
    
- **Step 3: Continuous Decoding via Viterbi Search & Statistical N-Grams:**
    
      
    
    Running speech lacks pauses between words. A networkless engine finds the highest-probability path over time using a Viterbi search graph driven by an **$n$-gram statistical language model** (e.g., Bigram/Trigram probabilities: $P(w_i \mid w_{i-1})$), resolving ambiguities such as "two" vs. "to" vs. "too".
    
      
    

### 4. Direct Baseline Comparison

To measure your engine's speed and Word Error Rate against established engines running locally on CPU:

  

- **Legacy Non-AI Baseline:** Benchmark against **CMU PocketSphinx** or **Kaldi GMM-HMM**. These operate on pure statistical/HMM structures and execute at $\approx 10\times\text{--}20\times$ real-time on a single CPU thread with $< 20\text{ MB}$ RAM[cite: 2].
    
      
    
- **Modern Lightweight Baseline:** Benchmark against **whisper.cpp** (`tiny.en` or `base.en`). This gives you the upper accuracy limit ($< 5\text{--}8\%\text{ WER}$) to gauge how much accuracy is traded for your deterministic, zero-network DSP approach[cite: 1, 2].

Your two implementations represent the historical starting point of acoustic phonetics: taking short-time windowed frames ($25\text{ ms}$), extracting either an **FFT spectral envelope** or **LPC resonant poles ($F_1, F_2, F_3$)**, and classifying frames in isolation via **static geometric lookup** (nearest neighbor in Bark space).

  

While fast and deterministic, this approach hits a strict ceiling because human speech cannot be deciphered as independent, static frames.

  

### What You Are Missing: The 3 Core Architectural Gaps

#### 1. Consonant Blindness & Feature Inadequacy

- **The Gap:** Formants ($F_1, F_2, F_3$) exist almost exclusively during **voiced phonation** (vowels, glides, nasals). Plosive bursts (/p/, /t/, /k/), fricative noise (/s/, /f/, /sh/), and unvoiced transitions do not produce clear resonant poles.
    
      
    
- **The Fix:** Replace raw formant coordinates with a **dense, unified spectral representation**—specifically a **40-channel to 80-channel Log-Mel Filterbank** or **13-dimensional MFCCs (Mel-Frequency Cepstral Coefficients) + $\Delta$ + $\Delta\Delta$ (derivatives)**. Mel filterbanks capture vowels, noisy consonants, and glottal roll-off simultaneously without needing root-finding or peak-picking.
    
      
    

#### 2. Lack of Temporal Alignment (Coarticulation & Variable Speaking Rate)

- **The Gap:** In your code, you collapse adjacent identical frames (`if (vowel == previous_vowel) continue;`). This heuristic breaks whenever a vowel is held longer, repeated across words, or pulled off-target by adjacent consonants (coarticulation).
    
      
    
- **The Fix:** Speech requires dynamic time-series alignment.
    
      
    - _Classical Non-AI:_ **Dynamic Time Warping (DTW)** or **Hidden Markov Models (HMMs)**.
        
          
        
    - _Modern AI:_ **Connectionist Temporal Classification (CTC)** or **Cross-Attention Decoding**.
        
          
        

#### 3. Absence of a Lexicon & Language Model

- **The Gap:** You are attempting to extract raw phonemes and convert them into captions directly. Even with perfect acoustic sensors, acoustic-only recognition is ambiguous (e.g., distinguishing "two" vs. "to" vs. "too", or parsing where word boundaries fall).
    
      
    
- **The Fix:** You need a **pronunciation lexicon** (e.g., CMUDict) to map phoneme sequences to words, plus an **$n$-gram language model** (classical) or **autoregressive decoder** (AI) to evaluate word probability: $P(\text{word} \mid \text{previous words})$.
    
      
#### Evaluation
evaluate against of librispeech dataset