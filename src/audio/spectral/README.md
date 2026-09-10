Here is an in-depth breakdown of the **Mel Filterbank**, **Pitch & Cepstrum**, and **Spectral Features** (Centroid & Tilt), along with the exact math and concrete C++ implementations using your current setup.

---

### 1. What is the Mel Filterbank and What Does It Look Like?

Human pitch perception is non-linear: we can easily detect a $50\text{ Hz}$ difference between $200\text{ Hz}$ and $250\text{ Hz}$, but cannot discern a $50\text{ Hz}$ difference between $8000\text{ Hz}$ and $8050\text{ Hz}$.

The **Mel scale** warps linear frequency ($f$) to a psychoacoustic pitch scale ($m$):


$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

#### What it Looks Like

A Mel filterbank is a collection of overlapping triangular filters (typically $40$ or $80$ filters, or $M$) spread across your FFT spectrum ($K = N/2 + 1$ bins):

```
Filter Response H_m(f)
1.0 ┼      /\          /\
    │     /  \        /  \
    │    /    \      /    \       /\             /\
    │   /  H1  \    /  H2  \     /  \           /  \
    │  /        \  /        \   / H3 \         / H4 \
0.0 ┴─┴───────────┴──────────┴─┴──────┴───────┴──────┴────► Linear Frequency (Hz)
      Narrow & dense at low Hz     Wider & sparser at high Hz

```

* **Low Frequencies ($< 1000\text{ Hz}$):** The triangles are narrow, closely packed, and linearly spaced.
* **High Frequencies ($> 1000\text{ Hz}$):** The triangles become progressively wider and farther apart (logarithmic spacing).

#### How It Operates Mathematically

It is a sparse matrix multiplication. If your FFT has $257$ bins (for an $N=512$ point FFT) and you construct $M=40$ Mel channels, your filterbank is an $(M \times K)$ matrix:


$$\text{Mel Energy}_m = \sum_{k=0}^{K-1} \vert{}X[k]\vert{}^2 \cdot H_m[k]$$

Taking $\log(\text{Mel Energy}_m)$ produces **Log-Mel Filterbank energies**, which are the exact input vectors modern speech recognizers (such as Whisper) ingest.

---

### 2. What is $X(f)$ and How Does Inverse FFT Extract Pitch?

In continuous math notation, $X(f)$ denotes the Fourier Transform of the signal.

In your discrete C++ pipeline:

* $X[k]$ is the complex spectrum output from `fftwf_execute` ($k = 0 \dots N/2$).
* $\vert{}X[k]\vert{}^2$ is the **power spectrum** array (what your `FastFFT::compute_power_spectrum` outputs).
* $\log(\vert{}X[k]\vert{}^2)$ means taking the **element-wise natural log of the power spectrum bins**.

#### Why IFFT of Log Power Spectrum Yields Pitch (The Cepstrum)

Recall the **Source-Filter Model**:


$$\text{Speech } s[n] = e[n] * h[n] \quad (\text{Excitation / Glottal Pulse } * \text{ Vocal Tract Filter})$$

In the frequency domain, convolution becomes multiplication:


$$\vert{}S(f)\vert{} = \vert{}E(f)\vert{} \cdot \vert{}H(f)\vert{}$$

Taking the logarithm turns multiplication into addition:


$$\log \vert{}S(f)\vert{} = \log \vert{}E(f)\vert{} + \log \vert{}H(f)\vert{}$$

Now you have:

* $\log \vert{}H(f)\vert{}$: A slow-moving, broad spectral envelope (formants).
* $\log \vert{}E(f)\vert{}$: Rapid periodic ripples caused by vocal cord vibration pitch harmonics.

Because these two components are added together, taking the **Inverse FFT (IFFT)** moves you into a pseudo-time domain called the **Quefrency Domain** (the **Real Cepstrum**):


$$c[n] = \text{IFFT}\left(\log \vert{}X[k]\vert{}^2\right)$$

```
Cepstrum c[n] Magnitude
│
│ █ (Near quefrency 0: Vocal tract envelope / Formants)
│ █ 
│ █
│ │
│ │                  █ <── Pitch Peak! (Index = Pitch Period T_0)
│ └──────────────────┴──────────────► Quefrency index n (time-lag)

```

* **Low Quefrency (small $n$):** Represents the vocal tract shape (if you take the DCT of this, you get MFCCs).
* **High Quefrency (larger $n$):** Displays a distinct spike at the fundamental pitch period ($T_0$).

$$\text{Pitch } (F_0) = \frac{\text{Sample Rate}}{\text{Peak Index } n}$$



---

### 3. Spectral Centroid & Spectral Tilt

Both metrics are derived directly from the power spectrum array $\vert{}X[k]\vert{}^2$ and classify phonetic features (such as separating vowels from fricative consonants like `/s/` or `/f/`).

#### A. Spectral Centroid (The "Center of Mass")

The spectral centroid identifies where the bulk of the spectral energy is concentrated:

$$\text{Centroid} = \frac{\sum_{k=0}^{K-1} f_k \cdot \vert{}X[k]\vert{}^2}{\sum_{k=0}^{K-1} \vert{}X[k]\vert{}^2}$$


where $f_k = k \cdot \frac{f_s}{N}$ is the linear frequency in Hz of bin $k$.

* **Voiced Vowels (`/a/`, `/u/`, `/i/`):** Most energy is in low formants. Centroid is typically **$500\text{ Hz} - 1800\text{ Hz}$**.
* **Unvoiced Fricatives (`/s/`, `/sh/`):** Characterized by high-frequency turbulent friction noise. Centroid shifts upwards to **$3500\text{ Hz} - 7000\text{ Hz}$**.

#### B. Spectral Tilt (The Spectral Slope)

Spectral tilt describes how rapidly energy decays as frequency increases.

* Glottal flow naturally rolls off at approximately **$-12\text{ dB/octave}$**.
* Voiced vowels show a strong downward tilt (high energy at low frequencies, low energy at high frequencies).
* Unvoiced consonants (`/s/`, `/t/` bursts) exhibit a flat or positive tilt (equal or greater high-frequency energy).

It is calculated by fitting a linear regression line across the log-power spectrum vs. frequency bins:


$$\text{Slope} = \frac{K \sum (k \cdot \log \vert{}X[k]\vert{}^2) - (\sum k)(\sum \log \vert{}X[k]\vert{}^2)}{K \sum k^2 - (\sum k)^2}$$

---

### 4. Implementation: Extracting Pitch & Centroid in C++

Here is how you can use the power spectrum from your `FastFFT` instance in `main()` to compute both the spectral centroid and cepstral pitch:

```cpp
#include <vector>
#include <cmath>
#include <algorithm>

struct SpectralFeatures {
    float centroid_hz;
    float pitch_f0_hz; // 0.0 if unvoiced
    bool is_voiced;
};

class SpectralAnalyzer {
public:
    // 1. Spectral Centroid directly from Power Spectrum
    static float compute_centroid(const std::vector<float>& power_spec, float sample_rate = 16000.0f, int n_fft = 512) {
        float weighted_sum = 0.0f;
        float total_energy = 0.0f;
        float bin_hz = sample_rate / n_fft;

        for (size_t k = 0; k < power_spec.size(); ++k) {
            float freq = k * bin_hz;
            weighted_sum += freq * power_spec[k];
            total_energy += power_spec[k];
        }

        return (total_energy > 1e-7f) ? (weighted_sum / total_energy) : 0.0f;
    }

    // 2. Fundamental Pitch (F0) via Cepstrum peak picking
    static float compute_pitch_cepstrum(const std::vector<float>& power_spec, float sample_rate = 16000.0f) {
        size_t K = power_spec.size();
        std::vector<float> log_power(K);

        for (size_t k = 0; k < K; ++k) {
            // Prevent log(0)
            log_power[k] = std::log(std::max(power_spec[k], 1e-7f));
        }

        // Compute IFFT / IDCT of log_power
        // Discrete Cosine Transform (DCT-II) serves as a fast real-to-real cepstrum approximation
        size_t N = (K - 1) * 2;
        std::vector<float> cepstrum(K, 0.0f);
        
        for (size_t n = 0; n < K; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += log_power[k] * std::cos(M_PI * n * (k + 0.5f) / K);
            }
            cepstrum[n] = sum;
        }

        // Search for peak within reasonable human vocal cord range:
        // 70 Hz to 400 Hz corresponds to lag indices:
        size_t min_lag = static_cast<size_t>(sample_rate / 400.0f); // ~40 samples
        size_t max_lag = static_cast<size_t>(sample_rate / 70.0f);  // ~228 samples
        max_lag = std::min(max_lag, K - 1);

        size_t best_lag = 0;
        float max_val = -1e9f;

        for (size_t lag = min_lag; lag <= max_lag; ++lag) {
            if (cepstrum[lag] > max_val) {
                max_val = cepstrum[lag];
                best_lag = lag;
            }
        }

        // If the cepstral peak is prominent relative to neighbors, it is voiced
        if (best_lag > 0 && max_val > 0.0f) {
            return sample_rate / static_cast<float>(best_lag);
        }

        return 0.0f; // Unvoiced frame
    }
};

```

### How to Connect This into Your Current Pipeline

In your processing loop:

1. Pass `power_spectrum` into `SpectralAnalyzer::compute_centroid()`. If the centroid is $> 3200\text{ Hz}$, you can flag the frame as a **consonant / fricative** (complementing or replacing ZCR).
2. Pass `power_spectrum` into `SpectralAnalyzer::compute_pitch_cepstrum()`. If `pitch_f0_hz > 0`, the vocal cords are vibrating—confirming the presence of a **true voiced vowel** before querying your `FormantVectorDB`.

You are computing the FFT power spectrum without using it because your current pipeline relies on **Linear Predictive Coding (LPC) in the time domain**, while the **FFT power spectrum represents an alternate, frequency-domain approach** to speech analysis.

Lines 74–75 explicitly discard the output:

```cpp
const std::vector<float> power_spectrum = fast_fft.compute_power_spectrum(frame);
(void)power_spectrum;

```

---

### Why the Two Methods Are Doing Parallel Work

Speech analysis decomposes audio into the **source-filter model**: the vocal cords generate a harmonic excitation signal (the source), and the vocal tract shapes it with resonant envelope peaks called formants (the filter).

* **LPC (`formant_tracker.extract_formants`) operates in the time domain:**
It calculates time-lag autocorrelations and uses Levinson-Durbin recursion with polynomial root finding. It bypasses Fourier transforms completely to estimate vocal-tract poles directly.
* **FFT (`fast_fft.compute_power_spectrum`) operates in the frequency domain:**
It computes the raw spectrum $\vert{}X(f)\vert{}^2$, containing both fine harmonic spikes (glottal pitch) and broader spectral peaks (formants) combined together.

Because your code sends the raw frame directly to `formant_tracker`, the FFT output is discarded.

---

### How to Use the Power Spectrum

If you want to use the FFT computation rather than deleting it, integrate it via one of these techniques:

#### 1. Mel-Frequency Spectral Coefficients (MFCCs / Filterbanks)

Instead of matching formants against a geometric table, standard Automated Speech Recognition (ASR) projects the FFT power spectrum directly onto a triangular Mel filterbank matrix:

$$\text{Mel Energy}_m = \sum_{k} \vert{}X[k]\vert{}^2 \cdot H_m[k]$$

Taking the log of those bin energies yields standard speech recognition feature vectors.

#### 2. Harmonic Pitch ($F_0$) Tracking via Cepstrum

To determine whether a frame is voiced or unvoiced, or to find pitch ($F_0$), perform an inverse transform on the log power spectrum to compute the **Real Cepstrum**:

$$c[n] = \text{IFFT}\left(\log\left(\vert{}X(f)\vert{}^2\right)\right)$$

The peak in the cepstral "quefrency" domain provides the fundamental vocal cord pitch period.

#### 3. Spectral Tilt / Spectral Centroid for Consonant Detection

Your current consonant gating uses Zero Crossing Rate (ZCR). Using the power spectrum, you can compute the **Spectral Centroid** directly to detect fricatives (`/s/`, `/f/`, `/θ/`):

$$\text{Centroid} = \frac{\sum_{k} f_k \cdot \vert{}X[k]\vert{}^2}{\sum_{k} \vert{}X[k]\vert{}^2}$$

Fricative consonants have centroids $>3500\text{ Hz}$, whereas voiced vowels remain concentrated below $2000\text{ Hz}$.

---

### Immediate Code Cleanup

If your goal is to keep the pipeline strictly focused on LPC formant tracking, remove `fast_fft` from the loop to eliminate unnecessary execution overhead:

```cpp
// Remove these lines entirely:
// const std::vector<float> power_spectrum = fast_fft.compute_power_spectrum(frame);
// (void)power_spectrum;

```