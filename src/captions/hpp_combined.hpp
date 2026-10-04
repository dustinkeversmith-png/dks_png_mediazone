// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\benchmarks\scoring.hpp ===
// Evaluation utilities: Levenshtein alignment, word/phone error rate and a
// real-time-factor timer. Kept separate from the models so the benchmark can
// score the new decoder and the old frame-level baselines with one implementation.
#pragma once

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace models {

struct EditCounts {
    int64_t substitutions = 0;
    int64_t deletions = 0;
    int64_t insertions = 0;
    int64_t reference_length = 0;

    int64_t errors() const { return substitutions + deletions + insertions; }
    double error_rate() const {
        return reference_length == 0 ? 0.0
                                     : static_cast<double>(errors()) / reference_length;
    }
    void operator+=(const EditCounts& other) {
        substitutions += other.substitutions;
        deletions += other.deletions;
        insertions += other.insertions;
        reference_length += other.reference_length;
    }
};

// Levenshtein with backtrace-free counting of S/D/I (two-row DP).
template <typename T>
EditCounts edit_distance(const std::vector<T>& reference, const std::vector<T>& hypothesis) {
    const size_t n = reference.size();
    const size_t m = hypothesis.size();

    struct Cell {
        int32_t cost = 0, sub = 0, del = 0, ins = 0;
    };
    std::vector<Cell> previous(m + 1), current(m + 1);
    for (size_t j = 0; j <= m; ++j) {
        previous[j] = {static_cast<int32_t>(j), 0, 0, static_cast<int32_t>(j)};
    }

    for (size_t i = 1; i <= n; ++i) {
        current[0] = {static_cast<int32_t>(i), 0, static_cast<int32_t>(i), 0};
        for (size_t j = 1; j <= m; ++j) {
            const bool match = reference[i - 1] == hypothesis[j - 1];
            Cell best = previous[j - 1];
            best.cost += match ? 0 : 1;
            if (!match) ++best.sub;

            Cell deletion = previous[j];
            deletion.cost += 1;
            ++deletion.del;
            if (deletion.cost < best.cost) best = deletion;

            Cell insertion = current[j - 1];
            insertion.cost += 1;
            ++insertion.ins;
            if (insertion.cost < best.cost) best = insertion;

            current[j] = best;
        }
        previous.swap(current);
    }

    EditCounts counts;
    counts.substitutions = previous[m].sub;
    counts.deletions = previous[m].del;
    counts.insertions = previous[m].ins;
    counts.reference_length = static_cast<int64_t>(n);
    return counts;
}

// Uppercase whitespace tokenisation, stripping punctuation that LibriSpeech
// transcripts do not contain but CMUDict spellings would trip over.
inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string token;
    while (stream >> token) {
        std::string cleaned;
        for (char c : token) {
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '\'') {
                cleaned.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
            }
        }
        if (!cleaned.empty()) tokens.push_back(cleaned);
    }
    return tokens;
}

// Wall-clock timer that also reports the real-time factor of processed audio.
class RtfTimer {
public:
    void start() { start_ = std::chrono::steady_clock::now(); }
    double stop() {
        const auto end = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(end - start_).count();
        elapsed_ += seconds;
        return seconds;
    }
    void add_audio(double seconds) { audio_ += seconds; }

    double elapsed() const { return elapsed_; }
    double audio() const { return audio_; }
    double rtf() const { return audio_ > 0.0 ? elapsed_ / audio_ : 0.0; }
    double times_real_time() const { return elapsed_ > 0.0 ? audio_ / elapsed_ : 0.0; }

private:
    std::chrono::steady_clock::time_point start_;
    double elapsed_ = 0.0;
    double audio_ = 0.0;
};

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\filter\mfcc.hpp ===
// Dense spectral front-end: log-Mel filterbank -> MFCC + delta + delta-delta.
//
// This replaces the formant coordinates (LPC method) and centroid/tilt triple
// (Fourier method) used by the frame-level vowel classifiers. Those features
// only exist during voiced phonation; a Mel filterbank encodes vowels,
// fricative noise and plosive bursts in one fixed-width vector, which is what
// every downstream model here (DTW, Gaussian acoustic model, Viterbi decoder)
// consumes.
//
// Layout: features are stored row-major in one flat vector (T frames x D dims)
// so a whole utterance is one contiguous allocation and frame access is a
// pointer offset - the decoder touches this array once per frame per active
// state, so cache locality matters more than convenience here.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <power_spectrum/fast_fft.hpp>

namespace models {

constexpr int kSampleRate = 16000;
constexpr int kFrameLength = 400;   // 25 ms
constexpr int kHopSize = 160;       // 10 ms
constexpr int kFftSize = 512;
constexpr int kNumMelFilters = 26;
constexpr int kNumCepstra = 13;     // c0..c12
constexpr int kFeatureDim = kNumCepstra * 3;  // + delta + delta-delta = 39

// Feature matrix: T frames of kFeatureDim floats, row-major and contiguous.
struct FeatureMatrix {
    std::vector<float> data;
    int num_frames = 0;
    int dim = kFeatureDim;

    const float* frame(int t) const { return data.data() + static_cast<size_t>(t) * dim; }
    float* frame(int t) { return data.data() + static_cast<size_t>(t) * dim; }
    bool empty() const { return num_frames == 0; }

    void resize(int frames, int dimension = kFeatureDim) {
        num_frames = frames;
        dim = dimension;
        data.assign(static_cast<size_t>(frames) * dimension, 0.0f);
    }
};

inline float hz_to_mel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
inline float mel_to_hz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

// Triangular Mel filters stored sparsely (first bin + weights) so the
// filterbank pass is O(non-zero bins) rather than O(filters * spectrum).
class MelFilterbank {
public:
    explicit MelFilterbank(int num_filters = kNumMelFilters,
                           int fft_size = kFftSize,
                           int sample_rate = kSampleRate,
                           float low_hz = 20.0f,
                           float high_hz = 7800.0f)
        : num_filters_(num_filters), num_bins_(fft_size / 2 + 1) {
        filters_.resize(num_filters_);

        const float low_mel = hz_to_mel(low_hz);
        const float high_mel = hz_to_mel(high_hz);
        std::vector<float> centers(num_filters_ + 2);
        for (int i = 0; i < num_filters_ + 2; ++i) {
            const float mel = low_mel + (high_mel - low_mel) * i / (num_filters_ + 1);
            centers[i] = mel_to_hz(mel);
        }

        const float bin_hz = static_cast<float>(sample_rate) / fft_size;
        for (int f = 0; f < num_filters_; ++f) {
            const float left = centers[f];
            const float center = centers[f + 1];
            const float right = centers[f + 2];

            Filter& filter = filters_[f];
            filter.first_bin = -1;
            for (int bin = 0; bin < num_bins_; ++bin) {
                const float hz = bin * bin_hz;
                float weight = 0.0f;
                if (hz > left && hz < center) {
                    weight = (hz - left) / (center - left);
                } else if (hz >= center && hz < right) {
                    weight = (right - hz) / (right - center);
                }
                if (weight > 0.0f) {
                    if (filter.first_bin < 0) filter.first_bin = bin;
                    filter.weights.push_back(weight);
                }
            }
            if (filter.first_bin < 0) filter.first_bin = 0;
        }
    }

    // Accumulates one frame's power spectrum into num_filters log-energies.
    void apply(const std::vector<float>& power_spectrum, std::vector<float>& out) const {
        out.resize(num_filters_);
        for (int f = 0; f < num_filters_; ++f) {
            const Filter& filter = filters_[f];
            float sum = 0.0f;
            for (size_t i = 0; i < filter.weights.size(); ++i) {
                const int bin = filter.first_bin + static_cast<int>(i);
                if (bin < static_cast<int>(power_spectrum.size())) {
                    sum += power_spectrum[bin] * filter.weights[i];
                }
            }
            out[f] = std::log(sum > 1e-10f ? sum : 1e-10f);
        }
    }

    int size() const { return num_filters_; }

private:
    struct Filter {
        int first_bin = 0;
        std::vector<float> weights;
    };
    int num_filters_;
    int num_bins_;
    std::vector<Filter> filters_;
};

// MFCC extractor. One instance owns an FFT plan and scratch buffers, so reuse
// it across an entire corpus rather than constructing per utterance.
class MfccExtractor {
public:
    MfccExtractor() : fft_(kFftSize), filterbank_() {
        // DCT-II basis, precomputed: cepstra = dct_ * log_mel.
        dct_.resize(static_cast<size_t>(kNumCepstra) * kNumMelFilters);
        const float scale = std::sqrt(2.0f / kNumMelFilters);
        for (int k = 0; k < kNumCepstra; ++k) {
            for (int m = 0; m < kNumMelFilters; ++m) {
                dct_[static_cast<size_t>(k) * kNumMelFilters + m] =
                    scale * std::cos(static_cast<float>(M_PI_F) * k * (m + 0.5f) / kNumMelFilters);
            }
        }
        // Cepstral liftering keeps higher quefrency terms on a comparable scale.
        lifter_.resize(kNumCepstra);
        constexpr float kLifter = 22.0f;
        for (int k = 0; k < kNumCepstra; ++k) {
            lifter_[k] = 1.0f + 0.5f * kLifter * std::sin(static_cast<float>(M_PI_F) * k / kLifter);
        }
        window_ = make_hamming(kFrameLength);
    }

    // Full front-end: pre-emphasis -> framing -> MFCC -> deltas -> CMVN.
    FeatureMatrix extract(const std::vector<float>& audio) {
        FeatureMatrix features;
        if (static_cast<int>(audio.size()) < kFrameLength) return features;

        // Pre-emphasis (in scratch, so the caller's buffer is untouched).
        emphasized_.resize(audio.size());
        emphasized_[0] = audio[0];
        constexpr float kAlpha = 0.97f;
        for (size_t i = 1; i < audio.size(); ++i) {
            emphasized_[i] = audio[i] - kAlpha * audio[i - 1];
        }

        const int num_frames =
            static_cast<int>((emphasized_.size() - kFrameLength) / kHopSize) + 1;
        features.resize(num_frames);

        frame_buffer_.assign(kFftSize, 0.0f);
        for (int t = 0; t < num_frames; ++t) {
            const size_t start = static_cast<size_t>(t) * kHopSize;
            for (int i = 0; i < kFrameLength; ++i) {
                frame_buffer_[i] = emphasized_[start + i] * window_[i];
            }
            const std::vector<float>& power = fft_.compute_power_spectrum(frame_buffer_);
            filterbank_.apply(power, log_mel_);

            float* out = features.frame(t);
            for (int k = 0; k < kNumCepstra; ++k) {
                const float* basis = dct_.data() + static_cast<size_t>(k) * kNumMelFilters;
                float sum = 0.0f;
                for (int m = 0; m < kNumMelFilters; ++m) sum += basis[m] * log_mel_[m];
                out[k] = sum * lifter_[k];
            }
        }

        add_derivatives(features);
        apply_cmvn(features);
        return features;
    }

    // Regression deltas over +/-2 frames, written into dims [13,26) and [26,39).
    static void add_derivatives(FeatureMatrix& features) {
        const int T = features.num_frames;
        if (T == 0) return;
        constexpr int kWindow = 2;
        constexpr float kNorm = 2.0f * (1 * 1 + 2 * 2);  // 2 * sum(n^2)

        for (int pass = 0; pass < 2; ++pass) {
            const int src = pass == 0 ? 0 : kNumCepstra;
            const int dst = pass == 0 ? kNumCepstra : 2 * kNumCepstra;
            for (int t = 0; t < T; ++t) {
                float* out = features.frame(t) + dst;
                for (int d = 0; d < kNumCepstra; ++d) {
                    float sum = 0.0f;
                    for (int n = 1; n <= kWindow; ++n) {
                        const int ahead = std::min(t + n, T - 1);
                        const int behind = std::max(t - n, 0);
                        sum += n * (features.frame(ahead)[src + d] -
                                    features.frame(behind)[src + d]);
                    }
                    out[d] = sum / kNorm;
                }
            }
        }
    }

    // Per-utterance cepstral mean and variance normalisation. This is what
    // makes a model trained on TIMIT usable on LibriSpeech: it removes the
    // channel/recording offset that would otherwise dominate the Gaussians.
    static void apply_cmvn(FeatureMatrix& features) {
        const int T = features.num_frames;
        if (T < 2) return;
        const int D = features.dim;

        std::vector<double> mean(D, 0.0), variance(D, 0.0);
        for (int t = 0; t < T; ++t) {
            const float* row = features.frame(t);
            for (int d = 0; d < D; ++d) mean[d] += row[d];
        }
        for (int d = 0; d < D; ++d) mean[d] /= T;
        for (int t = 0; t < T; ++t) {
            const float* row = features.frame(t);
            for (int d = 0; d < D; ++d) {
                const double diff = row[d] - mean[d];
                variance[d] += diff * diff;
            }
        }
        for (int d = 0; d < D; ++d) {
            variance[d] = std::sqrt(variance[d] / T);
            if (variance[d] < 1e-5) variance[d] = 1.0;
        }
        for (int t = 0; t < T; ++t) {
            float* row = features.frame(t);
            for (int d = 0; d < D; ++d) {
                row[d] = static_cast<float>((row[d] - mean[d]) / variance[d]);
            }
        }
    }

private:
    static constexpr float M_PI_F = 3.14159265358979323846f;

    static std::vector<float> make_hamming(int length) {
        std::vector<float> window(length);
        for (int n = 0; n < length; ++n) {
            window[n] = 0.54f - 0.46f * std::cos(2.0f * M_PI_F * n / (length - 1));
        }
        return window;
    }

    FastFFT fft_;
    MelFilterbank filterbank_;
    std::vector<float> dct_;
    std::vector<float> lifter_;
    std::vector<float> window_;
    std::vector<float> emphasized_;
    std::vector<float> frame_buffer_;
    std::vector<float> log_mel_;
};

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\filter\pre_emphasis_filter.hpp ===
// 1. **Pre-emphasis Filter:** Apply a first-order high-pass filter ($y[n] = x[n] - \alpha x[n-1]$, where $\alpha \approx 0.95 - 0.97$) to compensate for the $-6\text{ dB/octave}$ glottal roll-off.
#pragma once
#include <vector>

inline std::vector<float> pre_emphasis_filter(const std::vector<float>& buffer)
{
    if (buffer.empty()) return {};

    std::vector<float> out(buffer.size());
    const float alpha = 0.95f;

    // The first sample has no x[n-1], so we keep it as is or use a historical sample
    out[0] = buffer[0]; 

    // Start loop from index 1 to avoid buffer[-1] out-of-bounds crash
    for (size_t i = 1; i < buffer.size(); i++)
    {
        out[i] = buffer[i] - (alpha * buffer[i - 1]); // Multiplied by alpha
    }

    return out; // Added missing return statement
}

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\formants\find_formants.hpp ===
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

constexpr float kFormantPi = 3.14159265358979323846f;

struct Formant {
    float frequency; // in Hz
    float bandwidth; // in Hz
};

class FormantTracker {
public:
    // Levinson-Durbin Recursion
    static Eigen::VectorXf levinson_durbin(const Eigen::VectorXf& r, int order) {
        Eigen::VectorXf a(order + 1);
        a.setZero();
        a(0) = 1.0f;

        float e = r(0);
        if (e <= 0.0f) return a; // Avoid division by zero on silence

        for (int i = 1; i <= order; ++i) {
            float lambda = 0.0f;
            for (int j = 1; j < i; ++j) {
                lambda += a(j) * r(i - j);
            }
            lambda = (r(i) - lambda) / e;

            Eigen::VectorXf a_prev = a;
            a(i) = lambda;
            for (int j = 1; j < i; ++j) {
                a(j) = a_prev(j) - lambda * a_prev(i - j);
            }
            e *= (1.0f - lambda * lambda);
        }
        return a;
    }

    // Extract F1, F2, F3, etc.
    static std::vector<Formant> extract_formants(const std::vector<float>& frame, 
                                                 float sample_rate = 16000.0f, 
                                                 int order = 16) {
        int N = frame.size();

        std::vector<Formant> formants;
        
        

        

        // 1. Autocorrelation
        Eigen::VectorXf r(order + 1);
        r.setZero();
        for (int k = 0; k <= order; ++k) {
            for (int n = 0; n < N - k; ++n) {
                r(k) += frame[n] * frame[n + k];
            }
        }

        // 2. Levinson-Durbin
        Eigen::VectorXf a = levinson_durbin(r, order);

        // 3. Build Companion Matrix for Root Finding
        // Polynomial: z^p - a_1*z^(p-1) - a_2*z^(p-2) ... - a_p = 0
        Eigen::MatrixXf companion = Eigen::MatrixXf::Zero(order, order);
        for (int col = 0; col < order; ++col) {
            companion(0, col) = a(col + 1);
        }
        for (int row = 1; row < order; ++row) {
            companion(row, row - 1) = 1.0f;
        }

        // 4. Compute Eigenvalues
        Eigen::EigenSolver<Eigen::MatrixXf> solver(companion, /* computeEigenvectors = */ false);
        auto roots = solver.eigenvalues();

        // 5. Convert Roots to Formant Frequencies & Bandwidths
        
        for (int i = 0; i < roots.size(); ++i) {
            std::complex<float> z = roots(i);
            
            // Look only at the upper half of the unit circle (positive frequencies)
            if (z.imag() > 0.0f) {
                float freq = (std::atan2(z.imag(), z.real()) * sample_rate) / (2.0f * kFormantPi);
                float bw = -(sample_rate / kFormantPi) * std::log(std::abs(z));

                // Standard speech filters: Bandwidth should be sharp (< 400Hz) and Freq in range
                if (freq > 50.0f && freq < (sample_rate / 2.0f - 50.0f) && bw < 400.0f) {
                    formants.push_back({freq, bw});
                }
            }
        }

        // Sort ascending by frequency (F1 < F2 < F3)
        std::sort(formants.begin(), formants.end(), [](const Formant& a, const Formant& b) {
            return a.frequency < b.frequency;
        });

        // LPC can merge the two low back-vowel poles and expose F3 as F2.
        if (formants.size() >= 2 &&
            formants[0].frequency < 650.0f &&
            formants[1].frequency > 1850.0f) {
            const float estimated_f2 = formants[0].frequency * 2.2f;
            if (estimated_f2 > formants[0].frequency && estimated_f2 < 1200.0f) {
                formants.insert(formants.begin() + 1,
                                {estimated_f2, formants[0].bandwidth});
            }
        }

        return formants;
    }
};

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\formants\formant_to_vowel.hpp ===
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

struct FormantEntry {
    std::string key;          // "IY", "AE", etc.
    std::string ipa;          // "i", "æ"
    std::string example;      // "beet", "bat"
    float f1;                 // Target F1 in Hz
    float f2;                 // Target F2 in Hz
    float f3;                 // Target F3 in Hz
    float tolerance;          // Radius / Max confidence boundary
};

// Convert Linear Hz to Bark Scale (Traunmüller formula) for perceptual distance
inline float hz_to_bark(float f) {
    return (26.81f * f) / (1960.0f + f) - 0.53f;
}

class FormantVectorDB {
private:
    std::vector<FormantEntry> db;

public:
    FormantVectorDB() {
        // Initialize the vector dataset
        db = {
            // Front Vowels
            {"IY", "i",  "beet",   270.0f, 2290.0f, 3010.0f, 80.0f},
            {"IH", "ɪ",  "bit",    390.0f, 1990.0f, 2550.0f, 90.0f},
            {"EY", "eɪ", "bait",   530.0f, 1840.0f, 2480.0f, 90.0f},
            {"EH", "ɛ",  "bet",    660.0f, 1720.0f, 2410.0f, 100.0f},
            {"AE", "æ",  "bat",    730.0f, 1090.0f, 2440.0f, 110.0f},

            // Central Vowels
            {"AH", "ʌ",  "butt",   640.0f, 1190.0f, 2390.0f, 100.0f},
            {"ER", "ɝ",  "bird",   490.0f, 1350.0f, 1690.0f, 90.0f}, // Low F3
            {"AX", "ə",  "about",  500.0f, 1500.0f, 2500.0f, 100.0f},

            // Back Vowels
            {"AA", "ɑ",  "father", 730.0f, 1090.0f, 2440.0f, 110.0f},
            {"AO", "ɔ",  "bought", 570.0f,  840.0f, 2410.0f, 100.0f},
            {"OW", "oʊ", "boat",   500.0f, 1000.0f, 2350.0f, 90.0f},
            {"UH", "ʊ",  "book",   440.0f, 1020.0f, 2240.0f, 90.0f},
            {"UW", "u",  "boot",         300.0f,  870.0f, 2240.0f, 80.0f},
            {"UW", "u",  "two (fronted)", 360.0f, 1300.0f, 2200.0f, 110.0f}
        };
    }

    struct MatchResult {
        std::string key;
        std::string ipa;
        float distance;
        bool in_bounds;
    };

    // Finds the closest phoneme key using Bark-scaled acoustic distance
    MatchResult find_nearest_phoneme(float measured_f1, float measured_f2, float measured_f3) const {
        if (measured_f1 <= 0.0f || measured_f2 <= 0.0f) {
            return {"SIL", "", 0.0f, false}; // Silence / Invalid frame
        }

        float min_dist = 1.0e30f;
        const FormantEntry* best_match = nullptr;

        // Convert inputs to Bark scale
        float b1 = hz_to_bark(measured_f1);
        float b2 = hz_to_bark(measured_f2);
        float b3 = (measured_f3 > 0.0f) ? hz_to_bark(measured_f3) : 0.0f;

        for (const auto& entry : db) {
            float target_b1 = hz_to_bark(entry.f1);
            float target_b2 = hz_to_bark(entry.f2);
            float target_b3 = hz_to_bark(entry.f3);

            // Perceptual Bark distance: F1 and F2 drive the vowel identity most strongly
            float d1 = b1 - target_b1;
            float d2 = b2 - target_b2;
            
            // F3 weighting is reduced unless it's rhotic (/ɝ/)
            float d3 = (measured_f3 > 0.0f) ? (b3 - target_b3) * 0.35f : 0.0f;

            // Weighted Bark Distance
            float distance = std::sqrt(d1 * d1 * 1.5f + d2 * d2 * 1.0f + d3 * d3);

            if (distance < min_dist) {
                min_dist = distance;
                best_match = &entry;
            }
        }

        if (best_match) {
            // Check if within acceptable tolerance bounds in standard Hz space
            float f1_err = std::abs(measured_f1 - best_match->f1);
            float f2_err = std::abs(measured_f2 - best_match->f2);
            bool within_bounds = (f1_err <= best_match->tolerance * 1.5f) && 
                                 (f2_err <= best_match->tolerance * 2.0f);

            return {best_match->key, best_match->ipa, min_dist, within_bounds};
        }

        return {"UNK", "", min_dist, false};
    }
};

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\framing\audio_framing.hpp ===
#pragma once
#include <cmath>
#include <vector>

constexpr float kPi = 3.14159265358979323846f;

// Precompute Hamming Window coefficients: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N - 1))
std::vector<float> create_hamming_window(int frame_length) {
    std::vector<float> window(frame_length);
    for (int n = 0; n < frame_length; ++n) {
        window[n] = 0.54f - 0.46f * std::cos((2.0f * kPi * n) / (frame_length - 1));
    }
    return window;
}

// Slice the raw audio stream into windowed frames ready for FFT / LPC
//Frame Length ($N$): $25\text{ ms} \rightarrow 0.025 \times 16000 = \mathbf{400\text{ samples}}$ (padded with zeros to $512$ for power-of-two FFTs).Frame Step / Hop Size ($M$): $10\text{ ms} \rightarrow 0.010 \times 16000 = \mathbf{160\text{ samples}}$ (yields a 60% overlap).Windowing Function: Multiply by a Hamming or Hanning window before spectral analysis to eliminate spectral leakage at the frame edges.
std::vector<std::vector<float>> chop_into_frames(const std::vector<float>& audio, 
                                                 int frame_length = 400, 
                                                 int hop_size = 160, 
                                                 int fft_size = 512) {
    std::vector<float> window = create_hamming_window(frame_length);
    std::vector<std::vector<float>> frames;

    for (size_t start = 0; start + frame_length <= audio.size(); start += hop_size) {
        std::vector<float> frame(fft_size, 0.0f); // Zero-padded buffer for FFT
        
        for (int i = 0; i < frame_length; ++i) {
            frame[i] = audio[start + i] * window[i];
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\input\audio_loadnorm.hpp ===
#pragma once
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"


/**
 * Loads an audio file, converts it to 32-bit float, mixes to Mono, 
 * and resamples directly to 16 kHz.
 * * @param filepath Path to the input audio file (.wav, .mp3, .flac)
 * @param out_samples Target vector to store contiguous normalized float samples
 * @return true if successful, false otherwise
 */
bool load_and_preprocess_audio(const std::string& filepath, std::vector<float>& out_samples) {

    
    ma_decoder_config config = ma_decoder_config_init(
        ma_format_f32,   // Target format: 32-bit Float [-1.0, 1.0]
        1,               // Target channels: 1 (Mono)
        16000            // Target sample rate: 16 kHz
    );

    ma_decoder decoder;
    ma_result result = ma_decoder_init_file(filepath.c_str(), &config, &decoder);
    if (result != MA_SUCCESS) {
        std::cerr << "[Error] Failed to initialize decoder for file: " << filepath << "\n";
        return false;
    }

    // Retrieve total frame count at target 16 kHz
    ma_uint64 total_frames;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);
    
    if (result == MA_SUCCESS && total_frames > 0) {
        out_samples.resize(total_frames);
        ma_uint64 frames_read = 0;
        
        result = ma_decoder_read_pcm_frames(&decoder, out_samples.data(), total_frames, &frames_read);
        if (result != MA_SUCCESS) {
            std::cerr << "[Error] Failed reading PCM frames\n";
            ma_decoder_uninit(&decoder);
            return false;
        }
        
        // Truncate to exact frames read in case of stream differences
        out_samples.resize(frames_read);
    } else {
        // Fallback dynamic reading if length cannot be pre-calculated
        constexpr size_t CHUNK_SIZE = 4096;
        float chunk[CHUNK_SIZE];
        ma_uint64 frames_read_chunk = 0;
        
        while (ma_decoder_read_pcm_frames(&decoder, chunk, CHUNK_SIZE, &frames_read_chunk) == MA_SUCCESS 
               && frames_read_chunk > 0) {
            out_samples.insert(out_samples.end(), chunk, chunk + frames_read_chunk);
        }
    }



    ma_decoder_uninit(&decoder);
    
    
    return true;
}

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\input\wav_reader.hpp ===
// Minimal RIFF/WAVE reader for the caption engines.
//
// Why this exists rather than reusing input/audio_loadnorm.hpp: the streaming
// caption app compiles miniaudio with MA_NO_DECODING (it only wants the capture
// backend for --mic), so miniaudio cannot decode files in that translation
// unit. And why not reuse the TTS library's vocal::load_wav_mono: captions must
// not link the synthesis library just to open a file.
//
// Scope is deliberately narrow - PCM16 or float32 RIFF, any channel count,
// linear resampling to the target rate. That covers the evaluation corpora
// (16 kHz mono PCM16) and microphone dumps. For anything else (mp3, flac), use
// input/audio_loadnorm.hpp, which wraps a full miniaudio decoder.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace captions {

struct Wav {
    int sample_rate_hz = 16000;
    std::vector<float> samples;  // mono, normalized to [-1, 1]
};

namespace detail {

inline std::uint16_t read_u16(std::istream& in) {
    unsigned char b[2]{};
    if (!in.read(reinterpret_cast<char*>(b), 2)) throw std::runtime_error("truncated WAV");
    return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
}

inline std::uint32_t read_u32(std::istream& in) {
    unsigned char b[4]{};
    if (!in.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("truncated WAV");
    return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

}  // namespace detail

// Reads `path` as mono at `target_sample_rate_hz`. Throws on anything it cannot
// represent exactly rather than silently returning wrong-rate audio - a caption
// benchmark that quietly decodes at the wrong rate reports a plausible but
// meaningless WER.
[[nodiscard]] inline Wav read_wav_mono(const std::filesystem::path& path,
                                       int target_sample_rate_hz = 16000) {
    if (target_sample_rate_hz <= 0) throw std::invalid_argument("target sample rate must be positive");

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open WAV: " + path.string());

    char riff[4]{}, wave[4]{};
    in.read(riff, 4);
    (void)detail::read_u32(in);
    in.read(wave, 4);
    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("not a RIFF/WAVE file: " + path.string());
    }

    std::uint16_t format = 0, channels = 0, bits = 0;
    std::uint32_t sample_rate = 0;
    std::vector<unsigned char> data;
    while (in && (format == 0 || data.empty())) {
        char id[4]{};
        if (!in.read(id, 4)) break;
        const auto size = detail::read_u32(in);
        const auto chunk_start = in.tellg();
        if (std::string(id, 4) == "fmt ") {
            format = detail::read_u16(in);
            channels = detail::read_u16(in);
            sample_rate = detail::read_u32(in);
            (void)detail::read_u32(in);
            (void)detail::read_u16(in);
            bits = detail::read_u16(in);
        } else if (std::string(id, 4) == "data") {
            data.resize(size);
            if (!in.read(reinterpret_cast<char*>(data.data()), size)) {
                throw std::runtime_error("truncated WAV data: " + path.string());
            }
        }
        in.clear();
        in.seekg(chunk_start + static_cast<std::streamoff>(size + (size & 1U)));
    }

    if (channels == 0 || sample_rate == 0 || data.empty() ||
        !((format == 1 && bits == 16) || (format == 3 && bits == 32))) {
        throw std::runtime_error("WAV must be PCM16 or float32: " + path.string());
    }

    const std::size_t bytes_per_sample = bits / 8;
    const std::size_t frames = data.size() / (bytes_per_sample * channels);
    std::vector<float> mono(frames);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double sum = 0.0;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const auto offset = (frame * channels + channel) * bytes_per_sample;
            if (format == 1) {
                const auto raw = static_cast<std::uint16_t>(data[offset] | (data[offset + 1] << 8));
                sum += static_cast<std::int16_t>(raw) / 32768.0;
            } else {
                float sample{};
                std::memcpy(&sample, data.data() + offset, sizeof(float));
                sum += sample;
            }
        }
        mono[frame] = static_cast<float>(sum / channels);
    }

    if (static_cast<int>(sample_rate) == target_sample_rate_hz) {
        return {target_sample_rate_hz, std::move(mono)};
    }

    const double ratio = static_cast<double>(target_sample_rate_hz) / sample_rate;
    std::vector<float> resampled(static_cast<std::size_t>(std::floor(mono.size() * ratio)));
    for (std::size_t i = 0; i < resampled.size(); ++i) {
        const double source = static_cast<double>(i) / ratio;
        const auto left = std::min(static_cast<std::size_t>(source), mono.size() - 1);
        const auto right = std::min(left + 1, mono.size() - 1);
        const double fraction = source - static_cast<double>(left);
        resampled[i] = static_cast<float>(mono[left] * (1.0 - fraction) + mono[right] * fraction);
    }
    return {target_sample_rate_hz, std::move(resampled)};
}

}  // namespace captions

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\int8_zip\ort_graph.hpp ===
#pragma once
// ONNX session wrapper and filterbank settings shared by the streaming and
// batch Zipformer engines, so both read the same graph contract.
#include "ort_session_options.hpp"
#include <kaldi-native-fbank/csrc/online-feature.h>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace captions::detail {
struct Graph {
    Ort::Session session;
    std::vector<std::string> inputs, outputs;
    std::vector<const char*> in_names, out_names;
    Graph(Ort::Env& env, const std::filesystem::path& path, int threads, bool spin = false)
        : session(env, path.c_str(), cpu_session_options(threads, spin)) {
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session.GetInputCount(); ++i)
            inputs.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        for (size_t i = 0; i < session.GetOutputCount(); ++i)
            outputs.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        for (const auto& s : inputs) in_names.push_back(s.c_str());
        for (const auto& s : outputs) out_names.push_back(s.c_str());
    }
    std::string metadata(const char* key) const {
        Ort::AllocatorWithDefaultOptions allocator;
        auto value = session.GetModelMetadata().LookupCustomMetadataMapAllocated(key, allocator);
        if (!value) throw std::runtime_error(std::string("Missing ONNX metadata: ") + key);
        return value.get();
    }
    std::vector<Ort::Value> run(std::span<const Ort::Value> in) {
        return session.Run(Ort::RunOptions{nullptr}, in_names.data(), in.data(), in.size(),
                           out_names.data(), out_names.size());
    }
};

inline knf::FbankOptions fbank_options() {
    knf::FbankOptions o;
    o.frame_opts.samp_freq = 16000;
    o.frame_opts.dither = 0;
    o.frame_opts.snip_edges = false;
    o.mel_opts.num_bins = 80;
    o.mel_opts.low_freq = 20;
    o.mel_opts.high_freq = -400;
    return o;
}

// k2 token table; ids beyond the model vocabulary are disambiguation symbols.
inline std::vector<std::string> read_tokens(const std::filesystem::path& path, int vocabulary, int& unknown) {
    std::vector<std::string> tokens(vocabulary);
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open ASR tokens");
    std::string line;
    unknown = -1;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto at = line.find_last_of(" \t");
        if (at == std::string::npos) continue;
        const int id = std::stoi(line.substr(at + 1));
        if (id < 0 || id >= vocabulary) continue;
        tokens[id] = line.substr(0, at);
        if (tokens[id] == "<unk>") unknown = id;
    }
    if (tokens[0] != "<blk>")
        throw std::runtime_error("Token vocabulary does not match model");
    for (const auto& t : tokens) if (t.empty()) throw std::runtime_error("Token vocabulary does not match model");
    return tokens;
}
} // namespace captions::detail

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\int8_zip\ort_session_options.hpp ===
#pragma once
#include <onnxruntime_cxx_api.h>

namespace captions::detail {
inline Ort::SessionOptions cpu_session_options(int threads, bool spin = false) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetInterOpNumThreads(1);
    if (threads > 0) options.SetIntraOpNumThreads(threads);
    // Spinning trades idle CPU for lower wake-up latency; only worth it when
    // every core is saturated with decode work anyway (batch captioning).
    options.AddConfigEntry("session.intra_op.allow_spinning", spin ? "1" : "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    options.EnableCpuMemArena();
    options.EnableMemPattern();
    return options;
}
} // namespace captions::detail

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\int8_zip\include\captions\batch_asr.hpp ===
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace captions {

// Throughput engine for recorded audio (files, video soundtracks). It runs the
// same streaming Zipformer graphs as StreamingOnnxAsr, but decodes many
// independent segments at once: segments are sorted by length, stacked along
// the encoder's batch axis, and the batches are spread over worker threads that
// each own a private set of ONNX sessions.
struct BatchAsrConfig {
    std::filesystem::path encoder, decoder, joiner, tokens;
    int workers = 0;          // 0 = one per physical-core pair of the machine
    int threads_per_worker = 2;
    int batch = 16;           // segments stacked per encoder call
    bool spin = false;        // let ONNX Runtime worker threads spin between ops
    int beam = 1;             // 1 = greedy; >1 = modified beam search width
};

// Long-form splitting. Speech regions come from the same RMS gate as the
// streaming engine; regions longer than max_seconds are cut at the quietest
// 10 ms frame in [min_seconds, max_seconds] so cuts land in pauses.
struct SegmenterConfig {
    double max_seconds = 20.0;
    double min_seconds = 8.0;
    float gate_rms = 0.0003F;
    int hangover_ms = 1000;  // silence shorter than this stays inside a segment
    int pad_ms = 200;        // audio kept before/after each speech region
};

struct Span { std::size_t begin = 0, end = 0; };  // sample offsets, [begin, end)

struct Word { std::string text; double start = 0, end = 0; };

struct SegmentResult {
    double start = 0, end = 0;  // seconds in the source audio
    std::string text;
    std::vector<Word> words;
};

struct BatchAsrStats {
    double audio_seconds = 0;    // source duration
    double decoded_seconds = 0;  // audio actually sent to the model (after gating)
    double wall_seconds = 0;     // transcribe() wall time, all workers
    double worker_seconds = 0;   // summed per-worker busy time, split below
    double fbank_seconds = 0, encoder_seconds = 0, search_seconds = 0;
    std::uint64_t segments = 0, batches = 0, encoder_calls = 0;
};

std::vector<Span> segment_audio(std::span<const float> audio, const SegmenterConfig& config);

class BatchOnnxAsr {
public:
    explicit BatchOnnxAsr(BatchAsrConfig config);
    ~BatchOnnxAsr();
    BatchOnnxAsr(const BatchOnnxAsr&) = delete;
    BatchOnnxAsr& operator=(const BatchOnnxAsr&) = delete;

    // Independent clips (normalized 16 kHz mono); results keep input order.
    // Each clip is decoded exactly as StreamingOnnxAsr would decode it as one
    // gated segment: zero initial caches, right-context zero flush at the end.
    std::vector<SegmentResult> transcribe(const std::vector<std::span<const float>>& clips);

    // Whole recording: segment_audio() then transcribe(); timestamps are
    // relative to the start of `audio`.
    std::vector<SegmentResult> transcribe_long(std::span<const float> audio,
                                               const SegmenterConfig& segmenter = {});

    [[nodiscard]] const BatchAsrStats& stats() const;
    [[nodiscard]] int workers() const;
    [[nodiscard]] std::string model_type() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace captions

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\int8_zip\include\captions\streaming_asr.hpp ===
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace captions {
struct StreamingAsrConfig {
    std::filesystem::path encoder, decoder, joiner, tokens;
    int threads = 2;
    int packet_ms = 100;
    bool energy_gate = true;
    float gate_rms = 0.0003F;
    int gate_hangover_ms = 1000;
    int pre_roll_ms = 200;
};

struct StreamingAsrStats {
    std::uint64_t input_samples = 0, gated_samples = 0, encoder_calls = 0;
    double compute_seconds = 0;
    std::vector<double> encoder_ms;
    std::vector<double> packet_ms;
};

// One stream, single caller. Sessions survive reset(); acoustic caches do not.
// All samples are normalized float32, mono, 16 kHz. No whole-file lookahead.
class StreamingOnnxAsr {
public:
    explicit StreamingOnnxAsr(StreamingAsrConfig config);
    ~StreamingOnnxAsr();
    StreamingOnnxAsr(const StreamingOnnxAsr&) = delete;
    StreamingOnnxAsr& operator=(const StreamingOnnxAsr&) = delete;
    void accept(std::span<const float> samples);
    void finish();
    void reset();
    [[nodiscard]] std::string text() const;
    [[nodiscard]] const StreamingAsrStats& stats() const;
    [[nodiscard]] int model_chunk_ms() const;
    [[nodiscard]] double first_window_ms() const;
    [[nodiscard]] std::string model_type() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace captions

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\naive\baseline_frontends.hpp ===
// Wraps the two existing frame-level pipelines (lpc_method.cpp and
// fourier_method.cpp) so the benchmark can score them on the same task and the
// same split as the new models.
//
// The logic here mirrors those two programs exactly - same thresholds, same
// median smoothing, same run-length collapse - it is only lifted out of main()
// so it can be called per file instead of per process.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "audio_framing.hpp"

#include <filter/pre_emphasis_filter.hpp>
#include <formants/find_formants.hpp>
#include <formants/formant_to_vowel.hpp>
#include <power_spectrum/fast_fft.hpp>
#include <spectral/spectral_analysis.hpp>
#include <spectral/spectral_to_vowel.hpp>

namespace models {
namespace baseline {

inline float median_of_valid_values(const std::vector<float>& values) {
    std::vector<float> valid;
    for (float value : values) {
        if (value > 0.0f && std::isfinite(value)) valid.push_back(value);
    }
    if (valid.empty()) return 0.0f;
    std::sort(valid.begin(), valid.end());
    return valid[valid.size() / 2];
}

inline float zero_crossing_rate(const std::vector<float>& frame) {
    if (frame.size() < 2) return 0.0f;
    size_t crossings = 0;
    for (size_t i = 1; i < frame.size(); ++i) {
        const bool crossed = (frame[i] >= 0.0f && frame[i - 1] < 0.0f) ||
                             (frame[i] < 0.0f && frame[i - 1] >= 0.0f);
        if (crossed) ++crossings;
    }
    return static_cast<float>(crossings) / static_cast<float>(frame.size() - 1);
}

// Run-length collapse, as both legacy programs print it.
inline std::vector<std::string> collapse(const std::vector<std::string>& frames) {
    std::vector<std::string> out;
    std::string previous;
    for (const std::string& symbol : frames) {
        if (symbol == previous) continue;
        out.push_back(symbol);
        previous = symbol;
    }
    return out;
}

// LPC formant tracking -> nearest vowel in Bark space (lpc_method.cpp).
class LpcFrontend {
public:
    std::vector<std::string> vowel_string(const std::vector<float>& audio) {
        const std::vector<float> filtered = pre_emphasis_filter(audio);
        const std::vector<std::vector<float>> frames = chop_into_frames(filtered);
        std::vector<std::string> result;
        if (frames.empty()) return result;

        std::vector<std::array<float, 3>> formant_crop;
        std::vector<float> energies, zcrs;
        constexpr float kConsonantZcrThreshold = 0.35f;

        for (const std::vector<float>& frame : frames) {
            float energy = 0.0f;
            for (float sample : frame) energy += sample * sample;
            energies.push_back(std::sqrt(energy / frame.size()));
            zcrs.push_back(zero_crossing_rate(frame));

            const std::vector<Formant> formants = tracker_.extract_formants(frame);
            if (formants.size() < 2) {
                formant_crop.push_back({0.0f, 0.0f, 0.0f});
            } else {
                formant_crop.push_back({formants[0].frequency, formants[1].frequency,
                                        formants.size() > 2 ? formants[2].frequency : 0.0f});
            }
        }

        const float max_energy = *std::max_element(energies.begin(), energies.end());
        const float threshold = std::max(max_energy * 0.08f, 0.00015f);

        for (size_t i = 0; i < formant_crop.size(); ++i) {
            if (energies[i] < threshold || zcrs[i] > kConsonantZcrThreshold ||
                formant_crop[i][0] <= 0.0f || formant_crop[i][1] <= 0.0f) {
                result.push_back("SIL");
                continue;
            }
            const size_t first = i == 0 ? 0 : i - 1;
            const size_t last = i + 1 < formant_crop.size() ? i + 1 : formant_crop.size() - 1;
            std::array<float, 3> smoothed{};
            for (size_t f = 0; f < 3; ++f) {
                std::vector<float> neighborhood;
                for (size_t n = first; n <= last; ++n) neighborhood.push_back(formant_crop[n][f]);
                smoothed[f] = median_of_valid_values(neighborhood);
            }
            const FormantVectorDB::MatchResult match =
                database_.find_nearest_phoneme(smoothed[0], smoothed[1], smoothed[2]);
            result.push_back(match.key == "UNK" ? "SIL" : match.key);
        }
        return result;
    }

private:
    FormantTracker tracker_;
    FormantVectorDB database_;
};

// FFT spectral envelope -> nearest vowel (fourier_method.cpp).
class FourierFrontend {
public:
    std::vector<std::string> vowel_string(const std::vector<float>& audio) {
        const std::vector<float> filtered = pre_emphasis_filter(audio);
        const std::vector<std::vector<float>> frames = chop_into_frames(filtered);
        std::vector<std::string> result;
        if (frames.empty()) return result;

        std::vector<SpectralFeatures> crop;
        std::vector<float> energies, zcrs;
        constexpr float kCentroidThreshold = 3200.0f;
        constexpr float kZcrThreshold = 0.35f;

        for (const std::vector<float>& frame : frames) {
            float energy = 0.0f;
            for (float sample : frame) energy += sample * sample;
            energies.push_back(std::sqrt(energy / frame.size()));
            zcrs.push_back(zero_crossing_rate(frame));
            crop.push_back(SpectralAnalyzer::extract_features(fft_.compute_power_spectrum(frame)));
        }

        const float max_energy = *std::max_element(energies.begin(), energies.end());
        const float threshold = std::max(max_energy * 0.08f, 0.00015f);

        for (size_t i = 0; i < crop.size(); ++i) {
            if (energies[i] < threshold || zcrs[i] > kZcrThreshold ||
                crop[i].centroid_hz > kCentroidThreshold || !crop[i].is_voiced) {
                result.push_back("SIL");
                continue;
            }
            const size_t first = i == 0 ? 0 : i - 1;
            const size_t last = i + 1 < crop.size() ? i + 1 : crop.size() - 1;
            std::vector<float> centroids, pitches, tilts;
            for (size_t n = first; n <= last; ++n) {
                centroids.push_back(crop[n].centroid_hz);
                pitches.push_back(crop[n].pitch_f0_hz);
                tilts.push_back(crop[n].spectral_tilt);
            }
            SpectralFeatures smoothed{median_of_valid_values(centroids),
                                      median_of_valid_values(pitches), true,
                                      median_of_valid_values(tilts),
                                      crop[i].voicing_confidence};
            const SpectralVowelDB::MatchResult match = database_.find_nearest_phoneme(smoothed);
            result.push_back(match.key == "UNK" ? "SIL" : match.key);
        }
        return result;
    }

private:
    FastFFT fft_;
    SpectralVowelDB database_;
};

}  // namespace baseline
}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\toy_pruned_hmm\acoustic_model.hpp ===
// Monophone acoustic model: 3-state left-to-right HMM per phone, one diagonal
// covariance Gaussian per state. This is the GMM-HMM structure that legacy
// engines (PocketSphinx, Kaldi's mono stage) use, with a single mixture
// component - trained here from TIMIT's hand-aligned .phn boundaries, so no
// EM bootstrap or neural net is involved.
//
// Speed: the whole inventory is 40 phones * 3 states = 120 Gaussians. The
// decoder scores all 120 once per frame into a flat table, then every active
// token reads that table - so acoustic scoring costs ~120*39 multiply-adds per
// frame no matter how wide the search beam gets.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <filter/mfcc.hpp>
#include "phone_set.hpp"

namespace models {

class AcousticModel {
public:
    static constexpr int kMaxMixtures = 16;

    AcousticModel() { resize(); }

    // Number of emitting units: 120 monophone states by default, or the
    // number of tied triphone states (senones) once a decision tree exists.
    int units() const { return units_; }

    // ---- training -------------------------------------------------------

    void begin_training(int units = 0) {
        if (units > 0) units_ = units;
        resize();
        sum_.assign(static_cast<size_t>(units_) * kFeatureDim, 0.0);
        sum_squares_.assign(static_cast<size_t>(units_) * kFeatureDim, 0.0);
        counts_.assign(units_, 0.0);
        phone_frames_.assign(num_phones(), 0.0);
        phone_segments_.assign(num_phones(), 0.0);
    }

    // Accumulate one aligned phone segment: frames [start, end) belong to
    // `phone`, split evenly across its three states.
    void accumulate_segment(const FeatureMatrix& features, int start, int end, int phone) {
        if (phone < 0 || end <= start) return;
        const int length = end - start;
        phone_frames_[phone] += length;
        phone_segments_[phone] += 1.0;

        for (int t = start; t < end; ++t) {
            if (t < 0 || t >= features.num_frames) continue;
            int state = ((t - start) * kNumStatesPerPhone) / length;
            if (state >= kNumStatesPerPhone) state = kNumStatesPerPhone - 1;
            const int index = state_index(phone, state);
            const float* row = features.frame(t);
            double* sum = &sum_[static_cast<size_t>(index) * kFeatureDim];
            double* sq = &sum_squares_[static_cast<size_t>(index) * kFeatureDim];
            for (int d = 0; d < kFeatureDim; ++d) {
                sum[d] += row[d];
                sq[d] += static_cast<double>(row[d]) * row[d];
            }
            counts_[index] += 1.0;
        }
    }

    // Accumulate one frame against a state chosen by forced alignment, rather
    // than by splitting a labelled segment into equal thirds.
    void accumulate_frame(const float* row, int state) {
        if (state < 0 || state >= units_) return;
        double* sum = &sum_[static_cast<size_t>(state) * kFeatureDim];
        double* sq = &sum_squares_[static_cast<size_t>(state) * kFeatureDim];
        for (int d = 0; d < kFeatureDim; ++d) {
            sum[d] += row[d];
            sq[d] += static_cast<double>(row[d]) * row[d];
        }
        counts_[state] += 1.0;
    }

    // Record an aligned phone duration so transitions can be re-estimated.
    void accumulate_duration(int phone, int frames) {
        if (phone < 0 || frames <= 0) return;
        phone_frames_[phone] += frames;
        phone_segments_[phone] += 1.0;
    }

    // Turn accumulators into Gaussians and duration-derived transitions.
    // `variance_floor` guards states with too few frames from collapsing.
    void finish_training(double variance_floor = 0.01) {
        // Global variance is the fallback for starved states.
        std::vector<double> global_mean(kFeatureDim, 0.0), global_var(kFeatureDim, 0.0);
        double total = 0.0;
        for (int s = 0; s < units_; ++s) {
            total += counts_[s];
            for (int d = 0; d < kFeatureDim; ++d) {
                global_mean[d] += sum_[static_cast<size_t>(s) * kFeatureDim + d];
                global_var[d] += sum_squares_[static_cast<size_t>(s) * kFeatureDim + d];
            }
        }
        if (total > 0) {
            for (int d = 0; d < kFeatureDim; ++d) {
                global_mean[d] /= total;
                global_var[d] = global_var[d] / total - global_mean[d] * global_mean[d];
                if (global_var[d] < variance_floor) global_var[d] = variance_floor;
            }
        }

        for (int s = 0; s < units_; ++s) {
            float* mean = &means_[static_cast<size_t>(s) * kFeatureDim];
            float* inv_var = &inv_variances_[static_cast<size_t>(s) * kFeatureDim];
            const double count = counts_[s];
            double log_det = 0.0;

            for (int d = 0; d < kFeatureDim; ++d) {
                double m, v;
                if (count >= 3.0) {
                    m = sum_[static_cast<size_t>(s) * kFeatureDim + d] / count;
                    v = sum_squares_[static_cast<size_t>(s) * kFeatureDim + d] / count - m * m;
                    // Shrink toward the global model when the state is thin.
                    const double weight = count / (count + 10.0);
                    m = weight * m + (1.0 - weight) * global_mean[d];
                    v = weight * v + (1.0 - weight) * global_var[d];
                } else {
                    m = global_mean[d];
                    v = global_var[d];
                }
                if (v < variance_floor) v = variance_floor;
                mean[d] = static_cast<float>(m);
                inv_var[d] = static_cast<float>(1.0 / v);
                log_det += std::log(v);
            }
            // -0.5 * (D*log(2*pi) + log|Sigma|), the constant part of log N(x).
            constexpr double kLog2Pi = 1.8378770664093453;
            log_constants_[s] = static_cast<float>(-0.5 * (kFeatureDim * kLog2Pi + log_det));
            state_occupancy_[s] = static_cast<float>(count);
        }

        // Self-loop probability from the observed mean phone duration:
        // a state visited for `d` frames leaves with probability 1/d.
        for (int p = 0; p < num_phones(); ++p) {
            double frames_per_state = 3.0;
            if (phone_segments_[p] > 0.0) {
                frames_per_state =
                    (phone_frames_[p] / phone_segments_[p]) / kNumStatesPerPhone;
            }
            if (frames_per_state < 1.2) frames_per_state = 1.2;
            const double exit = 1.0 / frames_per_state;
            log_self_loop_[p] = static_cast<float>(std::log(1.0 - exit));
            log_exit_[p] = static_cast<float>(std::log(exit));
        }
    }

    // ---- mixture training ----------------------------------------------

    // Grow each state's single Gaussian into `mixtures` components and refine
    // them with EM on the frames assigned to that state.
    //
    // This is the classical GMM-HMM recipe: split the component with the most
    // occupancy along its principal axis (mean +/- 0.2 sigma), re-estimate,
    // repeat. A monophone with one Gaussian cannot model the fact that the
    // same phone sounds different across speakers and contexts; mixtures buy
    // exactly that, at a scoring cost of components x dimensions per state.
    void train_mixtures(const std::vector<float>& frames, const std::vector<int>& state_of_frame,
                        int mixtures, int em_iterations = 6, double variance_floor = 0.01) {
        if (mixtures <= 1) return;
        mixtures_ = mixtures;

        // Group frame indices by state.
        std::vector<std::vector<int>> by_state(units_);
        for (size_t i = 0; i < state_of_frame.size(); ++i) {
            const int state = state_of_frame[i];
            if (state >= 0 && state < units_) by_state[state].push_back(static_cast<int>(i));
        }

        std::vector<float> new_means(static_cast<size_t>(units_) * mixtures * kFeatureDim);
        std::vector<float> new_inv(static_cast<size_t>(units_) * mixtures * kFeatureDim);
        std::vector<float> new_const(static_cast<size_t>(units_) * mixtures);
        std::vector<float> new_weight(static_cast<size_t>(units_) * mixtures);

        std::vector<double> mean(kFeatureDim), variance(kFeatureDim);
        std::vector<double> posterior(mixtures);
        std::vector<double> occupancy(mixtures);
        std::vector<double> sum(static_cast<size_t>(mixtures) * kFeatureDim);
        std::vector<double> sum_squares(static_cast<size_t>(mixtures) * kFeatureDim);

        for (int s = 0; s < units_; ++s) {
            const std::vector<int>& indices = by_state[s];
            const size_t base = static_cast<size_t>(s) * mixtures;

            // Seed every component from the state's single Gaussian, then
            // perturb so EM has something to separate.
            for (int k = 0; k < mixtures; ++k) {
                const float shift = (k % 2 == 0 ? 1.0f : -1.0f) * 0.2f * (1 + k / 2);
                for (int d = 0; d < kFeatureDim; ++d) {
                    const float sigma =
                        1.0f / std::sqrt(inv_variances_[static_cast<size_t>(s) * kFeatureDim + d]);
                    new_means[(base + k) * kFeatureDim + d] =
                        means_[static_cast<size_t>(s) * kFeatureDim + d] +
                        (d % 3 == 0 ? shift * sigma : 0.0f);
                    new_inv[(base + k) * kFeatureDim + d] =
                        inv_variances_[static_cast<size_t>(s) * kFeatureDim + d];
                }
                new_weight[base + k] = static_cast<float>(std::log(1.0 / mixtures));
                new_const[base + k] = log_constants_[s];
            }

            // Too few frames to estimate a mixture: keep the copies as-is.
            if (indices.size() < static_cast<size_t>(mixtures) * 20) continue;

            for (int iteration = 0; iteration < em_iterations; ++iteration) {
                std::fill(occupancy.begin(), occupancy.end(), 0.0);
                std::fill(sum.begin(), sum.end(), 0.0);
                std::fill(sum_squares.begin(), sum_squares.end(), 0.0);

                for (int index : indices) {
                    const float* row = frames.data() + static_cast<size_t>(index) * kFeatureDim;

                    // E step: component posteriors via log-sum-exp.
                    double best = -1e30;
                    for (int k = 0; k < mixtures; ++k) {
                        const float* mu = &new_means[(base + k) * kFeatureDim];
                        const float* inv = &new_inv[(base + k) * kFeatureDim];
                        double acc = 0.0;
                        for (int d = 0; d < kFeatureDim; ++d) {
                            const double e = row[d] - mu[d];
                            acc += e * e * inv[d];
                        }
                        posterior[k] = new_weight[base + k] + new_const[base + k] - 0.5 * acc;
                        if (posterior[k] > best) best = posterior[k];
                    }
                    double total = 0.0;
                    for (int k = 0; k < mixtures; ++k) {
                        posterior[k] = std::exp(posterior[k] - best);
                        total += posterior[k];
                    }
                    for (int k = 0; k < mixtures; ++k) {
                        const double weight = posterior[k] / total;
                        if (weight < 1e-6) continue;
                        occupancy[k] += weight;
                        double* s_acc = &sum[static_cast<size_t>(k) * kFeatureDim];
                        double* q_acc = &sum_squares[static_cast<size_t>(k) * kFeatureDim];
                        for (int d = 0; d < kFeatureDim; ++d) {
                            s_acc[d] += weight * row[d];
                            q_acc[d] += weight * row[d] * row[d];
                        }
                    }
                }

                // M step.
                double total_occupancy = 0.0;
                for (int k = 0; k < mixtures; ++k) total_occupancy += occupancy[k];
                if (total_occupancy <= 0.0) break;

                for (int k = 0; k < mixtures; ++k) {
                    if (occupancy[k] < 5.0) {
                        // Starved component: park it with a tiny weight rather
                        // than letting its variance collapse onto one frame.
                        new_weight[base + k] = static_cast<float>(std::log(1e-5));
                        continue;
                    }
                    double log_det = 0.0;
                    for (int d = 0; d < kFeatureDim; ++d) {
                        const double m = sum[static_cast<size_t>(k) * kFeatureDim + d] / occupancy[k];
                        double v = sum_squares[static_cast<size_t>(k) * kFeatureDim + d] /
                                       occupancy[k] - m * m;
                        if (v < variance_floor) v = variance_floor;
                        new_means[(base + k) * kFeatureDim + d] = static_cast<float>(m);
                        new_inv[(base + k) * kFeatureDim + d] = static_cast<float>(1.0 / v);
                        log_det += std::log(v);
                    }
                    constexpr double kLog2Pi = 1.8378770664093453;
                    new_const[base + k] =
                        static_cast<float>(-0.5 * (kFeatureDim * kLog2Pi + log_det));
                    new_weight[base + k] =
                        static_cast<float>(std::log(occupancy[k] / total_occupancy));
                }
            }
        }

        means_.swap(new_means);
        inv_variances_.swap(new_inv);
        log_constants_.swap(new_const);
        log_weights_.swap(new_weight);
    }

    // ---- scoring --------------------------------------------------------

    // log p(x | state): a single Gaussian, or the log-sum-exp over mixture
    // components when the model was trained with more than one.
    float log_likelihood(int state, const float* features) const {
        const size_t base = static_cast<size_t>(state) * mixtures_;
        float best = -1e30f;
        float component[kMaxMixtures];

        for (int k = 0; k < mixtures_; ++k) {
            const float* mean = &means_[(base + k) * kFeatureDim];
            const float* inv_var = &inv_variances_[(base + k) * kFeatureDim];
            float acc0 = 0.0f, acc1 = 0.0f;
            int d = 0;
            for (; d + 2 <= kFeatureDim; d += 2) {
                const float e0 = features[d] - mean[d];
                const float e1 = features[d + 1] - mean[d + 1];
                acc0 += e0 * e0 * inv_var[d];
                acc1 += e1 * e1 * inv_var[d + 1];
            }
            float acc = acc0 + acc1;
            for (; d < kFeatureDim; ++d) {
                const float e = features[d] - mean[d];
                acc += e * e * inv_var[d];
            }
            float value = log_constants_[base + k] - 0.5f * acc;
            if (mixtures_ > 1) value += log_weights_[base + k];
            component[k] = value;
            if (value > best) best = value;
        }
        if (mixtures_ == 1) return best;

        float sum = 0.0f;
        for (int k = 0; k < mixtures_; ++k) sum += std::exp(component[k] - best);
        return best + std::log(sum);
    }

    int mixtures() const { return mixtures_; }

    // Score every state for one frame into `out` (size num_states()).
    void score_frame(const float* features, std::vector<float>& out) const {
        out.resize(units_);
        for (int s = 0; s < units_; ++s) out[s] = log_likelihood(s, features);
    }

    float log_self_loop(int phone) const { return log_self_loop_[phone]; }
    float log_exit(int phone) const { return log_exit_[phone]; }
    float occupancy(int state) const { return state_occupancy_[state]; }

    // ---- persistence ----------------------------------------------------

    bool save(const std::string& path) const {
        FILE* file = std::fopen(path.c_str(), "wb");
        if (!file) return false;
        const int32_t magic = 0x4D414D32;  // "MAM2" (mixture-capable)
        const int32_t states = units_;
        const int32_t dim = kFeatureDim;
        const int32_t mixtures = mixtures_;
        std::fwrite(&magic, sizeof(magic), 1, file);
        std::fwrite(&states, sizeof(states), 1, file);
        std::fwrite(&dim, sizeof(dim), 1, file);
        std::fwrite(&mixtures, sizeof(mixtures), 1, file);
        std::fwrite(means_.data(), sizeof(float), means_.size(), file);
        std::fwrite(inv_variances_.data(), sizeof(float), inv_variances_.size(), file);
        std::fwrite(log_constants_.data(), sizeof(float), log_constants_.size(), file);
        std::fwrite(log_weights_.data(), sizeof(float), log_weights_.size(), file);
        std::fwrite(state_occupancy_.data(), sizeof(float), state_occupancy_.size(), file);
        std::fwrite(log_self_loop_.data(), sizeof(float), log_self_loop_.size(), file);
        std::fwrite(log_exit_.data(), sizeof(float), log_exit_.size(), file);
        std::fclose(file);
        return true;
    }

    bool load(const std::string& path) {
        FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) return false;
        int32_t magic = 0, states = 0, dim = 0, mixtures = 1;
        bool ok = std::fread(&magic, sizeof(magic), 1, file) == 1 &&
                  std::fread(&states, sizeof(states), 1, file) == 1 &&
                  std::fread(&dim, sizeof(dim), 1, file) == 1 &&
                  std::fread(&mixtures, sizeof(mixtures), 1, file) == 1;
        if (!ok || magic != 0x4D414D32 || states < 1 || dim != kFeatureDim || mixtures < 1 ||
            mixtures > kMaxMixtures) {
            std::fclose(file);
            return false;
        }
        units_ = states;
        resize();
        mixtures_ = mixtures;
        const size_t components = static_cast<size_t>(units_) * mixtures;
        means_.assign(components * kFeatureDim, 0.0f);
        inv_variances_.assign(components * kFeatureDim, 1.0f);
        log_constants_.assign(components, 0.0f);
        log_weights_.assign(components, 0.0f);
        ok = std::fread(means_.data(), sizeof(float), means_.size(), file) == means_.size() &&
             std::fread(inv_variances_.data(), sizeof(float), inv_variances_.size(), file) ==
                 inv_variances_.size() &&
             std::fread(log_constants_.data(), sizeof(float), log_constants_.size(), file) ==
                 log_constants_.size() &&
             std::fread(log_weights_.data(), sizeof(float), log_weights_.size(), file) ==
                 log_weights_.size() &&
             std::fread(state_occupancy_.data(), sizeof(float), state_occupancy_.size(), file) ==
                 state_occupancy_.size() &&
             std::fread(log_self_loop_.data(), sizeof(float), log_self_loop_.size(), file) ==
                 log_self_loop_.size() &&
             std::fread(log_exit_.data(), sizeof(float), log_exit_.size(), file) == log_exit_.size();
        std::fclose(file);
        return ok;
    }

private:
    void resize() {
        mixtures_ = 1;
        means_.assign(static_cast<size_t>(units_) * kFeatureDim, 0.0f);
        inv_variances_.assign(static_cast<size_t>(units_) * kFeatureDim, 1.0f);
        log_constants_.assign(units_, 0.0f);
        log_weights_.assign(units_, 0.0f);
        state_occupancy_.assign(units_, 0.0f);
        log_self_loop_.assign(num_phones(), std::log(0.6f));
        log_exit_.assign(num_phones(), std::log(0.4f));
    }

    int mixtures_ = 1;
    int units_ = num_states();
    std::vector<float> means_;
    std::vector<float> inv_variances_;
    std::vector<float> log_constants_;
    std::vector<float> log_weights_;
    std::vector<float> state_occupancy_;
    std::vector<float> log_self_loop_;
    std::vector<float> log_exit_;

    // training accumulators
    std::vector<double> sum_;
    std::vector<double> sum_squares_;
    std::vector<double> counts_;
    std::vector<double> phone_frames_;
    std::vector<double> phone_segments_;
};

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\toy_pruned_hmm\forced_align.hpp ===
// Forced Viterbi alignment (Path 2): given an utterance and the word sequence
// that was actually spoken, find the maximum-likelihood assignment of frames to
// HMM states.
//
// Why this exists: TIMIT ships hand-marked phone boundaries, so the first
// acoustic model could be trained by splitting each labelled segment into three
// equal parts. LibriSpeech ships only sentence text, so there are no boundaries
// to split - they have to be inferred. Forced alignment is also better than
// uniform splitting even when boundaries *are* available, because real phone
// durations vary with stress and position: a stop burst is 2 frames, a stressed
// vowel is 20, and equal thirds smear the models of both.
//
// The search is a plain Viterbi over a linear chain (no LM, no beam): the word
// sequence is known, so the only unknowns are the boundaries. Cost is
// O(frames * states) with states = 3 * number of phones in the utterance, which
// for a 30 s utterance is ~3000 x ~900 - small enough to do exactly, with a
// full backtrace, rather than approximately.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <stdexcept>

#include "acoustic_model.hpp"
#include "g2p.hpp"
#include "lexicon.hpp"
#include <filter/mfcc.hpp>
#include "phone_set.hpp"
#include "triphone.hpp"

namespace models {

// One aligned segment of an utterance.
struct AlignedSegment {
    int phone = -1;
    int start_frame = 0;
    int stop_frame = 0;
};

// Maps words to phone sequences, inserting optional silence around words.
class PronunciationTable {
public:
    void build(const Lexicon& lexicon) {
        table_.clear();
        for (const Pronunciation& pron : lexicon.pronunciations()) {
            // Keep the first pronunciation per word: alignment does not need
            // variants, and a single choice keeps the chain deterministic.
            const std::string& word = lexicon.word(pron.word_id);
            table_.emplace(word, pron.phones);
        }
    }

    const std::vector<int>* find(const std::string& word) const {
        const auto it = table_.find(word);
        return it == table_.end() ? nullptr : &it->second;
    }

    // Word sequence -> phone sequence with leading/trailing silence.
    //
    // Words missing from CMUDict fall back to letter-to-sound rules and are
    // counted in `guessed`. Alignment needs *a* pronunciation for every word in
    // the audio: skipping the word would leave its frames to be absorbed by its
    // neighbours, which corrupts their models worse than an approximate
    // spelling of one rare proper noun does.
    bool phones_for(const std::vector<std::string>& words, std::vector<int>& out,
                    int* guessed = nullptr, bool silence_between_words = false,
                    std::vector<std::pair<int, int>>* contexts = nullptr) const {
        out.clear();
        if (contexts) contexts->clear();
        const int silence = phone_id("SIL");
        out.push_back(silence);
        if (contexts) contexts->push_back({silence, silence});
        for (const std::string& word : words) {
            const size_t start = out.size();
            const std::vector<int>* phones = find(word);
            if (phones) {
                out.insert(out.end(), phones->begin(), phones->end());
            } else {
                const std::vector<int> fallback = g2p_.convert(word);
                if (fallback.empty()) return false;
                out.insert(out.end(), fallback.begin(), fallback.end());
                if (guessed) ++*guessed;
            }
            if (contexts) {
                for (size_t i = start; i < out.size(); ++i) {
                    contexts->push_back({i > start ? out[i - 1] : silence,
                                         i + 1 < out.size() ? out[i + 1] : silence});
                }
            }
            if (silence_between_words) {
                out.push_back(silence);
                if (contexts) contexts->push_back({silence, silence});
            }
        }
        if (out.back() != silence) {
            out.push_back(silence);
            if (contexts) contexts->push_back({silence, silence});
        }
        return true;
    }

    size_t size() const { return table_.size(); }

private:
    std::unordered_map<std::string, std::vector<int>> table_;
    GraphemeToPhoneme g2p_;
};

class ForcedAligner {
public:
    // Aligns `features` to `phones`. Returns segments in time order, or an
    // empty vector if the utterance is too short for the phone sequence.
    // `tree` is optional: with it, each chain position emits from the tied
    // triphone state for its context instead of the monophone state, so
    // re-alignment benefits from the sharper context-dependent models.
    std::vector<AlignedSegment> align(const FeatureMatrix& features,
                                      const std::vector<int>& phones,
                                      const AcousticModel& model,
                                      const TriphoneTree* tree = nullptr,
                                      const std::vector<std::pair<int, int>>* contexts = nullptr) {
        state_path_.clear();
        phone_of_frame_.clear();
        if (contexts && contexts->size() != phones.size())
            throw std::invalid_argument("phone/context length mismatch");
        const int T = features.num_frames;
        const int S = static_cast<int>(phones.size()) * kNumStatesPerPhone;
        if (T == 0 || S == 0 || T < S) return {};

        constexpr float kNegInf = -std::numeric_limits<float>::infinity();
        scores_.assign(S, kNegInf);
        next_.assign(S, kNegInf);
        // Backpointer per (frame, state): 0 = self-loop, 1 = came from s-1.
        back_.assign(static_cast<size_t>(T) * S, 0);

        // Precomputed per-state emission indices and transition costs.
        state_phone_.resize(S);
        state_hmm_.resize(S);
        const int silence = phone_id("SIL");
        for (int s = 0; s < S; ++s) {
            const int index = s / kNumStatesPerPhone;
            const int sub = s % kNumStatesPerPhone;
            const int phone = phones[index];
            state_phone_[s] = phone;
            if (tree) {
                const int left = contexts ? (*contexts)[index].first
                                          : (index > 0 ? phones[index - 1] : silence);
                const int right = contexts ? (*contexts)[index].second
                    : (index + 1 < static_cast<int>(phones.size()) ? phones[index + 1] : silence);
                state_hmm_[s] = tree->senone(left, phone, sub, right);
            } else {
                state_hmm_[s] = state_index(phone, sub);
            }
        }

        // Score only units referenced by this transcript, once per frame.
        // The full senone inventory is much larger than a single phone chain.
        std::vector<int> used_units = state_hmm_;
        std::sort(used_units.begin(), used_units.end());
        used_units.erase(std::unique(used_units.begin(), used_units.end()), used_units.end());
        am_.resize(model.units());
        for (int t = 0; t < T; ++t) {
            for (int unit : used_units)
                am_[unit] = model.log_likelihood(unit, features.frame(t));
            uint8_t* back = back_.data() + static_cast<size_t>(t) * S;

            // A path must reach state s by frame t, and can still reach the
            // last state by frame T: outside that diagonal the cell is dead.
            const int lo = std::max(0, S - (T - t) * 2);
            const int hi = std::min(S - 1, t);

            for (int s = hi; s >= lo; --s) {
                float best;
                uint8_t choice;
                if (t == 0) {
                    // Only the first state can be occupied at frame 0.
                    best = (s == 0) ? 0.0f : kNegInf;
                    choice = 0;
                } else {
                    const float stay = scores_[s] == kNegInf
                                           ? kNegInf
                                           : scores_[s] + model.log_self_loop(state_phone_[s]);
                    const float enter =
                        (s > 0 && scores_[s - 1] != kNegInf)
                            ? scores_[s - 1] + model.log_exit(state_phone_[s - 1])
                            : kNegInf;
                    if (enter > stay) {
                        best = enter;
                        choice = 1;
                    } else {
                        best = stay;
                        choice = 0;
                    }
                }
                back[s] = choice;
                next_[s] = best == kNegInf ? kNegInf : best + am_[state_hmm_[s]];
            }
            for (int s = 0; s < lo; ++s) next_[s] = kNegInf;
            for (int s = hi + 1; s < S; ++s) next_[s] = kNegInf;
            scores_.swap(next_);
        }

        if (scores_[S - 1] == kNegInf) return {};  // no complete path

        // Backtrace over the chain. The path gives the exact state occupied at
        // every frame, including the within-phone state boundaries - keep them
        // rather than re-splitting each phone uniformly afterwards, which would
        // throw away most of what the alignment just computed.
        chain_position_.assign(T, 0);
        int state = S - 1;
        for (int t = T - 1; t >= 0; --t) {
            chain_position_[t] = state;
            if (back_[static_cast<size_t>(t) * S + state] == 1 && state > 0) --state;
        }

        state_path_.resize(T);
        phone_of_frame_.resize(T);
        for (int t = 0; t < T; ++t) {
            const int position = chain_position_[t];
            const int phone = phones[position / kNumStatesPerPhone];
            phone_of_frame_[t] = phone;
            state_path_[t] = state_index(phone, position % kNumStatesPerPhone);
        }

        // Phone-level segments, for triphone context and for reporting.
        std::vector<AlignedSegment> segments;
        int current = chain_position_[0] / kNumStatesPerPhone;
        int start = 0;
        for (int t = 1; t < T; ++t) {
            const int phone_index = chain_position_[t] / kNumStatesPerPhone;
            if (phone_index != current) {
                segments.push_back({phones[current], start, t});
                current = phone_index;
                start = t;
            }
        }
        segments.push_back({phones[current], start, T});
        return segments;
    }

    // Per-frame HMM state indices from the last align() call, in the model's
    // global state numbering - what GMM re-estimation consumes.
    const std::vector<int>& state_path() const { return state_path_; }
    const std::vector<int>& phone_of_frame() const { return phone_of_frame_; }

private:
    std::vector<int> chain_position_, state_path_, phone_of_frame_;
    std::vector<float> scores_, next_, am_;
    std::vector<uint8_t> back_;
    std::vector<int> state_phone_, state_hmm_;
};

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\toy_pruned_hmm\viterbi_decoder.hpp ===
// Single-pass token-passing Viterbi beam decoder (Step 3 of models/README.md).
//
// Search space: every pronunciation in the lexicon becomes a linear chain of
// 3-state phone HMMs. Tokens propagate through those chains frame by frame;
// when a token leaves a word's last state it is re-entered at the start of the
// next word with a bigram LM cost. Backtracking runs over word-link records,
// so the decoder emits a word sequence, not a frame-level phone string - which
// is what makes continuous captioning possible at all.
//
// The three costs that make this tractable ("optimality and speed"):
//   1. Acoustic scores are computed once per frame for all 120 HMM states into
//      a flat table; token propagation then reads that table, so widening the
//      beam never re-runs a Gaussian.
//   2. Beam + histogram pruning bound the active state set per frame.
//   3. LM expansion uses the backoff structure: explicit bigrams are scanned
//      only for the top-K surviving word ends, and the backed-off mass is
//      applied to the whole vocabulary in one pass via a single best-backoff
//      value. That makes exact-under-backoff entry scores cost
//      O(K * successors + V) instead of O(K * V).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "acoustic_model.hpp"
#include "lexicon.hpp"
#include <filter/mfcc.hpp>
#include "ngram_lm.hpp"
#include "phone_set.hpp"
#include "triphone.hpp"

namespace models {

struct DecoderConfig {
    float acoustic_scale = 0.06f;   // Gaussian log-likelihoods are sharp; scale them
    float word_insertion_penalty = 0.0f;
    float beam = 120.0f;            // log-prob width below the frame's best token
    float word_beam = 10.0f;        // tighter width for *entering* a new word
    int max_active = 4000;          // histogram pruning cap
    int max_word_ends = 24;         // word ends expanded per frame
    bool allow_silence = true;      // optional inter-word silence model
};

struct DecodeResult {
    std::vector<std::string> words;
    float score = 0.0f;
    int frames = 0;
    // search statistics, for the speed report
    int64_t states_visited = 0;
    int64_t word_entries = 0;
    int64_t gaussians_scored = 0;
    double decode_seconds = 0.0;
};

class ViterbiDecoder {
public:
    // Builds the search network. Words are addressed by LM id, so the lexicon
    // and the LM must agree on spelling; words missing from either side are
    // dropped (and counted) rather than silently mismatched.
    // `tree` is optional. With it, every phone in a word chain emits from the
    // tied triphone state for its word-internal context (SIL at word edges)
    // rather than from one context-independent monophone state.
    void build(const Lexicon& lexicon, const NgramLm& lm, const AcousticModel& model,
               const DecoderConfig& config = {}, const TriphoneTree* tree = nullptr) {
        lm_ = &lm;
        model_ = &model;
        config_ = config;
        tree_ = tree;

        state_phone_.clear();
        state_hmm_.clear();
        state_word_.clear();
        state_last_.clear();
        word_entry_.assign(lm.vocabulary_size(), std::vector<int>());
        skipped_words_ = 0;

        for (const Pronunciation& pron : lexicon.pronunciations()) {
            const std::string& spelling = lexicon.word(pron.word_id);
            const int lm_word = lm.word_id(spelling);
            if (lm_word < 0 || lm_word == lm.bos() || lm_word == lm.unk()) {
                ++skipped_words_;
                continue;
            }
            add_chain(pron.phones, lm_word);
        }

        // Optional silence chain: entered like a word, but leaves the LM
        // history untouched so "cat <sil> sat" still scores P(sat|cat).
        silence_entry_ = -1;
        if (config_.allow_silence) {
            silence_entry_ = static_cast<int>(state_phone_.size());
            add_chain({phone_id("SIL")}, kSilenceWord);
        }

        // Words ordered by unigram probability. The backed-off entry score is
        // best_backoff + log P_uni(w), which is monotonically decreasing in
        // this order, so the vocabulary sweep can stop as soon as it drops
        // below the beam floor instead of touching all 20k words every frame.
        unigram_order_.clear();
        for (int word = 0; word < lm.vocabulary_size(); ++word) {
            if (!word_entry_[word].empty()) unigram_order_.push_back(word);
        }
        std::sort(unigram_order_.begin(), unigram_order_.end(), [&lm](int a, int b) {
            return lm.log_unigram(a) > lm.log_unigram(b);
        });

        const int total = static_cast<int>(state_phone_.size());
        scores_.assign(total, kNegInf);
        next_scores_.assign(total, kNegInf);
        links_.assign(total, -1);
        next_links_.assign(total, -1);
        stamp_.assign(total, -1);
        entry_score_.assign(lm.vocabulary_size(), kNegInf);
        entry_link_.assign(lm.vocabulary_size(), -1);
        entry_mark_.assign(lm.vocabulary_size(), -1);
    }

    int num_states() const { return static_cast<int>(state_phone_.size()); }
    int skipped_words() const { return skipped_words_; }

    // Decode one utterance.
    DecodeResult decode(const FeatureMatrix& features) {
        DecodeResult result;
        result.frames = features.num_frames;
        if (!lm_ || !model_ || features.empty()) return result;

        const int total = num_states();
        std::fill(scores_.begin(), scores_.end(), kNegInf);
        std::fill(links_.begin(), links_.end(), -1);
        std::fill(stamp_.begin(), stamp_.end(), -1);
        active_.clear();
        word_links_.clear();

        // Frame -1: a virtual sentence-start word end seeds the first entries.
        ends_.clear();
        ends_.push_back({lm_->bos(), 0.0f, -1});
        previous_best_ = 0.0f;
        std::fill(entry_mark_.begin(), entry_mark_.end(), -1);

        // Lazy acoustic scoring: with 2500 tied states x 8 mixtures, scoring
        // every unit on every frame costs more than the search itself, yet the
        // beam only ever touches a few hundred of them. Score on demand and
        // cache per frame instead.
        am_scores_.assign(model_->units(), 0.0f);
        am_stamp_.assign(model_->units(), -1);

        for (int t = 0; t < features.num_frames; ++t) {
            const float* frame = features.frame(t);
            current_frame_ = frame;
            result_ = &result;

            next_active_.clear();
            ++frame_stamp_;

            // 1. Word entries from the previous frame's word ends.
            expand_entries(t, result);

            // 2. Transitions inside the active chains.
            for (int state : active_) {
                const float score = scores_[state];
                if (score == kNegInf) continue;
                const int phone = state_phone_[state];

                relax(state, score + model_->log_self_loop(phone), links_[state]);
                if (!state_last_[state]) {
                    relax(state + 1, score + model_->log_exit(phone), links_[state]);
                }
            }

            // 3. Emission + pruning.
            float best = kNegInf;
            for (int state : next_active_) {
                next_scores_[state] += config_.acoustic_scale * acoustic_score(state_hmm_[state]);
                if (next_scores_[state] > best) best = next_scores_[state];
            }
            previous_best_ = best;
            result.states_visited += static_cast<int64_t>(next_active_.size());
            prune(best);

            // 4. Collect word ends for the next frame's entries.
            collect_word_ends(t, best);

            scores_.swap(next_scores_);
            links_.swap(next_links_);
            active_.swap(next_active_);
            for (int state : next_active_) next_scores_[state] = kNegInf;
        }

        finalize(result);
        return result;
    }

    void set_config(const DecoderConfig& config) { config_ = config; }
    const DecoderConfig& config() const { return config_; }

private:
    static constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    static constexpr int kSilenceWord = -2;

    struct WordEnd {
        int word;    // LM id used as the history for the next word
        float score;
        int link;    // word-link record of the predecessor
        bool emit = true;  // false for silence: it ends no word, so it writes
                           // no link record and contributes no transcript token
    };
    struct WordLink {
        int32_t word;
        int32_t frame;
        int32_t parent;
    };

    void add_chain(const std::vector<int>& phones, int word) {
        const int first = static_cast<int>(state_phone_.size());
        const int silence = phone_id("SIL");
        for (size_t p = 0; p < phones.size(); ++p) {
            // Word-internal triphone context: neighbours inside the word, and
            // silence at the word edges. Cross-word context would need one
            // network copy per boundary context, which is not worth the size.
            const int left = p > 0 ? phones[p - 1] : silence;
            const int right = p + 1 < phones.size() ? phones[p + 1] : silence;
            for (int s = 0; s < kNumStatesPerPhone; ++s) {
                state_phone_.push_back(phones[p]);
                state_hmm_.push_back(tree_ ? tree_->senone(left, phones[p], s, right)
                                           : state_index(phones[p], s));
                state_word_.push_back(word);
                state_last_.push_back(false);
            }
        }
        state_last_.back() = true;
        if (word >= 0) word_entry_[word].push_back(first);
    }

    // Scores one emitting unit for the current frame, reusing the cache when
    // another active state already needed it.
    inline float acoustic_score(int unit) {
        if (am_stamp_[unit] != frame_stamp_) {
            am_stamp_[unit] = frame_stamp_;
            am_scores_[unit] = model_->log_likelihood(unit, current_frame_);
            ++result_->gaussians_scored;
        }
        return am_scores_[unit];
    }

    // Writes into the next frame's arrays, keeping only the best predecessor.
    inline void relax(int state, float score, int link) {
        if (stamp_[state] != frame_stamp_) {
            stamp_[state] = frame_stamp_;
            next_scores_[state] = score;
            next_links_[state] = link;
            next_active_.push_back(state);
        } else if (score > next_scores_[state]) {
            next_scores_[state] = score;
            next_links_[state] = link;
        }
    }

    // Bigram expansion over the surviving word ends. Explicit bigrams are
    // enumerated per end; the backed-off mass is folded into one best value
    // that is applied to every word in a single vocabulary sweep.
    void expand_entries(int frame, DecodeResult& result) {
        if (ends_.empty()) return;

        // Pass 1: explicit bigram successors of the surviving word ends, plus
        // the single best backed-off predecessor.
        touched_.clear();
        float best_backoff = kNegInf;
        int best_backoff_link = -1;
        for (const WordEnd& end : ends_) {
            const float backoff = end.score + lm_->log_backoff(end.word);
            if (backoff > best_backoff) {
                best_backoff = backoff;
                best_backoff_link = end.link;
            }
            for (const NgramLm::Bigram* it = lm_->successors_begin(end.word);
                 it != lm_->successors_end(end.word); ++it) {
                const int word = it->word;
                if (word_entry_[word].empty()) continue;
                const float candidate = end.score + it->log_prob;
                if (entry_mark_[word] != frame_stamp_) {
                    entry_mark_[word] = frame_stamp_;
                    entry_score_[word] = candidate;
                    entry_link_[word] = end.link;
                    touched_.push_back(word);
                } else if (candidate > entry_score_[word]) {
                    entry_score_[word] = candidate;
                    entry_link_[word] = end.link;
                }
            }
        }

        const float penalty = config_.word_insertion_penalty;

        // Pass 2: enter every word that an explicit bigram reached, taking the
        // better of the bigram and the backed-off score.
        for (int word : touched_) {
            float score = entry_score_[word];
            int link = entry_link_[word];
            const float backed_off = best_backoff + lm_->log_unigram(word);
            if (backed_off > score) {
                score = backed_off;
                link = best_backoff_link;
            }
            score += penalty;
            for (int state : word_entry_[word]) relax(state, score, link);
            ++result.word_entries;
        }

        // Pass 3: the rest of the vocabulary can only be entered through the
        // backoff path, whose score decreases monotonically in unigram order -
        // so stop as soon as it falls below the beam floor instead of sweeping
        // all 20k words on every frame.
        const float floor = previous_best_ - config_.word_beam;
        for (int word : unigram_order_) {
            const float score = best_backoff + lm_->log_unigram(word) + penalty;
            if (score < floor) break;
            if (entry_mark_[word] == frame_stamp_) continue;  // entered in pass 2
            for (int state : word_entry_[word]) relax(state, score, best_backoff_link);
            ++result.word_entries;
        }

        // Silence inherits the best word end without an LM cost.
        if (silence_entry_ >= 0) {
            const WordEnd* best = &ends_[0];
            for (const WordEnd& end : ends_) {
                if (end.score > best->score) best = &end;
            }
            relax(silence_entry_, best->score, best->link);
        }
    }

    void prune(float best) {
        if (next_active_.empty()) return;
        const float floor = best - config_.beam;

        size_t kept = 0;
        for (size_t i = 0; i < next_active_.size(); ++i) {
            const int state = next_active_[i];
            if (next_scores_[state] >= floor) {
                next_active_[kept++] = state;
            } else {
                next_scores_[state] = kNegInf;
                stamp_[state] = -1;
            }
        }
        next_active_.resize(kept);

        // Histogram pruning: cap the active set by score rank.
        if (static_cast<int>(next_active_.size()) > config_.max_active) {
            std::nth_element(next_active_.begin(),
                             next_active_.begin() + config_.max_active,
                             next_active_.end(),
                             [this](int a, int b) { return next_scores_[a] > next_scores_[b]; });
            for (size_t i = config_.max_active; i < next_active_.size(); ++i) {
                next_scores_[next_active_[i]] = kNegInf;
                stamp_[next_active_[i]] = -1;
            }
            next_active_.resize(config_.max_active);
        }
    }

    void collect_word_ends(int frame, float best) {
        ends_.clear();
        candidates_.clear();

        for (int state : next_active_) {
            if (!state_last_[state]) continue;
            const int word = state_word_[state];
            const float exit = next_scores_[state] + model_->log_exit(state_phone_[state]);
            if (exit < best - config_.beam) continue;

            if (word == kSilenceWord) {
                // A pause ends no word, so it emits no link record and no
                // transcript token - but it must still re-enter the word-entry
                // pool, otherwise silence is a trap that swallows the token.
                // Recover the preceding word from its link so a pause preserves
                // the bigram history; leading silence retains sentence start.
                const int link = next_links_[state];
                const int history = link >= 0 ? word_links_[link].word : lm_->bos();
                candidates_.push_back({history, exit, link, false});
                continue;
            }
            candidates_.push_back({word, exit, next_links_[state], true});
        }

        if (candidates_.empty()) return;

        // Keep only the best token per word, then the top-K words overall.
        std::sort(candidates_.begin(), candidates_.end(),
                  [](const WordEnd& a, const WordEnd& b) {
                      return a.word != b.word ? a.word < b.word : a.score > b.score;
                  });
        int previous = -1;
        for (const WordEnd& candidate : candidates_) {
            if (candidate.word == previous) continue;
            previous = candidate.word;
            ends_.push_back(candidate);
        }
        if (static_cast<int>(ends_.size()) > config_.max_word_ends) {
            std::nth_element(ends_.begin(), ends_.begin() + config_.max_word_ends, ends_.end(),
                             [](const WordEnd& a, const WordEnd& b) { return a.score > b.score; });
            ends_.resize(config_.max_word_ends);
        }

        // Record each surviving word end so the backtrace can recover the
        // transcript. Silence exits keep their predecessor's link instead.
        for (WordEnd& end : ends_) {
            if (!end.emit) continue;
            word_links_.push_back({static_cast<int32_t>(end.word), static_cast<int32_t>(frame),
                                   static_cast<int32_t>(end.link)});
            end.link = static_cast<int>(word_links_.size()) - 1;
        }
    }

    void finalize(DecodeResult& result) {
        float best = kNegInf;
        int best_link = -1;
        for (const WordEnd& end : ends_) {
            const float score = end.score + lm_->log_probability(end.word, lm_->eos());
            if (score > best) {
                best = score;
                best_link = end.link;
            }
        }
        if (best_link < 0) {
            // No word finished inside the beam; fall back to the best partial
            // hypothesis so the caller still gets a (short) transcript.
            for (int state : active_) {
                if (scores_[state] > best) {
                    best = scores_[state];
                    best_link = links_[state];
                }
            }
        }
        result.score = best;

        std::vector<std::string> reversed;
        int link = best_link;
        while (link >= 0 && link < static_cast<int>(word_links_.size())) {
            const WordLink& record = word_links_[link];
            reversed.push_back(lm_->word(record.word));
            link = record.parent;
        }
        result.words.assign(reversed.rbegin(), reversed.rend());
    }

    // network
    std::vector<int> state_phone_;
    std::vector<int> state_hmm_;
    std::vector<int> state_word_;
    std::vector<char> state_last_;
    std::vector<std::vector<int>> word_entry_;
    int silence_entry_ = -1;
    int skipped_words_ = 0;

    // per-utterance search state
    std::vector<float> scores_, next_scores_;
    std::vector<int> links_, next_links_;
    std::vector<int> stamp_;
    std::vector<int> active_, next_active_;
    std::vector<float> am_scores_;
    std::vector<int> am_stamp_;
    const float* current_frame_ = nullptr;
    DecodeResult* result_ = nullptr;
    std::vector<float> entry_score_;
    std::vector<int> entry_link_;
    std::vector<int> entry_mark_;
    std::vector<int> touched_;
    std::vector<int> unigram_order_;
    float previous_best_ = -std::numeric_limits<float>::infinity();
    std::vector<WordEnd> ends_, candidates_;
    std::vector<WordLink> word_links_;
    int frame_stamp_ = 0;

    const NgramLm* lm_ = nullptr;
    const AcousticModel* model_ = nullptr;
    const TriphoneTree* tree_ = nullptr;
    DecoderConfig config_;
};

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\toy_pruned_hmm\dtw\dtw.hpp ===
// Dynamic Time Warping over MFCC trajectories, plus a 1-NN template recogniser
// for isolated words (Tier 1: spoken digits).
//
// This is Step 1 of models/README.md: instead of collapsing repeated frames
// with `if (vowel == previous_vowel) continue;`, we align the whole feature
// trajectory against reference templates, which absorbs speaking-rate variation
// without any trained weights.
//
// Speed notes (the "optimality" half of this file):
//   * Sakoe-Chiba band: only |i*ratio - j| <= radius cells are visited, turning
//     the O(N*M) grid into a diagonal ribbon.
//   * Two-row rolling buffer: memory is O(min(N,M)) instead of O(N*M), so a
//     comparison stays in L1/L2 cache.
//   * Early abandoning: if the best cell in a row already exceeds the best
//     distance found so far, the comparison cannot win and is dropped.
//   * Envelope lower bound (LB_Keogh-style, on frame means): a cheap O(T) bound
//     that skips most templates before any DTW cell is touched.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <filter/mfcc.hpp>

namespace models {

// Squared euclidean distance between two frames, unrolled in blocks of 4.
inline float frame_distance(const float* a, const float* b, int dim) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int d = 0;
    for (; d + 4 <= dim; d += 4) {
        const float d0 = a[d] - b[d];
        const float d1 = a[d + 1] - b[d + 1];
        const float d2 = a[d + 2] - b[d + 2];
        const float d3 = a[d + 3] - b[d + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    float sum = s0 + s1 + s2 + s3;
    for (; d < dim; ++d) {
        const float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

// Banded DTW with early abandoning. Returns the length-normalised path cost,
// or `abandon_above` if the alignment provably cannot beat that threshold.
class DtwAligner {
public:
    float distance(const FeatureMatrix& query, const FeatureMatrix& reference,
                   int radius = 12,
                   float abandon_above = std::numeric_limits<float>::infinity()) {
        const int n = query.num_frames;
        const int m = reference.num_frames;
        if (n == 0 || m == 0) return std::numeric_limits<float>::infinity();
        const int dim = query.dim;

        // Band follows the diagonal of a non-square grid.
        const float ratio = static_cast<float>(m) / static_cast<float>(n);
        const int band = std::max(radius, std::abs(n - m) / 2 + radius);

        constexpr float kInf = std::numeric_limits<float>::infinity();
        previous_.assign(m + 1, kInf);
        current_.assign(m + 1, kInf);
        previous_[0] = 0.0f;

        for (int i = 1; i <= n; ++i) {
            const int center = static_cast<int>((i - 1) * ratio) + 1;
            const int lo = std::max(1, center - band);
            const int hi = std::min(m, center + band);

            std::fill(current_.begin(), current_.end(), kInf);
            const float* query_frame = query.frame(i - 1);

            float row_best = kInf;
            for (int j = lo; j <= hi; ++j) {
                const float best_predecessor =
                    std::min(previous_[j], std::min(current_[j - 1], previous_[j - 1]));
                if (best_predecessor == kInf) continue;
                const float cost = frame_distance(query_frame, reference.frame(j - 1), dim);
                const float value = cost + best_predecessor;
                current_[j] = value;
                if (value < row_best) row_best = value;
            }

            // Every remaining path passes through this row, so a row minimum
            // above the incumbent best is a proof of loss.
            if (row_best >= abandon_above * (n + m)) return kInf;
            previous_.swap(current_);
        }

        const float total = previous_[m];
        if (total == kInf) return kInf;
        return total / static_cast<float>(n + m);
    }

private:
    std::vector<float> previous_;
    std::vector<float> current_;
};

// One stored reference utterance.
struct Template {
    std::string label;
    FeatureMatrix features;
};

// LB_Kim lower bound on the *normalised* DTW cost.
//
// Every warping path must contain (1,1) and (N,M), so the aligned cost is at
// least the cost of those two pairs. Dividing by (N+M) - the same denominator
// the aligner uses - keeps the bound admissible, so pruning on it can never
// discard the true nearest neighbour. Cheap: O(D), independent of length.
inline float lower_bound_kim(const FeatureMatrix& query, const FeatureMatrix& reference) {
    if (query.empty() || reference.empty()) return 0.0f;
    const int dim = query.dim;
    const float first = frame_distance(query.frame(0), reference.frame(0), dim);
    const float last = frame_distance(query.frame(query.num_frames - 1),
                                      reference.frame(reference.num_frames - 1), dim);
    return (first + last) / static_cast<float>(query.num_frames + reference.num_frames);
}

// Band-aware LB_Keogh over the first `dims` cepstral coefficients.
//
// Why it is admissible: any warping path maps query frame i to at least one
// reference frame inside the Sakoe-Chiba band [lo_i, hi_i], so the path cost is
// at least sum_i min_{j in band} d(q_i, r_j). Replacing that inner minimum with
// the distance to the band's per-dimension envelope, and summing over a subset
// of dimensions, can only lower the value further - so the result never exceeds
// the true DTW cost and pruning on it stays exact.
//
// Why it is cheap: lo_i and hi_i are non-decreasing, so the envelope is
// maintained with monotonic deques in O(n + m) per dimension - roughly two
// orders of magnitude less work than the full 39-dimensional alignment.
inline float lower_bound_keogh(const FeatureMatrix& query, const FeatureMatrix& reference,
                               int radius, int dims = 4) {
    const int n = query.num_frames;
    const int m = reference.num_frames;
    if (n == 0 || m == 0) return 0.0f;
    dims = std::min(dims, query.dim);

    const float ratio = static_cast<float>(m) / static_cast<float>(n);
    const int band = std::max(radius, std::abs(n - m) / 2 + radius);

    static thread_local std::vector<int> max_deque, min_deque;
    float total = 0.0f;

    for (int d = 0; d < dims; ++d) {
        max_deque.clear();
        min_deque.clear();
        // Head indices instead of erase(begin()): popping the front of a
        // vector is O(size), which would make this bound cost more than the
        // alignment it is meant to avoid.
        size_t max_head = 0, min_head = 0;
        int filled = 0;  // reference frames pushed so far

        for (int i = 0; i < n; ++i) {
            const int center = static_cast<int>(i * ratio);
            const int lo = std::max(0, center - band);
            const int hi = std::min(m - 1, center + band);

            while (filled <= hi) {
                const float value = reference.frame(filled)[d];
                while (max_deque.size() > max_head &&
                       reference.frame(max_deque.back())[d] <= value) {
                    max_deque.pop_back();
                }
                max_deque.push_back(filled);
                while (min_deque.size() > min_head &&
                       reference.frame(min_deque.back())[d] >= value) {
                    min_deque.pop_back();
                }
                min_deque.push_back(filled);
                ++filled;
            }
            while (max_deque.size() > max_head && max_deque[max_head] < lo) ++max_head;
            while (min_deque.size() > min_head && min_deque[min_head] < lo) ++min_head;
            if (max_deque.size() == max_head || min_deque.size() == min_head) continue;

            const float upper = reference.frame(max_deque[max_head])[d];
            const float lower = reference.frame(min_deque[min_head])[d];
            const float value = query.frame(i)[d];
            if (value > upper) {
                const float diff = value - upper;
                total += diff * diff;
            } else if (value < lower) {
                const float diff = lower - value;
                total += diff * diff;
            }
        }
    }
    return total / static_cast<float>(n + m);
}

// Energy-based endpointing. Isolated-digit clips are padded to a fixed length
// with leading/trailing silence; warping silence against silence is both slow
// and a source of false matches, so trim to the spoken region first.
// Operates on c0 (frame log-energy), which is dimension 0 of the MFCC vector.
inline FeatureMatrix endpoint(const FeatureMatrix& features, float threshold = 0.35f,
                              int margin = 3) {
    if (features.num_frames < 5) return features;

    float low = features.frame(0)[0];
    float high = low;
    for (int t = 1; t < features.num_frames; ++t) {
        low = std::min(low, features.frame(t)[0]);
        high = std::max(high, features.frame(t)[0]);
    }
    if (high - low < 1e-3f) return features;

    const float cutoff = low + threshold * (high - low);
    int start = 0, stop = features.num_frames - 1;
    while (start < features.num_frames && features.frame(start)[0] < cutoff) ++start;
    while (stop > start && features.frame(stop)[0] < cutoff) --stop;

    start = std::max(0, start - margin);
    stop = std::min(features.num_frames - 1, stop + margin);
    if (stop - start + 1 < 5) return features;

    FeatureMatrix trimmed;
    trimmed.resize(stop - start + 1, features.dim);
    std::copy(features.frame(start), features.frame(stop) + features.dim,
              trimmed.data.begin());
    return trimmed;
}

// 1-NN / k-NN isolated word recogniser over stored templates.
class DtwRecognizer {
public:
    struct Result {
        std::string label;
        float distance = std::numeric_limits<float>::infinity();
        int templates_scored = 0;   // how many survived the bound
        int templates_pruned = 0;   // skipped by the admissible lower bound
        int templates_abandoned = 0;  // DTW started but proven hopeless mid-way
    };

    void add_template(const std::string& label, FeatureMatrix features) {
        Template entry;
        entry.label = label;
        entry.features = std::move(features);
        templates_.push_back(std::move(entry));
    }

    size_t size() const { return templates_.size(); }
    const std::vector<Template>& templates() const { return templates_; }

    // k-NN with distance-weighted vote; k=1 is plain nearest neighbour.
    //
    // Templates are visited in increasing lower-bound order so a tight `best`
    // is found early, which is what makes both the bound test and the DTW
    // early-abandon fire often.
    Result classify(const FeatureMatrix& query, int k = 1, int radius = 12,
                    bool use_lower_bound = false) {
        Result result;
        if (templates_.empty() || query.empty()) return result;

        // LB_Kim is always worth its O(D) cost: it orders the templates so a
        // tight incumbent appears early. LB_Keogh is opt-in - measured on
        // 39-dim CMVN features with a +/-12 frame band it prunes essentially
        // nothing while costing O(dims * (n + m)) per template, because the
        // per-dimension envelope over that many frames is almost never
        // violated. It pays off only for narrow bands or low-dim features.
        order_.clear();
        order_.reserve(templates_.size());
        for (size_t i = 0; i < templates_.size(); ++i) {
            const FeatureMatrix& reference = templates_[i].features;
            float bound = lower_bound_kim(query, reference);
            if (use_lower_bound) {
                bound = std::max(bound, lower_bound_keogh(query, reference, radius, 13));
            }
            order_.push_back({bound, i});
        }
        std::sort(order_.begin(), order_.end(),
                  [](const Bounded& a, const Bounded& b) { return a.bound < b.bound; });

        scored_.clear();
        float best = std::numeric_limits<float>::infinity();
        for (const Bounded& candidate : order_) {
            const Template& entry = templates_[candidate.index];
            // Admissible bound: if it already exceeds the incumbent, no
            // alignment of this template can win.
            if (k == 1 && candidate.bound >= best) {
                ++result.templates_pruned;
                continue;
            }
            const float distance = aligner_.distance(
                query, entry.features, radius,
                k == 1 ? best : std::numeric_limits<float>::infinity());
            ++result.templates_scored;
            if (distance == std::numeric_limits<float>::infinity()) {
                ++result.templates_abandoned;
                continue;
            }
            if (distance < best) best = distance;
            scored_.push_back({distance, &entry});
        }

        if (scored_.empty()) return result;
        const int neighbours = std::min<int>(k, static_cast<int>(scored_.size()));
        std::partial_sort(scored_.begin(), scored_.begin() + neighbours, scored_.end(),
                          [](const Scored& a, const Scored& b) { return a.distance < b.distance; });

        // Distance-weighted vote across the k nearest templates.
        std::vector<std::pair<std::string, float>> votes;
        for (int i = 0; i < neighbours; ++i) {
            const std::string& label = scored_[i].entry->label;
            const float weight = 1.0f / (scored_[i].distance + 1e-6f);
            auto it = std::find_if(votes.begin(), votes.end(),
                                   [&](const auto& v) { return v.first == label; });
            if (it == votes.end()) {
                votes.emplace_back(label, weight);
            } else {
                it->second += weight;
            }
        }
        const auto winner = std::max_element(
            votes.begin(), votes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        result.label = winner->first;
        result.distance = scored_[0].distance;
        return result;
    }

private:
    struct Scored {
        float distance;
        const Template* entry;
    };
    struct Bounded {
        float bound;
        size_t index;
    };

    std::vector<Template> templates_;
    std::vector<Scored> scored_;
    std::vector<Bounded> order_;
    DtwAligner aligner_;
};

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\toy_pruned_hmm\ngram\ngram_lm.hpp ===
// Interpolated absolute-discount bigram language model (Step 3 of models/README.md).
//
// Trained on LibriSpeech train-clean-100 transcripts, which are speaker- and
// book-disjoint from the test-clean set we evaluate on - training an LM on the
// evaluation transcripts would make the reported WER meaningless.
//
// Layout is built for the decoder's access pattern: bigrams are sorted by
// (history, word) and indexed by a per-history [begin, end) range, so
// enumerating "all successors of word w" - which the decoder does once per
// surviving word end per frame - is a contiguous scan.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace models {

class NgramLm {
public:
    struct Bigram {
        int32_t word;
        float log_prob;
    };

    // ---- training -------------------------------------------------------

    // One line = one utterance. `vocabulary_limit` keeps the most frequent
    // words; everything else maps to <unk>.
    bool train_from_text(const std::string& path, int vocabulary_limit = 20000,
                         double discount = 0.4) {
        std::ifstream file(path);
        if (!file) return false;

        std::unordered_map<std::string, int64_t> word_counts;
        std::vector<std::vector<std::string>> sentences;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream stream(line);
            std::vector<std::string> tokens;
            std::string token;
            while (stream >> token) {
                for (char& c : token) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                }
                tokens.push_back(token);
                ++word_counts[token];
            }
            if (!tokens.empty()) sentences.push_back(std::move(tokens));
        }
        if (sentences.empty()) return false;

        // Vocabulary: <s>, </s>, <unk> then the most frequent words.
        std::vector<std::pair<std::string, int64_t>> ranked(word_counts.begin(),
                                                           word_counts.end());
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first;
        });

        words_.clear();
        word_ids_.clear();
        add_word("<s>");
        add_word("</s>");
        add_word("<unk>");
        for (const auto& entry : ranked) {
            if (static_cast<int>(words_.size()) >= vocabulary_limit) break;
            if (word_ids_.count(entry.first)) continue;
            add_word(entry.first);
        }

        // Counts.
        std::vector<int64_t> unigram_counts(words_.size(), 0);
        std::unordered_map<int64_t, int64_t> bigram_counts;
        int64_t total_unigrams = 0;
        for (const std::vector<std::string>& sentence : sentences) {
            int previous = kBos;
            ++unigram_counts[kBos];
            ++total_unigrams;
            for (const std::string& token : sentence) {
                const int id = lookup_or_unk(token);
                ++unigram_counts[id];
                ++total_unigrams;
                ++bigram_counts[(static_cast<int64_t>(previous) << 32) | id];
                previous = id;
            }
            ++unigram_counts[kEos];
            ++total_unigrams;
            ++bigram_counts[(static_cast<int64_t>(previous) << 32) | kEos];
        }

        // Unigrams with add-one smoothing over the closed vocabulary.
        const double vocabulary = static_cast<double>(words_.size());
        log_unigram_.assign(words_.size(), 0.0f);
        for (size_t w = 0; w < words_.size(); ++w) {
            log_unigram_[w] = static_cast<float>(
                std::log((unigram_counts[w] + 1.0) / (total_unigrams + vocabulary)));
        }

        // Bigrams with absolute discounting; the discounted mass becomes the
        // per-history backoff weight.
        std::vector<std::vector<Bigram>> by_history(words_.size());
        std::vector<double> backoff_mass(words_.size(), 1.0);
        for (const auto& entry : bigram_counts) {
            const int history = static_cast<int>(entry.first >> 32);
            const int word = static_cast<int>(entry.first & 0xFFFFFFFF);
            const double count = static_cast<double>(entry.second);
            const double history_count = static_cast<double>(unigram_counts[history]);
            if (history_count <= 0.0 || count <= discount) continue;
            const double probability = (count - discount) / history_count;
            by_history[history].push_back({word, static_cast<float>(std::log(probability))});
            backoff_mass[history] -= probability;
        }

        bigrams_.clear();
        bigram_begin_.assign(words_.size() + 1, 0);
        log_backoff_.assign(words_.size(), 0.0f);
        for (size_t h = 0; h < words_.size(); ++h) {
            bigram_begin_[h] = static_cast<int32_t>(bigrams_.size());
            std::sort(by_history[h].begin(), by_history[h].end(),
                      [](const Bigram& a, const Bigram& b) { return a.word < b.word; });
            const double mass = backoff_mass[h] > 1e-6 ? backoff_mass[h] : 1e-6;
            // Interpolate the removed mass over ALL vocabulary entries.
            // Applying it only to unseen successors loses probability mass.
            // Explicit scores then dominate their backoff contribution, as
            // required by the decoder's shared best-backoff expansion.
            for (Bigram& entry : by_history[h]) {
                entry.log_prob = static_cast<float>(std::log(
                    std::exp(entry.log_prob) + mass * std::exp(log_unigram_[entry.word])));
            }
            bigrams_.insert(bigrams_.end(), by_history[h].begin(), by_history[h].end());
            log_backoff_[h] = static_cast<float>(std::log(mass));
        }
        bigram_begin_[words_.size()] = static_cast<int32_t>(bigrams_.size());
        return true;
    }

    // ---- query ----------------------------------------------------------

    int word_id(const std::string& word) const {
        const auto it = word_ids_.find(word);
        return it == word_ids_.end() ? -1 : it->second;
    }
    const std::string& word(int id) const { return words_[id]; }
    int vocabulary_size() const { return static_cast<int>(words_.size()); }
    int bos() const { return kBos; }
    int eos() const { return kEos; }
    int unk() const { return kUnk; }

    float log_unigram(int word) const { return log_unigram_[word]; }
    float log_backoff(int history) const { return log_backoff_[history]; }

    // Contiguous successor range for a history word.
    const Bigram* successors_begin(int history) const {
        return bigrams_.data() + bigram_begin_[history];
    }
    const Bigram* successors_end(int history) const {
        return bigrams_.data() + bigram_begin_[history + 1];
    }

    float log_probability(int history, int word) const {
        const Bigram* begin = successors_begin(history);
        const Bigram* end = successors_end(history);
        const Bigram* hit = std::lower_bound(
            begin, end, word, [](const Bigram& b, int w) { return b.word < w; });
        if (hit != end && hit->word == word) return hit->log_prob;
        return log_backoff_[history] + log_unigram_[word];
    }

    // Perplexity on held-out text, for sanity-checking the LM in isolation.
    double perplexity(const std::vector<std::vector<std::string>>& sentences) const {
        double log_sum = 0.0;
        int64_t tokens = 0;
        for (const auto& sentence : sentences) {
            int previous = kBos;
            for (const std::string& token : sentence) {
                const int id = lookup_or_unk_const(token);
                log_sum += log_probability(previous, id);
                previous = id;
                ++tokens;
            }
            log_sum += log_probability(previous, kEos);
            ++tokens;
        }
        if (tokens == 0) return 0.0;
        return std::exp(-log_sum / tokens);
    }

    // ---- persistence ----------------------------------------------------

    bool save(const std::string& path) const {
        FILE* file = std::fopen(path.c_str(), "wb");
        if (!file) return false;
        const int32_t magic = 0x4C4D4231;  // "LMB1"
        const int32_t vocabulary = static_cast<int32_t>(words_.size());
        const int32_t bigrams = static_cast<int32_t>(bigrams_.size());
        std::fwrite(&magic, sizeof(magic), 1, file);
        std::fwrite(&vocabulary, sizeof(vocabulary), 1, file);
        std::fwrite(&bigrams, sizeof(bigrams), 1, file);
        for (const std::string& word : words_) {
            const int32_t length = static_cast<int32_t>(word.size());
            std::fwrite(&length, sizeof(length), 1, file);
            std::fwrite(word.data(), 1, word.size(), file);
        }
        std::fwrite(log_unigram_.data(), sizeof(float), log_unigram_.size(), file);
        std::fwrite(log_backoff_.data(), sizeof(float), log_backoff_.size(), file);
        std::fwrite(bigram_begin_.data(), sizeof(int32_t), bigram_begin_.size(), file);
        std::fwrite(bigrams_.data(), sizeof(Bigram), bigrams_.size(), file);
        std::fclose(file);
        return true;
    }

    bool load(const std::string& path) {
        FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) return false;
        int32_t magic = 0, vocabulary = 0, bigrams = 0;
        if (std::fread(&magic, sizeof(magic), 1, file) != 1 ||
            std::fread(&vocabulary, sizeof(vocabulary), 1, file) != 1 ||
            std::fread(&bigrams, sizeof(bigrams), 1, file) != 1 || magic != 0x4C4D4231) {
            std::fclose(file);
            return false;
        }
        words_.clear();
        word_ids_.clear();
        words_.reserve(vocabulary);
        for (int i = 0; i < vocabulary; ++i) {
            int32_t length = 0;
            if (std::fread(&length, sizeof(length), 1, file) != 1 || length < 0) {
                std::fclose(file);
                return false;
            }
            std::string word(static_cast<size_t>(length), '\0');
            if (length > 0 && std::fread(&word[0], 1, length, file) != static_cast<size_t>(length)) {
                std::fclose(file);
                return false;
            }
            word_ids_[word] = static_cast<int>(words_.size());
            words_.push_back(std::move(word));
        }
        log_unigram_.resize(vocabulary);
        log_backoff_.resize(vocabulary);
        bigram_begin_.resize(static_cast<size_t>(vocabulary) + 1);
        bigrams_.resize(bigrams);
        const bool ok =
            std::fread(log_unigram_.data(), sizeof(float), log_unigram_.size(), file) ==
                log_unigram_.size() &&
            std::fread(log_backoff_.data(), sizeof(float), log_backoff_.size(), file) ==
                log_backoff_.size() &&
            std::fread(bigram_begin_.data(), sizeof(int32_t), bigram_begin_.size(), file) ==
                bigram_begin_.size() &&
            std::fread(bigrams_.data(), sizeof(Bigram), bigrams_.size(), file) == bigrams_.size();
        std::fclose(file);
        return ok;
    }

private:
    static constexpr int kBos = 0;
    static constexpr int kEos = 1;
    static constexpr int kUnk = 2;

    void add_word(const std::string& word) {
        word_ids_[word] = static_cast<int>(words_.size());
        words_.push_back(word);
    }
    int lookup_or_unk(const std::string& word) {
        const auto it = word_ids_.find(word);
        return it == word_ids_.end() ? kUnk : it->second;
    }
    int lookup_or_unk_const(const std::string& word) const {
        const auto it = word_ids_.find(word);
        return it == word_ids_.end() ? kUnk : it->second;
    }

    std::vector<std::string> words_;
    std::unordered_map<std::string, int> word_ids_;
    std::vector<float> log_unigram_;
    std::vector<float> log_backoff_;
    std::vector<int32_t> bigram_begin_;
    std::vector<Bigram> bigrams_;
};

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\toy_pruned_hmm\phonemes\g2p.hpp ===
// Rule-based grapheme-to-phoneme fallback for words missing from CMUDict.
//
// Why this is worth having: in LibriSpeech train-clean-100, only ~1.8 % of word
// tokens are outside CMUDict - almost all of them proper nouns from 19th
// century novels (CHAUVELIN, PENCROFT, KENNICOTT). But utterances are ~34 words
// long, so that 1.8 % lands in 41 % of utterances. Dropping those utterances
// throws away 41 % of the acoustic training audio to avoid mis-spelling 1.8 %
// of the words; guessing a pronunciation is the better trade.
//
// This is deliberately a small letter-to-sound ruleset, not a trained G2P: it
// handles English digraphs, the silent-final-E rule and common suffixes. It is
// used only for forced alignment of training audio, never for decoding - the
// decoding lexicon stays CMUDict-only, so a bad guess here cannot invent a
// word in a transcript.
#pragma once

#include <string>
#include <vector>

#include "phone_set.hpp"

namespace models {

class GraphemeToPhoneme {
public:
    // Returns phone ids for an uppercase A-Z (+ apostrophe) word.
    std::vector<int> convert(const std::string& word) const {
        std::vector<std::string> phones;
        std::string text;
        for (char c : word) {
            if (c >= 'A' && c <= 'Z') text.push_back(c);
        }
        if (text.empty()) return {};

        const size_t n = text.size();
        for (size_t i = 0; i < n;) {
            const char c = text[i];
            const char next = i + 1 < n ? text[i + 1] : '\0';
            const char after = i + 2 < n ? text[i + 2] : '\0';
            const bool at_end = i + 1 == n;

            // --- two- and three-letter graphemes -------------------------
            if (c == 'C' && next == 'H') { phones.push_back("CH"); i += 2; continue; }
            if (c == 'S' && next == 'H') { phones.push_back("SH"); i += 2; continue; }
            if (c == 'T' && next == 'H') { phones.push_back("TH"); i += 2; continue; }
            if (c == 'P' && next == 'H') { phones.push_back("F");  i += 2; continue; }
            if (c == 'W' && next == 'H') { phones.push_back("W");  i += 2; continue; }
            if (c == 'C' && next == 'K') { phones.push_back("K");  i += 2; continue; }
            if (c == 'N' && next == 'G' && at_end_after(text, i + 1)) {
                phones.push_back("NG"); i += 2; continue;
            }
            if (c == 'Q' && next == 'U') { phones.push_back("K"); phones.push_back("W"); i += 2; continue; }
            if (c == 'G' && next == 'H') {
                // "GH" is silent word-finally (THROUGH) and /f/ in LAUGH-like
                // endings; treat the common cases and otherwise drop it.
                if (i + 2 == n && i > 0 && (text[i - 1] == 'U')) { i += 2; continue; }
                phones.push_back("F"); i += 2; continue;
            }
            if (c == 'P' && next == 'S' && i == 0) { phones.push_back("S"); i += 2; continue; }
            if (c == 'K' && next == 'N' && i == 0) { phones.push_back("N"); i += 2; continue; }
            if (c == 'W' && next == 'R' && i == 0) { phones.push_back("R"); i += 2; continue; }

            // --- vowels --------------------------------------------------
            if (is_vowel(c)) {
                // Common vowel digraphs.
                if (c == 'E' && next == 'E') { phones.push_back("IY"); i += 2; continue; }
                if (c == 'E' && next == 'A') { phones.push_back("IY"); i += 2; continue; }
                if (c == 'O' && next == 'O') { phones.push_back("UW"); i += 2; continue; }
                if (c == 'O' && next == 'U') { phones.push_back("AW"); i += 2; continue; }
                if (c == 'O' && next == 'W') { phones.push_back("OW"); i += 2; continue; }
                if (c == 'O' && next == 'I') { phones.push_back("OY"); i += 2; continue; }
                if (c == 'A' && next == 'I') { phones.push_back("EY"); i += 2; continue; }
                if (c == 'A' && next == 'Y') { phones.push_back("EY"); i += 2; continue; }
                if (c == 'A' && next == 'U') { phones.push_back("AA"); i += 2; continue; }
                if (c == 'E' && next == 'I') { phones.push_back("EY"); i += 2; continue; }
                if (c == 'I' && next == 'E') { phones.push_back("IY"); i += 2; continue; }

                // Silent final E (NAME, HOPE) - lengthens the previous vowel,
                // which the digraph rules above already approximate.
                if (c == 'E' && at_end && phones.size() > 0 && i > 0 && !is_vowel(text[i - 1])) {
                    ++i;
                    continue;
                }
                // R-coloured vowels.
                if (next == 'R' && (after == '\0' || !is_vowel(after))) {
                    phones.push_back("ER");
                    i += 2;
                    continue;
                }
                // Long vowel when a single consonant is followed by a final E.
                const bool magic_e = (i + 3 == n && !is_vowel(next) && after == 'E');
                phones.push_back(vowel_phone(c, magic_e));
                ++i;
                continue;
            }

            // --- consonants ----------------------------------------------
            switch (c) {
                case 'B': phones.push_back("B"); break;
                case 'C': phones.push_back(is_front_vowel(next) ? "S" : "K"); break;
                case 'D': phones.push_back("D"); break;
                case 'F': phones.push_back("F"); break;
                case 'G': phones.push_back(is_front_vowel(next) ? "JH" : "G"); break;
                case 'H': phones.push_back("HH"); break;
                case 'J': phones.push_back("JH"); break;
                case 'K': phones.push_back("K"); break;
                case 'L': phones.push_back("L"); break;
                case 'M': phones.push_back("M"); break;
                case 'N': phones.push_back("N"); break;
                case 'P': phones.push_back("P"); break;
                case 'R': phones.push_back("R"); break;
                case 'S': phones.push_back("S"); break;
                case 'T': phones.push_back("T"); break;
                case 'V': phones.push_back("V"); break;
                case 'W': phones.push_back("W"); break;
                case 'X': phones.push_back("K"); phones.push_back("S"); break;
                case 'Y': phones.push_back(i == 0 ? "Y" : "IY"); break;
                case 'Z': phones.push_back("Z"); break;
                default: break;
            }
            // Collapse doubled consonants (LL, TT, SS).
            if (next == c) ++i;
            ++i;
        }

        std::vector<int> ids;
        ids.reserve(phones.size());
        for (const std::string& phone : phones) {
            const int id = phone_id(phone);
            if (id >= 0) ids.push_back(id);
        }
        return ids;
    }

private:
    static bool is_vowel(char c) {
        return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
    }
    static bool is_front_vowel(char c) { return c == 'E' || c == 'I' || c == 'Y'; }
    static bool at_end_after(const std::string& text, size_t index) {
        return index + 1 >= text.size() || !is_vowel(text[index + 1]);
    }
    static const char* vowel_phone(char c, bool magic_e) {
        switch (c) {
            case 'A': return magic_e ? "EY" : "AE";
            case 'E': return magic_e ? "IY" : "EH";
            case 'I': return magic_e ? "AY" : "IH";
            case 'O': return magic_e ? "OW" : "AA";
            case 'U': return magic_e ? "UW" : "AH";
            default: return "AH";
        }
    }
};

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\toy_pruned_hmm\phonemes\lexicon.hpp ===
// Pronunciation lexicon (Step 2 of models/README.md).
//
// Loads the CMU Pronouncing Dictionary and turns every word into a sequence of
// phone ids from phone_set.hpp. The decoder never sees text until a word HMM
// finishes; this table is the only bridge between acoustics and orthography.
#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "phone_set.hpp"

namespace models {

struct Pronunciation {
    int word_id = -1;
    std::vector<int> phones;
};

class Lexicon {
public:
    // `vocabulary` restricts the lexicon to words we actually decode (empty =
    // keep everything). A 200k-word lexicon is mostly dead weight for a
    // read-speech test set and slows the search down for nothing.
    bool load_cmudict(const std::string& path,
                      const std::unordered_set<std::string>& vocabulary = {}) {
        std::ifstream file(path);
        if (!file) return false;

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == ';') continue;

            std::istringstream stream(line);
            std::string word;
            stream >> word;
            if (word.empty()) continue;

            // "word(2)" marks an alternate pronunciation of the same word.
            const size_t paren = word.find('(');
            if (paren != std::string::npos) word = word.substr(0, paren);
            for (char& c : word) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            if (!vocabulary.empty() && vocabulary.find(word) == vocabulary.end()) continue;

            std::vector<int> phones;
            std::string token;
            bool usable = true;
            while (stream >> token) {
                if (token[0] == '#') break;  // trailing comment
                const int phone = phone_id(strip_stress(token));
                if (phone < 0) {
                    usable = false;
                    break;
                }
                phones.push_back(phone);
            }
            if (!usable || phones.empty()) continue;

            int id;
            const auto it = word_ids_.find(word);
            if (it == word_ids_.end()) {
                id = static_cast<int>(words_.size());
                word_ids_[word] = id;
                words_.push_back(word);
            } else {
                id = it->second;
            }
            pronunciations_.push_back({id, std::move(phones)});
        }
        return !pronunciations_.empty();
    }

    // Programmatic entry, used to build non-word inventories - the Tier 2
    // phone loop is just a lexicon whose "words" are single phones.
    void add_entry(const std::string& word, const std::vector<int>& phones) {
        if (phones.empty()) return;
        int id;
        const auto it = word_ids_.find(word);
        if (it == word_ids_.end()) {
            id = static_cast<int>(words_.size());
            word_ids_[word] = id;
            words_.push_back(word);
        } else {
            id = it->second;
        }
        pronunciations_.push_back({id, phones});
    }

    // Words with no CMUDict entry are unrecoverable for the decoder; report
    // them so evaluation can separate lexicon gaps from search errors.
    int word_id(const std::string& word) const {
        const auto it = word_ids_.find(word);
        return it == word_ids_.end() ? -1 : it->second;
    }

    const std::string& word(int id) const { return words_[id]; }
    int num_words() const { return static_cast<int>(words_.size()); }
    const std::vector<Pronunciation>& pronunciations() const { return pronunciations_; }
    const std::vector<std::string>& words() const { return words_; }

private:
    std::vector<std::string> words_;
    std::unordered_map<std::string, int> word_ids_;
    std::vector<Pronunciation> pronunciations_;
};

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\toy_pruned_hmm\phonemes\phone_set.hpp ===
// Phone inventory shared by the acoustic model, the lexicon and the decoder.
//
// TIMIT labels its alignments with 61 symbols; CMUDict spells words with the
// 39 ARPAbet phones. Training on TIMIT and decoding with CMUDict only works if
// both collapse onto one inventory, so this header owns that folding (the
// standard Lee & Hon 61 -> 39 map) plus a silence unit.
#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace models {

// 39 ARPAbet phones + SIL. Index 0 is silence so an empty/zero state is safe.
inline const std::vector<std::string>& phone_names() {
    static const std::vector<std::string> names = {
        "SIL",
        "AA", "AE", "AH", "AO", "AW", "AY", "B",  "CH", "D",  "DH",
        "EH", "ER", "EY", "F",  "G",  "HH", "IH", "IY", "JH", "K",
        "L",  "M",  "N",  "NG", "OW", "OY", "P",  "R",  "S",  "SH",
        "T",  "TH", "UH", "UW", "V",  "W",  "Y",  "Z",  "ZH"};
    return names;
}

constexpr int kNumStatesPerPhone = 3;  // left-to-right HMM, no skips

inline int num_phones() { return static_cast<int>(phone_names().size()); }
inline int num_states() { return num_phones() * kNumStatesPerPhone; }

inline int phone_id(const std::string& name) {
    static const std::unordered_map<std::string, int> index = [] {
        std::unordered_map<std::string, int> map;
        const std::vector<std::string>& names = phone_names();
        for (int i = 0; i < static_cast<int>(names.size()); ++i) map[names[i]] = i;
        return map;
    }();
    const auto it = index.find(name);
    return it == index.end() ? -1 : it->second;
}

// TIMIT's 61 symbols folded onto the inventory above. Closures, pauses and the
// glottal stop become silence; allophones merge with their base phone.
inline int timit_phone_id(const std::string& label) {
    static const std::unordered_map<std::string, std::string> fold = {
        // silence, closures, pauses, glottal stop
        {"h#", "SIL"},   {"pau", "SIL"},  {"epi", "SIL"},  {"q", "SIL"},
        {"bcl", "SIL"},  {"dcl", "SIL"},  {"gcl", "SIL"},
        {"pcl", "SIL"},  {"tcl", "SIL"},  {"kcl", "SIL"},
        // vowels
        {"aa", "AA"},    {"ao", "AA"},    {"ae", "AE"},
        {"ah", "AH"},    {"ax", "AH"},    {"ax-h", "AH"},
        {"aw", "AW"},    {"ay", "AY"},    {"eh", "EH"},
        {"er", "ER"},    {"axr", "ER"},   {"ey", "EY"},
        {"ih", "IH"},    {"ix", "IH"},    {"iy", "IY"},
        {"ow", "OW"},    {"oy", "OY"},    {"uh", "UH"},
        {"uw", "UW"},    {"ux", "UW"},
        // consonants
        {"b", "B"},      {"ch", "CH"},    {"d", "D"},      {"dx", "D"},
        {"dh", "DH"},    {"f", "F"},      {"g", "G"},
        {"hh", "HH"},    {"hv", "HH"},    {"jh", "JH"},    {"k", "K"},
        {"l", "L"},      {"el", "L"},     {"m", "M"},      {"em", "M"},
        {"n", "N"},      {"en", "N"},     {"nx", "N"},
        {"ng", "NG"},    {"eng", "NG"},   {"p", "P"},      {"r", "R"},
        {"s", "S"},      {"sh", "SH"},    {"zh", "SH"},    {"t", "T"},
        {"th", "TH"},    {"v", "V"},      {"w", "W"},      {"y", "Y"},
        {"z", "Z"}};

    const auto it = fold.find(label);
    if (it == fold.end()) return -1;
    return phone_id(it->second);
}

// CMUDict spells phones with stress digits ("AH0", "EY1"); strip them.
inline std::string strip_stress(const std::string& phone) {
    std::string out;
    out.reserve(phone.size());
    for (char c : phone) {
        if (c < '0' || c > '9') out.push_back(c);
    }
    return out;
}

inline int state_index(int phone, int state) { return phone * kNumStatesPerPhone + state; }

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\toy_pruned_hmm\triphone\triphone.hpp ===
// Context-dependent triphones with phonetic decision-tree state tying
// (Path 3 - "senones").
//
// The problem this solves: a monophone gives /k/ one distribution, but /k/
// before /iy/ ("keep") and before /uw/ ("cool") have F2 loci hundreds of Hz
// apart. Frames from both are averaged into one Gaussian, which is why the
// monophone system's errors are 75 % substitutions.
//
// The naive fix does not work: 40^3 triphones x 3 states is ~192,000 states,
// far more than 5 hours of audio can estimate. The classical answer is to build
// one decision tree per (centre phone, state position) and split contexts with
// binary phonetic questions ("is the right context a front vowel?"), pooling
// every context that lands in the same leaf into a single tied state. Leaves
// with too little data are never created, so every senone is backed by real
// frames, and unseen triphones still reach a leaf by answering the questions -
// which is what makes the model usable on words it never saw in training.
//
// Splitting criterion (standard HTK/Kaldi form): for a diagonal Gaussian fitted
// to a pooled set of frames, the log-likelihood of that set is
//
//     L = -0.5 * N * sum_d (log var_d + 1 + log 2pi)
//
// so a split is worth making when L(yes) + L(no) - L(parent) is largest. Only
// counts, sums and sums of squares are needed, which are additive over contexts.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include <filter/mfcc.hpp>
#include "phone_set.hpp"

namespace models {

// Sufficient statistics for one context (left, centre, right, state).
struct ContextStats {
    int left = 0;
    int right = 0;
    double count = 0.0;
    std::vector<double> sum;
    std::vector<double> sum_squares;

    ContextStats() : sum(kFeatureDim, 0.0), sum_squares(kFeatureDim, 0.0) {}

    void add(const float* row) {
        for (int d = 0; d < kFeatureDim; ++d) {
            sum[d] += row[d];
            sum_squares[d] += static_cast<double>(row[d]) * row[d];
        }
        count += 1.0;
    }
};

// Phonetic questions: each is a set of phones. A context answers "yes" when the
// relevant neighbour is in the set. Broad classes come first so early splits
// are linguistically meaningful, then every phone as a singleton question.
inline const std::vector<std::pair<std::string, std::vector<std::string>>>& question_sets() {
    static const std::vector<std::pair<std::string, std::vector<std::string>>> questions = {
        {"silence", {"SIL"}},
        {"vowel", {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW",
                   "OY", "UH", "UW"}},
        {"front_vowel", {"IY", "IH", "EH", "AE", "EY"}},
        {"back_vowel", {"UW", "UH", "OW", "AO", "AA"}},
        {"central_vowel", {"AH", "ER"}},
        {"high_vowel", {"IY", "IH", "UW", "UH"}},
        {"low_vowel", {"AE", "AA", "AO", "AW"}},
        {"rounded", {"UW", "UH", "OW", "AO", "OY", "W"}},
        {"diphthong", {"AY", "EY", "OY", "AW", "OW"}},
        {"consonant", {"B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG",
                       "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"}},
        {"stop", {"B", "D", "G", "P", "T", "K"}},
        {"voiced_stop", {"B", "D", "G"}},
        {"unvoiced_stop", {"P", "T", "K"}},
        {"fricative", {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}},
        {"sibilant", {"S", "Z", "SH", "ZH", "CH", "JH"}},
        {"voiced_fricative", {"V", "DH", "Z", "ZH"}},
        {"affricate", {"CH", "JH"}},
        {"nasal", {"M", "N", "NG"}},
        {"liquid", {"L", "R"}},
        {"glide", {"W", "Y"}},
        {"approximant", {"L", "R", "W", "Y"}},
        {"labial", {"B", "P", "M", "F", "V", "W"}},
        {"alveolar", {"D", "T", "N", "S", "Z", "L", "R"}},
        {"velar", {"K", "G", "NG", "W"}},
        {"dental", {"TH", "DH"}},
        {"palatal", {"SH", "ZH", "CH", "JH", "Y"}},
        {"voiced", {"B", "D", "G", "V", "DH", "Z", "ZH", "JH", "M", "N", "NG", "L", "R", "W",
                    "Y"}},
    };
    return questions;
}

// Bitmask over the phone inventory, so a question is answered with one AND.
struct Question {
    std::string name;
    uint64_t mask = 0;
    bool on_left = true;

    bool answer(const ContextStats& context) const {
        const int phone = on_left ? context.left : context.right;
        return phone >= 0 && (mask >> phone) & 1ull;
    }
};

inline std::vector<Question> build_questions() {
    std::vector<Question> questions;
    for (int side = 0; side < 2; ++side) {
        for (const auto& entry : question_sets()) {
            Question question;
            question.name = (side == 0 ? "L:" : "R:") + entry.first;
            question.on_left = side == 0;
            for (const std::string& phone : entry.second) {
                const int id = phone_id(phone);
                if (id >= 0) question.mask |= 1ull << id;
            }
            if (question.mask) questions.push_back(question);
        }
        // Singleton questions let the tree isolate one specific context.
        for (int p = 0; p < num_phones(); ++p) {
            Question question;
            question.name = (side == 0 ? "L=" : "R=") + phone_names()[p];
            question.on_left = side == 0;
            question.mask = 1ull << p;
            questions.push_back(question);
        }
    }
    return questions;
}

// One decision tree node. Leaves carry a senone id.
struct TreeNode {
    int question = -1;   // index into the question list; -1 for a leaf
    int yes = -1;
    int no = -1;
    int senone = -1;
};

// Trees for every (centre phone, state position) pair, plus the senone table.
class TriphoneTree {
public:
    // ---- training -------------------------------------------------------

    // `contexts[(phone, state)]` holds the statistics of every observed
    // context. Splitting stops when a split would gain less than
    // `min_gain` or leave a leaf with fewer than `min_occupancy` frames.
    void build(const std::vector<std::vector<ContextStats>>& contexts, double min_gain = 1200.0,
               double min_occupancy = 600.0, int max_senones = 2200) {
        questions_ = build_questions();
        nodes_.clear();
        roots_.assign(contexts.size(), -1);
        num_senones_ = 0;

        // Grow every tree greedily, best-gain-first across all trees, so the
        // senone budget goes where it buys the most likelihood.
        struct Pending {
            int node;
            std::vector<const ContextStats*> members;
        };
        std::vector<Pending> frontier;

        for (size_t key = 0; key < contexts.size(); ++key) {
            if (contexts[key].empty()) continue;
            Pending pending;
            pending.node = static_cast<int>(nodes_.size());
            nodes_.push_back(TreeNode{});
            roots_[key] = pending.node;
            for (const ContextStats& stats : contexts[key]) {
                if (stats.count > 0.0) pending.members.push_back(&stats);
            }
            if (!pending.members.empty()) frontier.push_back(std::move(pending));
        }

        while (!frontier.empty() && num_senones_ + static_cast<int>(frontier.size()) < max_senones) {
            // Find the best split available anywhere in the frontier.
            double best_gain = min_gain;
            size_t best_index = frontier.size();
            int best_question = -1;

            for (size_t i = 0; i < frontier.size(); ++i) {
                int question = -1;
                const double gain = best_split(frontier[i].members, min_occupancy, &question);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_index = i;
                    best_question = question;
                }
            }
            if (best_index == frontier.size()) break;  // nothing worth splitting

            Pending parent = std::move(frontier[best_index]);
            frontier.erase(frontier.begin() + best_index);

            Pending yes_branch, no_branch;
            for (const ContextStats* stats : parent.members) {
                if (questions_[best_question].answer(*stats)) {
                    yes_branch.members.push_back(stats);
                } else {
                    no_branch.members.push_back(stats);
                }
            }

            yes_branch.node = static_cast<int>(nodes_.size());
            nodes_.push_back(TreeNode{});
            no_branch.node = static_cast<int>(nodes_.size());
            nodes_.push_back(TreeNode{});

            nodes_[parent.node].question = best_question;
            nodes_[parent.node].yes = yes_branch.node;
            nodes_[parent.node].no = no_branch.node;

            frontier.push_back(std::move(yes_branch));
            frontier.push_back(std::move(no_branch));
        }

        // Everything left in the frontier becomes a leaf.
        for (Pending& pending : frontier) {
            nodes_[pending.node].senone = num_senones_++;
        }
        // Trees that never entered the frontier (no data) still need a leaf.
        for (TreeNode& node : nodes_) {
            if (node.question < 0 && node.senone < 0) node.senone = num_senones_++;
        }
    }

    // ---- lookup ---------------------------------------------------------

    int senone(int left, int centre, int state, int right) const {
        const int key = tree_key(centre, state);
        if (key < 0 || key >= static_cast<int>(roots_.size()) || roots_[key] < 0) return 0;

        ContextStats probe;
        probe.left = left;
        probe.right = right;

        int node = roots_[key];
        while (nodes_[node].question >= 0) {
            node = questions_[nodes_[node].question].answer(probe) ? nodes_[node].yes
                                                                   : nodes_[node].no;
        }
        return nodes_[node].senone;
    }

    int num_senones() const { return num_senones_; }
    static int tree_key(int phone, int state) { return phone * kNumStatesPerPhone + state; }
    static int num_tree_keys() { return num_phones() * kNumStatesPerPhone; }

    // ---- persistence ----------------------------------------------------

    bool save(const std::string& path) const {
        FILE* file = std::fopen(path.c_str(), "wb");
        if (!file) return false;
        const int32_t magic = 0x54524931;  // "TRI1"
        const int32_t nodes = static_cast<int32_t>(nodes_.size());
        const int32_t roots = static_cast<int32_t>(roots_.size());
        const int32_t senones = num_senones_;
        std::fwrite(&magic, sizeof(magic), 1, file);
        std::fwrite(&nodes, sizeof(nodes), 1, file);
        std::fwrite(&roots, sizeof(roots), 1, file);
        std::fwrite(&senones, sizeof(senones), 1, file);
        std::fwrite(nodes_.data(), sizeof(TreeNode), nodes_.size(), file);
        std::fwrite(roots_.data(), sizeof(int), roots_.size(), file);
        std::fclose(file);
        return true;
    }

    bool load(const std::string& path) {
        FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) return false;
        int32_t magic = 0, nodes = 0, roots = 0, senones = 0;
        if (std::fread(&magic, sizeof(magic), 1, file) != 1 ||
            std::fread(&nodes, sizeof(nodes), 1, file) != 1 ||
            std::fread(&roots, sizeof(roots), 1, file) != 1 ||
            std::fread(&senones, sizeof(senones), 1, file) != 1 || magic != 0x54524931) {
            std::fclose(file);
            return false;
        }
        nodes_.resize(nodes);
        roots_.resize(roots);
        const bool ok =
            std::fread(nodes_.data(), sizeof(TreeNode), nodes_.size(), file) == nodes_.size() &&
            std::fread(roots_.data(), sizeof(int), roots_.size(), file) == roots_.size();
        std::fclose(file);
        num_senones_ = senones;
        questions_ = build_questions();
        return ok;
    }

private:
    // Log-likelihood of a pooled set under a single diagonal Gaussian.
    static double pooled_likelihood(double count, const std::vector<double>& sum,
                                    const std::vector<double>& sum_squares) {
        if (count < 1.0) return 0.0;
        constexpr double kLog2Pi = 1.8378770664093453;
        double log_var = 0.0;
        for (int d = 0; d < kFeatureDim; ++d) {
            const double mean = sum[d] / count;
            double variance = sum_squares[d] / count - mean * mean;
            if (variance < 0.01) variance = 0.01;
            log_var += std::log(variance);
        }
        return -0.5 * count * (log_var + kFeatureDim * (1.0 + kLog2Pi));
    }

    static void accumulate(const std::vector<const ContextStats*>& members, double& count,
                           std::vector<double>& sum, std::vector<double>& sum_squares) {
        count = 0.0;
        sum.assign(kFeatureDim, 0.0);
        sum_squares.assign(kFeatureDim, 0.0);
        for (const ContextStats* stats : members) {
            count += stats->count;
            for (int d = 0; d < kFeatureDim; ++d) {
                sum[d] += stats->sum[d];
                sum_squares[d] += stats->sum_squares[d];
            }
        }
    }

    double best_split(const std::vector<const ContextStats*>& members, double min_occupancy,
                      int* out_question) const {
        if (members.size() < 2) return 0.0;

        double parent_count;
        std::vector<double> parent_sum, parent_squares;
        accumulate(members, parent_count, parent_sum, parent_squares);
        if (parent_count < 2.0 * min_occupancy) return 0.0;
        const double parent_likelihood =
            pooled_likelihood(parent_count, parent_sum, parent_squares);

        double best_gain = 0.0;
        std::vector<double> yes_sum(kFeatureDim), yes_squares(kFeatureDim);

        for (size_t q = 0; q < questions_.size(); ++q) {
            double yes_count = 0.0;
            std::fill(yes_sum.begin(), yes_sum.end(), 0.0);
            std::fill(yes_squares.begin(), yes_squares.end(), 0.0);

            for (const ContextStats* stats : members) {
                if (!questions_[q].answer(*stats)) continue;
                yes_count += stats->count;
                for (int d = 0; d < kFeatureDim; ++d) {
                    yes_sum[d] += stats->sum[d];
                    yes_squares[d] += stats->sum_squares[d];
                }
            }
            const double no_count = parent_count - yes_count;
            if (yes_count < min_occupancy || no_count < min_occupancy) continue;

            std::vector<double> no_sum(kFeatureDim), no_squares(kFeatureDim);
            for (int d = 0; d < kFeatureDim; ++d) {
                no_sum[d] = parent_sum[d] - yes_sum[d];
                no_squares[d] = parent_squares[d] - yes_squares[d];
            }

            const double gain = pooled_likelihood(yes_count, yes_sum, yes_squares) +
                                pooled_likelihood(no_count, no_sum, no_squares) -
                                parent_likelihood;
            if (gain > best_gain) {
                best_gain = gain;
                *out_question = static_cast<int>(q);
            }
        }
        return best_gain;
    }

    std::vector<Question> questions_;
    std::vector<TreeNode> nodes_;
    std::vector<int> roots_;
    int num_senones_ = 0;
};

}  // namespace models

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\power_spectrum\fast_fft.hpp ===
#pragma once
#include <fftw3.h>
#include <vector>
#include <cmath>

class FastFFT {
public:
    int n_fft;
    int num_bins;
    float* in_buf;
    fftwf_complex* out_buf;
    fftwf_plan plan;

public:
    FastFFT(int fft_size = 512) : n_fft(fft_size), num_bins(fft_size / 2 + 1) {
        // SIMD-aligned allocation
        in_buf = fftwf_alloc_real(n_fft);
        out_buf = fftwf_alloc_complex(num_bins);
        
        // Plan r2c 1D transform
        plan = fftwf_plan_dft_r2c_1d(n_fft, in_buf, out_buf, FFTW_ESTIMATE);
    }

    ~FastFFT() {
        fftwf_destroy_plan(plan);
        fftwf_free(in_buf);
        fftwf_free(out_buf);
    }

    // Computes Power Spectrum: |X(f)|^2
    std::vector<float> compute_power_spectrum(const std::vector<float>& frame) {
        // Copy frame into aligned FFTW input buffer
        for (int i = 0; i < n_fft; ++i) {
            in_buf[i] = (i < frame.size()) ? frame[i] : 0.0f;
        }

        fftwf_execute(plan);

        std::vector<float> power_spec(num_bins);
        for (int k = 0; k < num_bins; ++k) {
            float re = out_buf[k][0];
            float im = out_buf[k][1];
            power_spec[k] = (re * re + im * im) / n_fft; // Normalized power
        }
        return power_spec;
    }
};

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\spectral\spectral_analysis.hpp ===
#pragma once
#ifndef SPECTRAL_ANALYSIS_HPP
#define SPECTRAL_ANALYSIS_HPP

#include <vector>
#include <cmath>
#include <algorithm>

// Ensure M_PI is defined for Windows (MSVC)
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct SpectralFeatures {
    float centroid_hz;
    float pitch_f0_hz;      // 0.0 if unvoiced
    bool is_voiced;
    float spectral_tilt;    // Slope in dB/octave (negative = rolloff)
    float voicing_confidence; // 0.0 to 1.0 (how prominent the pitch peak is)
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

    // 3. Spectral Tilt (slope of log-power spectrum vs. frequency in dB/octave)
    static float compute_spectral_tilt(const std::vector<float>& power_spec, float sample_rate = 16000.0f, int n_fft = 512) {
        size_t K = power_spec.size();
        float bin_hz = sample_rate / n_fft;
        
        // Use linear regression on log-power vs. log-frequency
        float sum_xy = 0.0f, sum_x = 0.0f, sum_y = 0.0f, sum_x2 = 0.0f;
        int count = 0;

        for (size_t k = 1; k < K; ++k) { // Skip k=0 (DC component)
            float freq = k * bin_hz;
            float log_freq = std::log2(freq); // Log2 for octave scaling
            float log_power = std::log10(std::max(power_spec[k], 1e-7f)); // Log10 for dB

            sum_x += log_freq;
            sum_y += log_power;
            sum_xy += log_freq * log_power;
            sum_x2 += log_freq * log_freq;
            ++count;
        }

        if (count < 2) return 0.0f;

        // Linear regression: slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        float slope = (count * sum_xy - sum_x * sum_y) / (count * sum_x2 - sum_x * sum_x);
        
        // Convert from log2-scale to dB/octave: slope * 20*log10(2) ≈ slope * 6.02
        return slope * 20.0f * std::log10(2.0f);
    }

    // 4. Voicing Confidence from cepstral peak prominence
    static float compute_voicing_confidence(const std::vector<float>& power_spec, float sample_rate = 16000.0f) {
        size_t K = power_spec.size();
        std::vector<float> log_power(K);

        for (size_t k = 0; k < K; ++k) {
            log_power[k] = std::log(std::max(power_spec[k], 1e-7f));
        }

        // Compute cepstrum
        std::vector<float> cepstrum(K, 0.0f);
        for (size_t n = 0; n < K; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += log_power[k] * std::cos(M_PI * n * (k + 0.5f) / K);
            }
            cepstrum[n] = sum;
        }

        // Search for pitch peak
        size_t min_lag = static_cast<size_t>(sample_rate / 400.0f);
        size_t max_lag = static_cast<size_t>(sample_rate / 70.0f);
        max_lag = std::min(max_lag, K - 1);

        float max_peak = -1e9f;
        float avg_neighbor = 0.0f;

        for (size_t lag = min_lag; lag <= max_lag; ++lag) {
            max_peak = std::max(max_peak, cepstrum[lag]);
        }

        // Average of neighbors (for comparison)
        for (size_t lag = min_lag; lag <= max_lag; ++lag) {
            avg_neighbor += std::abs(cepstrum[lag]);
        }
        avg_neighbor /= (max_lag - min_lag + 1);

        // Confidence: ratio of peak to average (normalized to 0-1)
        float confidence = (avg_neighbor > 1e-7f) ? (max_peak / avg_neighbor) : 0.0f;
        return std::min(1.0f, confidence / 3.0f); // Normalize to ~0-1 range
    }

    // 5. Extract all spectral features from power spectrum
    static SpectralFeatures extract_features(const std::vector<float>& power_spec, float sample_rate = 16000.0f, int n_fft = 512) {
        float centroid = compute_centroid(power_spec, sample_rate, n_fft);
        float pitch = compute_pitch_cepstrum(power_spec, sample_rate);
        float tilt = compute_spectral_tilt(power_spec, sample_rate, n_fft);
        float confidence = compute_voicing_confidence(power_spec, sample_rate);

        return {
            centroid,
            pitch,
            pitch > 0.0f && confidence > 0.4f,  // is_voiced: pitch present and confidence high
            tilt,
            confidence
        };
    }
};

#endif // SPECTRAL_ANALYSIS_HPP

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\spectral\spectral_to_vowel.hpp ===
#pragma once
#ifndef SPECTRAL_TO_VOWEL_HPP
#define SPECTRAL_TO_VOWEL_HPP

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include "spectral_analysis.hpp"

struct SpectralVowelEntry {
    std::string key;           // "IY", "AE", etc.
    std::string ipa;           // "i", "æ"
    std::string example;       // "beet", "bat"
    float centroid_hz;         // Expected spectral centroid in Hz
    float pitch_range_low;     // Typical F0 range (male speaker)
    float pitch_range_high;
    float spectral_tilt_db;    // Expected tilt in dB/octave (negative = rolloff)
    float tilt_tolerance;      // Acceptable variation in tilt
};

class SpectralVowelDB {
private:
    std::vector<SpectralVowelEntry> db;

public:
    SpectralVowelDB() {
        // Initialize spectral vowel dataset
        // Centroid values based on frequency distributions of formants
        // Tilt values: vowels have -12 to -6 dB/octave rolloff; high consonants near 0 or positive
        db = {
            // Front Vowels (high F2, rising spectral content)
            {"IY", "i",  "beet",   1850.0f,  100.0f, 130.0f,  -9.0f,  2.0f},
            {"IH", "ɪ",  "bit",    1700.0f,   95.0f, 125.0f,  -9.0f,  2.0f},
            {"EY", "eɪ", "bait",   1600.0f,   90.0f, 120.0f, -10.0f,  2.0f},
            {"EH", "ɛ",  "bet",    1500.0f,   85.0f, 115.0f, -10.0f,  2.0f},
            {"AE", "æ",  "bat",    1400.0f,   85.0f, 115.0f, -11.0f,  2.0f},

            // Central Vowels
            {"AH", "ʌ",  "butt",   1200.0f,   85.0f, 115.0f, -11.0f,  2.0f},
            {"ER", "ɝ",  "bird",   1150.0f,   90.0f, 120.0f, -11.0f,  2.0f}, // Rhotic
            {"AX", "ə",  "about",  1300.0f,   80.0f, 110.0f, -11.0f,  2.0f},

            // Back Vowels (lower centroid, strong rolloff)
            {"AA", "ɑ",  "father",  900.0f,   80.0f, 110.0f, -12.0f,  2.0f},
            {"AO", "ɔ",  "bought",  950.0f,   75.0f, 105.0f, -12.0f,  2.0f},
            {"OW", "oʊ", "boat",   1050.0f,   75.0f, 105.0f, -12.0f,  2.0f},
            {"UH", "ʊ",  "book",   1100.0f,   70.0f, 100.0f, -12.0f,  2.0f},
            {"UW", "u",  "boot",   1150.0f,   70.0f, 100.0f, -12.0f,  2.0f},
        };
    }

    struct MatchResult {
        std::string key;
        std::string ipa;
        float distance;
        bool in_bounds;
    };

    // Find closest vowel using spectral features
    MatchResult find_nearest_phoneme(const SpectralFeatures& features) const {
        if (!features.is_voiced) {
            return {"SIL", "", 0.0f, false}; // Unvoiced / Silence
        }

        float min_dist = 1.0e30f;
        const SpectralVowelEntry* best_match = nullptr;

        for (const auto& entry : db) {
            // Distance components: normalized L2 norm in feature space
            
            // 1. Centroid distance (Hz scale, normalized by typical range ~500-2000 Hz)
            float centroid_diff = features.centroid_hz - entry.centroid_hz;
            float centroid_dist = (centroid_diff * centroid_diff) / (400.0f * 400.0f); // ~400 Hz tolerance

            // 2. Pitch distance (Hz scale, normalized by typical range ~70-400 Hz for male speaker)
            float pitch_tolerance = (entry.pitch_range_high - entry.pitch_range_low) / 2.0f + 20.0f;
            float pitch_mid = (entry.pitch_range_low + entry.pitch_range_high) / 2.0f;
            float pitch_diff = features.pitch_f0_hz - pitch_mid;
            float pitch_dist = (pitch_diff * pitch_diff) / (pitch_tolerance * pitch_tolerance);

            // 3. Spectral tilt distance (dB/octave scale, normalized by ~2 dB tolerance)
            float tilt_diff = features.spectral_tilt - entry.spectral_tilt_db;
            float tilt_dist = (tilt_diff * tilt_diff) / (entry.tilt_tolerance * entry.tilt_tolerance);

            // Weighted combination (centroid most important, tilt less so)
            float distance = std::sqrt(centroid_dist * 2.0f + pitch_dist * 0.5f + tilt_dist * 0.3f);

            if (distance < min_dist) {
                min_dist = distance;
                best_match = &entry;
            }
        }

        if (best_match) {
            // Check bounds: centroid within ±500 Hz, pitch within typical range
            float centroid_err = std::abs(features.centroid_hz - best_match->centroid_hz);
            float pitch_range = best_match->pitch_range_high - best_match->pitch_range_low;
            float pitch_err = std::abs(features.pitch_f0_hz - (best_match->pitch_range_low + best_match->pitch_range_high) / 2.0f);
            
            bool within_bounds = (centroid_err <= 500.0f) && (pitch_err <= pitch_range);

            return {best_match->key, best_match->ipa, min_dist, within_bounds};
        }

        return {"UNK", "", min_dist, false};
    }
};

#endif // SPECTRAL_TO_VOWEL_HPP

