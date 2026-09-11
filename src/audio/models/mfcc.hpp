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
