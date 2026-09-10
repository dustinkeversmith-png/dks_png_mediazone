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