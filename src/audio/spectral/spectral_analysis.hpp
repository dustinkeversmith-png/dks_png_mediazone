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