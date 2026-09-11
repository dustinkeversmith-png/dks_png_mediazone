#pragma once

#include "vocal/audio.hpp"
#include "vocal/metrics.hpp"

#include <cstddef>
#include <filesystem>

namespace vocal {

struct FeatureConfig {
    int sample_rate_hz{24'000};
    std::size_t fft_size{1024};
    std::size_t hop_samples{256};
    std::size_t mel_bins{80};
    std::size_t mcep_coefficients{25};
    double minimum_hz{40.0};
    double maximum_hz{12'000.0};
    double log_floor{1e-10};
};

struct AcousticFeatures {
    metrics::Frames log_mel;
    metrics::Frames mcep;
};

[[nodiscard]] Waveform load_wav_mono(const std::filesystem::path& path,
                                     int target_sample_rate_hz = 24'000);
[[nodiscard]] AcousticFeatures extract_features(const Waveform& audio,
                                                const FeatureConfig& config = {});
void write_feature_matrix(const std::filesystem::path& path,
                          const metrics::Frames& matrix);

}  // namespace vocal
