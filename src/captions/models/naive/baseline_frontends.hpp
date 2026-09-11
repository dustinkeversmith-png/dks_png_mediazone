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
