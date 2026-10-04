
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "audio_loadnorm.hpp"
#include "audio_framing.hpp"

#include <power_spectrum/fast_fft.hpp>
#include <filter/pre_emphasis_filter.hpp>
#include <spectral/spectral_analysis.hpp>
#include <spectral/spectral_to_vowel.hpp>

static float median_of_valid_values(const std::vector<float>& values) {
    std::vector<float> valid_values;
    for (float value : values) {
        if (value > 0.0f && std::isfinite(value)) {
            valid_values.push_back(value);
        }
    }
    if (valid_values.empty()) {
        return 0.0f;
    }

    std::sort(valid_values.begin(), valid_values.end());
    return valid_values[valid_values.size() / 2];
}

static float compute_zero_crossing_rate(const std::vector<float>& frame) {
    if (frame.size() < 2) {
        return 0.0f;
    }

    size_t crossings = 0;
    for (size_t index = 1; index < frame.size(); ++index) {
        const bool crossed_zero = (frame[index] >= 0.0f && frame[index - 1] < 0.0f) ||
                                  (frame[index] < 0.0f && frame[index - 1] >= 0.0f);
        if (crossed_zero) {
            ++crossings;
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(frame.size() - 1);
}

int main(int argc, char* argv[]) {
    const std::string sample_file = argc > 1
        ? argv[1]
        : "data/audiomnist/digit_7_sample_0000.wav";

    std::vector<float> audio_buffer;

    std::cout << "Loading: " << sample_file << "\n";

    const bool loaded_and_normalized = load_and_preprocess_audio(sample_file, audio_buffer);

    if (!loaded_and_normalized || audio_buffer.empty()) {
        std::cerr << "No audio samples were loaded.\n";
        return 1;
    }

    std::cout << "Loaded " << audio_buffer.size() << " samples at 16 kHz\n";

    std::vector<float> filtered_audio_buffer = pre_emphasis_filter(audio_buffer);

    const std::vector<std::vector<float>> frames = chop_into_frames(filtered_audio_buffer);
    if (frames.empty()) {
        std::cerr << "Audio is shorter than one analysis frame.\n";
        return 1;
    }

    FastFFT fast_fft;
    SpectralVowelDB vowel_database;
    std::vector<std::string> vowel_string;

    std::vector<SpectralFeatures> spectral_features_crop;
    std::vector<float> frame_energies;
    std::vector<float> frame_zcrs;
    constexpr float kConsonantSpectralCentroidThreshold = 3200.0f; // Hz - above this is fricative
    constexpr float kConsonantZcrThreshold = 0.35f;

    // Batch collect spectral features for all frames
    for (const std::vector<float>& frame : frames) {
        // Compute energy
        float energy = 0.0f;
        for (float s : frame) energy += s * s;
        energy = std::sqrt(energy / frame.size());
        frame_energies.push_back(energy);
        frame_zcrs.push_back(compute_zero_crossing_rate(frame));

        // Compute power spectrum and extract spectral features
        const std::vector<float> power_spectrum = fast_fft.compute_power_spectrum(frame);
        SpectralFeatures features = SpectralAnalyzer::extract_features(power_spectrum);
        spectral_features_crop.push_back(features);
    }

    const float maximum_energy = *std::max_element(frame_energies.begin(), frame_energies.end());
    const float scaled_energy_threshold = maximum_energy * 0.08f;
    const float energy_threshold = scaled_energy_threshold > 0.00015f
        ? scaled_energy_threshold
        : 0.00015f;

    // Median smoothing reduces frame-to-frame spectral jitter while preserving silence.
    for (size_t frame_index = 0; frame_index < spectral_features_crop.size(); ++frame_index) {
        const SpectralFeatures& raw_features = spectral_features_crop[frame_index];

        // Frame rejection criteria:
        // 1. Too quiet (below energy threshold) -> Silence
        // 2. High ZCR or high spectral centroid -> Consonant/Fricative
        // 3. Not voiced -> Unvoiced consonant/silence
        if (frame_energies[frame_index] < energy_threshold ||
            frame_zcrs[frame_index] > kConsonantZcrThreshold ||
            raw_features.centroid_hz > kConsonantSpectralCentroidThreshold ||
            !raw_features.is_voiced) {
            vowel_string.push_back("SIL");
            continue;
        }

        // Smooth spectral features using median of neighborhood (frame_index-1, frame_index, frame_index+1)
        const size_t first = frame_index == 0 ? 0 : frame_index - 1;
        const size_t last = frame_index + 1 < spectral_features_crop.size()
            ? frame_index + 1
            : spectral_features_crop.size() - 1;

        // Collect neighborhood centroids and pitches
        std::vector<float> neighbor_centroids;
        std::vector<float> neighbor_pitches;
        std::vector<float> neighbor_tilts;

        for (size_t neighbor = first; neighbor <= last; ++neighbor) {
            neighbor_centroids.push_back(spectral_features_crop[neighbor].centroid_hz);
            neighbor_pitches.push_back(spectral_features_crop[neighbor].pitch_f0_hz);
            neighbor_tilts.push_back(spectral_features_crop[neighbor].spectral_tilt);
        }

        // Median smoothing
        SpectralFeatures smoothed_features{
            median_of_valid_values(neighbor_centroids),
            median_of_valid_values(neighbor_pitches),
            true, // Already determined to be voiced
            median_of_valid_values(neighbor_tilts),
            raw_features.voicing_confidence
        };

        // Match to vowel database
        const SpectralVowelDB::MatchResult result = vowel_database.find_nearest_phoneme(smoothed_features);
        vowel_string.push_back(result.key == "UNK" ? "SIL" : result.key);
    }

    std::cout << "\n=== FOURIER/SPECTRAL METHOD RESULTS ===\n";
    std::cout << "Vowel string (all predicted frames): ";
    if (vowel_string.empty()) {
        std::cout << "(none)";
    } else {
        for (const std::string& vowel : vowel_string) {
            std::cout << vowel << ' ';
        }
    }
    std::cout << "\n";

    std::cout << "Vowel string (collapsed): ";
    std::string previous_vowel;
    bool printed_vowel = false;
    for (const std::string& vowel : vowel_string) {
        if (vowel == previous_vowel) {
            continue;
        }
        std::cout << vowel << ' ';
        previous_vowel = vowel;
        printed_vowel = true;
    }
    if (!printed_vowel) {
        std::cout << "(none)";
    }
    std::cout << "\n";

    return 0;
}


