
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "audio_loadnorm.hpp"
#include "audio_framing.hpp"

#include <power_spectrum/fast_fft.hpp>
#include <formants/find_formants.hpp>
#include <formants/formant_to_vowel.hpp>
#include <filter/pre_emphasis_filter.hpp>

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
    FormantTracker formant_tracker;
    FormantVectorDB vowel_database;
    std::vector<std::string> vowel_string;

    

    std::vector<std::array<float, 3>> formant_crop;
    std::vector<float> frame_energies;
    std::vector<float> frame_zcrs;
    constexpr float kConsonantZcrThreshold = 0.35f;

    
    // Batch collect the formants into this crop.
    // Compute all of the formants for the entire frames as well as muting the formants giving a moving average.
    for (const std::vector<float>& frame : frames) {

        const std::vector<float> power_spectrum = fast_fft.compute_power_spectrum(frame);
        (void)power_spectrum;

        float energy = 0.0f;
        for (float s : frame) energy += s * s;
        energy = std::sqrt(energy / frame.size());
        frame_energies.push_back(energy);
        frame_zcrs.push_back(compute_zero_crossing_rate(frame));

        const std::vector<Formant> formants = formant_tracker.extract_formants(frame);
        if (formants.size() < 2) {
            formant_crop.push_back({0.0f, 0.0f, 0.0f});
        } else {
            formant_crop.push_back({
                formants[0].frequency,
                formants[1].frequency,
                formants.size() > 2 ? formants[2].frequency : 0.0f
            });
        }
    }

    const float maximum_energy = *std::max_element(frame_energies.begin(), frame_energies.end());
    const float scaled_energy_threshold = maximum_energy * 0.08f;
    const float energy_threshold = scaled_energy_threshold > 0.00015f
        ? scaled_energy_threshold
        : 0.00015f;

    // Median smoothing reduces frame-to-frame LPC jitter while preserving silence.
    for (size_t frame_index = 0; frame_index < formant_crop.size(); ++frame_index) {
        const std::array<float, 3>& raw_formants = formant_crop[frame_index];
        if (frame_energies[frame_index] < energy_threshold ||
            frame_zcrs[frame_index] > kConsonantZcrThreshold ||
            raw_formants[0] <= 0.0f || raw_formants[1] <= 0.0f) {
            vowel_string.push_back("SIL");
            continue;
        }

        const size_t first = frame_index == 0 ? 0 : frame_index - 1;
        const size_t last = frame_index + 1 < formant_crop.size()
            ? frame_index + 1
            : formant_crop.size() - 1;
        std::array<float, 3> smoothed_formants{};
        for (size_t formant_index = 0; formant_index < smoothed_formants.size(); ++formant_index) {
            std::vector<float> neighborhood;
            for (size_t neighbor = first; neighbor <= last; ++neighbor) {
                neighborhood.push_back(formant_crop[neighbor][formant_index]);
            }
            smoothed_formants[formant_index] = median_of_valid_values(neighborhood);
        }

        const FormantVectorDB::MatchResult result = vowel_database.find_nearest_phoneme(
            smoothed_formants[0], smoothed_formants[1], smoothed_formants[2]);
        vowel_string.push_back(result.key == "UNK" ? "SIL" : result.key);
    }

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