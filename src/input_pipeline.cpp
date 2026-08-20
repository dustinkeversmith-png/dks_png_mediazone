
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "audio_loadnorm.hpp"
#include "audio_framing.hpp"

#include <power_spectrum/fast_fft.hpp>
#include <formants/find_formants.hpp>
#include <formants/formant_to_vowel.hpp>

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

    const std::vector<std::vector<float>> frames = chop_into_frames(audio_buffer);
    if (frames.empty()) {
        std::cerr << "Audio is shorter than one analysis frame.\n";
        return 1;
    }

    FastFFT fast_fft;
    FormantTracker formant_tracker;
    FormantVectorDB vowel_database;
    std::vector<std::string> vowel_string;

    for (const std::vector<float>& frame : frames) {
        const std::vector<float> power_spectrum = fast_fft.compute_power_spectrum(frame);
        (void)power_spectrum;

        const std::vector<Formant> formants = formant_tracker.extract_formants(frame);
        if (formants.size() < 2) {
            continue;
        }

        const float f1 = formants[0].frequency;
        const float f2 = formants[1].frequency;
        const float f3 = formants.size() > 2 ? formants[2].frequency : 0.0f;
        const FormantVectorDB::MatchResult result =
            vowel_database.find_nearest_phoneme(f1, f2, f3);

        if (result.key != "SIL" && result.key != "UNK") {
            vowel_string.push_back(result.key);
        }
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