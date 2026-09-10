
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



    std::vector<std::array<float, 3>> formant_crop;
    std::vector<float> frame_energies;
    std::vector<float> frame_zcrs;
    constexpr float kConsonantZcrThreshold = 0.35f;

    
    // Batch collect the formants into this crop.
    // Compute all of the formants for the entire frames as well as muting the formants giving a moving average.
    for (const std::vector<float>& frame : frames) {

        const std::vector<float> power_spectrum = fast_fft.compute_power_spectrum(frame);
        (void)power_spectrum;
    }

    return 0;
}


