
#include <iostream>
#include <vector>
#include <string>

#include "audio_loadnorm.hpp"
#include "audio_framing.hpp"

#include <power_spectrum/fast_fft.hpp>
#include <formants/find_formants.hpp>

int main(int argc, char* argv[]) {
    // Example test path from your extracted AudioMNIST dataset
    std::string sample_file = argv[0];

    std::vector<float> audio_buffer;

    std::cout << "Loading: " << sample_file << "\n";

    bool loaded_and_normalized = load_and_preprocess_audio(sample_file, audio_buffer)

    if (loaded_and_normalized) {
        std::cout << "[✓] Successfully loaded and converted audio!\n";
        std::cout << "    - Total Samples (at 16 kHz): " << audio_buffer.size() << "\n";
        std::cout << "    - Duration: " << static_cast<float>(audio_buffer.size()) / 16000.0f << " seconds\n";
        
        if (!audio_buffer.empty()) {
            std::cout << "    - First 5 samples: ";
            for (size_t i = 0; i < std::min<size_t>(5, audio_buffer.size()); ++i) {
                std::cout << audio_buffer[i] << " ";
            }
            std::cout << "\n";
        }
    }

    // Chop into little frames then use the FFTW for the FFT of these
    std::vector<std::vector<float>> frames = chop_into_frames(audio_buffer);


    FastFFT fast_fft;
    std::vector<float> frame = frames[0];
    std::vector<float> power_spectrum = fast_fft.compute_power_spectrum(frame);


    FormantTracker format_tracker;
    std::vector<Formant> format = format_tracker.extract_formants(frame);


    // [Glottal Source e[n]] ──► [Vocal Tract Filter H(z)] ──► [Speech Waveform s[n]]
    // (Pitch / F0 / Dirac)         (Formants F1, F2 / LPC)        (Acoustic Output)



    return 0;
}