#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <iostream>
#include <vector>
#include <string>

/**
 * Loads an audio file, converts it to 32-bit float, mixes to Mono, 
 * and resamples directly to 16 kHz.
 * * @param filepath Path to the input audio file (.wav, .mp3, .flac)
 * @param out_samples Target vector to store contiguous normalized float samples
 * @return true if successful, false otherwise
 */
bool load_and_preprocess_audio(const std::string& filepath, std::vector<float>& out_samples) {

    
    ma_decoder_config config = ma_decoder_config_init(
        ma_format_f32,   // Target format: 32-bit Float [-1.0, 1.0]
        1,               // Target channels: 1 (Mono)
        16000            // Target sample rate: 16 kHz
    );

    ma_decoder decoder;
    ma_result result = ma_decoder_init_file(filepath.c_str(), &config, &decoder);
    if (result != MA_SUCCESS) {
        std::cerr << "[Error] Failed to initialize decoder for file: " << filepath << "\n";
        return false;
    }

    // Retrieve total frame count at target 16 kHz
    ma_uint64 total_frames;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);
    
    if (result == MA_SUCCESS && total_frames > 0) {
        out_samples.resize(total_frames);
        ma_uint64 frames_read = 0;
        
        result = ma_decoder_read_pcm_frames(&decoder, out_samples.data(), total_frames, &frames_read);
        if (result != MA_SUCCESS) {
            std::cerr << "[Error] Failed reading PCM frames\n";
            ma_decoder_uninit(&decoder);
            return false;
        }
        
        // Truncate to exact frames read in case of stream differences
        out_samples.resize(frames_read);
    } else {
        // Fallback dynamic reading if length cannot be pre-calculated
        constexpr size_t CHUNK_SIZE = 4096;
        float chunk[CHUNK_SIZE];
        ma_uint64 frames_read_chunk = 0;
        
        while (ma_decoder_read_pcm_frames(&decoder, chunk, CHUNK_SIZE, &frames_read_chunk) == MA_SUCCESS 
               && frames_read_chunk > 0) {
            out_samples.insert(out_samples.end(), chunk, chunk + frames_read_chunk);
        }
    }

    ma_decoder_uninit(&decoder);
    return true;
}

int main() {
    // Example test path from your extracted AudioMNIST dataset
    std::string sample_file = "data/audiomnist/digit_0_sample_0106.wav";
    std::vector<float> audio_buffer;

    std::cout << "Loading: " << sample_file << "\n";

    if (load_and_preprocess_audio(sample_file, audio_buffer)) {
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

    return 0;
}