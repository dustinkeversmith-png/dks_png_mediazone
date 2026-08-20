#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"


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
