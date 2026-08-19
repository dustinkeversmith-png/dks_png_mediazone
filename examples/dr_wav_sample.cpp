#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include <vector>
#include <iostream>

std::vector<float> load_audio_mono_f32(const char* filename, unsigned int* sampleRate) {
    drwav wav;
    if (!drwav_init_file(&wav, filename, NULL)) {
        std::cerr << "Failed to open WAV file\n";
        return {};
    }

    *sampleRate = wav.sampleRate;
    size_t total_samples = wav.totalPCMFrameCount * wav.channels;
    
    // Allocate temporary buffer for multi-channel data
    std::vector<float> raw_buffer(total_samples);
    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, raw_buffer.data());
    drwav_uninit(&wav);

    // Convert to Mono (average channels)
    std::vector<float> mono_buffer(wav.totalPCMFrameCount);
    if (wav.channels == 1) {
        mono_buffer = std::move(raw_buffer);
    } else {
        for (size_t i = 0; i < wav.totalPCMFrameCount; ++i) {
            float sum = 0.0f;
            for (size_t c = 0; c < wav.channels; ++c) {
                sum += raw_buffer[i * wav.channels + c];
            }
            mono_buffer[i] = sum / wav.channels;
        }
    }
    return mono_buffer;
}