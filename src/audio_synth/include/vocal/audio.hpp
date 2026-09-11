#pragma once

#include <filesystem>
#include <span>
#include <vector>

namespace vocal {

struct Waveform {
    int sample_rate_hz{24'000};
    std::vector<float> samples;
};

void normalize_peak(Waveform& audio, float peak = 0.92F);
void write_wav_pcm16(const std::filesystem::path& path, const Waveform& audio);

}  // namespace vocal

