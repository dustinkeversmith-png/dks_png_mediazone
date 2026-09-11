#include "vocal/audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace vocal {
namespace {

void u16(std::ostream& out, std::uint16_t value) {
    const char bytes[] = {static_cast<char>(value & 0xff), static_cast<char>((value >> 8) & 0xff)};
    out.write(bytes, 2);
}

void u32(std::ostream& out, std::uint32_t value) {
    const char bytes[] = {static_cast<char>(value & 0xff), static_cast<char>((value >> 8) & 0xff),
                          static_cast<char>((value >> 16) & 0xff), static_cast<char>((value >> 24) & 0xff)};
    out.write(bytes, 4);
}

}  // namespace

void normalize_peak(Waveform& audio, float peak) {
    float maximum = 0.0F;
    for (float sample : audio.samples) maximum = std::max(maximum, std::abs(sample));
    if (maximum <= 0.0F) return;
    const float scale = peak / maximum;
    for (float& sample : audio.samples) sample *= scale;
}

void write_wav_pcm16(const std::filesystem::path& path, const Waveform& audio) {
    if (audio.sample_rate_hz <= 0 || audio.samples.size() > 0x7fffffffU) {
        throw std::invalid_argument("invalid audio for WAV output");
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create WAV file: " + path.string());
    const auto data_bytes = static_cast<std::uint32_t>(audio.samples.size() * 2);
    out.write("RIFF", 4); u32(out, 36 + data_bytes); out.write("WAVE", 4);
    out.write("fmt ", 4); u32(out, 16); u16(out, 1); u16(out, 1);
    u32(out, static_cast<std::uint32_t>(audio.sample_rate_hz));
    u32(out, static_cast<std::uint32_t>(audio.sample_rate_hz * 2));
    u16(out, 2); u16(out, 16); out.write("data", 4); u32(out, data_bytes);
    for (float sample : audio.samples) {
        const auto clipped = std::clamp(sample, -1.0F, 1.0F);
        const auto pcm = static_cast<std::int16_t>(std::lrint(clipped * 32767.0F));
        u16(out, static_cast<std::uint16_t>(pcm));
    }
}

}  // namespace vocal

