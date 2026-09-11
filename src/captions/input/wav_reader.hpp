// Minimal RIFF/WAVE reader for the caption engines.
//
// Why this exists rather than reusing input/audio_loadnorm.hpp: the streaming
// caption app compiles miniaudio with MA_NO_DECODING (it only wants the capture
// backend for --mic), so miniaudio cannot decode files in that translation
// unit. And why not reuse the TTS library's vocal::load_wav_mono: captions must
// not link the synthesis library just to open a file.
//
// Scope is deliberately narrow - PCM16 or float32 RIFF, any channel count,
// linear resampling to the target rate. That covers the evaluation corpora
// (16 kHz mono PCM16) and microphone dumps. For anything else (mp3, flac), use
// input/audio_loadnorm.hpp, which wraps a full miniaudio decoder.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace captions {

struct Wav {
    int sample_rate_hz = 16000;
    std::vector<float> samples;  // mono, normalized to [-1, 1]
};

namespace detail {

inline std::uint16_t read_u16(std::istream& in) {
    unsigned char b[2]{};
    if (!in.read(reinterpret_cast<char*>(b), 2)) throw std::runtime_error("truncated WAV");
    return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
}

inline std::uint32_t read_u32(std::istream& in) {
    unsigned char b[4]{};
    if (!in.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("truncated WAV");
    return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

}  // namespace detail

// Reads `path` as mono at `target_sample_rate_hz`. Throws on anything it cannot
// represent exactly rather than silently returning wrong-rate audio - a caption
// benchmark that quietly decodes at the wrong rate reports a plausible but
// meaningless WER.
[[nodiscard]] inline Wav read_wav_mono(const std::filesystem::path& path,
                                       int target_sample_rate_hz = 16000) {
    if (target_sample_rate_hz <= 0) throw std::invalid_argument("target sample rate must be positive");

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open WAV: " + path.string());

    char riff[4]{}, wave[4]{};
    in.read(riff, 4);
    (void)detail::read_u32(in);
    in.read(wave, 4);
    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("not a RIFF/WAVE file: " + path.string());
    }

    std::uint16_t format = 0, channels = 0, bits = 0;
    std::uint32_t sample_rate = 0;
    std::vector<unsigned char> data;
    while (in && (format == 0 || data.empty())) {
        char id[4]{};
        if (!in.read(id, 4)) break;
        const auto size = detail::read_u32(in);
        const auto chunk_start = in.tellg();
        if (std::string(id, 4) == "fmt ") {
            format = detail::read_u16(in);
            channels = detail::read_u16(in);
            sample_rate = detail::read_u32(in);
            (void)detail::read_u32(in);
            (void)detail::read_u16(in);
            bits = detail::read_u16(in);
        } else if (std::string(id, 4) == "data") {
            data.resize(size);
            if (!in.read(reinterpret_cast<char*>(data.data()), size)) {
                throw std::runtime_error("truncated WAV data: " + path.string());
            }
        }
        in.clear();
        in.seekg(chunk_start + static_cast<std::streamoff>(size + (size & 1U)));
    }

    if (channels == 0 || sample_rate == 0 || data.empty() ||
        !((format == 1 && bits == 16) || (format == 3 && bits == 32))) {
        throw std::runtime_error("WAV must be PCM16 or float32: " + path.string());
    }

    const std::size_t bytes_per_sample = bits / 8;
    const std::size_t frames = data.size() / (bytes_per_sample * channels);
    std::vector<float> mono(frames);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double sum = 0.0;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const auto offset = (frame * channels + channel) * bytes_per_sample;
            if (format == 1) {
                const auto raw = static_cast<std::uint16_t>(data[offset] | (data[offset + 1] << 8));
                sum += static_cast<std::int16_t>(raw) / 32768.0;
            } else {
                float sample{};
                std::memcpy(&sample, data.data() + offset, sizeof(float));
                sum += sample;
            }
        }
        mono[frame] = static_cast<float>(sum / channels);
    }

    if (static_cast<int>(sample_rate) == target_sample_rate_hz) {
        return {target_sample_rate_hz, std::move(mono)};
    }

    const double ratio = static_cast<double>(target_sample_rate_hz) / sample_rate;
    std::vector<float> resampled(static_cast<std::size_t>(std::floor(mono.size() * ratio)));
    for (std::size_t i = 0; i < resampled.size(); ++i) {
        const double source = static_cast<double>(i) / ratio;
        const auto left = std::min(static_cast<std::size_t>(source), mono.size() - 1);
        const auto right = std::min(left + 1, mono.size() - 1);
        const double fraction = source - static_cast<double>(left);
        resampled[i] = static_cast<float>(mono[left] * (1.0 - fraction) + mono[right] * fraction);
    }
    return {target_sample_rate_hz, std::move(resampled)};
}

}  // namespace captions
