#pragma once

#include "vocal/audio.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace vocal {

struct PiperVoiceConfig {
    std::filesystem::path model_path;
    std::filesystem::path properties_path;
    std::filesystem::path dictionary_path;
    std::filesystem::path token_map_path;
    std::int64_t speaker_id{0};
    float noise_scale{0.667F};
    float length_scale{1.0F};
    float noise_w{0.8F};
    int intra_op_threads{0};
    std::size_t maximum_chunk_characters{240};
    double sentence_silence_seconds{0.16};
};

struct PiperDiagnostics {
    double inference_ms{};
    double audio_seconds{};
    double real_time_factor{};
    std::size_t chunks{};
    std::size_t phoneme_tokens{};
    std::size_t words{};
    std::size_t dictionary_hits{};
    std::size_t fallback_words{};
    std::size_t missing_model_symbols{};
    std::int64_t speaker_id{};
};

class PiperVoiceSynthesizer {
public:
    explicit PiperVoiceSynthesizer(PiperVoiceConfig config);
    ~PiperVoiceSynthesizer();
    PiperVoiceSynthesizer(PiperVoiceSynthesizer&&) noexcept;
    PiperVoiceSynthesizer& operator=(PiperVoiceSynthesizer&&) noexcept;
    PiperVoiceSynthesizer(const PiperVoiceSynthesizer&) = delete;
    PiperVoiceSynthesizer& operator=(const PiperVoiceSynthesizer&) = delete;

    [[nodiscard]] Waveform synthesize(std::string_view text);
    [[nodiscard]] const PiperDiagnostics& diagnostics() const noexcept;
    [[nodiscard]] static bool compiled_with_runtime() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vocal
