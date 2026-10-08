#pragma once

#include "vocal/acoustic_model.hpp"

#include <filesystem>
#include <cstdint>
#include <memory>

namespace vocal {
// Native public icefall Matcha export, using x/x_length/noise_scale/length_scale.
// This checkpoint predicts its own durations and does not expose F0/energy targets.
class MatchaAcousticGenerator {
public:
    explicit MatchaAcousticGenerator(const std::filesystem::path& path, int threads = 4);
    ~MatchaAcousticGenerator();
    MatchaAcousticGenerator(MatchaAcousticGenerator&&) noexcept;
    MatchaAcousticGenerator& operator=(MatchaAcousticGenerator&&) noexcept;
    [[nodiscard]] MelSpectrogram infer(std::span<const std::int64_t> cmu_token_ids,
                                       float speed = 1.0F, float noise_scale = 1.0F);
    [[nodiscard]] double inference_ms() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace vocal
