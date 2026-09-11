#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace vocal {

struct MelSpectrogram {
    std::size_t frames{};
    std::size_t bins{};
    std::vector<float> log_mel;

    [[nodiscard]] std::span<const float> frame(std::size_t index) const;
};

struct SynthesisRequest {
    std::string_view text;
    std::string_view speaker_id;
    int sample_rate_hz{24'000};
    std::size_t mel_bins{80};
};

class AcousticModel {
public:
    virtual ~AcousticModel() = default;
    [[nodiscard]] virtual MelSpectrogram infer(const SynthesisRequest& request) = 0;
};

}  // namespace vocal

