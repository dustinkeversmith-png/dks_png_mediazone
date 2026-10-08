#pragma once

#include "vocal/control_params.hpp"

namespace vocal {
struct ExpandedHiddenStates {
    std::size_t frames{};
    std::size_t hidden_dim{};
    std::vector<float> values; // Flattened [1, frames, hidden_dim].
};

[[nodiscard]] ExpandedHiddenStates length_regulate(
    std::span<const float> hidden_states, std::size_t hidden_dim,
    std::span<const std::int64_t> durations,
    std::size_t maximum_frames = maximum_prosody_frames);
} // namespace vocal
