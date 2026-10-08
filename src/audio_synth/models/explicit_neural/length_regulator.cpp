#include "include/length_regulator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace vocal {
std::size_t duration_frames(std::span<const std::int64_t> durations, std::size_t maximum) {
    std::size_t total = 0;
    for (auto duration : durations) {
        if (duration < 0 || static_cast<std::uint64_t>(duration) > maximum - total)
            throw std::invalid_argument("negative duration or duration sum exceeds frame limit");
        total += static_cast<std::size_t>(duration);
    }
    return total;
}

void validate_prosody(const ProsodyControls& controls, std::size_t tokens, std::size_t maximum) {
    if (tokens == 0 || tokens > 4096 || controls.durations.size() != tokens)
        throw std::invalid_argument("durations must have one entry for each of 1..4096 tokens");
    const auto frames = duration_frames(controls.durations, maximum);
    if (frames == 0 || controls.f0_contour.size() != frames || controls.energy_contour.size() != frames)
        throw std::invalid_argument("F0 and energy must each match the positive sum of durations");
    if (controls.speaker_id < 0) throw std::invalid_argument("speaker ID must be nonnegative");
    for (float value : controls.f0_contour)
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("F0 must be finite and nonnegative");
    for (float value : controls.energy_contour)
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("energy must be finite and nonnegative");
}

ProsodyControls baseline_prosody(std::size_t tokens, std::int64_t frames_per_token, float f0_hz, float energy) {
    if (tokens == 0 || tokens > 4096 || frames_per_token <= 0)
        throw std::invalid_argument("baseline needs 1..4096 tokens and positive frames per token");
    ProsodyControls controls;
    controls.durations.assign(tokens, frames_per_token);
    const auto frames = duration_frames(controls.durations);
    controls.f0_contour.assign(frames, f0_hz);
    controls.energy_contour.assign(frames, energy);
    validate_prosody(controls, tokens);
    return controls;
}

ProsodyControls apply_prosody_sliders(const ProsodyControls& controls, const ProsodySliders& sliders) {
    validate_prosody(controls, controls.durations.size());
    if (!std::isfinite(sliders.pitch_scale) || sliders.pitch_scale <= 0 ||
        !std::isfinite(sliders.speed) || sliders.speed <= 0 ||
        !std::isfinite(sliders.energy_scale) || sliders.energy_scale < 0 ||
        !std::isfinite(sliders.energy_variance) || sliders.energy_variance < 0)
        throw std::invalid_argument("pitch/speed must be positive, energy sliders nonnegative, all finite");
    ProsodyControls result;
    result.speaker_id = controls.speaker_id;
    for (auto duration : controls.durations) {
        const double scaled = static_cast<double>(duration) / sliders.speed;
        if (scaled > maximum_prosody_frames)
            throw std::invalid_argument("speed exceeds frame limit");
        result.durations.push_back(duration == 0 ? 0 : std::max<std::int64_t>(1, std::llround(scaled)));
    }
    const auto frames = duration_frames(result.durations);
    result.f0_contour.reserve(frames);
    result.energy_contour.reserve(frames);
    const double mean = std::accumulate(controls.energy_contour.begin(), controls.energy_contour.end(), 0.0) /
                        static_cast<double>(controls.energy_contour.size());
    std::size_t offset = 0;
    for (std::size_t token = 0; token < controls.durations.size(); ++token) {
        const auto old_count = static_cast<std::size_t>(controls.durations[token]);
        const auto new_count = static_cast<std::size_t>(result.durations[token]);
        for (std::size_t frame = 0; frame < new_count; ++frame) {
            const auto source = offset + std::min(old_count - 1, frame * old_count / new_count);
            result.f0_contour.push_back(controls.f0_contour[source] * sliders.pitch_scale);
            const double energy = mean + (controls.energy_contour[source] - mean) * sliders.energy_variance;
            result.energy_contour.push_back(static_cast<float>(std::max(0.0, energy) * sliders.energy_scale));
        }
        offset += old_count;
    }
    validate_prosody(result, result.durations.size());
    return result;
}

ExpandedHiddenStates length_regulate(std::span<const float> hidden_states, std::size_t hidden_dim,
                                     std::span<const std::int64_t> durations, std::size_t maximum_frames) {
    if (hidden_dim == 0 || durations.size() > std::numeric_limits<std::size_t>::max() / hidden_dim ||
        hidden_states.size() != durations.size() * hidden_dim)
        throw std::invalid_argument("hidden states must have shape [1, durations.size(), hidden_dim]");
    const auto frames = duration_frames(durations, maximum_frames);
    if (frames > std::numeric_limits<std::size_t>::max() / hidden_dim)
        throw std::overflow_error("expanded hidden states size overflows");
    ExpandedHiddenStates result{frames, hidden_dim, {}};
    result.values.reserve(frames * hidden_dim);
    for (std::size_t token = 0; token < durations.size(); ++token) {
        const auto state = hidden_states.subspan(token * hidden_dim, hidden_dim);
        for (std::int64_t frame = 0; frame < durations[token]; ++frame)
            result.values.insert(result.values.end(), state.begin(), state.end());
    }
    return result;
}
} // namespace vocal
