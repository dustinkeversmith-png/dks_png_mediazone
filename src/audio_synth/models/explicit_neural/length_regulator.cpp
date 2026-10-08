#include "include/length_regulator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace vocal {
namespace {
ProsodyControls remap_contours(const ProsodyControls& controls, std::vector<std::int64_t> durations) {
    ProsodyControls result;
    result.durations = std::move(durations);
    result.speaker_id = controls.speaker_id;
    result.token_kinds = controls.token_kinds;
    const auto frames = duration_frames(result.durations);
    result.f0_contour.reserve(frames);
    result.energy_contour.reserve(frames);
    std::size_t offset = 0;
    for (std::size_t token = 0; token < controls.durations.size(); ++token) {
        const auto old_count = static_cast<std::size_t>(controls.durations[token]);
        const auto new_count = static_cast<std::size_t>(result.durations[token]);
        for (std::size_t frame = 0; frame < new_count; ++frame) {
            const auto source = offset + std::min(old_count - 1, frame * old_count / new_count);
            result.f0_contour.push_back(controls.f0_contour[source]);
            result.energy_contour.push_back(controls.energy_contour[source]);
        }
        offset += old_count;
    }
    return result;
}
} // namespace

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
    if (!controls.token_kinds.empty() && controls.token_kinds.size() != tokens)
        throw std::invalid_argument("token annotations must be empty or match durations");
    for (auto kind : controls.token_kinds)
        if (kind < ProsodyTokenKind::Unknown || kind > ProsodyTokenKind::Boundary)
            throw std::invalid_argument("invalid token annotation");
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
    std::vector<std::int64_t> durations;
    for (auto duration : controls.durations) {
        const double scaled = static_cast<double>(duration) / sliders.speed;
        if (scaled > maximum_prosody_frames)
            throw std::invalid_argument("speed exceeds frame limit");
        durations.push_back(duration == 0 ? 0 : std::max<std::int64_t>(1, std::llround(scaled)));
    }
    auto result = remap_contours(controls, std::move(durations));
    const double mean = std::accumulate(controls.energy_contour.begin(), controls.energy_contour.end(), 0.0) /
                        static_cast<double>(controls.energy_contour.size());
    for (std::size_t frame = 0; frame < result.f0_contour.size(); ++frame) {
        result.f0_contour[frame] *= sliders.pitch_scale;
        const double energy = mean + (result.energy_contour[frame] - mean) * sliders.energy_variance;
        result.energy_contour[frame] = static_cast<float>(std::max(0.0, energy) * sliders.energy_scale);
    }
    validate_prosody(result, result.durations.size());
    return result;
}

ProsodyControls apply_emotion_preset(const ProsodyControls& base, VocalEmotion emotion) {
    validate_prosody(base, base.durations.size());
    if (emotion == VocalEmotion::Neutral) return base;
    if (emotion < VocalEmotion::Neutral || emotion > VocalEmotion::Authoritative)
        throw std::invalid_argument("invalid emotion preset");
    auto result = apply_prosody_sliders(base, {1.0F,
        emotion == VocalEmotion::Excited ? 1.15F : emotion == VocalEmotion::Somber ? .88F : 1.0F, 1.0F, 1.0F});
    if (!base.token_kinds.empty()) {
        auto durations = result.durations;
        for (std::size_t token = 0; token < durations.size(); ++token) {
            if (durations[token] == 0) continue;
            if (emotion == VocalEmotion::Whisper && base.token_kinds[token] == ProsodyTokenKind::UnvoicedConsonant)
                durations[token] = static_cast<std::int64_t>(std::ceil(durations[token] * 1.1));
            if (emotion == VocalEmotion::Authoritative && base.token_kinds[token] == ProsodyTokenKind::Boundary)
                durations[token] = std::max<std::int64_t>(1, durations[token] / 2);
        }
        result = remap_contours(result, std::move(durations));
    }
    double voiced_sum = 0;
    std::size_t voiced_frames = 0;
    for (float f0 : base.f0_contour) if (f0 > 0) { voiced_sum += f0; ++voiced_frames; }
    const double mean_f0 = voiced_frames ? voiced_sum / voiced_frames : 0;
    const double mean_energy = std::accumulate(base.energy_contour.begin(), base.energy_contour.end(), 0.0) /
                               base.energy_contour.size();
    for (std::size_t frame = 0; frame < result.f0_contour.size(); ++frame) {
        auto& f0 = result.f0_contour[frame];
        auto& energy = result.energy_contour[frame];
        double pitch = f0;
        if (f0 > 0) {
            if (emotion == VocalEmotion::Whisper) pitch = mean_f0 + (f0 - mean_f0) * .15;
            else if (emotion == VocalEmotion::Excited) pitch = mean_f0 + 35 + (f0 - mean_f0) * 1.6;
            else if (emotion == VocalEmotion::Somber) pitch = mean_f0 - 25 + (f0 - mean_f0) * .65;
            else pitch = std::clamp(mean_f0, 140.0, 190.0) + (f0 - mean_f0) * .35;
            f0 = static_cast<float>(std::max(1.0, pitch));
        }
        double level = energy;
        if (emotion == VocalEmotion::Whisper) level *= .6;
        else if (emotion == VocalEmotion::Excited) level = (mean_energy + (level - mean_energy) * 1.35) * 1.1;
        else if (emotion == VocalEmotion::Somber) level = mean_energy * .85 + (level - mean_energy) * .4;
        else level *= 1.05;
        energy = static_cast<float>(std::max(0.0, level));
    }
    std::size_t offset = 0, phrase_begin = 0;
    bool initial_stress = true;
    auto falling_tail = [&](std::size_t end) {
        if (end <= phrase_begin) return;
        const auto tail = std::max<std::size_t>(1, (end - phrase_begin) / 5);
        for (std::size_t frame = end - tail; frame < end; ++frame)
            if (result.f0_contour[frame] > 0)
                result.f0_contour[frame] = std::max(1.0F, result.f0_contour[frame] *
                    static_cast<float>(1.0 - .1 * (frame - (end - tail) + 1) / tail));
    };
    for (std::size_t token = 0; token < result.durations.size(); ++token) {
        const auto count = static_cast<std::size_t>(result.durations[token]);
        const auto kind = result.token_kinds.empty() ? ProsodyTokenKind::Unknown : result.token_kinds[token];
        if (emotion == VocalEmotion::Authoritative && initial_stress && count && kind == ProsodyTokenKind::StressedVowel) {
            for (std::size_t frame = offset; frame < offset + count; ++frame) result.energy_contour[frame] *= 1.25F;
            // Adjacent diphthong vowel tokens share this first stressed nucleus.
            if (token + 1 == result.durations.size() || result.token_kinds[token + 1] != ProsodyTokenKind::StressedVowel)
                initial_stress = false;
        }
        if (kind == ProsodyTokenKind::Boundary) {
            if (emotion == VocalEmotion::Somber) falling_tail(offset);
            if (emotion == VocalEmotion::Authoritative)
                for (std::size_t frame = offset; frame < offset + count; ++frame) result.energy_contour[frame] *= .35F;
            phrase_begin = offset + count;
            initial_stress = true;
        }
        offset += count;
    }
    if (emotion == VocalEmotion::Somber) falling_tail(offset);
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
