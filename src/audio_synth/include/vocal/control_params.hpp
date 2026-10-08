#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vocal {

enum class VocalEmotion { Neutral, Whisper, Excited, Somber, Authoritative };
// Optional frontend annotations; do not infer consonant identity from zero F0.
enum class ProsodyTokenKind { Unknown, UnvoicedConsonant, Vowel, StressedVowel, Boundary };

struct ProsodyControls {
    std::vector<std::int64_t> durations; // One frame count per token, including special tokens.
    std::vector<float> f0_contour;       // Hz; zero means unvoiced.
    std::vector<float> energy_contour;   // Normalized nonnegative energy, not dB.
    std::int64_t speaker_id{0};
    std::vector<ProsodyTokenKind> token_kinds; // Empty or one annotation per token.
};

struct ProsodySliders {
    float pitch_scale{1.0F};
    float speed{1.0F};                  // > 1 is faster; durations are divided by speed.
    float energy_scale{1.0F};
    float energy_variance{1.0F};        // Scale deviations from mean energy.
};

inline constexpr std::size_t maximum_prosody_frames = 15'000;
[[nodiscard]] std::size_t duration_frames(std::span<const std::int64_t> durations,
                                         std::size_t maximum = maximum_prosody_frames);
void validate_prosody(const ProsodyControls& controls, std::size_t tokens,
                      std::size_t maximum = maximum_prosody_frames);
[[nodiscard]] ProsodyControls baseline_prosody(std::size_t tokens, std::int64_t frames_per_token = 6,
                                              float f0_hz = 180.0F, float energy = 1.0F);
// Resample contours inside each token when cadence changes; preserve unvoiced zeros.
[[nodiscard]] ProsodyControls apply_prosody_sliders(const ProsodyControls& controls,
                                                   const ProsodySliders& sliders);
[[nodiscard]] ProsodyControls apply_emotion_preset(const ProsodyControls& base, VocalEmotion emotion);
} // namespace vocal
