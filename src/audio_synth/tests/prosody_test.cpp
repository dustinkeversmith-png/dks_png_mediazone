#include <explicit_neural/include/length_regulator.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int failures{};
void check(bool ok, const char* label) { if (!ok) { ++failures; std::cerr << label << '\n'; } }
template<class F> void rejects(F call, const char* label) {
    try { call(); check(false, label); } catch (const std::exception&) {}
}
}

int main() {
    const std::vector<float> hidden{1, 2, 3, 4, 5, 6};
    const std::vector<std::int64_t> durations{2, 0, 3};
    const auto expanded = vocal::length_regulate(hidden, 2, durations);
    check(expanded.frames == 5 && expanded.hidden_dim == 2 &&
          expanded.values == std::vector<float>{1, 2, 1, 2, 5, 6, 5, 6, 5, 6}, "token ordering / zero-duration expansion");
    check(vocal::length_regulate(hidden, 2, std::vector<std::int64_t>{0, 0, 0}).values.empty(), "all-zero regulator");
    rejects([&] { (void)vocal::length_regulate(hidden, 2, std::vector<std::int64_t>{-1, 1, 1}); }, "negative duration");
    rejects([&] { (void)vocal::length_regulate(hidden, 2, durations, 4); }, "frame budget");
    rejects([&] { (void)vocal::length_regulate(hidden, 0, durations); }, "zero hidden dimension");
    rejects([&] { (void)vocal::length_regulate(hidden, 3, durations); }, "hidden shape mismatch");
    rejects([&] { (void)vocal::duration_frames(std::vector<std::int64_t>{std::numeric_limits<std::int64_t>::max(), 1}); }, "duration overflow");
    vocal::ProsodyControls controls{{2, 0, 2}, {0, 100, 200, 0}, {1, 3, 1, 3}, 2};
    const auto scaled = vocal::apply_prosody_sliders(controls, {1.5F, 2.0F, 2.0F, 0.0F});
    check(scaled.durations == std::vector<std::int64_t>{1, 0, 1}, "cadence preserves skipped tokens");
    check(scaled.f0_contour == std::vector<float>{0, 300}, "pitch preserves unvoiced zeros and token boundaries");
    check(scaled.energy_contour == std::vector<float>{4, 4} && scaled.speaker_id == 2, "energy mean and speaker");
    rejects([&] { (void)vocal::apply_prosody_sliders(controls, {1, 0, 1, 1}); }, "zero speed");
    rejects([&] { (void)vocal::apply_prosody_sliders(controls, {std::numeric_limits<float>::quiet_NaN(), 1, 1, 1}); }, "nonfinite pitch");
    controls.f0_contour.pop_back();
    rejects([&] { vocal::validate_prosody(controls, 3); }, "contour mismatch");
    controls = vocal::baseline_prosody(3);
    check(controls.f0_contour.size() == 18 && controls.energy_contour.size() == 18, "baseline contour length");
    controls.energy_contour[0] = std::numeric_limits<float>::infinity();
    rejects([&] { vocal::validate_prosody(controls, 3); }, "nonfinite energy");
    using Kind = vocal::ProsodyTokenKind;
    using Emotion = vocal::VocalEmotion;
    const vocal::ProsodyControls expressive{{10, 10, 0, 10, 4},
        std::vector<float>(34, 150), std::vector<float>(34, 1), 7,
        {Kind::UnvoicedConsonant, Kind::StressedVowel, Kind::Unknown, Kind::Vowel, Kind::Boundary}};
    auto base = expressive;
    for (std::size_t i = 0; i < 10; ++i) base.f0_contour[i] = 0;
    for (std::size_t i = 10; i < 20; ++i) base.f0_contour[i] = 130;
    for (std::size_t i = 20; i < 30; ++i) base.f0_contour[i] = 170;
    for (std::size_t i = 30; i < 34; ++i) base.f0_contour[i] = 0;
    base.energy_contour[12] = 2;
    const auto neutral = vocal::apply_emotion_preset(base, Emotion::Neutral);
    check(neutral.durations == base.durations && neutral.f0_contour == base.f0_contour &&
          neutral.energy_contour == base.energy_contour && neutral.token_kinds == base.token_kinds,
          "neutral is an identity transform");
    const auto whisper = vocal::apply_emotion_preset(base, Emotion::Whisper);
    check(whisper.durations == std::vector<std::int64_t>{11, 10, 0, 10, 4}, "whisper lengthens only annotated unvoiced consonants");
    check(whisper.f0_contour[0] == 0 && std::abs(whisper.f0_contour[11] - 147) < 1e-4F &&
          std::abs(whisper.f0_contour[21] - 153) < 1e-4F, "whisper flattens voiced range and preserves unvoiced zeros");
    check(std::abs(whisper.energy_contour[13] - 1.2F) < 1e-5F, "whisper energy falls forty percent");
    const auto excited = vocal::apply_emotion_preset(base, Emotion::Excited);
    check(excited.durations == std::vector<std::int64_t>{9, 9, 0, 9, 3}, "excited speeds cadence with per-token rounding");
    check(std::abs(excited.f0_contour[9] - 153) < 1e-4F && std::abs(excited.f0_contour[18] - 217) < 1e-4F,
          "excited raises voiced mean by 35 Hz and range by 1.6");
    check(excited.energy_contour[11] > 2 && excited.f0_contour[0] == 0, "excited boosts peaks without voicing consonants");
    const auto somber = vocal::apply_emotion_preset(base, Emotion::Somber);
    check(somber.durations == std::vector<std::int64_t>{11, 11, 0, 11, 5}, "somber extends cadence at speed 0.88");
    check(std::abs(somber.f0_contour[11] - 112) < 1e-4F && somber.f0_contour[32] < somber.f0_contour[22],
          "somber lowers voiced mean and falls at sentence ending");
    check(somber.energy_contour[14] < base.energy_contour[12], "somber dampens energy peak");
    const auto authoritative = vocal::apply_emotion_preset(base, Emotion::Authoritative);
    check(authoritative.durations == std::vector<std::int64_t>{10, 10, 0, 10, 2}, "authoritative sharpens boundary cadence");
    check(std::abs(authoritative.f0_contour[10] - 143) < 1e-4F && authoritative.energy_contour[10] > authoritative.energy_contour[20]
          && authoritative.energy_contour[30] < authoritative.energy_contour[20], "authoritative stress and boundary dynamics");
    for (auto emotion : {Emotion::Neutral, Emotion::Whisper, Emotion::Excited, Emotion::Somber, Emotion::Authoritative}) {
        const auto result = vocal::apply_emotion_preset(base, emotion);
        vocal::validate_prosody(result, base.durations.size());
        check(result.speaker_id == 7 && result.durations[2] == 0 && result.token_kinds == base.token_kinds,
              "every preset preserves speaker, skipped token and annotations");
    }
    check(base.durations == expressive.durations && base.energy_contour[12] == 2, "presets leave base unmodified");
    auto unvoiced = vocal::baseline_prosody(2, 5, 0);
    unvoiced.token_kinds = {Kind::Unknown, Kind::Boundary};
    for (auto emotion : {Emotion::Whisper, Emotion::Excited, Emotion::Somber, Emotion::Authoritative}) {
        const auto result = vocal::apply_emotion_preset(unvoiced, emotion);
        for (auto value : result.f0_contour) check(value == 0, "all-unvoiced input stays unvoiced");
    }
    auto no_annotations = base;
    no_annotations.token_kinds.clear();
    check(vocal::apply_emotion_preset(no_annotations, Emotion::Whisper).durations == base.durations,
          "missing annotations never guess consonant identity");
    const vocal::ProsodyControls phrases{{2, 2, 0, 2, 2}, std::vector<float>(8, 150),
        std::vector<float>(8, 1), 0,
        {Kind::StressedVowel, Kind::Boundary, Kind::Unknown, Kind::StressedVowel, Kind::Boundary}};
    const auto phrase_controls = vocal::apply_emotion_preset(phrases, Emotion::Authoritative);
    check(std::abs(phrase_controls.energy_contour[0] - 1.3125F) < 1e-5F &&
          std::abs(phrase_controls.energy_contour[3] - 1.3125F) < 1e-5F,
          "authoritative emphasis resets at each phrase boundary");
    const auto low_pitch = vocal::apply_emotion_preset(vocal::baseline_prosody(1, 2, 10), Emotion::Somber);
    for (auto value : low_pitch.f0_contour) check(value > 0, "somber floor never converts voiced input to unvoiced");
    auto invalid = base;
    invalid.token_kinds.pop_back();
    rejects([&] { (void)vocal::apply_emotion_preset(invalid, Emotion::Whisper); }, "annotation mismatch rejected");
    invalid = base;
    invalid.token_kinds[0] = static_cast<Kind>(99);
    rejects([&] { (void)vocal::apply_emotion_preset(invalid, Emotion::Whisper); }, "unknown token annotation rejected");
    rejects([&] { (void)vocal::apply_emotion_preset(base, static_cast<Emotion>(99)); }, "unknown emotion rejected");
    invalid = vocal::baseline_prosody(1, vocal::maximum_prosody_frames);
    invalid.token_kinds = {Kind::UnvoicedConsonant};
    rejects([&] { (void)vocal::apply_emotion_preset(invalid, Emotion::Whisper); }, "emotion expansion respects frame limit");
    return failures ? 1 : 0;
}
