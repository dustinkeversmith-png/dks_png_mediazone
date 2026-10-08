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
    return failures ? 1 : 0;
}
