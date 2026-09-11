#pragma once

#include "vocal/audio.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace vocal {

enum class SynthesizerKind {
    formant,
    statistical_hmm,
    articulatory,
    lpc,
    sine_wave,
    unit_selection,
    homebrew_neural,
    onnx_style
};

[[nodiscard]] std::string_view name(SynthesizerKind kind);
[[nodiscard]] Waveform synthesize_demo(SynthesizerKind kind, std::string_view text,
                                       int sample_rate_hz = 24'000);
[[nodiscard]] std::vector<std::filesystem::path> render_comparison_suite(
    const std::filesystem::path& output_directory, std::string_view text,
    int sample_rate_hz = 24'000);

}  // namespace vocal

