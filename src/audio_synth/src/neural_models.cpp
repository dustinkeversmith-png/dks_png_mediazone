#include "vocal/neural_models.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace vocal {
namespace {

float hash_weight(std::size_t a, std::size_t b, std::uint32_t seed) {
    std::uint32_t x = static_cast<std::uint32_t>(a * 0x9e3779b9U) ^
                      static_cast<std::uint32_t>(b * 0x85ebca6bU) ^ seed;
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    return (static_cast<float>(x & 0xffffU) / 32767.5F - 1.0F) * 0.22F;
}

std::array<float, 16> embed(unsigned char token, std::uint32_t seed) {
    std::array<float, 16> value{};
    for (std::size_t i = 0; i < value.size(); ++i) {
        value[i] = std::sin((static_cast<float>(token) + 1.0F) *
                            (static_cast<float>(i) + 1.0F) * 0.071F +
                            static_cast<float>(seed % 17U));
    }
    return value;
}

MelSpectrogram run_graph(const SynthesisRequest& request, bool variance_adaptor) {
    const auto bins = std::max<std::size_t>(request.mel_bins, 8);
    const std::string_view text = request.text.empty() ? std::string_view{" "} : request.text;
    std::vector<std::array<float, 16>> expanded;
    for (std::size_t token_index = 0; token_index < text.size(); ++token_index) {
        auto hidden = embed(static_cast<unsigned char>(text[token_index]),
                            variance_adaptor ? 91U : 37U);
        // A compact encoder block: fixed dense projection plus residual nonlinearity.
        for (std::size_t i = 0; i < hidden.size(); ++i) {
            float projected = 0.0F;
            for (std::size_t j = 0; j < hidden.size(); ++j) {
                projected += hidden[j] * hash_weight(i, j, variance_adaptor ? 109U : 43U);
            }
            hidden[i] = std::tanh(hidden[i] + projected);
        }
        const unsigned char c = static_cast<unsigned char>(text[token_index]);
        std::size_t duration = 5 + (c % 3);
        if (c == ' ') duration = 4;
        if (variance_adaptor) {
            // FastSpeech2-like explicit duration/pitch/energy conditioning.
            duration += std::isalpha(c) && std::string_view("aeiouAEIOU").find(c) !=
                                            std::string_view::npos ? 3U : 0U;
            const float pitch = 0.15F * std::sin(static_cast<float>(token_index) * 0.7F);
            const float energy = c == ' ' ? -0.7F : 0.1F;
            hidden[0] += pitch;
            hidden[1] += energy;
        }
        for (std::size_t frame = 0; frame < duration; ++frame) expanded.push_back(hidden);
    }

    MelSpectrogram mel{expanded.size(), bins, {}};
    mel.log_mel.resize(mel.frames * mel.bins);
    for (std::size_t frame = 0; frame < mel.frames; ++frame) {
        for (std::size_t bin = 0; bin < mel.bins; ++bin) {
            float value = -5.0F;
            for (std::size_t h = 0; h < expanded[frame].size(); ++h) {
                value += expanded[frame][h] * hash_weight(bin, h, variance_adaptor ? 211U : 127U);
            }
            const float envelope = -0.018F * static_cast<float>(bin);
            mel.log_mel[frame * mel.bins + bin] = std::clamp(value + envelope, -11.5F, 2.0F);
        }
    }
    return mel;
}

}  // namespace

MelSpectrogram HomebrewAcousticModel::infer(const SynthesisRequest& request) {
    return run_graph(request, false);
}

MelSpectrogram OnnxStyleAcousticModel::infer(const SynthesisRequest& request) {
    return run_graph(request, true);
}

AcousticComparison compare_neural_models(const SynthesisRequest& request) {
    HomebrewAcousticModel homebrew_model;
    OnnxStyleAcousticModel onnx_model;
    const auto homebrew = homebrew_model.infer(request);
    const auto onnx = onnx_model.infer(request);
    double absolute = 0.0, squared = 0.0;
    std::size_t count = 0;
    // Compare at normalized utterance time because the duration predictors differ.
    const std::size_t comparison_frames = std::max(homebrew.frames, onnx.frames);
    for (std::size_t frame = 0; frame < comparison_frames; ++frame) {
        const auto homebrew_index = std::min(homebrew.frames - 1,
            frame * homebrew.frames / comparison_frames);
        const auto onnx_index = std::min(onnx.frames - 1,
            frame * onnx.frames / comparison_frames);
        const auto a = homebrew.frame(homebrew_index);
        const auto b = onnx.frame(onnx_index);
        for (std::size_t bin = 0; bin < std::min(a.size(), b.size()); ++bin) {
            const double delta = static_cast<double>(a[bin]) - static_cast<double>(b[bin]);
            absolute += std::abs(delta);
            squared += delta * delta;
            ++count;
        }
    }
    return {homebrew.frames, onnx.frames, std::min(homebrew.bins, onnx.bins),
            static_cast<double>(onnx.frames) / static_cast<double>(homebrew.frames),
            absolute / static_cast<double>(count), std::sqrt(squared / static_cast<double>(count))};
}

}  // namespace vocal
