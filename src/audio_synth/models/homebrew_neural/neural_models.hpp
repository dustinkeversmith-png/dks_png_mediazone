#pragma once

#include "vocal/acoustic_model.hpp"

namespace vocal {

struct AcousticComparison {
    std::size_t homebrew_frames{};
    std::size_t onnx_style_frames{};
    std::size_t mel_bins{};
    double duration_ratio{};
    double log_mel_mae{};
    double log_mel_rmse{};
};

// Tiny deterministic network used as the project's transparent homebrew model.
class HomebrewAcousticModel final : public AcousticModel {
public:
    [[nodiscard]] MelSpectrogram infer(const SynthesisRequest& request) override;
};

// Dependency-free FastSpeech2-shaped baseline. It mirrors an ONNX deployment
// graph's inputs/outputs but is not an ONNX Runtime implementation or pretrained.
class OnnxStyleAcousticModel final : public AcousticModel {
public:
    [[nodiscard]] MelSpectrogram infer(const SynthesisRequest& request) override;
};

[[nodiscard]] AcousticComparison compare_neural_models(const SynthesisRequest& request);

}  // namespace vocal
