#pragma once

#include "vocal/acoustic_model.hpp"

#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vocal {

struct OnnxModelConfig {
    std::string token_input{"input_ids"};
    std::string mel_output{"mel"};
    std::string duration_output;
    std::string alignment_output;
    std::size_t mel_bins{80};
    int intra_op_threads{0};
};

struct AlignmentDiagnostics {
    double inference_ms{};
    std::vector<float> durations;
    std::vector<float> alignment;
    std::vector<std::int64_t> alignment_shape;
    std::size_t input_tokens{};
    std::size_t output_frames{};
    double duration_frame_error{};
    std::size_t zero_duration_tokens{};
    std::size_t alignment_monotonic_violations{};
    double alignment_token_coverage{};
    [[nodiscard]] double frames_per_token() const;
};

class OnnxRuntimeAcousticModel final : public AcousticModel {
public:
    OnnxRuntimeAcousticModel(const std::filesystem::path& model_path,
                             OnnxModelConfig config = {});
    ~OnnxRuntimeAcousticModel() override;
    OnnxRuntimeAcousticModel(OnnxRuntimeAcousticModel&&) noexcept;
    OnnxRuntimeAcousticModel& operator=(OnnxRuntimeAcousticModel&&) noexcept;
    OnnxRuntimeAcousticModel(const OnnxRuntimeAcousticModel&) = delete;
    OnnxRuntimeAcousticModel& operator=(const OnnxRuntimeAcousticModel&) = delete;

    [[nodiscard]] MelSpectrogram infer(const SynthesisRequest& request) override;
    [[nodiscard]] const AlignmentDiagnostics& diagnostics() const noexcept;
    [[nodiscard]] static bool compiled_with_runtime() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vocal
