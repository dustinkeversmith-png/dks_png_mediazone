#include "vocal/onnx_runtime_model.hpp"

#include <chrono>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#if VA_HAS_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#include "ort_session_options.hpp"
#endif

namespace vocal {

double AlignmentDiagnostics::frames_per_token() const {
    return input_tokens ? static_cast<double>(output_frames) / input_tokens : 0.0;
}

struct OnnxRuntimeAcousticModel::Impl {
    OnnxModelConfig config;
    AlignmentDiagnostics diagnostics;
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "vocal-acoustics"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};

    Impl(const std::filesystem::path& model_path, OnnxModelConfig supplied)
        : config(std::move(supplied)) {
        options = detail::cpu_session_options(config.intra_op_threads);
        session = Ort::Session(environment, model_path.c_str(), options);
    }
#else
    Impl(const std::filesystem::path&, OnnxModelConfig supplied) : config(std::move(supplied)) {
        throw std::runtime_error(
            "ONNX Runtime support is disabled; configure with "
            "-DVA_ENABLE_ONNX_RUNTIME=ON -DONNXRUNTIME_ROOT=<installation>");
    }
#endif
};

OnnxRuntimeAcousticModel::OnnxRuntimeAcousticModel(
    const std::filesystem::path& model_path, OnnxModelConfig config)
    : impl_(std::make_unique<Impl>(model_path, std::move(config))) {}
OnnxRuntimeAcousticModel::~OnnxRuntimeAcousticModel() = default;
OnnxRuntimeAcousticModel::OnnxRuntimeAcousticModel(OnnxRuntimeAcousticModel&&) noexcept = default;
OnnxRuntimeAcousticModel& OnnxRuntimeAcousticModel::operator=(OnnxRuntimeAcousticModel&&) noexcept = default;

bool OnnxRuntimeAcousticModel::compiled_with_runtime() noexcept {
#if VA_HAS_ONNX_RUNTIME
    return true;
#else
    return false;
#endif
}

const AlignmentDiagnostics& OnnxRuntimeAcousticModel::diagnostics() const noexcept {
    return impl_->diagnostics;
}

MelSpectrogram OnnxRuntimeAcousticModel::infer(const SynthesisRequest& request) {
#if VA_HAS_ONNX_RUNTIME
    std::vector<std::int64_t> tokens;
    tokens.reserve(request.text.size());
    // The export contract uses byte-level IDs 1..256 and 0 for padding. Replace
    // this tokenizer at the application boundary for phoneme/token-ID exports.
    for (unsigned char c : request.text) tokens.push_back(static_cast<std::int64_t>(c) + 1);
    if (tokens.empty()) tokens.push_back(1);
    const std::array<std::int64_t, 2> input_shape{1, static_cast<std::int64_t>(tokens.size())};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto token_tensor = Ort::Value::CreateTensor<std::int64_t>(
        memory, tokens.data(), tokens.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {impl_->config.token_input.c_str()};
    std::vector<const char*> output_names{impl_->config.mel_output.c_str()};
    if (!impl_->config.duration_output.empty()) output_names.push_back(impl_->config.duration_output.c_str());
    if (!impl_->config.alignment_output.empty()) output_names.push_back(impl_->config.alignment_output.c_str());

    const auto started = std::chrono::steady_clock::now();
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names, &token_tensor, 1,
                                      output_names.data(), output_names.size());
    const auto finished = std::chrono::steady_clock::now();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2 && shape.size() != 3) throw std::runtime_error("mel output must have rank 2 or 3");
    const std::size_t frames = static_cast<std::size_t>(shape[shape.size() - 2]);
    const std::size_t bins = static_cast<std::size_t>(shape.back());
    if (bins != impl_->config.mel_bins) throw std::runtime_error("ONNX mel-bin count does not match configuration");
    const float* mel_data = outputs[0].GetTensorData<float>();
    MelSpectrogram mel{frames, bins, std::vector<float>(mel_data, mel_data + frames * bins)};

    impl_->diagnostics = {};
    impl_->diagnostics.inference_ms =
        std::chrono::duration<double, std::milli>(finished - started).count();
    impl_->diagnostics.input_tokens = tokens.size();
    impl_->diagnostics.output_frames = frames;
    std::size_t output_index = 1;
    if (!impl_->config.duration_output.empty()) {
        const auto count = outputs[output_index].GetTensorTypeAndShapeInfo().GetElementCount();
        const auto type = outputs[output_index].GetTensorTypeAndShapeInfo().GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* values = outputs[output_index].GetTensorData<float>();
            impl_->diagnostics.durations.assign(values, values + count);
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            const auto* values = outputs[output_index].GetTensorData<std::int64_t>();
            impl_->diagnostics.durations.reserve(count);
            for (std::size_t i = 0; i < count; ++i) impl_->diagnostics.durations.push_back(static_cast<float>(values[i]));
        }
        ++output_index;
    }
    if (!impl_->config.alignment_output.empty()) {
        auto info = outputs[output_index].GetTensorTypeAndShapeInfo();
        impl_->diagnostics.alignment_shape = info.GetShape();
        const auto count = info.GetElementCount();
        const float* values = outputs[output_index].GetTensorData<float>();
        impl_->diagnostics.alignment.assign(values, values + count);
    }
    if (!impl_->diagnostics.durations.empty()) {
        double duration_sum = 0.0;
        for (float duration : impl_->diagnostics.durations) {
            duration_sum += duration;
            impl_->diagnostics.zero_duration_tokens += duration <= 0.0F;
        }
        impl_->diagnostics.duration_frame_error = std::abs(duration_sum - static_cast<double>(frames));
    }
    if (impl_->diagnostics.alignment_shape.size() >= 2 && !impl_->diagnostics.alignment.empty()) {
        const auto rows = static_cast<std::size_t>(impl_->diagnostics.alignment_shape[
            impl_->diagnostics.alignment_shape.size() - 2]);
        const auto columns = static_cast<std::size_t>(impl_->diagnostics.alignment_shape.back());
        if (rows && columns && rows * columns <= impl_->diagnostics.alignment.size()) {
            std::vector<bool> covered(columns);
            std::size_t previous = 0;
            for (std::size_t row = 0; row < rows; ++row) {
                std::size_t best = 0;
                for (std::size_t column = 1; column < columns; ++column)
                    if (impl_->diagnostics.alignment[row * columns + column] >
                        impl_->diagnostics.alignment[row * columns + best]) best = column;
                if (row && best < previous) ++impl_->diagnostics.alignment_monotonic_violations;
                previous = best;
                covered[best] = true;
            }
            impl_->diagnostics.alignment_token_coverage =
                static_cast<double>(std::count(covered.begin(), covered.end(), true)) / columns;
        }
    }
    return mel;
#else
    (void)request;
    throw std::runtime_error("ONNX Runtime support was not compiled");
#endif
}

}  // namespace vocal
