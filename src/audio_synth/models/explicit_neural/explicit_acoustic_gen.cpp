#include "include/explicit_acoustic_gen.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#if VA_HAS_ONNX_RUNTIME
#include "ort_contract.hpp"
#endif

namespace vocal {
struct ExplicitAcousticGenerator::Impl {
    ExplicitAcousticConfig config;
    double elapsed_ms{};
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "explicit-acoustic"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    Impl(const std::filesystem::path& path, ExplicitAcousticConfig supplied) : config(std::move(supplied)) {
        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error("missing acoustic generator: " + path.string());
        options = detail::explicit_session_options(config.intra_op_threads, config.thread_affinities);
        session = Ort::Session(environment, path.c_str(), options);
        if (session.GetInputCount() != 5)
            throw std::runtime_error("explicit acoustic generator must expose exactly five inputs");
        const std::array<std::int64_t, 2> dynamic{1, -1};
        const std::array<std::int64_t, 1> speaker{1};
        detail::require_input(session, "input_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, dynamic);
        detail::require_input(session, "durations", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, dynamic);
        detail::require_input(session, "f0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dynamic);
        detail::require_input(session, "energy", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dynamic);
        detail::require_input(session, "sid", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, speaker);
        detail::require_output(session, config.mel_output);
    }
#else
    Impl(const std::filesystem::path&, ExplicitAcousticConfig supplied) : config(std::move(supplied)) {
        throw std::runtime_error("explicit-tts requires an ONNX Runtime-enabled build; use msvc-onnx");
    }
#endif
};

ExplicitAcousticGenerator::ExplicitAcousticGenerator(const std::filesystem::path& path, ExplicitAcousticConfig config)
    : impl_(std::make_unique<Impl>(path, std::move(config))) {}
ExplicitAcousticGenerator::~ExplicitAcousticGenerator() = default;
ExplicitAcousticGenerator::ExplicitAcousticGenerator(ExplicitAcousticGenerator&&) noexcept = default;
ExplicitAcousticGenerator& ExplicitAcousticGenerator::operator=(ExplicitAcousticGenerator&&) noexcept = default;
double ExplicitAcousticGenerator::inference_ms() const noexcept { return impl_->elapsed_ms; }

MelSpectrogram ExplicitAcousticGenerator::infer(std::span<const std::int64_t> token_ids, const ProsodyControls& controls) {
    validate_prosody(controls, token_ids.size());
    for (auto token : token_ids) if (token < 0) throw std::invalid_argument("token IDs must be nonnegative");
#if VA_HAS_ONNX_RUNTIME
    const auto frames = controls.f0_contour.size();
    const std::array<std::int64_t, 2> tokens_shape{1, static_cast<std::int64_t>(token_ids.size())};
    const std::array<std::int64_t, 2> frames_shape{1, static_cast<std::int64_t>(frames)};
    const std::array<std::int64_t, 1> speaker_shape{1};
    const std::array<const char*, 5> names{"input_ids", "durations", "f0", "energy", "sid"};
    detail::require_input(impl_->session, names[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, tokens_shape);
    detail::require_input(impl_->session, names[1], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, tokens_shape);
    detail::require_input(impl_->session, names[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, frames_shape);
    detail::require_input(impl_->session, names[3], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, frames_shape);
    // Owned, isolated writable buffers outlive all input tensors and Run().
    std::vector<std::int64_t> tokens(token_ids.begin(), token_ids.end());
    auto durations = controls.durations;
    auto f0 = controls.f0_contour;
    auto energy = controls.energy_contour;
    std::array<std::int64_t, 1> speaker{controls.speaker_id};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, tokens.data(), tokens.size(), tokens_shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, durations.data(), durations.size(), tokens_shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(memory, f0.data(), f0.size(), frames_shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(memory, energy.data(), energy.size(), frames_shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, speaker.data(), 1, speaker_shape.data(), 1));
    const char* output = impl_->config.mel_output.c_str();
    const auto started = std::chrono::steady_clock::now();
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, names.data(), inputs.data(), inputs.size(), &output, 1);
    impl_->elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const auto shape = detail::float_output_shape(outputs[0]);
    if (shape != std::vector<std::int64_t>{1, 80, static_cast<std::int64_t>(frames)})
        throw std::runtime_error("acoustic mel output must be [1, 80, sum(durations)]");
    // The framework uses frame-major layout; ONNX exports use channel-major.
    const float* data = outputs[0].GetTensorData<float>();
    MelSpectrogram mel{frames, 80, std::vector<float>(frames * 80)};
    for (std::size_t bin = 0; bin < 80; ++bin)
        for (std::size_t frame = 0; frame < frames; ++frame) {
            const float value = data[bin * frames + frame];
            if (!std::isfinite(value)) throw std::runtime_error("non-finite acoustic mel output");
            mel.log_mel[frame * 80 + bin] = value;
        }
    return mel;
#else
    throw std::runtime_error("ONNX Runtime support was not compiled");
#endif
}
} // namespace vocal
