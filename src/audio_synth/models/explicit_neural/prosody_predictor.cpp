#include "include/prosody_predictor.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>

#if VA_HAS_ONNX_RUNTIME
#include "ort_contract.hpp"
#endif

namespace vocal {
struct NeuralProsodyPredictor::Impl {
    double elapsed_ms{};
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "explicit-prosody"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    Impl(const std::filesystem::path& path, int threads, const std::string& affinities) {
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("missing trained prosody predictor: " + path.string());
        options = detail::explicit_session_options(threads, affinities);
        session = Ort::Session(environment, path.c_str(), options);
        if (session.GetInputCount() != 2) throw std::runtime_error("prosody predictor needs input_ids and sid");
        detail::require_input(session, "input_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::array<std::int64_t, 2>{1, -1});
        detail::require_input(session, "sid", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::array<std::int64_t, 1>{1});
        for (const auto* name : {"durations", "token_f0_hz", "token_energy"}) detail::require_output(session, name);
    }
#else
    Impl(const std::filesystem::path&, int, const std::string&) {
        throw std::runtime_error("trained prosody predictor requires ONNX Runtime");
    }
#endif
};

NeuralProsodyPredictor::NeuralProsodyPredictor(const std::filesystem::path& path, int threads, const std::string& affinities)
    : impl_(std::make_unique<Impl>(path, threads, affinities)) {}
NeuralProsodyPredictor::~NeuralProsodyPredictor() = default;
double NeuralProsodyPredictor::inference_ms() const noexcept { return impl_->elapsed_ms; }

ProsodyControls NeuralProsodyPredictor::predict(std::span<const std::int64_t> token_ids, std::int64_t speaker,
                                               std::span<const std::int64_t> duration_override) {
    if (token_ids.empty() || token_ids.size() > 4096 || speaker < 0)
        throw std::invalid_argument("prosody predictor needs 1..4096 tokens and a nonnegative speaker ID");
    if (!duration_override.empty() && duration_override.size() != token_ids.size())
        throw std::invalid_argument("durations must match the model token count");
#if VA_HAS_ONNX_RUNTIME
    std::vector<std::int64_t> tokens(token_ids.begin(), token_ids.end());
    const std::array<std::int64_t, 2> token_shape{1, static_cast<std::int64_t>(tokens.size())};
    const std::array<std::int64_t, 1> sid_shape{1};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, tokens.data(), tokens.size(), token_shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, &speaker, 1, sid_shape.data(), 1));
    const char* names[] = {"input_ids", "sid"};
    const char* outputs[] = {"durations", "token_f0_hz", "token_energy"};
    const auto started = std::chrono::steady_clock::now();
    auto values = impl_->session.Run(Ort::RunOptions{nullptr}, names, inputs.data(), 2, outputs, 3);
    impl_->elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const std::vector<std::int64_t> expected{1, static_cast<std::int64_t>(tokens.size())};
    if (!values[0].IsTensor() || values[0].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        values[0].GetTensorTypeAndShapeInfo().GetShape() != expected ||
        detail::float_output_shape(values[1]) != expected || detail::float_output_shape(values[2]) != expected)
        throw std::runtime_error("prosody outputs must have one duration/F0/energy value per token");
    const auto* durations = values[0].GetTensorData<std::int64_t>();
    const auto* f0 = values[1].GetTensorData<float>();
    const auto* energy = values[2].GetTensorData<float>();
    ProsodyControls result;
    result.speaker_id = speaker;
    if (duration_override.empty()) result.durations.assign(durations, durations + tokens.size());
    else result.durations.assign(duration_override.begin(), duration_override.end());
    const auto frames = duration_frames(result.durations);
    result.f0_contour.reserve(frames); result.energy_contour.reserve(frames);
    for (std::size_t token = 0; token < tokens.size(); ++token) {
        if (!std::isfinite(f0[token]) || f0[token] < 0 || !std::isfinite(energy[token]) || energy[token] < 0)
            throw std::runtime_error("prosody predictor returned invalid F0 or energy");
        result.f0_contour.insert(result.f0_contour.end(), static_cast<std::size_t>(result.durations[token]), f0[token]);
        result.energy_contour.insert(result.energy_contour.end(), static_cast<std::size_t>(result.durations[token]), energy[token]);
    }
    validate_prosody(result, tokens.size());
    return result;
#else
    throw std::runtime_error("trained prosody predictor requires ONNX Runtime");
#endif
}
} // namespace vocal
