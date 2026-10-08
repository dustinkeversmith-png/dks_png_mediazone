#include "matcha_acoustic_gen.hpp"
#include "vocal/control_params.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#if VA_HAS_ONNX_RUNTIME
#include <explicit_neural/ort_contract.hpp>
#endif

namespace vocal {
struct MatchaAcousticGenerator::Impl {
    double elapsed_ms{};
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "matcha-acoustic"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    Impl(const std::filesystem::path& path, int threads) {
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("missing Matcha acoustic model: " + path.string());
        options = detail::explicit_session_options(threads, {});
        session = Ort::Session(environment, path.c_str(), options);
        if (session.GetInputCount() != 4) throw std::runtime_error("Matcha export must expose four inputs");
        const std::array<std::int64_t, 2> tokens{1, -1};
        const std::array<std::int64_t, 1> scalar{1};
        detail::require_input(session, "x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, tokens);
        detail::require_input(session, "x_length", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, scalar);
        detail::require_input(session, "noise_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, scalar);
        detail::require_input(session, "length_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, scalar);
        detail::require_output(session, "mel");
        // This runner deliberately accepts only the selected English single-speaker export.
        Ort::AllocatorWithDefaultOptions allocator;
        auto metadata = session.GetModelMetadata();
        for (const auto& [key, expected] : std::array<std::pair<const char*, const char*>, 4>{
                 {{"sample_rate", "22050"}, {"pad_id", "0"}, {"use_eos_bos", "1"}, {"n_speakers", "1"}}}) {
            const auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
            if (!value || std::string(value.get()) != expected)
                throw std::runtime_error(std::string("unexpected Matcha metadata: ") + key);
        }
    }
#else
    Impl(const std::filesystem::path&, int) {
        throw std::runtime_error("matcha-tts requires an ONNX Runtime-enabled build; use msvc-onnx");
    }
#endif
};

MatchaAcousticGenerator::MatchaAcousticGenerator(const std::filesystem::path& path, int threads)
    : impl_(std::make_unique<Impl>(path, threads)) {}
MatchaAcousticGenerator::~MatchaAcousticGenerator() = default;
MatchaAcousticGenerator::MatchaAcousticGenerator(MatchaAcousticGenerator&&) noexcept = default;
MatchaAcousticGenerator& MatchaAcousticGenerator::operator=(MatchaAcousticGenerator&&) noexcept = default;
double MatchaAcousticGenerator::inference_ms() const noexcept { return impl_->elapsed_ms; }

MelSpectrogram MatchaAcousticGenerator::infer(std::span<const std::int64_t> cmu_token_ids, float speed, float noise_scale) {
    if (cmu_token_ids.size() < 3 || cmu_token_ids.size() > 400 || cmu_token_ids.front() != 1 || cmu_token_ids.back() != 2)
        throw std::invalid_argument("Matcha requires 3..400 CMU/Piper-formatted tokens with BOS=1 and EOS=2; split longer text");
    if (!std::isfinite(speed) || speed < .25F || speed > 4 || !std::isfinite(noise_scale) || noise_scale < 0 || noise_scale > 2)
        throw std::invalid_argument("Matcha speed must be in 0.25..4, noise in 0..2, both finite");
#if VA_HAS_ONNX_RUNTIME
    // CMU already pads between symbols. Matcha's AddBlank also pads before BOS and after EOS.
    std::vector<std::int64_t> tokens{0};
    tokens.insert(tokens.end(), cmu_token_ids.begin(), cmu_token_ids.end());
    tokens.push_back(0);
    for (auto token : tokens) if (token < 0) throw std::invalid_argument("token IDs must be nonnegative");
    const std::array<std::int64_t, 2> shape{1, static_cast<std::int64_t>(tokens.size())};
    const std::array<std::int64_t, 1> scalar_shape{1};
    std::array<std::int64_t, 1> length{shape[1]};
    std::array<float, 1> noise{noise_scale}, cadence{1.0F / speed};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, tokens.data(), tokens.size(), shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, length.data(), 1, scalar_shape.data(), 1));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(memory, noise.data(), 1, scalar_shape.data(), 1));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(memory, cadence.data(), 1, scalar_shape.data(), 1));
    const std::array<const char*, 4> names{"x", "x_length", "noise_scale", "length_scale"};
    detail::require_input(impl_->session, "x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, shape);
    const char* output = "mel";
    const auto started = std::chrono::steady_clock::now();
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, names.data(), inputs.data(), inputs.size(), &output, 1);
    impl_->elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const auto actual = detail::float_output_shape(outputs[0]);
    if (actual.size() != 3 || actual[0] != 1 || actual[1] != 80 || actual[2] <= 0 ||
        actual[2] > static_cast<std::int64_t>(maximum_prosody_frames))
        throw std::runtime_error("Matcha must return a bounded [1, 80, frames] mel tensor");
    const auto frames = static_cast<std::size_t>(actual[2]);
    MelSpectrogram mel{frames, 80, std::vector<float>(frames * 80)};
    const float* data = outputs[0].GetTensorData<float>();
    for (std::size_t bin = 0; bin < 80; ++bin)
        for (std::size_t frame = 0; frame < frames; ++frame) {
            const float value = data[bin * frames + frame];
            if (!std::isfinite(value)) throw std::runtime_error("non-finite Matcha mel output");
            mel.log_mel[frame * 80 + bin] = value;
        }
    return mel;
#else
    throw std::runtime_error("ONNX Runtime support was not compiled");
#endif
}
} // namespace vocal
