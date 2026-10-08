#include "include/neural_vocoder.hpp"
#include "vocal/control_params.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#if VA_HAS_ONNX_RUNTIME
#include "ort_contract.hpp"
#endif

namespace vocal {
struct NeuralVocoder::Impl {
    NeuralVocoderConfig config;
    double elapsed_ms{};
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "explicit-vocoder"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    Impl(const std::filesystem::path& path, NeuralVocoderConfig supplied) : config(std::move(supplied)) {
        if ((config.sample_rate_hz != 24'000 && config.sample_rate_hz != 22'050) ||
            config.hop_length == 0 || config.hop_length > 4096)
            throw std::invalid_argument("vocoder requires 24 kHz or 22.05 kHz and a hop length in 1..4096");
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("missing neural vocoder: " + path.string());
        options = detail::explicit_session_options(config.intra_op_threads, config.thread_affinities);
        session = Ort::Session(environment, path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        const auto model_rate = session.GetModelMetadata().LookupCustomMetadataMapAllocated("sample_rate", allocator);
        if (model_rate && std::stoi(model_rate.get()) != config.sample_rate_hz)
            throw std::runtime_error("vocoder sample-rate metadata does not match configuration");
        if (session.GetInputCount() != 1) throw std::runtime_error("vocoder must expose one mel input");
        const std::array<std::int64_t, 3> shape{1, 80, -1};
        detail::require_input(session, config.mel_input, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape);
        detail::require_output(session, config.audio_output);
    }
#else
    Impl(const std::filesystem::path&, NeuralVocoderConfig supplied) : config(std::move(supplied)) {
        throw std::runtime_error("neural vocoder requires an ONNX Runtime-enabled build; use msvc-onnx");
    }
#endif
};

NeuralVocoder::NeuralVocoder(const std::filesystem::path& path, NeuralVocoderConfig config)
    : impl_(std::make_unique<Impl>(path, std::move(config))) {}
NeuralVocoder::~NeuralVocoder() = default;
NeuralVocoder::NeuralVocoder(NeuralVocoder&&) noexcept = default;
NeuralVocoder& NeuralVocoder::operator=(NeuralVocoder&&) noexcept = default;
double NeuralVocoder::inference_ms() const noexcept { return impl_->elapsed_ms; }

Waveform NeuralVocoder::synthesize(const MelSpectrogram& mel) {
    if (mel.bins != 80 || mel.frames == 0 || mel.frames > maximum_prosody_frames ||
        mel.log_mel.size() != mel.frames * mel.bins)
        throw std::invalid_argument("vocoder expects a bounded, nonempty 80-bin mel spectrogram");
#if VA_HAS_ONNX_RUNTIME
    std::vector<float> channels(mel.log_mel.size());
    for (std::size_t frame = 0; frame < mel.frames; ++frame)
        for (std::size_t bin = 0; bin < mel.bins; ++bin) {
            const float value = mel.log_mel[frame * mel.bins + bin];
            if (!std::isfinite(value)) throw std::invalid_argument("vocoder mel input contains non-finite values");
            channels[bin * mel.frames + frame] = value;
        }
    const std::array<std::int64_t, 3> shape{1, 80, static_cast<std::int64_t>(mel.frames)};
    detail::require_input(impl_->session, impl_->config.mel_input, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory, channels.data(), channels.size(), shape.data(), shape.size());
    const char* input_name = impl_->config.mel_input.c_str();
    const char* output_name = impl_->config.audio_output.c_str();
    const auto started = std::chrono::steady_clock::now();
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, &input_name, &input, 1, &output_name, 1);
    impl_->elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const auto actual = detail::float_output_shape(outputs[0]);
    const auto samples = mel.frames * impl_->config.hop_length;
    if (actual != std::vector<std::int64_t>{1, static_cast<std::int64_t>(samples)} &&
        actual != std::vector<std::int64_t>{1, 1, static_cast<std::int64_t>(samples)})
        throw std::runtime_error("vocoder audio output must be [1, frames * hop_length] or [1, 1, frames * hop_length]");
    const float* data = outputs[0].GetTensorData<float>();
    Waveform audio{impl_->config.sample_rate_hz, std::vector<float>(data, data + samples)};
    for (float value : audio.samples)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite vocoder audio output");
    return audio;
#else
    throw std::runtime_error("ONNX Runtime support was not compiled");
#endif
}
} // namespace vocal
