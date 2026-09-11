#include <piper_onnx/piper_voice.hpp>

#include <piper_onnx/phonemizer.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if VA_HAS_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace vocal {
namespace {

[[maybe_unused]] std::vector<std::string> chunks(std::string_view text, std::size_t maximum) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t end = std::min(text.size(), begin + maximum);
        if (end < text.size()) {
            const auto punctuation = text.find_last_of(".!?;:", end);
            const auto space = text.find_last_of(' ', end);
            const auto split = punctuation != std::string_view::npos && punctuation >= begin ? punctuation + 1 : space;
            if (split != std::string_view::npos && split > begin) end = split;
        }
        auto part = std::string(text.substr(begin, end - begin));
        if (!part.empty()) result.push_back(std::move(part));
        begin = end;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    }
    if (result.empty()) result.emplace_back(" ");
    return result;
}

std::unordered_map<std::string, std::string> properties(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open voice properties: " + path.string());
    std::unordered_map<std::string, std::string> values;
    for (std::string line; std::getline(in, line);) {
        const auto equals = line.find('=');
        if (equals != std::string::npos) values[line.substr(0, equals)] = line.substr(equals + 1);
    }
    return values;
}

}  // namespace

struct PiperVoiceSynthesizer::Impl {
    PiperVoiceConfig config;
    CmuPhonemizer phonemizer;
    PiperDiagnostics diagnostics;
    int sample_rate{22'050};
    int num_speakers{1};
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "vocal-piper"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
#endif

    explicit Impl(PiperVoiceConfig supplied)
        : config(std::move(supplied)),
          phonemizer(config.dictionary_path, config.token_map_path) {
        const auto values = properties(config.properties_path);
        if (values.contains("sample_rate")) sample_rate = std::stoi(values.at("sample_rate"));
        if (values.contains("num_speakers")) num_speakers = std::stoi(values.at("num_speakers"));
        if (config.speaker_id < 0 || config.speaker_id >= num_speakers)
            throw std::out_of_range("speaker ID is outside the voice model's range");
#if VA_HAS_ONNX_RUNTIME
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.EnableCpuMemArena();
        options.EnableMemPattern();
        const int threads = config.intra_op_threads > 0 ? config.intra_op_threads :
            static_cast<int>(std::max(1U, std::thread::hardware_concurrency() / 2));
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        session = Ort::Session(environment, config.model_path.c_str(), options);
#else
        throw std::runtime_error("Piper/VITS requires an ONNX Runtime-enabled build");
#endif
    }
};

PiperVoiceSynthesizer::PiperVoiceSynthesizer(PiperVoiceConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
PiperVoiceSynthesizer::~PiperVoiceSynthesizer() = default;
PiperVoiceSynthesizer::PiperVoiceSynthesizer(PiperVoiceSynthesizer&&) noexcept = default;
PiperVoiceSynthesizer& PiperVoiceSynthesizer::operator=(PiperVoiceSynthesizer&&) noexcept = default;

bool PiperVoiceSynthesizer::compiled_with_runtime() noexcept {
#if VA_HAS_ONNX_RUNTIME
    return true;
#else
    return false;
#endif
}

const PiperDiagnostics& PiperVoiceSynthesizer::diagnostics() const noexcept { return impl_->diagnostics; }

Waveform PiperVoiceSynthesizer::synthesize(std::string_view text) {
#if VA_HAS_ONNX_RUNTIME
    impl_->diagnostics = {};
    impl_->diagnostics.speaker_id = impl_->config.speaker_id;
    Waveform waveform{impl_->sample_rate, {}};
    const auto parts = chunks(text, impl_->config.maximum_chunk_characters);
    for (std::size_t part_index = 0; part_index < parts.size(); ++part_index) {
        auto phonemes = impl_->phonemizer.phonemize(parts[part_index]);
        impl_->diagnostics.words += phonemes.words;
        impl_->diagnostics.dictionary_hits += phonemes.dictionary_hits;
        impl_->diagnostics.fallback_words += phonemes.fallback_words;
        impl_->diagnostics.missing_model_symbols += phonemes.missing_model_symbols;
        impl_->diagnostics.phoneme_tokens += phonemes.token_ids.size();
        std::vector<std::int64_t> lengths{static_cast<std::int64_t>(phonemes.token_ids.size())};
        std::vector<float> scales{impl_->config.noise_scale, impl_->config.length_scale, impl_->config.noise_w};
        std::vector<std::int64_t> speaker{impl_->config.speaker_id};
        const std::array<std::int64_t, 2> token_shape{1, lengths[0]};
        const std::array<std::int64_t, 1> vector_shape{1};
        const std::array<std::int64_t, 1> scales_shape{3};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory,
            phonemes.token_ids.data(), phonemes.token_ids.size(),
            token_shape.data(), token_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, lengths.data(), lengths.size(),
            vector_shape.data(), vector_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(memory, scales.data(), scales.size(),
            scales_shape.data(), scales_shape.size()));
        std::vector<const char*> input_names{"input", "input_lengths", "scales"};
        if (impl_->num_speakers > 1) {
            inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, speaker.data(), speaker.size(),
                vector_shape.data(), vector_shape.size()));
            input_names.push_back("sid");
        }
        const char* output_names[] = {"output"};
        const auto started = std::chrono::steady_clock::now();
        auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                                          output_names, 1);
        const auto stopped = std::chrono::steady_clock::now();
        impl_->diagnostics.inference_ms +=
            std::chrono::duration<double, std::milli>(stopped - started).count();
        const auto count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const float* samples = outputs[0].GetTensorData<float>();
        waveform.samples.insert(waveform.samples.end(), samples, samples + count);
        if (part_index + 1 < parts.size()) {
            waveform.samples.insert(waveform.samples.end(),
                static_cast<std::size_t>(impl_->sample_rate * impl_->config.sentence_silence_seconds), 0.0F);
        }
        ++impl_->diagnostics.chunks;
    }
    impl_->diagnostics.audio_seconds = static_cast<double>(waveform.samples.size()) / waveform.sample_rate_hz;
    impl_->diagnostics.real_time_factor = impl_->diagnostics.audio_seconds > 0.0 ?
        (impl_->diagnostics.inference_ms / 1000.0) / impl_->diagnostics.audio_seconds : 0.0;
    return waveform;
#else
    (void)text;
    throw std::runtime_error("Piper/VITS requires an ONNX Runtime-enabled build");
#endif
}

}  // namespace vocal
