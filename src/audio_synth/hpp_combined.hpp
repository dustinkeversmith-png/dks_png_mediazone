// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\app\cli_options.hpp ===
#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vocal::cli {
class Options {
public:
    bool help{};
    Options(int argc, char** argv, std::initializer_list<std::string> allowed) {
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help" || key == "-h") { help = true; continue; }
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
                throw std::invalid_argument("unknown option: " + key);
            if (++i == argc || std::string(argv[i]).rfind("--", 0) == 0)
                throw std::invalid_argument("missing value for " + key);
            if (!values_.emplace(key, argv[i]).second) throw std::invalid_argument("duplicate option: " + key);
        }
    }
    [[nodiscard]] bool has(const std::string& key) const { return values_.contains(key); }
    [[nodiscard]] std::string get(const std::string& key, std::string fallback = {}) const {
        const auto value = values_.find(key);
        return value == values_.end() ? fallback : value->second;
    }
    [[nodiscard]] std::string required(const std::string& key) const {
        const auto value = get(key);
        if (value.empty()) throw std::invalid_argument("required option: " + key);
        return value;
    }
    template<class T> [[nodiscard]] T number(const std::string& key, T fallback) const {
        return has(key) ? parse<T>(get(key)) : fallback;
    }
    template<class T> static T parse(const std::string& value) {
        T result{};
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
            throw std::invalid_argument("invalid numeric value: " + value);
        return result;
    }
private:
    std::map<std::string, std::string> values_;
};

template<class T> std::vector<T> read_values(const std::filesystem::path& path, std::size_t limit) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read control file: " + path.string());
    std::vector<T> values;
    for (std::string line; std::getline(input, line);) {
        line.resize(line.find('#') == std::string::npos ? line.size() : line.find('#'));
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        for (std::string value; row >> value;) {
            if (values.size() == limit) throw std::invalid_argument("control file exceeds value limit");
            values.push_back(Options::parse<T>(value));
        }
    }
    return values;
}

inline std::ofstream report_file(std::filesystem::path output) {
    output.replace_extension(".diagnostics.json");
    std::ofstream report(output);
    if (!report) throw std::runtime_error("cannot write diagnostics: " + output.string());
    return report;
}

template<class T> void array(std::ostream& out, const std::vector<T>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}
} // namespace vocal::cli

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\include\vocal\acoustic_model.hpp ===
#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace vocal {

struct MelSpectrogram {
    std::size_t frames{};
    std::size_t bins{};
    std::vector<float> log_mel;

    [[nodiscard]] std::span<const float> frame(std::size_t index) const;
};

struct SynthesisRequest {
    std::string_view text;
    std::string_view speaker_id;
    int sample_rate_hz{24'000};
    std::size_t mel_bins{80};
};

class AcousticModel {
public:
    virtual ~AcousticModel() = default;
    [[nodiscard]] virtual MelSpectrogram infer(const SynthesisRequest& request) = 0;
};

}  // namespace vocal


// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\include\vocal\audio.hpp ===
#pragma once

#include <filesystem>
#include <span>
#include <vector>

namespace vocal {

struct Waveform {
    int sample_rate_hz{24'000};
    std::vector<float> samples;
};

void normalize_peak(Waveform& audio, float peak = 0.92F);
[[nodiscard]] Waveform resample_waveform(const Waveform& audio, int sample_rate_hz = 24'000);
void write_wav_pcm16(const std::filesystem::path& path, const Waveform& audio);

}  // namespace vocal


// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\include\vocal\control_params.hpp ===
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vocal {

enum class VocalEmotion { Neutral, Whisper, Excited, Somber, Authoritative };
// Optional frontend annotations; do not infer consonant identity from zero F0.
enum class ProsodyTokenKind { Unknown, UnvoicedConsonant, Vowel, StressedVowel, Boundary };

struct ProsodyControls {
    std::vector<std::int64_t> durations; // One frame count per token, including special tokens.
    std::vector<float> f0_contour;       // Hz; zero means unvoiced.
    std::vector<float> energy_contour;   // Normalized nonnegative energy, not dB.
    std::int64_t speaker_id{0};
    std::vector<ProsodyTokenKind> token_kinds; // Empty or one annotation per token.
};

struct ProsodySliders {
    float pitch_scale{1.0F};
    float speed{1.0F};                  // > 1 is faster; durations are divided by speed.
    float energy_scale{1.0F};
    float energy_variance{1.0F};        // Scale deviations from mean energy.
};

inline constexpr std::size_t maximum_prosody_frames = 15'000;
[[nodiscard]] std::size_t duration_frames(std::span<const std::int64_t> durations,
                                         std::size_t maximum = maximum_prosody_frames);
void validate_prosody(const ProsodyControls& controls, std::size_t tokens,
                      std::size_t maximum = maximum_prosody_frames);
[[nodiscard]] ProsodyControls baseline_prosody(std::size_t tokens, std::int64_t frames_per_token = 6,
                                              float f0_hz = 180.0F, float energy = 1.0F);
// Resample contours inside each token when cadence changes; preserve unvoiced zeros.
[[nodiscard]] ProsodyControls apply_prosody_sliders(const ProsodyControls& controls,
                                                   const ProsodySliders& sliders);
[[nodiscard]] ProsodyControls apply_emotion_preset(const ProsodyControls& base, VocalEmotion emotion);
} // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\include\vocal\dataset.hpp ===
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vocal {

struct Utterance {
    std::string id;
    std::string speaker_id;
    std::string chapter_id;
    std::string normalized_text;
    std::filesystem::path audio_path;
};

class LibriTtsDataset {
public:
    explicit LibriTtsDataset(std::filesystem::path root);
    [[nodiscard]] std::vector<Utterance> scan() const;

private:
    std::filesystem::path root_;
};

}  // namespace vocal


// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\include\vocal\evaluator.hpp ===
#pragma once

#include "vocal/metrics.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vocal {

struct EvaluationCase {
    std::string utterance_id;
    std::filesystem::path reference_mcep;
    std::filesystem::path synthesized_mcep;
    std::optional<std::filesystem::path> reference_f0;
    std::optional<std::filesystem::path> synthesized_f0;
    std::optional<std::string> reference_text;
    std::optional<std::string> hypothesis_text;
};

struct EvaluationResult {
    std::string utterance_id;
    double mcd_db{};
    std::optional<metrics::PitchScore> pitch;
    std::optional<double> wer;
};

class Evaluator {
public:
    [[nodiscard]] EvaluationResult evaluate(const EvaluationCase& item) const;
    [[nodiscard]] static metrics::Frames load_matrix(const std::filesystem::path& path);
    [[nodiscard]] static std::vector<double> load_vector(const std::filesystem::path& path);
};

}  // namespace vocal


// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\include\vocal\features.hpp ===
#pragma once

#include "vocal/audio.hpp"
#include "vocal/metrics.hpp"

#include <cstddef>
#include <filesystem>

namespace vocal {

struct FeatureConfig {
    int sample_rate_hz{24'000};
    std::size_t fft_size{1024};
    std::size_t hop_samples{256};
    std::size_t mel_bins{80};
    std::size_t mcep_coefficients{25};
    double minimum_hz{40.0};
    double maximum_hz{12'000.0};
    double log_floor{1e-10};
};

struct AcousticFeatures {
    metrics::Frames log_mel;
    metrics::Frames mcep;
};

[[nodiscard]] Waveform load_wav_mono(const std::filesystem::path& path,
                                     int target_sample_rate_hz = 24'000);
[[nodiscard]] AcousticFeatures extract_features(const Waveform& audio,
                                                const FeatureConfig& config = {});
void write_feature_matrix(const std::filesystem::path& path,
                          const metrics::Frames& matrix);

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\include\vocal\metrics.hpp ===
#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace vocal::metrics {

using Frames = std::vector<std::vector<double>>;

struct PitchScore {
    double f0_rmse_hz{};
    double vuv_error_rate{};
    std::size_t jointly_voiced_frames{};
};

[[nodiscard]] double mcd_db(const Frames& reference, const Frames& synthesized,
                            bool use_dtw = true);
[[nodiscard]] PitchScore pitch(std::span<const double> reference_hz,
                               std::span<const double> synthesized_hz);
[[nodiscard]] double word_error_rate(std::string_view reference,
                                     std::string_view hypothesis);

}  // namespace vocal::metrics


// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\include\vocal\phonemizer.hpp ===
#pragma once

#include "vocal/control_params.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vocal {

struct PhonemizationResult {
    std::vector<std::int64_t> token_ids;
    std::string ipa;
    std::size_t words{};
    std::size_t dictionary_hits{};
    std::size_t fallback_words{};
    std::size_t missing_model_symbols{};
    std::vector<ProsodyTokenKind> token_kinds;
};

class CmuPhonemizer {
public:
    CmuPhonemizer(const std::filesystem::path& dictionary_path,
                  const std::filesystem::path& token_map_path);
    [[nodiscard]] PhonemizationResult phonemize(std::string_view text) const;

private:
    std::unordered_map<std::string, std::vector<std::string>> dictionary_;
    std::unordered_map<std::uint32_t, std::vector<std::int64_t>> token_map_;
};

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\include\vocal\reporting.hpp ===
#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vocal {

struct ScoredUtterance {
    std::string utterance_id;
    std::string speaker_id;
    std::optional<double> mcd_db;
    std::optional<double> f0_rmse_hz;
    std::optional<double> vuv_error;
    std::optional<double> wer;
};

struct ConfidenceInterval {
    std::size_t count{};
    double mean{};
    double lower_95{};
    double upper_95{};
};

struct SpeakerSummary {
    std::string speaker_id;
    std::size_t utterances{};
    std::optional<ConfidenceInterval> mcd_db;
    std::optional<ConfidenceInterval> f0_rmse_hz;
    std::optional<ConfidenceInterval> vuv_error;
    std::optional<ConfidenceInterval> wer;
};

[[nodiscard]] std::vector<ScoredUtterance> load_score_csv(const std::filesystem::path& path);
[[nodiscard]] std::vector<SpeakerSummary> summarize_by_speaker(
    const std::vector<ScoredUtterance>& scores, std::size_t bootstrap_samples = 2000,
    std::uint64_t seed = 0x564f43414cULL);
void write_summaries_csv(const std::filesystem::path& path,
                         const std::vector<SpeakerSummary>& summaries);
void write_summaries_json(const std::filesystem::path& path,
                          const std::vector<SpeakerSummary>& summaries,
                          std::size_t bootstrap_samples, std::uint64_t seed);

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\include\vocal\study.hpp ===
#pragma once

#include <cstdint>
#include <filesystem>

namespace vocal {

struct StudyManifestOptions {
    std::uint64_t seed{0x4d5553485241ULL};
    bool include_mos{true};
    bool include_mushra{true};
};

void write_listening_study(const std::filesystem::path& wav_directory,
                           const std::filesystem::path& json_path,
                           const std::filesystem::path& csv_path,
                           StudyManifestOptions options = {});

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\explicit_neural\ort_contract.hpp ===
#pragma once

#include <piper_onnx/ort_session_options.hpp>

#include <algorithm>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>

namespace vocal::detail {
inline Ort::SessionOptions explicit_session_options(int threads, const std::string& affinities) {
    if (threads <= 0) throw std::invalid_argument("ONNX thread count must be positive");
    auto options = cpu_session_options(threads);
    if (!affinities.empty()) options.AddConfigEntry("session.intra_op_thread_affinities", affinities.c_str());
    return options;
}

// Also checks static dimensions at each call so fixed-size exports fail clearly.
inline void require_input(const Ort::Session& session, const std::string& name,
                          ONNXTensorElementDataType type, std::span<const std::int64_t> shape) {
    Ort::AllocatorWithDefaultOptions allocator;
    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
        const auto input_name = session.GetInputNameAllocated(i, allocator);
        if (name != input_name.get()) continue;
        const auto info = session.GetInputTypeInfo(i);
        if (info.GetONNXType() != ONNX_TYPE_TENSOR)
            throw std::runtime_error("input " + name + " must be a tensor");
        const auto tensor = info.GetTensorTypeAndShapeInfo();
        const auto actual = tensor.GetShape();
        if (tensor.GetElementType() != type || actual.size() != shape.size())
            throw std::runtime_error("input " + name + " has incompatible type or rank");
        for (std::size_t dim = 0; dim < actual.size(); ++dim)
            if (actual[dim] >= 0 && shape[dim] >= 0 && actual[dim] != shape[dim])
                throw std::runtime_error("input " + name + " has incompatible dimensions");
        return;
    }
    throw std::runtime_error("missing required ONNX input: " + name);
}

inline void require_output(const Ort::Session& session, const std::string& name) {
    Ort::AllocatorWithDefaultOptions allocator;
    for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
        const auto output_name = session.GetOutputNameAllocated(i, allocator);
        if (name == output_name.get()) return;
    }
    throw std::runtime_error("missing required ONNX output: " + name);
}

inline std::vector<std::int64_t> float_output_shape(const Ort::Value& value) {
    if (!value.IsTensor()) throw std::runtime_error("ONNX output must be a float32 tensor");
    const auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("ONNX output must be a float32 tensor");
    return info.GetShape();
}
} // namespace vocal::detail

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\explicit_neural\include\explicit_acoustic_gen.hpp ===
#pragma once

#include "vocal/acoustic_model.hpp"
#include "vocal/control_params.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace vocal {
struct ExplicitAcousticConfig {
    int intra_op_threads{4};
    std::string thread_affinities; // ORT format, one affinity group per worker thread.
    std::string mel_output{"mel"};
};

class ExplicitAcousticGenerator {
public:
    explicit ExplicitAcousticGenerator(const std::filesystem::path& model_path,
                                       ExplicitAcousticConfig config = {});
    ~ExplicitAcousticGenerator();
    ExplicitAcousticGenerator(ExplicitAcousticGenerator&&) noexcept;
    ExplicitAcousticGenerator& operator=(ExplicitAcousticGenerator&&) noexcept;
    [[nodiscard]] MelSpectrogram infer(std::span<const std::int64_t> token_ids,
                                       const ProsodyControls& controls);
    [[nodiscard]] double inference_ms() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\explicit_neural\include\length_regulator.hpp ===
#pragma once

#include "vocal/control_params.hpp"

namespace vocal {
struct ExpandedHiddenStates {
    std::size_t frames{};
    std::size_t hidden_dim{};
    std::vector<float> values; // Flattened [1, frames, hidden_dim].
};

[[nodiscard]] ExpandedHiddenStates length_regulate(
    std::span<const float> hidden_states, std::size_t hidden_dim,
    std::span<const std::int64_t> durations,
    std::size_t maximum_frames = maximum_prosody_frames);
} // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\explicit_neural\include\neural_vocoder.hpp ===
#pragma once

#include "vocal/acoustic_model.hpp"
#include "vocal/audio.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace vocal {
struct NeuralVocoderConfig {
    int sample_rate_hz{24'000};
    std::size_t hop_length{256};
    int intra_op_threads{4};
    std::string thread_affinities;
    std::string mel_input{"mel"};
    std::string audio_output{"audio"};
};

class NeuralVocoder {
public:
    explicit NeuralVocoder(const std::filesystem::path& model_path, NeuralVocoderConfig config = {});
    ~NeuralVocoder();
    NeuralVocoder(NeuralVocoder&&) noexcept;
    NeuralVocoder& operator=(NeuralVocoder&&) noexcept;
    [[nodiscard]] Waveform synthesize(const MelSpectrogram& mel);
    [[nodiscard]] double inference_ms() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\piper_onnx\onnx_runtime_model.hpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\piper_onnx\ort_session_options.hpp ===
#pragma once
#include <onnxruntime_cxx_api.h>

namespace vocal::detail {
inline Ort::SessionOptions cpu_session_options(int threads) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetInterOpNumThreads(1);
    if (threads > 0) options.SetIntraOpNumThreads(threads);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    options.EnableCpuMemArena();
    options.EnableMemPattern();
    return options;
}
} // namespace vocal::detail

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\piper_onnx\phonemizer.hpp ===
#pragma once

// Compatibility include; the shared frontend is part of the public framework API.
#include "vocal/phonemizer.hpp"

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\piper_onnx\piper_voice.hpp ===
#pragma once

#include "vocal/audio.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace vocal {

struct PiperVoiceConfig {
    std::filesystem::path model_path;
    std::filesystem::path properties_path;
    std::filesystem::path dictionary_path;
    std::filesystem::path token_map_path;
    std::int64_t speaker_id{0};
    float noise_scale{0.667F};
    float length_scale{1.0F};
    float noise_w{0.8F};
    int intra_op_threads{0};
    std::size_t maximum_chunk_characters{240};
    double sentence_silence_seconds{0.16};
};

struct PiperDiagnostics {
    double inference_ms{};
    double audio_seconds{};
    double real_time_factor{};
    std::size_t chunks{};
    std::size_t phoneme_tokens{};
    std::size_t words{};
    std::size_t dictionary_hits{};
    std::size_t fallback_words{};
    std::size_t missing_model_symbols{};
    std::int64_t speaker_id{};
};

class PiperVoiceSynthesizer {
public:
    explicit PiperVoiceSynthesizer(PiperVoiceConfig config);
    ~PiperVoiceSynthesizer();
    PiperVoiceSynthesizer(PiperVoiceSynthesizer&&) noexcept;
    PiperVoiceSynthesizer& operator=(PiperVoiceSynthesizer&&) noexcept;
    PiperVoiceSynthesizer(const PiperVoiceSynthesizer&) = delete;
    PiperVoiceSynthesizer& operator=(const PiperVoiceSynthesizer&) = delete;

    [[nodiscard]] Waveform synthesize(std::string_view text);
    [[nodiscard]] const PiperDiagnostics& diagnostics() const noexcept;
    [[nodiscard]] static bool compiled_with_runtime() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vocal

