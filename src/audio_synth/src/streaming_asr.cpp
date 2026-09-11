#include "vocal/streaming_asr.hpp"
#include "ort_session_options.hpp"
#include <kaldi-native-fbank/csrc/online-feature.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vocal {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Graph {
    Ort::Session session;
    std::vector<std::string> inputs, outputs;
    std::vector<const char*> in_names, out_names;
    Graph(Ort::Env& env, const std::filesystem::path& path, int threads)
        : session(env, path.c_str(), detail::cpu_session_options(threads)) {
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session.GetInputCount(); ++i)
            inputs.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        for (size_t i = 0; i < session.GetOutputCount(); ++i)
            outputs.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        for (const auto& s : inputs) in_names.push_back(s.c_str());
        for (const auto& s : outputs) out_names.push_back(s.c_str());
    }
    std::string metadata(const char* key) const {
        Ort::AllocatorWithDefaultOptions allocator;
        auto value = session.GetModelMetadata().LookupCustomMetadataMapAllocated(key, allocator);
        if (!value) throw std::runtime_error(std::string("Missing ONNX metadata: ") + key);
        return value.get();
    }
    std::vector<Ort::Value> run(std::span<const Ort::Value> in) {
        return session.Run(Ort::RunOptions{nullptr}, in_names.data(), in.data(), in.size(),
                           out_names.data(), out_names.size());
    }
};
knf::FbankOptions fbank_options() {
    knf::FbankOptions o;
    o.frame_opts.samp_freq = 16000;
    o.frame_opts.dither = 0;
    o.frame_opts.snip_edges = false;
    o.mel_opts.num_bins = 80;
    o.mel_opts.low_freq = 20;
    o.mel_opts.high_freq = -400;
    return o;
}
void append_text(std::string& destination, const std::string& source) {
    if (source.empty()) return;
    if (!destination.empty()) destination += ' ';
    destination += source;
}
} // namespace

struct StreamingOnnxAsr::Impl {
    StreamingAsrConfig config;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "streaming-asr"};
    Graph encoder, decoder, joiner;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> states;
    Ort::Value decoder_out{nullptr};
    std::unique_ptr<knf::OnlineFbank> fbank;
    std::vector<float> features, pending, pre_roll;
    std::vector<std::int64_t> history;
    std::vector<std::string> tokens;
    std::string committed;
    StreamingAsrStats stats;
    int window = 0, shift = 0, context = 0, vocabulary = 0, frame_offset = 0, unknown = -1;
    int quiet_samples = 0;
    bool active = false, finished = false;
    std::string type;

    explicit Impl(StreamingAsrConfig c)
        : config(std::move(c)), encoder(env, config.encoder, checked_threads(config.threads)),
          decoder(env, config.decoder, 1), joiner(env, config.joiner, 1) {
        if (config.packet_ms < 10 || config.packet_ms > 160 ||
            config.pre_roll_ms < 0 || config.pre_roll_ms > 1000 ||
            config.gate_hangover_ms < 200 || config.gate_hangover_ms > 10000 ||
            !std::isfinite(config.gate_rms) || config.gate_rms < 0)
            throw std::invalid_argument("Invalid streaming packet/gate configuration");
        type = encoder.metadata("model_type");
        if (type != "zipformer" && type != "zipformer2")
            throw std::runtime_error("Expected streaming Zipformer transducer graphs");
        window = std::stoi(encoder.metadata("T"));
        shift = std::stoi(encoder.metadata("decode_chunk_len"));
        context = std::stoi(decoder.metadata("context_size"));
        vocabulary = std::stoi(decoder.metadata("vocab_size"));
        if (window < 1 || window > 1000 || shift < 1 || shift > window || context < 1 || context > 32 ||
            vocabulary < 3 || encoder.inputs.size() != encoder.outputs.size() ||
            decoder.inputs.size() != 1 || joiner.inputs.size() != 2)
            throw std::runtime_error("Unsupported streaming graph contract");
        tokens.resize(vocabulary);
        std::ifstream file(config.tokens);
        if (!file) throw std::runtime_error("Cannot open ASR tokens");
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const auto at = line.find_last_of(" \t");
            if (at == std::string::npos) continue;
            const int id = std::stoi(line.substr(at + 1));
            if (id < 0 || id >= vocabulary) continue; // k2 disambiguation symbols
            tokens[id] = line.substr(0, at);
            if (tokens[id] == "<unk>") unknown = id;
        }
        if (tokens[0] != "<blk>" || std::any_of(tokens.begin(), tokens.end(),
                                                [](const auto& t) { return t.empty(); }))
            throw std::runtime_error("Token vocabulary does not match model");
        features.resize(static_cast<size_t>(window) * 80);
        pending.reserve(static_cast<size_t>(config.packet_ms) * 16);
        pre_roll.reserve(static_cast<size_t>(config.pre_roll_ms + config.packet_ms) * 16);
        new_segment();
    }
    static int checked_threads(int n) {
        if (n < 1 || n > 4) throw std::invalid_argument("ASR threads must be 1..4");
        return n;
    }
    void new_segment() {
        fbank = std::make_unique<knf::OnlineFbank>(fbank_options());
        states.clear();
        states.reserve(encoder.inputs.size() - 1);
        for (size_t i = 1; i < encoder.inputs.size(); ++i) {
            auto type_info = encoder.session.GetInputTypeInfo(i);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            for (auto& d : shape) { if (d == -1) d = 1; }
            if (std::any_of(shape.begin(), shape.end(), [](auto d) { return d <= 0; }))
                throw std::runtime_error("Unsupported dynamic cache shape");
            if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                auto v = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
                std::fill_n(v.GetTensorMutableData<float>(), v.GetTensorTypeAndShapeInfo().GetElementCount(), 0.F);
                states.push_back(std::move(v));
            } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                auto v = Ort::Value::CreateTensor<std::int64_t>(allocator, shape.data(), shape.size());
                std::fill_n(v.GetTensorMutableData<std::int64_t>(), v.GetTensorTypeAndShapeInfo().GetElementCount(), 0);
                states.push_back(std::move(v));
            } else throw std::runtime_error("Unsupported cache element type");
        }
        // The original Zipformer export embeds y directly (Gather): -1 would
        // select the last vocabulary entry. Zipformer2 masks negative padding.
        history.assign(context, type == "zipformer" ? 0 : -1);
        history.back() = 0;
        decoder_out = Ort::Value{nullptr};
        frame_offset = 0;
        quiet_samples = 0;
        active = false;
    }
    void run_decoder() {
        const std::array<std::int64_t, 2> shape{1, context};
        auto value = Ort::Value::CreateTensor<std::int64_t>(memory,
            history.data() + history.size() - context, context, shape.data(), shape.size());
        auto out = decoder.run(std::span<const Ort::Value>(&value, 1));
        decoder_out = std::move(out[0]);
    }
    void decode_ready() {
        if (!decoder_out) run_decoder();
        while (fbank->NumFramesReady() - frame_offset >= window) {
            for (int t = 0; t < window; ++t)
                std::copy_n(fbank->GetFrame(frame_offset + t), 80, features.data() + t * 80);
            const std::array<std::int64_t, 3> shape{1, window, 80};
            std::vector<Ort::Value> in;
            in.reserve(1 + states.size());
            in.push_back(Ort::Value::CreateTensor<float>(memory, features.data(), features.size(), shape.data(), shape.size()));
            for (auto& state : states) in.push_back(std::move(state));
            auto started = Clock::now();
            auto out = encoder.run(in);
            if (stats.encoder_ms.size() < 100000) stats.encoder_ms.push_back(seconds(started) * 1000);
            ++stats.encoder_calls;
            for (size_t i = 0; i < states.size(); ++i) states[i] = std::move(out[i + 1]);
            const auto dims = out[0].GetTensorTypeAndShapeInfo().GetShape();
            if (dims.size() != 3 || dims[0] != 1 || dims[2] != 512)
                throw std::runtime_error("Unexpected encoder output dimensions");
            const float* encoded = out[0].GetTensorData<float>();
            const std::array<std::int64_t, 2> join_shape{1, dims[2]};
            for (std::int64_t t = 0; t < dims[1]; ++t) {
                std::array<Ort::Value, 2> join_inputs{
                    Ort::Value::CreateTensor<float>(memory, const_cast<float*>(encoded + t * dims[2]),
                        static_cast<size_t>(dims[2]), join_shape.data(), join_shape.size()),
                    Ort::Value::CreateTensor<float>(memory, decoder_out.GetTensorMutableData<float>(),
                        static_cast<size_t>(dims[2]), join_shape.data(), join_shape.size())};
                auto logits = joiner.run(join_inputs);
                if (logits[0].GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(vocabulary))
                    throw std::runtime_error("Joiner/token vocabulary mismatch");
                const float* values = logits[0].GetTensorData<float>();
                const auto id = std::max_element(values, values + vocabulary) - values;
                // Icefall streaming transducers use at most one symbol/frame.
                if (id != 0 && id != unknown) { history.push_back(id); run_decoder(); }
            }
            frame_offset += shift;
            fbank->Pop(shift); // retain only overlap, never the whole recording
        }
    }
    void feed(std::span<const float> samples) {
        fbank->AcceptWaveform(16000, samples.data(), static_cast<int>(samples.size()));
        decode_ready();
    }
    std::string current_text() const {
        std::string s;
        for (size_t i = static_cast<size_t>(context); i < history.size(); ++i) s += tokens[history[i]];
        const std::string marker = "\xE2\x96\x81";
        size_t p = 0;
        while ((p = s.find(marker, p)) != std::string::npos) { s.replace(p, marker.size(), " "); ++p; }
        const auto begin = s.find_first_not_of(' '), end = s.find_last_not_of(' ');
        return begin == std::string::npos ? "" : s.substr(begin, end - begin + 1);
    }
    void close_segment() {
        if (!active) return;
        // The export needs right context to flush final speech frames. Padding
        // is synthesized here and included in compute time, never audio duration.
        std::vector<float> tail(static_cast<size_t>(window) * 160 + 4800, 0.F);
        feed(tail);
        fbank->InputFinished();
        decode_ready();
        append_text(committed, current_text());
        new_segment();
    }
    void packet(std::span<const float> samples) {
        const auto started = Clock::now();
        double power = 0;
        for (float value : samples) {
            if (!std::isfinite(value)) throw std::invalid_argument("Non-finite audio sample");
            power += static_cast<double>(value) * value;
        }
        const bool voiced = !config.energy_gate || power > samples.size() * config.gate_rms * config.gate_rms;
        if (!active && !voiced) {
            stats.gated_samples += samples.size();
            pre_roll.insert(pre_roll.end(), samples.begin(), samples.end());
            const size_t cap = static_cast<size_t>(config.pre_roll_ms) * 16;
            if (pre_roll.size() > cap) pre_roll.erase(pre_roll.begin(), pre_roll.end() - cap);
        } else {
            if (!active) { active = true; if (!pre_roll.empty()) feed(pre_roll); pre_roll.clear(); }
            feed(samples);
            quiet_samples = voiced ? 0 : quiet_samples + static_cast<int>(samples.size());
            if (config.energy_gate && quiet_samples >= config.gate_hangover_ms * 16) close_segment();
        }
        if (stats.packet_ms.size() < 100000) stats.packet_ms.push_back(seconds(started) * 1000);
    }
};

StreamingOnnxAsr::StreamingOnnxAsr(StreamingAsrConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
StreamingOnnxAsr::~StreamingOnnxAsr() = default;
void StreamingOnnxAsr::accept(std::span<const float> samples) {
    auto& s = *impl_;
    if (s.finished) throw std::logic_error("accept() after finish(); call reset()");
    const auto started = Clock::now();
    s.stats.input_samples += samples.size();
    const size_t packet = static_cast<size_t>(s.config.packet_ms) * 16;
    while (!samples.empty()) {
        const size_t n = std::min(packet - s.pending.size(), samples.size());
        s.pending.insert(s.pending.end(), samples.begin(), samples.begin() + n);
        samples = samples.subspan(n);
        if (s.pending.size() == packet) { s.packet(s.pending); s.pending.clear(); }
    }
    s.stats.compute_seconds += seconds(started);
}
void StreamingOnnxAsr::finish() {
    auto& s = *impl_;
    if (s.finished) return;
    const auto started = Clock::now();
    if (!s.pending.empty()) { s.packet(s.pending); s.pending.clear(); }
    s.close_segment();
    s.pre_roll.clear();
    s.finished = true;
    s.stats.compute_seconds += seconds(started);
}
void StreamingOnnxAsr::reset() {
    auto& s = *impl_;
    s.new_segment(); s.committed.clear(); s.pending.clear(); s.pre_roll.clear();
    s.finished = false; s.stats = {};
}
std::string StreamingOnnxAsr::text() const {
    std::string result = impl_->committed;
    append_text(result, impl_->current_text());
    return result;
}
const StreamingAsrStats& StreamingOnnxAsr::stats() const { return impl_->stats; }
int StreamingOnnxAsr::model_chunk_ms() const { return impl_->shift * 10; }
double StreamingOnnxAsr::first_window_ms() const { return (impl_->window - 1) * 10 + 17.5; }
std::string StreamingOnnxAsr::model_type() const { return impl_->type; }
} // namespace vocal
