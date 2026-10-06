#include <captions/streaming_asr.hpp>
#include "ort_graph.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace captions {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
using detail::Graph;
void append_text(std::string& destination, const std::string& source) {
    if (source.empty()) return;
    if (!destination.empty()) destination += ' ';
    destination += source;
}
std::vector<Ort::Value> zero_tensors(Graph& graph, size_t first_input, Ort::AllocatorWithDefaultOptions& allocator,
                                     const std::vector<std::vector<std::int64_t>>& shapes = {}) {
    std::vector<Ort::Value> out;
    for (size_t i = first_input; i < graph.inputs.size(); ++i) {
        auto info = graph.session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto shape = i - first_input < shapes.size() ? shapes[i - first_input] : info.GetShape();
        for (auto& d : shape) if (d == -1) d = 1;
        if (std::any_of(shape.begin(), shape.end(), [](auto d) { return d <= 0; }))
        {
            std::string dims;
            for (auto d : shape) dims += " " + std::to_string(d);
            throw std::runtime_error("Unsupported cache shape for " + graph.inputs[i] + ":" + dims);
        }
        if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            auto v = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
            std::fill_n(v.GetTensorMutableData<float>(), v.GetTensorTypeAndShapeInfo().GetElementCount(), 0.F);
            out.push_back(std::move(v));
        } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            auto v = Ort::Value::CreateTensor<std::int64_t>(allocator, shape.data(), shape.size());
            std::fill_n(v.GetTensorMutableData<std::int64_t>(), v.GetTensorTypeAndShapeInfo().GetElementCount(), 0);
            out.push_back(std::move(v));
        } else throw std::runtime_error("Unsupported cache element type");
    }
    return out;
}
} // namespace

// Two model families share the gate, packetizer and endpointing:
//   zipformer2  icefall streaming Zipformer2 transducer (stateless decoder)
//   nemo        NeMo cache-aware streaming FastConformer, RNN-T head (LSTM
//               prediction network), unnormalized NeMo features
struct StreamingOnnxAsr::Impl {
    StreamingAsrConfig config;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "streaming-asr"};
    Graph encoder, decoder, joiner;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> states;
    std::unique_ptr<knf::OnlineFbank> fbank;
    std::vector<float> features, pending, pre_roll;
    std::vector<std::string> tokens;
    std::vector<int> emitted;  // token ids of the open segment
    std::string committed;
    StreamingAsrStats stats;
    HotwordBiaser hotwords;
    int hot_state = 0;
    int window = 0, shift = 0, vocabulary = 0, frame_offset = 0, unknown = -1, mel = 80;
    int quiet_samples = 0;
    bool active = false, finished = false, nemo = false;
    std::string type;
    // zipformer2: decoder context window and current decoder output.
    int context = 0;
    std::vector<std::int64_t> history;
    Ort::Value decoder_out{nullptr};
    // nemo: blank id, LSTM state and prediction-network output.
    int blank = 0;
    size_t hidden = 0;
    std::vector<float> h, c, pred;
    std::vector<std::vector<std::int64_t>> cache_shapes;

    explicit Impl(StreamingAsrConfig cfg)
        : config(std::move(cfg)), encoder(env, config.encoder, checked_threads(config.threads)),
          decoder(env, config.decoder, 1), joiner(env, config.joiner, 1) {
        if (config.packet_ms < 10 || config.packet_ms > 160 ||
            config.pre_roll_ms < 0 || config.pre_roll_ms > 1000 ||
            config.gate_hangover_ms < 200 || config.gate_hangover_ms > 10000 ||
            !std::isfinite(config.gate_rms) || config.gate_rms < 0)
            throw std::invalid_argument("Invalid streaming packet/gate configuration");
        type = encoder.metadata("model_type");
        if (type.rfind("EncDec", 0) == 0) setup_nemo();
        else if (type == "zipformer2") setup_zipformer();
        else throw std::runtime_error("Unsupported streaming model type '" + type + "'");
        hotwords = HotwordBiaser(config.hotwords, tokens, config.hotword_boost, config.hotword_start);
        features.resize(static_cast<size_t>(window) * mel);
        pending.reserve(static_cast<size_t>(config.packet_ms) * 16);
        pre_roll.reserve(static_cast<size_t>(config.pre_roll_ms + config.packet_ms) * 16);
        new_segment();
    }
    static int checked_threads(int n) {
        if (n < 1 || n > 4) throw std::invalid_argument("ASR threads must be 1..4");
        return n;
    }
    void setup_zipformer() {
        window = std::stoi(encoder.metadata("T"));
        shift = std::stoi(encoder.metadata("decode_chunk_len"));
        context = std::stoi(decoder.metadata("context_size"));
        vocabulary = std::stoi(decoder.metadata("vocab_size"));
        if (window < 1 || window > 1000 || shift < 1 || shift > window || context < 1 || context > 32 ||
            vocabulary < 3 || encoder.inputs.size() != encoder.outputs.size() ||
            decoder.inputs.size() != 1 || joiner.inputs.size() != 2)
            throw std::runtime_error("Unsupported streaming graph contract");
        tokens = detail::read_tokens(config.tokens, vocabulary, unknown);
    }
    void setup_nemo() {
        nemo = true;
        window = std::stoi(encoder.metadata("window_size"));
        shift = std::stoi(encoder.metadata("chunk_shift"));
        vocabulary = std::stoi(encoder.metadata("vocab_size"));
        hidden = static_cast<size_t>(std::stoi(encoder.metadata("pred_hidden")));
        mel = static_cast<int>(encoder.session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1]);
        if (!encoder.metadata("normalize_type").empty())
            throw std::runtime_error("Expected a cache-aware streaming NeMo export (no feature normalization)");
        if (window < 1 || window > 1000 || shift < 1 || shift > window || encoder.inputs.size() != 5 ||
            encoder.outputs.size() != 5 || decoder.inputs.size() != 4 || joiner.inputs.size() != 2)
            throw std::runtime_error("Unsupported NeMo streaming graph contract");
        // Cache shapes come from metadata: [batch, layers, frames, dim] etc.
        auto dims = [&](const char* name) {
            std::vector<std::int64_t> s{1};
            for (int d = 1; d <= 3; ++d) s.push_back(std::stoll(encoder.metadata((std::string(name) + "_dim" + std::to_string(d)).c_str())));
            return s;
        };
        cache_shapes = {dims("cache_last_channel"), dims("cache_last_time"), {1}};
        tokens = detail::read_nemo_tokens(config.tokens, vocabulary);
        blank = vocabulary;
        unknown = tokens[0] == "<unk>" ? 0 : -1;
        h.resize(hidden); c.resize(hidden); pred.resize(hidden);
    }

    void new_segment() {
        fbank = std::make_unique<knf::OnlineFbank>(nemo ? detail::nemo_fbank_options(mel) : detail::fbank_options());
        emitted.clear();
        hot_state = 0;
        frame_offset = 0;
        quiet_samples = 0;
        active = false;
        if (nemo) {
            states = zero_tensors(encoder, 2, allocator, cache_shapes);
            std::fill(h.begin(), h.end(), 0.F);
            std::fill(c.begin(), c.end(), 0.F);
            predict(blank);  // blank doubles as start-of-sequence
            // The first window's leading frames are only left context: give it
            // silence there rather than the start of the speech.
            std::vector<float> lead(static_cast<size_t>(window - shift) * 160, 0.F);
            fbank->AcceptWaveform(16000, lead.data(), static_cast<int>(lead.size()));
        } else {
            states = zero_tensors(encoder, 1, allocator);
            // Decoder context: Zipformer2 masks the -1 padding; the last slot is blank.
            history.assign(context, -1);
            history.back() = 0;
            decoder_out = Ort::Value{nullptr};
        }
    }

    // ---- zipformer2 ----------------------------------------------------------
    void run_decoder() {
        const std::array<std::int64_t, 2> shape{1, context};
        auto value = Ort::Value::CreateTensor<std::int64_t>(memory,
            history.data() + history.size() - context, context, shape.data(), shape.size());
        auto out = decoder.run(std::span<const Ort::Value>(&value, 1));
        decoder_out = std::move(out[0]);
    }
    void zipformer_window() {
        if (!decoder_out) run_decoder();
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
        std::vector<float> logits_copy(static_cast<size_t>(vocabulary));
        for (std::int64_t t = 0; t < dims[1]; ++t) {
            std::array<Ort::Value, 2> join_inputs{
                Ort::Value::CreateTensor<float>(memory, const_cast<float*>(encoded + t * dims[2]),
                    static_cast<size_t>(dims[2]), join_shape.data(), join_shape.size()),
                Ort::Value::CreateTensor<float>(memory, decoder_out.GetTensorMutableData<float>(),
                    static_cast<size_t>(dims[2]), join_shape.data(), join_shape.size())};
            auto logits = joiner.run(join_inputs);
            if (logits[0].GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(vocabulary))
                throw std::runtime_error("Joiner/token vocabulary mismatch");
            std::copy_n(logits[0].GetTensorData<float>(), vocabulary, logits_copy.data());
            hotwords.bias(hot_state, logits_copy.data(), logits_copy.size());
            const auto id = static_cast<int>(std::max_element(logits_copy.begin(), logits_copy.end()) - logits_copy.begin());
            // Icefall streaming transducers use at most one symbol/frame.
            if (id != 0 && id != unknown) {
                history.push_back(id);
                emitted.push_back(id);
                hot_state = hotwords.advance(hot_state, id);
                run_decoder();
            }
        }
    }

    // ---- nemo ---------------------------------------------------------------
    void predict(int token) {
        std::int32_t target = token, target_len = 1;
        const std::array<std::int64_t, 2> ts{1, 1};
        const std::array<std::int64_t, 1> ls{1};
        const std::array<std::int64_t, 3> ss{1, 1, static_cast<std::int64_t>(hidden)};
        std::array<Ort::Value, 4> in{
            Ort::Value::CreateTensor<std::int32_t>(memory, &target, 1, ts.data(), 2),
            Ort::Value::CreateTensor<std::int32_t>(memory, &target_len, 1, ls.data(), 1),
            Ort::Value::CreateTensor<float>(memory, h.data(), hidden, ss.data(), 3),
            Ort::Value::CreateTensor<float>(memory, c.data(), hidden, ss.data(), 3)};
        auto out = decoder.run(in);
        std::copy_n(out[0].GetTensorData<float>(), hidden, pred.data());
        std::copy_n(out[2].GetTensorData<float>(), hidden, h.data());
        std::copy_n(out[3].GetTensorData<float>(), hidden, c.data());
    }
    void nemo_window() {
        // Mel-major [1, mel, window] as the encoder expects.
        for (int t = 0; t < window; ++t) {
            const float* f = fbank->GetFrame(frame_offset + t);
            for (int m = 0; m < mel; ++m) features[static_cast<size_t>(m) * window + t] = f[m];
        }
        const std::array<std::int64_t, 3> xs{1, mel, window};
        std::int64_t length = window;
        const std::array<std::int64_t, 1> ls{1};
        std::vector<Ort::Value> in;
        in.reserve(5);
        in.push_back(Ort::Value::CreateTensor<float>(memory, features.data(), features.size(), xs.data(), xs.size()));
        in.push_back(Ort::Value::CreateTensor<std::int64_t>(memory, &length, 1, ls.data(), 1));
        for (auto& state : states) in.push_back(std::move(state));
        auto started = Clock::now();
        auto out = encoder.run(in);
        if (stats.encoder_ms.size() < 100000) stats.encoder_ms.push_back(seconds(started) * 1000);
        ++stats.encoder_calls;
        for (size_t i = 0; i < states.size(); ++i) states[i] = std::move(out[i + 2]);
        const auto dims = out[0].GetTensorTypeAndShapeInfo().GetShape();  // [1, D, T']
        const size_t D = static_cast<size_t>(dims[1]), U = static_cast<size_t>(dims[2]);
        const size_t valid = std::min<size_t>(U, static_cast<size_t>(out[1].GetTensorData<std::int64_t>()[0]));
        const float* encoded = out[0].GetTensorData<float>();
        std::vector<float> frame(D), logits(static_cast<size_t>(vocabulary) + 1);
        constexpr int kMaxSymbolsPerFrame = 5;
        for (size_t t = 0; t < valid; ++t) {
            for (size_t d = 0; d < D; ++d) frame[d] = encoded[d * U + t];
            for (int symbols = 0; symbols < kMaxSymbolsPerFrame; ++symbols) {
                const std::array<std::int64_t, 3> es{1, static_cast<std::int64_t>(D), 1};
                const std::array<std::int64_t, 3> ps{1, static_cast<std::int64_t>(hidden), 1};
                std::array<Ort::Value, 2> join_in{
                    Ort::Value::CreateTensor<float>(memory, frame.data(), D, es.data(), 3),
                    Ort::Value::CreateTensor<float>(memory, pred.data(), hidden, ps.data(), 3)};
                auto result = joiner.run(join_in);
                if (result[0].GetTensorTypeAndShapeInfo().GetElementCount() != logits.size())
                    throw std::runtime_error("Joiner/token vocabulary mismatch");
                std::copy_n(result[0].GetTensorData<float>(), logits.size(), logits.data());
                hotwords.bias(hot_state, logits.data(), static_cast<size_t>(vocabulary));
                const int id = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
                if (id == blank) break;  // RNN-T: blank advances to the next frame
                if (id != unknown) {
                    emitted.push_back(id);
                    hot_state = hotwords.advance(hot_state, id);
                }
                predict(id);
            }
        }
    }

    void decode_ready() {
        while (fbank->NumFramesReady() - frame_offset >= window) {
            if (nemo) nemo_window(); else zipformer_window();
            frame_offset += shift;
            fbank->Pop(shift);  // retain only overlap, never the whole recording
        }
    }
    void feed(std::span<const float> samples) {
        fbank->AcceptWaveform(16000, samples.data(), static_cast<int>(samples.size()));
        decode_ready();
    }
    std::string current_text() const {
        std::string s;
        for (int id : emitted) s += tokens[static_cast<size_t>(id)];
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
double StreamingOnnxAsr::first_window_ms() const {
    // NeMo's first window starts with silence padding, so speech fills only `shift` frames of it.
    return impl_->nemo ? impl_->shift * 10 + 15.0 : (impl_->window - 1) * 10 + 17.5;
}
std::string StreamingOnnxAsr::model_type() const { return impl_->type; }
const HotwordBiaser& StreamingOnnxAsr::hotwords() const { return impl_->hotwords; }
} // namespace captions
