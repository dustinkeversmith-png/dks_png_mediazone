#include <captions/batch_asr.hpp>
#include "ort_graph.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace captions {
namespace {
using Clock = std::chrono::steady_clock;
using detail::Graph;
using detail::nemo_fbank_options;
constexpr int kMel = 80;
constexpr size_t kFrame = 160;  // 10 ms at 16 kHz
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
} // namespace

std::vector<Span> segment_audio(std::span<const float> audio, const SegmenterConfig& c,
                                const std::vector<float>* speech_prob) {
    if (!(c.max_seconds > 0) || c.min_seconds < 0 || c.min_seconds >= c.max_seconds ||
        c.hangover_ms < 0 || c.pad_ms < 0 || !std::isfinite(c.gate_rms) || c.gate_rms < 0 ||
        !(c.vad_threshold > 0 && c.vad_threshold < 1))
        throw std::invalid_argument("Invalid segmenter configuration");
    const size_t frames = (audio.size() + kFrame - 1) / kFrame;
    if (frames == 0) return {};
    // Per 10 ms frame: is it speech, and how good a cut point is it (lower is
    // better). Without a VAD both come from RMS energy; with one, from the
    // speech probability (hysteresis: enter at p >= t, leave below t / 2).
    std::vector<float> energy(frames);
    std::vector<char> voiced(frames);
    for (size_t f = 0; f < frames; ++f) {
        const size_t b = f * kFrame, e = std::min(audio.size(), b + kFrame);
        double power = 0;
        for (size_t i = b; i < e; ++i) power += static_cast<double>(audio[i]) * audio[i];
        energy[f] = static_cast<float>(power / static_cast<double>(e - b));
        voiced[f] = energy[f] > c.gate_rms * c.gate_rms;
    }
    if (speech_prob && !speech_prob->empty()) {
        bool speaking = false;
        for (size_t f = 0; f < frames; ++f) {
            const size_t chunk = std::min(speech_prob->size() - 1, f * kFrame / 512);
            const float p = (*speech_prob)[chunk];
            speaking = speaking ? p >= c.vad_threshold * 0.5F : p >= c.vad_threshold;
            voiced[f] = speaking;
            energy[f] = p;  // cut score: least speech-like point
        }
    }
    const size_t hang = static_cast<size_t>(c.hangover_ms) / 10, pad = static_cast<size_t>(c.pad_ms) / 10;

    // Voiced regions, bridging pauses shorter than the hangover.
    std::vector<std::pair<size_t, size_t>> regions;  // frame [begin, end)
    for (size_t f = 0; f < frames; ++f) {
        if (!voiced[f]) continue;
        if (!regions.empty() && f - regions.back().second <= hang) regions.back().second = f + 1;
        else regions.push_back({f, f + 1});
    }
    std::vector<std::pair<size_t, size_t>> padded;
    for (auto [b, e] : regions) {
        b = b > pad ? b - pad : 0;
        e = std::min(frames, e + pad);
        if (!padded.empty() && b <= padded.back().second) padded.back().second = e;
        else padded.push_back({b, e});
    }

    // Cut long regions at the quietest point (200 ms moving average) in the
    // allowed window, so a cut falls in a pause rather than a stop closure.
    std::vector<double> prefix(frames + 1, 0.0);
    for (size_t f = 0; f < frames; ++f) prefix[f + 1] = prefix[f] + energy[f];
    auto smoothed = [&](size_t f) {
        const size_t b = f > 10 ? f - 10 : 0, e = std::min(frames, f + 10);
        return (prefix[e] - prefix[b]) / static_cast<double>(e - b);
    };
    const size_t max_frames = static_cast<size_t>(c.max_seconds * 100);
    const size_t min_frames = static_cast<size_t>(c.min_seconds * 100);
    std::vector<Span> spans;
    for (auto [b, e] : padded) {
        while (e - b > max_frames) {
            size_t cut = b + max_frames;
            double best = smoothed(cut);
            for (size_t f = b + std::max<size_t>(min_frames, 1); f < b + max_frames; ++f) {
                const double v = smoothed(f);
                if (v < best) { best = v; cut = f; }
            }
            spans.push_back({b * kFrame, cut * kFrame});
            b = cut;
        }
        spans.push_back({b * kFrame, std::min(audio.size(), e * kFrame)});
    }
    return spans;
}

namespace {
struct Job {
    std::span<const float> samples;
    double offset = 0;  // seconds of samples[0] in the source
    SegmentResult* out = nullptr;
};

// One worker's private sessions. ONNX Runtime sessions are thread-safe, but
// sharing one would also share its intra-op pool; private sessions keep every
// worker's matrix kernels on their own threads.
// A CTC model has only the encoder graph; transducers add decoder and joiner.
struct Worker {
    Graph encoder;
    std::unique_ptr<Graph> decoder, joiner;
    Worker(Ort::Env& env, const BatchAsrConfig& c) : encoder(env, c.encoder, c.threads_per_worker, c.spin) {
        if (!c.decoder.empty()) decoder = std::make_unique<Graph>(env, c.decoder, 1);
        if (!c.joiner.empty()) joiner = std::make_unique<Graph>(env, c.joiner, 1);
    }
};

enum class ModelKind { zipformer2, nemo_ctc, nemo_tdt };

// One emitted token: id, encoder output frame, and its posterior probability.
struct Emission { int token = 0, frame = 0; float prob = 1.F; };

// Softmax probability of row[index] over row[0, n).
float posterior(const float* row, size_t n, size_t index) {
    const float m = *std::max_element(row, row + n);
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += std::exp(static_cast<double>(row[i] - m));
    return static_cast<float>(std::exp(static_cast<double>(row[index] - m)) / sum);
}
} // namespace

struct BatchOnnxAsr::Impl {
    BatchAsrConfig config;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "batch-asr"};
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<std::string> tokens;
    std::string type;
    int window = 0, shift = 0, context = 0, vocabulary = 0, unknown = -1, joiner_dim = 0;
    std::vector<int> batch_axis;  // per encoder cache input
    ModelKind kind = ModelKind::zipformer2;
    // NeMo: blank is the last token id; TDT appends duration logits after it.
    int blank = 0, mel_bins = 80, subsampling = 8, pred_hidden = 0;
    std::vector<int> durations;
    HotwordBiaser hotwords;  // domain terms (greedy/TDT decoding)

    void setup_nemo(Worker& w) {
        vocabulary = std::stoi(w.encoder.metadata("vocab_size"));
        subsampling = std::stoi(w.encoder.metadata("subsampling_factor"));
        if (w.encoder.metadata("normalize_type") != "per_feature")
            throw std::runtime_error("Only per_feature-normalized NeMo models are supported");
        mel_bins = static_cast<int>(w.encoder.session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1]);
        blank = vocabulary;
        // tokens.txt lists vocab_size pieces plus <blk>.
        tokens = detail::read_nemo_tokens(config.tokens, vocabulary);
        hotwords = HotwordBiaser(config.hotwords, tokens, config.hotword_boost, config.hotword_start);
        unknown = tokens[0] == "<unk>" ? 0 : -1;
        if (!w.decoder) {
            kind = ModelKind::nemo_ctc;
            if (w.encoder.outputs.size() != 1) throw std::runtime_error("Expected a NeMo CTC graph with one output");
            return;
        }
        if (!w.joiner) throw std::runtime_error("NeMo transducer needs a joiner graph");
        kind = ModelKind::nemo_tdt;
        pred_hidden = std::stoi(w.encoder.metadata("pred_hidden"));
        joiner_dim = static_cast<int>(w.encoder.session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1]);
        const auto out = w.joiner->session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back();
        // RNNT joiners emit vocab+1 logits; TDT adds one logit per duration 0..n-1.
        for (int d = 0; d < static_cast<int>(out) - (vocabulary + 1); ++d) durations.push_back(d);
        if (durations.empty()) durations.push_back(1);  // plain RNNT: always advance one frame
    }
    BatchAsrStats stats;
    std::mutex stats_mutex;

    explicit Impl(BatchAsrConfig c) : config(std::move(c)) {
        if (config.threads_per_worker < 1 || config.threads_per_worker > 16 ||
            config.batch < 0 || config.batch > 256 || config.beam < 1 || config.beam > 64 || config.workers < 0 || config.workers > 64)
            throw std::invalid_argument("Invalid batch ASR configuration");
        if (config.workers == 0) {
            const int hw = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
            config.workers = std::max(1, hw / config.threads_per_worker);
        }
        // Only the first worker's sessions load up front (they carry the graph
        // metadata); the rest load inside their threads the first time a job
        // needs them, so short files never pay for idle workers.
        workers.resize(config.workers);
        workers.front() = std::make_unique<Worker>(env, config);

        auto& w = *workers.front();
        type = w.encoder.metadata("model_type");
        // Offline NeMo encoders see a whole segment per call, so batching only
        // adds padding; the streaming Zipformer2 needs it to fill its GEMMs.
        if (type.rfind("EncDec", 0) == 0) { setup_nemo(w); if (config.batch == 0) config.batch = 1; return; }
        if (config.batch == 0) config.batch = 16;
        if (type != "zipformer2")
            throw std::runtime_error("Unsupported model type '" + type + "' (Zipformer2 or NeMo FastConformer)");
        if (!w.decoder || !w.joiner) throw std::runtime_error("Zipformer2 needs decoder and joiner graphs");
        kind = ModelKind::zipformer2;
        window = std::stoi(w.encoder.metadata("T"));
        shift = std::stoi(w.encoder.metadata("decode_chunk_len"));
        context = std::stoi(w.decoder->metadata("context_size"));
        vocabulary = std::stoi(w.decoder->metadata("vocab_size"));
        if (window < 1 || window > 1000 || shift < 1 || shift > window || context < 1 || context > 32 ||
            vocabulary < 3 || w.encoder.inputs.size() != w.encoder.outputs.size() ||
            w.decoder->inputs.size() != 1 || w.joiner->inputs.size() != 2)
            throw std::runtime_error("Unsupported streaming graph contract");
        joiner_dim = static_cast<int>(w.joiner->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back());
        tokens = detail::read_tokens(config.tokens, vocabulary, unknown);
        hotwords = HotwordBiaser(config.hotwords, tokens, config.hotword_boost, config.hotword_start);
        // Every cache input has exactly one dynamic dimension: the batch axis.
        for (size_t i = 1; i < w.encoder.inputs.size(); ++i) {
            const auto shape = w.encoder.session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            int axis = -1;
            for (size_t d = 0; d < shape.size(); ++d) {
                if (shape[d] == -1) {
                    if (axis != -1) throw std::runtime_error("Cache tensor with two dynamic axes");
                    axis = static_cast<int>(d);
                } else if (shape[d] <= 0) throw std::runtime_error("Unsupported cache shape");
            }
            if (axis == -1) throw std::runtime_error("Cache tensor without a batch axis");
            batch_axis.push_back(axis);
        }
    }

    // Features for one segment, identical to the streaming engine's single
    // gated segment: samples, then right-context zeros, then InputFinished.
    std::vector<float> features(std::span<const float> samples, int& frames) const {
        knf::OnlineFbank fbank(detail::fbank_options());
        if (!samples.empty()) fbank.AcceptWaveform(16000, samples.data(), static_cast<int>(samples.size()));
        std::vector<float> tail(static_cast<size_t>(window) * 160 + 4800, 0.F);
        fbank.AcceptWaveform(16000, tail.data(), static_cast<int>(tail.size()));
        fbank.InputFinished();
        frames = fbank.NumFramesReady();
        std::vector<float> out(static_cast<size_t>(frames) * kMel);
        for (int t = 0; t < frames; ++t) std::copy_n(fbank.GetFrame(t), kMel, out.data() + static_cast<size_t>(t) * kMel);
        return out;
    }

    std::vector<Ort::Value> zero_states(Worker& w, std::int64_t batch) const {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<Ort::Value> states;
        states.reserve(batch_axis.size());
        for (size_t i = 0; i < batch_axis.size(); ++i) {
            auto info = w.encoder.session.GetInputTypeInfo(i + 1).GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            shape[batch_axis[i]] = batch;
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
        return states;
    }

    // Lockstep decode of one batch. Rows whose segment has ended keep running
    // on zero features (their outputs are discarded); jobs are length-sorted,
    // so that padding is small.
    struct Timing { double fbank = 0, encoder = 0, search = 0; };
    std::uint64_t decode_batch(Worker& w, std::span<Job> jobs, Timing& timing) const {
        if (kind != ModelKind::zipformer2) return decode_nemo(w, jobs, timing);
        auto t0 = Clock::now();
        const auto B = static_cast<std::int64_t>(jobs.size());
        const size_t D = static_cast<size_t>(joiner_dim);
        std::vector<std::vector<float>> feats(jobs.size());
        std::vector<int> chunks(jobs.size());
        int max_chunks = 0;
        for (size_t b = 0; b < jobs.size(); ++b) {
            int frames = 0;
            feats[b] = features(jobs[b].samples, frames);
            chunks[b] = frames >= window ? (frames - window) / shift + 1 : 0;
            max_chunks = std::max(max_chunks, chunks[b]);
        }
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        timing.fbank += seconds(t0);
        auto states = zero_states(w, B);

        std::vector<std::vector<std::int64_t>> history(jobs.size());
        std::vector<std::vector<Emission>> emitted(jobs.size());  // (token, output frame)
        // Decoder context: Zipformer2 masks the -1 padding; the last slot is blank.
        for (auto& h : history) { h.assign(context, -1); h.back() = 0; }
        std::vector<float> dec_out(static_cast<size_t>(B) * D);
        std::vector<std::int64_t> dec_in;
        std::vector<size_t> rows;
        auto run_decoder = [&](const std::vector<size_t>& which) {
            dec_in.clear();
            for (size_t b : which) dec_in.insert(dec_in.end(), history[b].end() - context, history[b].end());
            const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(which.size()), context};
            auto in = Ort::Value::CreateTensor<std::int64_t>(memory, dec_in.data(), dec_in.size(), shape.data(), shape.size());
            auto out = w.decoder->run(std::span<const Ort::Value>(&in, 1));
            const float* p = out[0].GetTensorData<float>();
            for (size_t i = 0; i < which.size(); ++i) std::copy_n(p + i * D, D, dec_out.data() + which[i] * D);
        };
        rows.resize(jobs.size());
        std::iota(rows.begin(), rows.end(), size_t{0});
        run_decoder(rows);

        // Beam state (config.beam > 1). Each hypothesis keeps its own decoder
        // output; hypotheses of every active row share one joiner call.
        struct Hyp {
            std::vector<std::int64_t> ys;
            std::vector<Emission> timeline;
            double score = 0;
            std::shared_ptr<const std::vector<float>> dec;
        };
        std::vector<std::vector<Hyp>> beams;
        if (config.beam > 1)
            for (size_t b = 0; b < jobs.size(); ++b)
                beams.push_back({Hyp{history[b], {}, 0.0, std::make_shared<const std::vector<float>>(
                    dec_out.begin() + b * D, dec_out.begin() + (b + 1) * D)}});

        std::vector<float> x(static_cast<size_t>(B) * window * kMel);
        std::vector<float> enc_rows, dec_rows;
        std::vector<size_t> active, fired;
        std::vector<int> hot(jobs.size(), 0);  // hotword prefix-tree state per row (greedy)
        std::vector<float> biased(static_cast<size_t>(vocabulary));
        int out_frames = 0;

        // Modified beam search (icefall): at most one symbol per frame; the
        // top `beam` (hypothesis, token) pairs survive, equal token sequences
        // merge by log-add.
        auto beam_step = [&](const float* encoded, int U, int t, int k) {
            const size_t K = static_cast<size_t>(config.beam);
            size_t R = 0;
            for (size_t b : active) R += beams[b].size();
            enc_rows.resize(R * D);
            dec_rows.resize(R * D);
            size_t r = 0;
            for (size_t b : active)
                for (const auto& h : beams[b]) {
                    std::copy_n(encoded + (b * U + t) * D, D, enc_rows.data() + r * D);
                    std::copy_n(h.dec->data(), D, dec_rows.data() + r * D);
                    ++r;
                }
            const std::array<std::int64_t, 2> join_shape{static_cast<std::int64_t>(R), joiner_dim};
            std::array<Ort::Value, 2> join_in{
                Ort::Value::CreateTensor<float>(memory, enc_rows.data(), enc_rows.size(), join_shape.data(), 2),
                Ort::Value::CreateTensor<float>(memory, dec_rows.data(), dec_rows.size(), join_shape.data(), 2)};
            auto logits = w.joiner->run(join_in);
            if (logits[0].GetTensorTypeAndShapeInfo().GetElementCount() != R * vocabulary)
                throw std::runtime_error("Joiner/token vocabulary mismatch");
            float* values = logits[0].GetTensorMutableData<float>();

            struct Candidate { double score; size_t hyp; int token; float prob; };
            std::vector<Candidate> candidates;
            std::vector<int> order(static_cast<size_t>(vocabulary));
            std::vector<std::pair<size_t, size_t>> pending;  // (row, hyp) needing a decoder run
            r = 0;
            for (size_t b : active) {
                auto& hyps = beams[b];
                candidates.clear();
                for (size_t i = 0; i < hyps.size(); ++i, ++r) {
                    float* row = values + r * vocabulary;
                    if (unknown >= 0) row[unknown] = -std::numeric_limits<float>::infinity();
                    const float m = *std::max_element(row, row + vocabulary);
                    double sum = 0;
                    for (int v = 0; v < vocabulary; ++v) sum += std::exp(static_cast<double>(row[v] - m));
                    const double norm = m + std::log(sum);
                    std::iota(order.begin(), order.end(), 0);
                    const size_t take = std::min(K, order.size());
                    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(take), order.end(),
                                      [&](int a, int c) { return row[a] > row[c]; });
                    for (size_t j = 0; j < take; ++j)
                        candidates.push_back({hyps[i].score + row[order[j]] - norm, i, order[j],
                                              static_cast<float>(std::exp(row[order[j]] - norm))});
                }
                const size_t keep = std::min(K, candidates.size());
                std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep),
                                  candidates.end(), [](const auto& a, const auto& c) { return a.score > c.score; });
                std::vector<Hyp> next;
                for (size_t j = 0; j < keep; ++j) {
                    const auto& c = candidates[j];
                    Hyp h = hyps[c.hyp];
                    h.score = c.score;
                    if (c.token != 0) {
                        h.ys.push_back(c.token);
                        h.timeline.push_back({c.token, k * U + t, c.prob});
                        h.dec.reset();
                    }
                    auto same = std::find_if(next.begin(), next.end(), [&](const Hyp& o) { return o.ys == h.ys; });
                    if (same == next.end()) { next.push_back(std::move(h)); continue; }
                    const double hi = std::max(same->score, h.score), lo = std::min(same->score, h.score);
                    same->score = hi + std::log1p(std::exp(lo - hi));  // candidates arrive best-first
                }
                hyps = std::move(next);
                for (size_t i = 0; i < hyps.size(); ++i) if (!hyps[i].dec) pending.push_back({b, i});
            }
            if (pending.empty()) return;
            dec_in.clear();
            for (auto [b, i] : pending) {
                const auto& ys = beams[b][i].ys;
                dec_in.insert(dec_in.end(), ys.end() - context, ys.end());
            }
            const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(pending.size()), context};
            auto in = Ort::Value::CreateTensor<std::int64_t>(memory, dec_in.data(), dec_in.size(), shape.data(), shape.size());
            auto out = w.decoder->run(std::span<const Ort::Value>(&in, 1));
            const float* p = out[0].GetTensorData<float>();
            for (size_t j = 0; j < pending.size(); ++j)
                beams[pending[j].first][pending[j].second].dec =
                    std::make_shared<const std::vector<float>>(p + j * D, p + (j + 1) * D);
        };
        for (int k = 0; k < max_chunks; ++k) {
            std::fill(x.begin(), x.end(), 0.F);
            active.clear();
            for (size_t b = 0; b < jobs.size(); ++b) {
                if (k >= chunks[b]) continue;
                active.push_back(b);
                std::copy_n(feats[b].data() + static_cast<size_t>(k) * shift * kMel,
                            static_cast<size_t>(window) * kMel, x.data() + b * window * kMel);
            }
            const std::array<std::int64_t, 3> shape{B, window, kMel};
            std::vector<Ort::Value> in;
            in.reserve(1 + states.size());
            in.push_back(Ort::Value::CreateTensor<float>(memory, x.data(), x.size(), shape.data(), shape.size()));
            for (auto& s : states) in.push_back(std::move(s));
            t0 = Clock::now();
            auto out = w.encoder.run(in);
            timing.encoder += seconds(t0);
            t0 = Clock::now();
            for (size_t i = 0; i < states.size(); ++i) states[i] = std::move(out[i + 1]);
            const auto dims = out[0].GetTensorTypeAndShapeInfo().GetShape();
            if (dims.size() != 3 || dims[0] != B || dims[2] != joiner_dim)
                throw std::runtime_error("Unexpected encoder output dimensions");
            const auto U = static_cast<int>(dims[1]);
            out_frames = U;
            const float* encoded = out[0].GetTensorData<float>();

            if (config.beam > 1) {
                for (int t = 0; t < U; ++t) beam_step(encoded, U, t, k);
                timing.search += seconds(t0);
                continue;
            }
            for (int t = 0; t < U; ++t) {
                const auto A = static_cast<std::int64_t>(active.size());
                enc_rows.resize(active.size() * D);
                dec_rows.resize(active.size() * D);
                for (size_t i = 0; i < active.size(); ++i) {
                    std::copy_n(encoded + (active[i] * U + t) * D, D, enc_rows.data() + i * D);
                    std::copy_n(dec_out.data() + active[i] * D, D, dec_rows.data() + i * D);
                }
                const std::array<std::int64_t, 2> join_shape{A, joiner_dim};
                std::array<Ort::Value, 2> join_in{
                    Ort::Value::CreateTensor<float>(memory, enc_rows.data(), enc_rows.size(), join_shape.data(), 2),
                    Ort::Value::CreateTensor<float>(memory, dec_rows.data(), dec_rows.size(), join_shape.data(), 2)};
                auto logits = w.joiner->run(join_in);
                if (logits[0].GetTensorTypeAndShapeInfo().GetElementCount() != active.size() * vocabulary)
                    throw std::runtime_error("Joiner/token vocabulary mismatch");
                const float* values = logits[0].GetTensorData<float>();
                fired.clear();
                for (size_t i = 0; i < active.size(); ++i) {
                    const float* row = values + i * vocabulary;
                    std::copy_n(row, vocabulary, biased.data());
                    hotwords.bias(hot[active[i]], biased.data(), static_cast<size_t>(vocabulary));
                    const auto id = static_cast<int>(std::max_element(biased.begin(), biased.end()) - biased.begin());
                    // Icefall streaming transducers use at most one symbol/frame.
                    if (id == 0 || id == unknown) continue;
                    history[active[i]].push_back(id);
                    emitted[active[i]].push_back({id, k * U + t, posterior(row, static_cast<size_t>(vocabulary), static_cast<size_t>(id))});
                    hot[active[i]] = hotwords.advance(hot[active[i]], id);
                    fired.push_back(active[i]);
                }
                if (!fired.empty()) run_decoder(fired);
            }
            timing.search += seconds(t0);
        }

        // Output frames tile each chunk's shift evenly (40 ms for these exports).
        const double frame_seconds = out_frames > 0 ? shift * 0.01 / out_frames : 0.04;
        for (size_t b = 0; b < beams.size(); ++b)
            emitted[b] = std::max_element(beams[b].begin(), beams[b].end(), [](const Hyp& a, const Hyp& c) {
                return a.score < c.score;
            })->timeline;
        for (size_t b = 0; b < jobs.size(); ++b) assemble(jobs[b], emitted[b], frame_seconds);
        return static_cast<std::uint64_t>(max_chunks);
    }

    // SentencePiece pieces -> words. A piece starting with U+2581 opens a new
    // word; a word's time is the output frame of its first piece.
    void assemble(Job& job, const std::vector<Emission>& emitted, double frame_seconds) const {
        const std::string marker = "\xE2\x96\x81";
        auto& r = *job.out;
        r.start = job.offset;
        r.end = job.offset + job.samples.size() / 16000.0;
        r.words.clear();
        for (const auto& [id, frame, prob] : emitted) {
            std::string piece = tokens[static_cast<size_t>(id)];
            const double at = r.start + frame * frame_seconds;
            const bool starts_word = piece.rfind(marker, 0) == 0;
            if (starts_word) piece.erase(0, marker.size());
            if (starts_word || r.words.empty()) r.words.push_back({piece, at, at + frame_seconds, prob});
            else {
                auto& w = r.words.back();
                w.text += piece;
                w.end = at + frame_seconds;
                w.confidence = std::min(w.confidence, prob);  // a word is as sure as its least sure piece
            }
        }
        std::erase_if(r.words, [](const Word& wd) { return wd.text.empty(); });
        r.text.clear();
        for (const auto& wd : r.words) { if (!r.text.empty()) r.text += ' '; r.text += wd.text; }
    }

    // ---- NeMo FastConformer (offline, full context) -------------------------
    // Normalized features, mel-major ([bins][frames]) as the encoder expects.
    std::vector<float> nemo_features(std::span<const float> samples, int& frames) const {
        knf::OnlineFbank fbank(nemo_fbank_options(mel_bins));
        if (!samples.empty()) fbank.AcceptWaveform(16000, samples.data(), static_cast<int>(samples.size()));
        fbank.InputFinished();
        frames = fbank.NumFramesReady();
        const size_t T = static_cast<size_t>(frames), M = static_cast<size_t>(mel_bins);
        std::vector<float> out(M * T);
        for (size_t t = 0; t < T; ++t) {
            const float* f = fbank.GetFrame(static_cast<int>(t));
            for (size_t m = 0; m < M; ++m) out[m * T + t] = f[m];
        }
        for (size_t m = 0; m < M && T > 0; ++m) {
            float* row = out.data() + m * T;
            double sum = 0, sq = 0;
            for (size_t t = 0; t < T; ++t) sum += row[t];
            const double mean = sum / static_cast<double>(T);
            for (size_t t = 0; t < T; ++t) sq += (row[t] - mean) * (row[t] - mean);
            const double std = std::sqrt(sq / std::max<double>(1.0, static_cast<double>(T) - 1)) + 1e-5;
            for (size_t t = 0; t < T; ++t) row[t] = static_cast<float>((row[t] - mean) / std);
        }
        return out;
    }
    // Encoder output length after three stride-2 convolutions (kernel 3, pad 1).
    int subsampled(int frames) const {
        for (int f = subsampling; f > 1; f /= 2) frames = (frames + 2 - 3) / 2 + 1;
        return frames;
    }

    std::uint64_t decode_nemo(Worker& w, std::span<Job> jobs, Timing& timing) const {
        auto t0 = Clock::now();
        const auto B = static_cast<std::int64_t>(jobs.size());
        const size_t M = static_cast<size_t>(mel_bins);
        std::vector<std::vector<float>> feats(jobs.size());
        std::vector<int> frames(jobs.size());
        int T = 0;
        for (size_t b = 0; b < jobs.size(); ++b) {
            feats[b] = nemo_features(jobs[b].samples, frames[b]);
            T = std::max(T, frames[b]);
        }
        std::vector<float> x(static_cast<size_t>(B) * M * T, 0.F);
        std::vector<std::int64_t> lengths(jobs.size());
        for (size_t b = 0; b < jobs.size(); ++b) {
            lengths[b] = frames[b];
            for (size_t m = 0; m < M; ++m)
                std::copy_n(feats[b].data() + m * frames[b], frames[b], x.data() + (b * M + m) * T);
        }
        timing.fbank += seconds(t0);

        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<std::int64_t, 3> xs{B, static_cast<std::int64_t>(M), T};
        const std::array<std::int64_t, 1> ls{B};
        std::array<Ort::Value, 2> in{
            Ort::Value::CreateTensor<float>(memory, x.data(), x.size(), xs.data(), xs.size()),
            Ort::Value::CreateTensor<std::int64_t>(memory, lengths.data(), lengths.size(), ls.data(), ls.size())};
        t0 = Clock::now();
        auto out = w.encoder.run(in);
        timing.encoder += seconds(t0);
        t0 = Clock::now();
        const double frame_seconds = subsampling * 0.01;
        std::vector<std::vector<Emission>> emitted(jobs.size());

        if (kind == ModelKind::nemo_ctc) {
            // logprobs [B, T', vocab+1]: greedy CTC, collapse repeats, drop blanks.
            const auto dims = out[0].GetTensorTypeAndShapeInfo().GetShape();
            const auto U = dims[1], V = dims[2];
            if (V != vocabulary + 1) throw std::runtime_error("CTC output/token vocabulary mismatch");
            const float* lp = out[0].GetTensorData<float>();
            for (size_t b = 0; b < jobs.size(); ++b) {
                const int valid = std::min<int>(subsampled(frames[b]), static_cast<int>(U));
                int previous = blank;
                for (int t = 0; t < valid; ++t) {
                    const float* row = lp + (b * U + t) * V;
                    const int id = static_cast<int>(std::max_element(row, row + V) - row);
                    if (id != blank && id != previous && id != unknown) emitted[b].push_back({id, t, std::exp(row[id])});
                    previous = id;
                }
            }
        } else {
            decode_tdt(w, out, jobs.size(), emitted, memory);
        }
        timing.search += seconds(t0);
        for (size_t b = 0; b < jobs.size(); ++b) assemble(jobs[b], emitted[b], frame_seconds);
        return 1;
    }

    // Greedy TDT (token-and-duration transducer): each joiner step picks a
    // token and how many encoder frames to advance. Rows advance independently;
    // the joiner runs once per step for every row still inside its utterance.
    void decode_tdt(Worker& w, std::vector<Ort::Value>& enc_out, size_t rows,
                    std::vector<std::vector<Emission>>& emitted, Ort::MemoryInfo& memory) const {
        const auto shape = enc_out[0].GetTensorTypeAndShapeInfo().GetShape();  // [B, D, T']
        const size_t D = static_cast<size_t>(shape[1]), U = static_cast<size_t>(shape[2]);
        const float* enc = enc_out[0].GetTensorData<float>();
        const std::int64_t* enc_len = enc_out[1].GetTensorData<std::int64_t>();
        const size_t H = static_cast<size_t>(pred_hidden);
        const bool tdt = !(durations.size() == 1 && durations[0] == 1);
        constexpr int kMaxSymbolsPerFrame = 10;

        // Prediction network per row (its exported cell state has batch 1).
        std::vector<float> dec_out(rows * H), h(rows * H, 0.F), c(rows * H, 0.F);
        auto predict = [&](size_t b, int token) {
            std::int32_t target = token, target_len = 1;
            const std::array<std::int64_t, 2> ts{1, 1};
            const std::array<std::int64_t, 1> ls{1};
            const std::array<std::int64_t, 3> ss{1, 1, static_cast<std::int64_t>(H)};
            std::array<Ort::Value, 4> in{
                Ort::Value::CreateTensor<std::int32_t>(memory, &target, 1, ts.data(), 2),
                Ort::Value::CreateTensor<std::int32_t>(memory, &target_len, 1, ls.data(), 1),
                Ort::Value::CreateTensor<float>(memory, h.data() + b * H, H, ss.data(), 3),
                Ort::Value::CreateTensor<float>(memory, c.data() + b * H, H, ss.data(), 3)};
            auto out = w.decoder->run(in);
            std::copy_n(out[0].GetTensorData<float>(), H, dec_out.data() + b * H);
            std::copy_n(out[2].GetTensorData<float>(), H, h.data() + b * H);
            std::copy_n(out[3].GetTensorData<float>(), H, c.data() + b * H);
        };
        for (size_t b = 0; b < rows; ++b) predict(b, blank);  // blank doubles as start-of-sequence

        std::vector<size_t> t(rows, 0);
        std::vector<int> hot(rows, 0);  // hotword prefix-tree state per row
        std::vector<float> biased(static_cast<size_t>(vocabulary) + 1);
        std::vector<int> symbols(rows, 0);
        std::vector<size_t> active;
        std::vector<float> enc_rows, dec_rows;
        const size_t V = static_cast<size_t>(vocabulary) + 1;
        for (;;) {
            active.clear();
            for (size_t b = 0; b < rows; ++b)
                if (t[b] < std::min<size_t>(U, static_cast<size_t>(enc_len[b]))) active.push_back(b);
            if (active.empty()) break;
            const auto A = static_cast<std::int64_t>(active.size());
            enc_rows.resize(active.size() * D);
            dec_rows.resize(active.size() * H);
            for (size_t i = 0; i < active.size(); ++i) {
                const size_t b = active[i];
                for (size_t d = 0; d < D; ++d) enc_rows[i * D + d] = enc[(b * D + d) * U + t[b]];
                std::copy_n(dec_out.data() + b * H, H, dec_rows.data() + i * H);
            }
            const std::array<std::int64_t, 3> es{A, static_cast<std::int64_t>(D), 1};
            const std::array<std::int64_t, 3> ds{A, static_cast<std::int64_t>(H), 1};
            std::array<Ort::Value, 2> in{
                Ort::Value::CreateTensor<float>(memory, enc_rows.data(), enc_rows.size(), es.data(), 3),
                Ort::Value::CreateTensor<float>(memory, dec_rows.data(), dec_rows.size(), ds.data(), 3)};
            auto logits = w.joiner->run(in);
            const size_t width = logits[0].GetTensorTypeAndShapeInfo().GetShape().back();
            const float* values = logits[0].GetTensorData<float>();
            for (size_t i = 0; i < active.size(); ++i) {
                const size_t b = active[i];
                const float* row = values + i * width;
                // Domain-term bias applies to the choice only; confidence stays the model's own.
                std::copy_n(row, V, biased.data());
                hotwords.bias(hot[b], biased.data(), V);
                const int token = static_cast<int>(std::max_element(biased.begin(), biased.end()) - biased.begin());
                int advance = token == blank ? 1 : 0;
                if (tdt) {
                    advance = durations[static_cast<size_t>(std::max_element(row + V, row + width) - (row + V))];
                    if (token == blank && advance == 0) advance = 1;
                }
                if (token != blank && token != unknown) {
                    emitted[b].push_back({token, static_cast<int>(t[b]), posterior(row, V, static_cast<size_t>(token))});
                    hot[b] = hotwords.advance(hot[b], token);
                    predict(b, token);
                }
                if (token != blank && ++symbols[b] >= kMaxSymbolsPerFrame) advance = std::max(advance, 1);
                if (advance > 0) { t[b] += static_cast<size_t>(advance); symbols[b] = 0; }
            }
        }
    }

    void run(std::vector<Job>& jobs) {
        const auto started = Clock::now();
        // Longest first: similar lengths share a batch, and the longest
        // batches start early so no worker is left with a straggler.
        std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
            return a.samples.size() > b.samples.size();
        });
        // Round the batch count up to a multiple of the worker count so the
        // last round does not leave workers idle; batches shrink to match.
        const size_t W = workers.size();
        size_t count = (jobs.size() + config.batch - 1) / config.batch;
        count = std::min(jobs.size(), (count + W - 1) / W * W);
        std::vector<std::span<Job>> batches;
        for (size_t i = 0, b = 0; b < count; ++b) {
            const size_t size = (jobs.size() - i) / (count - b) + ((jobs.size() - i) % (count - b) ? 1 : 0);
            batches.emplace_back(jobs.data() + i, size);
            i += size;
        }
        std::atomic<size_t> next{0};
        std::atomic<std::uint64_t> calls{0};
        std::vector<double> busy(workers.size(), 0.0);
        std::vector<Timing> timing(workers.size());
        std::exception_ptr failure;
        std::mutex failure_mutex;
        std::vector<std::thread> threads;
        const size_t used = std::min(workers.size(), std::max<size_t>(batches.size(), 1));
        for (size_t wi = 0; wi < used; ++wi)
            threads.emplace_back([&, wi] {
                try {
                    if (!config.share_sessions && !workers[wi]) workers[wi] = std::make_unique<Worker>(env, config);
                    Worker& sessions = config.share_sessions ? *workers.front() : *workers[wi];
                    for (size_t i; (i = next.fetch_add(1)) < batches.size();) {
                        const auto t0 = Clock::now();
                        calls += decode_batch(sessions, batches[i], timing[wi]);
                        busy[wi] += seconds(t0);
                    }
                } catch (...) {
                    std::lock_guard lock(failure_mutex);
                    failure = std::current_exception();
                    next = batches.size();
                }
            });
        for (auto& t : threads) t.join();
        if (failure) std::rethrow_exception(failure);
        std::lock_guard lock(stats_mutex);
        stats.wall_seconds += seconds(started);
        stats.worker_seconds += std::accumulate(busy.begin(), busy.end(), 0.0);
        stats.encoder_calls += calls;
        for (const auto& t : timing) { stats.fbank_seconds += t.fbank; stats.encoder_seconds += t.encoder; stats.search_seconds += t.search; }
        stats.batches += batches.size();
        stats.segments += jobs.size();
        for (const auto& j : jobs) stats.decoded_seconds += j.samples.size() / 16000.0;
    }
};

BatchOnnxAsr::BatchOnnxAsr(BatchAsrConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
BatchOnnxAsr::~BatchOnnxAsr() = default;

std::vector<SegmentResult> BatchOnnxAsr::transcribe(const std::vector<std::span<const float>>& clips) {
    std::vector<SegmentResult> results(clips.size());
    std::vector<Job> jobs;
    for (size_t i = 0; i < clips.size(); ++i) {
        for (float v : clips[i]) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite audio sample");
        jobs.push_back({clips[i], 0.0, &results[i]});
        impl_->stats.audio_seconds += clips[i].size() / 16000.0;
    }
    impl_->run(jobs);
    return results;
}

std::vector<SegmentResult> BatchOnnxAsr::transcribe_long(std::span<const float> audio,
                                                         const SegmenterConfig& segmenter,
                                                         const std::vector<float>* speech_prob) {
    for (float v : audio) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite audio sample");
    const auto started = Clock::now();
    const auto spans = segment_audio(audio, segmenter, speech_prob);
    std::vector<SegmentResult> results(spans.size());
    std::vector<Job> jobs;
    for (size_t i = 0; i < spans.size(); ++i)
        jobs.push_back({audio.subspan(spans[i].begin, spans[i].end - spans[i].begin),
                        spans[i].begin / 16000.0, &results[i]});
    impl_->stats.audio_seconds += audio.size() / 16000.0;
    const double segment_seconds = seconds(started);
    impl_->run(jobs);
    impl_->stats.wall_seconds += segment_seconds;
    return results;
}

const BatchAsrStats& BatchOnnxAsr::stats() const { return impl_->stats; }
int BatchOnnxAsr::workers() const { return static_cast<int>(impl_->workers.size()); }
int BatchOnnxAsr::batch() const { return impl_->config.batch; }
const HotwordBiaser& BatchOnnxAsr::hotwords() const { return impl_->hotwords; }
std::string BatchOnnxAsr::model_type() const { return impl_->type; }
} // namespace captions
