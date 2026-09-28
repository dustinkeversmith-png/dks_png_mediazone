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
constexpr int kMel = 80;
constexpr size_t kFrame = 160;  // 10 ms at 16 kHz
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
} // namespace

std::vector<Span> segment_audio(std::span<const float> audio, const SegmenterConfig& c) {
    if (!(c.max_seconds > 0) || c.min_seconds < 0 || c.min_seconds >= c.max_seconds ||
        c.hangover_ms < 0 || c.pad_ms < 0 || !std::isfinite(c.gate_rms) || c.gate_rms < 0)
        throw std::invalid_argument("Invalid segmenter configuration");
    const size_t frames = (audio.size() + kFrame - 1) / kFrame;
    if (frames == 0) return {};
    std::vector<float> energy(frames);
    for (size_t f = 0; f < frames; ++f) {
        const size_t b = f * kFrame, e = std::min(audio.size(), b + kFrame);
        double power = 0;
        for (size_t i = b; i < e; ++i) power += static_cast<double>(audio[i]) * audio[i];
        energy[f] = static_cast<float>(power / static_cast<double>(e - b));
    }
    const float threshold = c.gate_rms * c.gate_rms;
    const size_t hang = static_cast<size_t>(c.hangover_ms) / 10, pad = static_cast<size_t>(c.pad_ms) / 10;

    // Voiced regions, bridging pauses shorter than the hangover.
    std::vector<std::pair<size_t, size_t>> regions;  // frame [begin, end)
    for (size_t f = 0; f < frames; ++f) {
        if (energy[f] <= threshold) continue;
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
struct Worker {
    Graph encoder, decoder, joiner;
    Worker(Ort::Env& env, const BatchAsrConfig& c)
        : encoder(env, c.encoder, c.threads_per_worker, c.spin), decoder(env, c.decoder, 1),
          joiner(env, c.joiner, 1) {}
};
} // namespace

struct BatchOnnxAsr::Impl {
    BatchAsrConfig config;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "batch-asr"};
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<std::string> tokens;
    std::string type;
    int window = 0, shift = 0, context = 0, vocabulary = 0, unknown = -1, joiner_dim = 0;
    std::vector<int> batch_axis;  // per encoder cache input
    BatchAsrStats stats;
    std::mutex stats_mutex;

    explicit Impl(BatchAsrConfig c) : config(std::move(c)) {
        if (config.threads_per_worker < 1 || config.threads_per_worker > 16 ||
            config.batch < 1 || config.batch > 256 || config.beam < 1 || config.beam > 64 || config.workers < 0 || config.workers > 64)
            throw std::invalid_argument("Invalid batch ASR configuration");
        if (config.workers == 0) {
            const int hw = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
            config.workers = std::max(1, hw / config.threads_per_worker);
        }
        workers.resize(config.workers);
        std::vector<std::thread> loaders;
        std::exception_ptr failure;
        std::mutex failure_mutex;
        for (int i = 0; i < config.workers; ++i)
            loaders.emplace_back([&, i] {
                try { workers[i] = std::make_unique<Worker>(env, config); }
                catch (...) { std::lock_guard lock(failure_mutex); failure = std::current_exception(); }
            });
        for (auto& t : loaders) t.join();
        if (failure) std::rethrow_exception(failure);

        auto& w = *workers.front();
        type = w.encoder.metadata("model_type");
        if (type != "zipformer" && type != "zipformer2")
            throw std::runtime_error("Expected streaming Zipformer transducer graphs");
        window = std::stoi(w.encoder.metadata("T"));
        shift = std::stoi(w.encoder.metadata("decode_chunk_len"));
        context = std::stoi(w.decoder.metadata("context_size"));
        vocabulary = std::stoi(w.decoder.metadata("vocab_size"));
        if (window < 1 || window > 1000 || shift < 1 || shift > window || context < 1 || context > 32 ||
            vocabulary < 3 || w.encoder.inputs.size() != w.encoder.outputs.size() ||
            w.decoder.inputs.size() != 1 || w.joiner.inputs.size() != 2)
            throw std::runtime_error("Unsupported streaming graph contract");
        joiner_dim = static_cast<int>(w.joiner.session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back());
        tokens = detail::read_tokens(config.tokens, vocabulary, unknown);
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

        // Decoder context per row; the original Zipformer export embeds y with
        // Gather, so it pads with blank (0); Zipformer2 masks -1 padding.
        std::vector<std::vector<std::int64_t>> history(jobs.size());
        std::vector<std::vector<std::pair<int, int>>> emitted(jobs.size());  // (token, output frame)
        for (auto& h : history) { h.assign(context, type == "zipformer" ? 0 : -1); h.back() = 0; }
        std::vector<float> dec_out(static_cast<size_t>(B) * D);
        std::vector<std::int64_t> dec_in;
        std::vector<size_t> rows;
        auto run_decoder = [&](const std::vector<size_t>& which) {
            dec_in.clear();
            for (size_t b : which) dec_in.insert(dec_in.end(), history[b].end() - context, history[b].end());
            const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(which.size()), context};
            auto in = Ort::Value::CreateTensor<std::int64_t>(memory, dec_in.data(), dec_in.size(), shape.data(), shape.size());
            auto out = w.decoder.run(std::span<const Ort::Value>(&in, 1));
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
            std::vector<std::pair<int, int>> timeline;
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
            auto logits = w.joiner.run(join_in);
            if (logits[0].GetTensorTypeAndShapeInfo().GetElementCount() != R * vocabulary)
                throw std::runtime_error("Joiner/token vocabulary mismatch");
            float* values = logits[0].GetTensorMutableData<float>();

            struct Candidate { double score; size_t hyp; int token; };
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
                        candidates.push_back({hyps[i].score + row[order[j]] - norm, i, order[j]});
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
                        h.timeline.push_back({c.token, k * U + t});
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
            auto out = w.decoder.run(std::span<const Ort::Value>(&in, 1));
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
                auto logits = w.joiner.run(join_in);
                if (logits[0].GetTensorTypeAndShapeInfo().GetElementCount() != active.size() * vocabulary)
                    throw std::runtime_error("Joiner/token vocabulary mismatch");
                const float* values = logits[0].GetTensorData<float>();
                fired.clear();
                for (size_t i = 0; i < active.size(); ++i) {
                    const float* row = values + i * vocabulary;
                    const auto id = static_cast<int>(std::max_element(row, row + vocabulary) - row);
                    // Icefall streaming transducers use at most one symbol/frame.
                    if (id == 0 || id == unknown) continue;
                    history[active[i]].push_back(id);
                    emitted[active[i]].push_back({id, k * U + t});
                    fired.push_back(active[i]);
                }
                if (!fired.empty()) run_decoder(fired);
            }
            timing.search += seconds(t0);
        }

        // Output frames tile each chunk's shift evenly (40 ms for these exports).
        const double frame_seconds = out_frames > 0 ? shift * 0.01 / out_frames : 0.04;
        const std::string marker = "\xE2\x96\x81";
        for (size_t b = 0; b < beams.size(); ++b)
            emitted[b] = std::max_element(beams[b].begin(), beams[b].end(), [](const Hyp& a, const Hyp& c) {
                return a.score < c.score;
            })->timeline;
        for (size_t b = 0; b < jobs.size(); ++b) {
            auto& r = *jobs[b].out;
            r.start = jobs[b].offset;
            r.end = jobs[b].offset + jobs[b].samples.size() / 16000.0;
            r.words.clear();
            for (auto [id, frame] : emitted[b]) {
                std::string piece = tokens[static_cast<size_t>(id)];
                const double at = r.start + frame * frame_seconds;
                const bool starts_word = piece.rfind(marker, 0) == 0;
                if (starts_word) piece.erase(0, marker.size());
                if (starts_word || r.words.empty()) r.words.push_back({piece, at, at + frame_seconds});
                else { r.words.back().text += piece; r.words.back().end = at + frame_seconds; }
            }
            std::erase_if(r.words, [](const Word& wd) { return wd.text.empty(); });
            r.text.clear();
            for (const auto& wd : r.words) { if (!r.text.empty()) r.text += ' '; r.text += wd.text; }
        }
        return static_cast<std::uint64_t>(max_chunks);
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
                    for (size_t i; (i = next.fetch_add(1)) < batches.size();) {
                        const auto t0 = Clock::now();
                        calls += decode_batch(*workers[wi], batches[i], timing[wi]);
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
                                                         const SegmenterConfig& segmenter) {
    for (float v : audio) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite audio sample");
    const auto started = Clock::now();
    const auto spans = segment_audio(audio, segmenter);
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
std::string BatchOnnxAsr::model_type() const { return impl_->type; }
} // namespace captions
