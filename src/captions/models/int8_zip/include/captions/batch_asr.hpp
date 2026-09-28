#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace captions {

// Throughput engine for recorded audio (files, video soundtracks). It runs the
// same streaming Zipformer graphs as StreamingOnnxAsr, but decodes many
// independent segments at once: segments are sorted by length, stacked along
// the encoder's batch axis, and the batches are spread over worker threads that
// each own a private set of ONNX sessions.
struct BatchAsrConfig {
    std::filesystem::path encoder, decoder, joiner, tokens;
    int workers = 0;          // 0 = one per physical-core pair of the machine
    int threads_per_worker = 2;
    int batch = 16;           // segments stacked per encoder call
    bool spin = false;        // let ONNX Runtime worker threads spin between ops
    int beam = 1;             // 1 = greedy; >1 = modified beam search width
};

// Long-form splitting. Speech regions come from the same RMS gate as the
// streaming engine; regions longer than max_seconds are cut at the quietest
// 10 ms frame in [min_seconds, max_seconds] so cuts land in pauses.
struct SegmenterConfig {
    double max_seconds = 20.0;
    double min_seconds = 8.0;
    float gate_rms = 0.0003F;
    int hangover_ms = 1000;  // silence shorter than this stays inside a segment
    int pad_ms = 200;        // audio kept before/after each speech region
};

struct Span { std::size_t begin = 0, end = 0; };  // sample offsets, [begin, end)

struct Word { std::string text; double start = 0, end = 0; };

struct SegmentResult {
    double start = 0, end = 0;  // seconds in the source audio
    std::string text;
    std::vector<Word> words;
};

struct BatchAsrStats {
    double audio_seconds = 0;    // source duration
    double decoded_seconds = 0;  // audio actually sent to the model (after gating)
    double wall_seconds = 0;     // transcribe() wall time, all workers
    double worker_seconds = 0;   // summed per-worker busy time, split below
    double fbank_seconds = 0, encoder_seconds = 0, search_seconds = 0;
    std::uint64_t segments = 0, batches = 0, encoder_calls = 0;
};

std::vector<Span> segment_audio(std::span<const float> audio, const SegmenterConfig& config);

class BatchOnnxAsr {
public:
    explicit BatchOnnxAsr(BatchAsrConfig config);
    ~BatchOnnxAsr();
    BatchOnnxAsr(const BatchOnnxAsr&) = delete;
    BatchOnnxAsr& operator=(const BatchOnnxAsr&) = delete;

    // Independent clips (normalized 16 kHz mono); results keep input order.
    // Each clip is decoded exactly as StreamingOnnxAsr would decode it as one
    // gated segment: zero initial caches, right-context zero flush at the end.
    std::vector<SegmentResult> transcribe(const std::vector<std::span<const float>>& clips);

    // Whole recording: segment_audio() then transcribe(); timestamps are
    // relative to the start of `audio`.
    std::vector<SegmentResult> transcribe_long(std::span<const float> audio,
                                               const SegmenterConfig& segmenter = {});

    [[nodiscard]] const BatchAsrStats& stats() const;
    [[nodiscard]] int workers() const;
    [[nodiscard]] std::string model_type() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace captions
