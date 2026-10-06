#pragma once

#include <captions/hotwords.hpp>
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
    int workers = 0;          // 0 = hardware threads / threads_per_worker
    int threads_per_worker = 2;
    int batch = 0;            // segments per encoder call; 0 = per model (Zipformer2 16, NeMo 1)
    bool spin = false;        // let ONNX Runtime worker threads spin between ops
    int beam = 4;             // modified beam search width; 1 = greedy
    // All workers call one set of sessions concurrently (load once). Use with
    // threads_per_worker = 1 so each Run executes on its calling thread.
    bool share_sessions = false;
    // Domain terms favoured during greedy/TDT decoding (not Zipformer2 beam search).
    std::vector<HotwordPhrase> hotwords;
    float hotword_boost = 2.0F;
    float hotword_start = 0.25F;  // fraction of the boost given to a phrase's first token
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
    // Speech probability that opens a region (with a VAD). Deliberately low:
    // Silero scores sung or music-backed vocals around 0.05-0.3, and stretches
    // below 0.02 are reliably silence or steady noise.
    float vad_threshold = 0.02F;
};

struct Span { std::size_t begin = 0, end = 0; };  // sample offsets, [begin, end)

// confidence: lowest posterior probability among the word's tokens (0..1).
struct Word { std::string text; double start = 0, end = 0; float confidence = 1.F; };

struct SegmentResult {
    float speech_probability = 1.F;  // mean VAD probability over the segment (1 without a VAD)
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
    double vad_seconds = 0;      // voice activity detection (CaptionEngine), included in wall_seconds
    std::uint64_t segments = 0, batches = 0, encoder_calls = 0;
};

// speech_prob: optional per-512-sample speech probabilities (SileroVad); when
// given they replace the RMS gate for both speech detection and cut points.
std::vector<Span> segment_audio(std::span<const float> audio, const SegmenterConfig& config,
                                const std::vector<float>* speech_prob = nullptr);

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
                                               const SegmenterConfig& segmenter = {},
                                               const std::vector<float>* speech_prob = nullptr);

    [[nodiscard]] const BatchAsrStats& stats() const;
    [[nodiscard]] int workers() const;
    [[nodiscard]] int batch() const;  // resolved segments per encoder call
    [[nodiscard]] const HotwordBiaser& hotwords() const;
    [[nodiscard]] std::string model_type() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace captions
