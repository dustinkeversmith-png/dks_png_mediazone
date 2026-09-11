#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vocal {
struct StreamingAsrConfig {
    std::filesystem::path encoder, decoder, joiner, tokens;
    int threads = 2;
    int packet_ms = 100;
    bool energy_gate = true;
    float gate_rms = 0.0003F;
    int gate_hangover_ms = 1000;
    int pre_roll_ms = 200;
};

struct StreamingAsrStats {
    std::uint64_t input_samples = 0, gated_samples = 0, encoder_calls = 0;
    double compute_seconds = 0;
    std::vector<double> encoder_ms;
    std::vector<double> packet_ms;
};

// One stream, single caller. Sessions survive reset(); acoustic caches do not.
// All samples are normalized float32, mono, 16 kHz. No whole-file lookahead.
class StreamingOnnxAsr {
public:
    explicit StreamingOnnxAsr(StreamingAsrConfig config);
    ~StreamingOnnxAsr();
    StreamingOnnxAsr(const StreamingOnnxAsr&) = delete;
    StreamingOnnxAsr& operator=(const StreamingOnnxAsr&) = delete;
    void accept(std::span<const float> samples);
    void finish();
    void reset();
    [[nodiscard]] std::string text() const;
    [[nodiscard]] const StreamingAsrStats& stats() const;
    [[nodiscard]] int model_chunk_ms() const;
    [[nodiscard]] double first_window_ms() const;
    [[nodiscard]] std::string model_type() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace vocal
