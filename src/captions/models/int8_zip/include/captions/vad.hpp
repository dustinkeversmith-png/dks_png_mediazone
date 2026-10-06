#pragma once
// Neural voice activity detection (Silero VAD v5, ONNX, 2 MB). Replaces the
// RMS gate for segmentation: steady noise, music and room tone are rejected
// instead of being decoded (wasted time, invented words), and segment cuts
// land where speech is least likely rather than where the signal is quietest.

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace captions {

class SileroVad {
public:
    static constexpr int kChunk = 512;  // samples per probability (32 ms at 16 kHz)
    explicit SileroVad(const std::filesystem::path& model, int threads = 4);
    ~SileroVad();
    SileroVad(const SileroVad&) = delete;
    SileroVad& operator=(const SileroVad&) = delete;

    // Speech probability per 32 ms chunk of 16 kHz mono audio. The recording
    // is cut into stripes decoded together along the batch axis (each with
    // 0.5 s of warm-up), so an hour takes a few hundred model calls.
    std::vector<float> probabilities(std::span<const float> audio);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace captions
