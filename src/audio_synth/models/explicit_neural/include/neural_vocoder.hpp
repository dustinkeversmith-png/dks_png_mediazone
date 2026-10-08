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
