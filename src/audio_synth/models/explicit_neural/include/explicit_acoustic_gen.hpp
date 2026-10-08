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
