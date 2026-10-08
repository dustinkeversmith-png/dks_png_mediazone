#pragma once
#include "vocal/control_params.hpp"
#include <filesystem>
#include <memory>
#include <string>

namespace vocal {
class NeuralProsodyPredictor {
public:
    explicit NeuralProsodyPredictor(const std::filesystem::path& path, int threads = 4,
                                   const std::string& affinities = {});
    ~NeuralProsodyPredictor();
    [[nodiscard]] ProsodyControls predict(std::span<const std::int64_t> tokens, std::int64_t speaker,
                                         std::span<const std::int64_t> duration_override = {});
    [[nodiscard]] double inference_ms() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace vocal
