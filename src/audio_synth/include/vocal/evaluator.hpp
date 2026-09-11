#pragma once

#include "vocal/metrics.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vocal {

struct EvaluationCase {
    std::string utterance_id;
    std::filesystem::path reference_mcep;
    std::filesystem::path synthesized_mcep;
    std::optional<std::filesystem::path> reference_f0;
    std::optional<std::filesystem::path> synthesized_f0;
    std::optional<std::string> reference_text;
    std::optional<std::string> hypothesis_text;
};

struct EvaluationResult {
    std::string utterance_id;
    double mcd_db{};
    std::optional<metrics::PitchScore> pitch;
    std::optional<double> wer;
};

class Evaluator {
public:
    [[nodiscard]] EvaluationResult evaluate(const EvaluationCase& item) const;
    [[nodiscard]] static metrics::Frames load_matrix(const std::filesystem::path& path);
    [[nodiscard]] static std::vector<double> load_vector(const std::filesystem::path& path);
};

}  // namespace vocal

