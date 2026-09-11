#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vocal {

struct ScoredUtterance {
    std::string utterance_id;
    std::string speaker_id;
    std::optional<double> mcd_db;
    std::optional<double> f0_rmse_hz;
    std::optional<double> vuv_error;
    std::optional<double> wer;
};

struct ConfidenceInterval {
    std::size_t count{};
    double mean{};
    double lower_95{};
    double upper_95{};
};

struct SpeakerSummary {
    std::string speaker_id;
    std::size_t utterances{};
    std::optional<ConfidenceInterval> mcd_db;
    std::optional<ConfidenceInterval> f0_rmse_hz;
    std::optional<ConfidenceInterval> vuv_error;
    std::optional<ConfidenceInterval> wer;
};

[[nodiscard]] std::vector<ScoredUtterance> load_score_csv(const std::filesystem::path& path);
[[nodiscard]] std::vector<SpeakerSummary> summarize_by_speaker(
    const std::vector<ScoredUtterance>& scores, std::size_t bootstrap_samples = 2000,
    std::uint64_t seed = 0x564f43414cULL);
void write_summaries_csv(const std::filesystem::path& path,
                         const std::vector<SpeakerSummary>& summaries);
void write_summaries_json(const std::filesystem::path& path,
                          const std::vector<SpeakerSummary>& summaries,
                          std::size_t bootstrap_samples, std::uint64_t seed);

}  // namespace vocal
