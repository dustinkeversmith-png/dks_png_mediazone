#include "vocal/reporting.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace vocal {
namespace {

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') { field += '"'; ++i; }
            else quoted = !quoted;
        } else if (c == ',' && !quoted) { fields.push_back(field); field.clear(); }
        else field += c;
    }
    fields.push_back(field);
    return fields;
}

std::optional<double> number(const std::string& value) {
    if (value.empty()) return std::nullopt;
    double result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error("invalid numeric score: " + value);
    }
    return result;
}

ConfidenceInterval bootstrap(std::vector<double> values, std::size_t samples, std::uint64_t seed) {
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    ConfidenceInterval result{values.size(), sum / values.size(), 0.0, 0.0};
    if (values.size() == 1 || samples == 0) {
        result.lower_95 = result.upper_95 = result.mean;
        return result;
    }
    std::mt19937_64 random(seed);
    std::uniform_int_distribution<std::size_t> choose(0, values.size() - 1);
    std::vector<double> means(samples);
    for (double& mean : means) {
        double sample_sum = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) sample_sum += values[choose(random)];
        mean = sample_sum / values.size();
    }
    std::sort(means.begin(), means.end());
    result.lower_95 = means[static_cast<std::size_t>(.025 * (means.size() - 1))];
    result.upper_95 = means[static_cast<std::size_t>(.975 * (means.size() - 1))];
    return result;
}

std::optional<ConfidenceInterval> metric(const std::vector<const ScoredUtterance*>& rows,
    const std::function<std::optional<double>(const ScoredUtterance&)>& getter,
    std::size_t samples, std::uint64_t seed) {
    std::vector<double> values;
    for (const auto* row : rows) if (auto value = getter(*row)) values.push_back(*value);
    if (values.empty()) return std::nullopt;
    return bootstrap(std::move(values), samples, seed);
}

std::string escape_json(const std::string& value) {
    std::string result;
    for (char c : value) {
        if (c == '"' || c == '\\') result += '\\';
        result += c;
    }
    return result;
}

void csv_metric(std::ostream& out, const std::optional<ConfidenceInterval>& value) {
    if (value) out << value->mean << ',' << value->lower_95 << ',' << value->upper_95 << ',' << value->count;
    else out << ",,,0";
}

void json_metric(std::ostream& out, std::string_view name,
                 const std::optional<ConfidenceInterval>& value, bool comma) {
    out << "      \"" << name << "\": ";
    if (!value) out << "null";
    else out << "{\"n\": " << value->count << ", \"mean\": " << value->mean
             << ", \"lower_95\": " << value->lower_95 << ", \"upper_95\": " << value->upper_95 << '}';
    out << (comma ? ",\n" : "\n");
}

}  // namespace

std::vector<ScoredUtterance> load_score_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open score CSV: " + path.string());
    std::string header;
    if (!std::getline(in, header)) return {};
    const auto columns = split_csv(header);
    std::map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < columns.size(); ++i) index[columns[i]] = i;
    if (!index.contains("utterance_id") || !index.contains("speaker_id"))
        throw std::runtime_error("score CSV requires utterance_id and speaker_id columns");
    auto field = [&](const std::vector<std::string>& row, std::string_view name) -> std::string {
        const auto found = index.find(std::string(name));
        return found == index.end() || found->second >= row.size() ? "" : row[found->second];
    };
    std::vector<ScoredUtterance> scores;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        const auto row = split_csv(line);
        scores.push_back({field(row, "utterance_id"), field(row, "speaker_id"),
                          number(field(row, "mcd_db")), number(field(row, "f0_rmse_hz")),
                          number(field(row, "vuv_error")), number(field(row, "wer"))});
    }
    return scores;
}

std::vector<SpeakerSummary> summarize_by_speaker(const std::vector<ScoredUtterance>& scores,
                                                  std::size_t samples, std::uint64_t seed) {
    std::map<std::string, std::vector<const ScoredUtterance*>> groups;
    for (const auto& score : scores) { groups[score.speaker_id].push_back(&score); groups["__all__"].push_back(&score); }
    std::vector<SpeakerSummary> summaries;
    for (const auto& [speaker, rows] : groups) {
        auto make = [&](auto member, std::uint64_t salt) {
            return metric(rows, [member](const ScoredUtterance& row) { return row.*member; }, samples, seed ^ salt);
        };
        summaries.push_back({speaker, rows.size(), make(&ScoredUtterance::mcd_db, 1),
            make(&ScoredUtterance::f0_rmse_hz, 2), make(&ScoredUtterance::vuv_error, 3),
            make(&ScoredUtterance::wer, 4)});
    }
    return summaries;
}

void write_summaries_csv(const std::filesystem::path& path, const std::vector<SpeakerSummary>& summaries) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create summary CSV: " + path.string());
    out << "speaker_id,utterances,mcd_mean,mcd_lower95,mcd_upper95,mcd_n,"
           "f0_mean,f0_lower95,f0_upper95,f0_n,vuv_mean,vuv_lower95,vuv_upper95,vuv_n,"
           "wer_mean,wer_lower95,wer_upper95,wer_n\n" << std::setprecision(10);
    for (const auto& row : summaries) {
        out << row.speaker_id << ',' << row.utterances << ','; csv_metric(out, row.mcd_db); out << ',';
        csv_metric(out, row.f0_rmse_hz); out << ','; csv_metric(out, row.vuv_error); out << ',';
        csv_metric(out, row.wer); out << '\n';
    }
}

void write_summaries_json(const std::filesystem::path& path, const std::vector<SpeakerSummary>& summaries,
                          std::size_t samples, std::uint64_t seed) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create summary JSON: " + path.string());
    out << std::setprecision(10) << "{\n  \"bootstrap_samples\": " << samples
        << ",\n  \"seed\": " << seed << ",\n  \"speakers\": [\n";
    for (std::size_t i = 0; i < summaries.size(); ++i) {
        const auto& row = summaries[i];
        out << "    {\n      \"speaker_id\": \"" << escape_json(row.speaker_id)
            << "\",\n      \"utterances\": " << row.utterances << ",\n";
        json_metric(out, "mcd_db", row.mcd_db, true); json_metric(out, "f0_rmse_hz", row.f0_rmse_hz, true);
        json_metric(out, "vuv_error", row.vuv_error, true); json_metric(out, "wer", row.wer, false);
        out << "    }" << (i + 1 == summaries.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
}

}  // namespace vocal
