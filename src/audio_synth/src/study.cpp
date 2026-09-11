#include "vocal/study.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace vocal {
namespace {

struct Stimulus { std::string utterance; std::string system; std::filesystem::path path; };

std::string json_escape(const std::string& value) {
    std::string result;
    for (char c : value) { if (c == '"' || c == '\\') result += '\\'; result += c; }
    return result;
}

std::vector<Stimulus> discover(const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory)) throw std::runtime_error("study input is not a directory");
    std::vector<Stimulus> result;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".wav") continue;
        const auto stem = entry.path().stem().string();
        const auto separator = stem.find("__");
        result.push_back({separator == std::string::npos ? "demo" : stem.substr(0, separator),
                          separator == std::string::npos ? stem : stem.substr(separator + 2),
                          std::filesystem::absolute(entry.path())});
    }
    if (result.empty()) throw std::runtime_error("no WAV files found for listening study");
    return result;
}

bool reference_name(const std::string& system) {
    return system == "reference" || system == "natural" || system == "ground_truth";
}

}  // namespace

void write_listening_study(const std::filesystem::path& wav_directory,
                           const std::filesystem::path& json_path,
                           const std::filesystem::path& csv_path,
                           StudyManifestOptions options) {
    auto stimuli = discover(wav_directory);
    std::map<std::string, std::vector<Stimulus>> groups;
    for (auto& stimulus : stimuli) groups[stimulus.utterance].push_back(std::move(stimulus));
    std::mt19937_64 random(options.seed);
    for (auto& [utterance, rows] : groups) { (void)utterance; std::shuffle(rows.begin(), rows.end(), random); }

    std::ofstream json(json_path), csv(csv_path);
    if (!json || !csv) throw std::runtime_error("cannot create listening-study outputs");
    csv << "trial_index,utterance_id,position,blind_id,system,file,role\n";
    json << "{\n  \"schema_version\": 1,\n  \"seed\": " << options.seed
         << ",\n  \"mos\": {\"enabled\": " << (options.include_mos ? "true" : "false")
         << ", \"scale\": [1, 5], \"labels\": [\"bad\", \"poor\", \"fair\", \"good\", \"excellent\"]},\n"
         << "  \"mushra\": {\"enabled\": " << (options.include_mushra ? "true" : "false")
         << ", \"scale\": [0, 100], \"requires_hidden_reference\": true, \"requires_anchor\": true},\n"
         << "  \"warning\": \"Trials without natural/reference and anchor systems are marked incomplete.\",\n"
         << "  \"trials\": [\n";
    std::size_t trial_index = 0;
    for (auto group = groups.begin(); group != groups.end(); ++group, ++trial_index) {
        const auto& utterance = group->first;
        const auto& rows = group->second;
        bool has_reference = false, has_anchor = false;
        for (const auto& row : rows) { has_reference |= reference_name(row.system); has_anchor |= row.system.find("anchor") != std::string::npos; }
        json << "    {\"trial_index\": " << trial_index << ", \"utterance_id\": \""
             << json_escape(utterance) << "\", \"mushra_complete\": "
             << (has_reference && has_anchor ? "true" : "false") << ", \"stimuli\": [\n";
        for (std::size_t position = 0; position < rows.size(); ++position) {
            const auto blind = "T" + std::to_string(trial_index + 1) + "_S" + std::to_string(position + 1);
            const std::string role = reference_name(rows[position].system) ? "hidden_reference" :
                (rows[position].system.find("anchor") != std::string::npos ? "anchor" : "candidate");
            json << "      {\"blind_id\": \"" << blind << "\", \"system\": \""
                 << json_escape(rows[position].system) << "\", \"file\": \""
                 << json_escape(rows[position].path.generic_string()) << "\", \"role\": \"" << role << "\"}"
                 << (position + 1 == rows.size() ? "\n" : ",\n");
            csv << trial_index << ',' << utterance << ',' << position << ',' << blind << ','
                << rows[position].system << ",\"" << rows[position].path.string() << "\"," << role << '\n';
        }
        json << "    ]}" << (std::next(group) == groups.end() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";
}

}  // namespace vocal
