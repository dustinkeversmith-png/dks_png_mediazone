#pragma once

#include "datasets/io/vision_io.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace vision {

struct AtomConfig {
    std::string input_type;
    std::vector<std::string> pre_processing_stages;
    std::vector<std::string> artifacts;
    std::string preferred_dataset;
};

inline std::string extract_json_string(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return {};
    }
    const size_t colon = json.find(':', pos + needle.size());
    if (colon == std::string::npos) {
        return {};
    }
    const size_t q0 = json.find('"', colon);
    if (q0 == std::string::npos) {
        return {};
    }
    const size_t q1 = json.find('"', q0 + 1);
    if (q1 == std::string::npos) {
        return {};
    }
    return json.substr(q0 + 1, q1 - q0 - 1);
}

inline std::vector<std::string> extract_json_string_array(const std::string& json, const std::string& key) {
    std::vector<std::string> out;
    const std::string needle = "\"" + key + "\"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return out;
    }
    const size_t lb = json.find('[', pos);
    const size_t rb = json.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) {
        return out;
    }
    const std::string body = json.substr(lb + 1, rb - lb - 1);
    size_t i = 0;
    while (i < body.size()) {
        const size_t q0 = body.find('"', i);
        if (q0 == std::string::npos) {
            break;
        }
        const size_t q1 = body.find('"', q0 + 1);
        if (q1 == std::string::npos) {
            break;
        }
        out.push_back(body.substr(q0 + 1, q1 - q0 - 1));
        i = q1 + 1;
    }
    return out;
}

inline AtomConfig load_atom_config(const std::filesystem::path& config_path) {
    AtomConfig cfg;
    const std::string text = read_text_file(config_path.string());
    if (text.empty()) {
        return cfg;
    }
    cfg.input_type = extract_json_string(text, "input_type");
    cfg.pre_processing_stages = extract_json_string_array(text, "pre_processing_stages");
    cfg.artifacts = extract_json_string_array(text, "artifacts");
    cfg.preferred_dataset = extract_json_string(text, "preferred_dataset");
    return cfg;
}

inline AtomConfig load_atom_config_near(const char* source_file) {
#ifdef VISION_ATOM_CONFIG_PATH
    return load_atom_config(VISION_ATOM_CONFIG_PATH);
#endif
    if (source_file == nullptr) {
        return {};
    }
    const std::filesystem::path p(source_file);
    const auto cfg = p.parent_path() / "atom.config.json";
    if (std::filesystem::exists(cfg)) {
        return load_atom_config(cfg);
    }
    return {};
}

inline std::filesystem::path default_artifact_dir(const char* source_file) {
#ifdef VISION_TEST_ARTIFACT_ROOT
    return std::filesystem::path(VISION_TEST_ARTIFACT_ROOT);
#endif
    if (source_file != nullptr) {
        return std::filesystem::path(source_file).parent_path() / "artifacts";
    }
    return std::filesystem::current_path() / "artifacts";
}

}  // namespace vision
