#include "vocal/dataset.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vocal {

LibriTtsDataset::LibriTtsDataset(std::filesystem::path root) : root_(std::move(root)) {}

std::vector<Utterance> LibriTtsDataset::scan() const {
    if (!std::filesystem::exists(root_)) {
        throw std::runtime_error("dataset root does not exist: " + root_.string());
    }

    std::vector<Utterance> items;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".wav") {
            continue;
        }

        const auto stem = entry.path().stem().string();
        auto text_path = entry.path().parent_path() / (stem + ".normalized.txt");
        if (!std::filesystem::exists(text_path)) {
            continue;
        }

        std::ifstream text_file(text_path);
        std::ostringstream text;
        text << text_file.rdbuf();
        auto normalized = text.str();
        while (!normalized.empty() && (normalized.back() == '\n' || normalized.back() == '\r')) {
            normalized.pop_back();
        }

        const auto chapter_dir = entry.path().parent_path();
        const auto speaker_dir = chapter_dir.parent_path();
        items.push_back({stem, speaker_dir.filename().string(), chapter_dir.filename().string(),
                         std::move(normalized), entry.path()});
    }
    return items;
}

}  // namespace vocal

