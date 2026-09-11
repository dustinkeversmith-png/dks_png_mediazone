#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vocal {

struct Utterance {
    std::string id;
    std::string speaker_id;
    std::string chapter_id;
    std::string normalized_text;
    std::filesystem::path audio_path;
};

class LibriTtsDataset {
public:
    explicit LibriTtsDataset(std::filesystem::path root);
    [[nodiscard]] std::vector<Utterance> scan() const;

private:
    std::filesystem::path root_;
};

}  // namespace vocal

