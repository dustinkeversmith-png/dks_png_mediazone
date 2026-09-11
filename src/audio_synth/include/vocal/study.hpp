#pragma once

#include <cstdint>
#include <filesystem>

namespace vocal {

struct StudyManifestOptions {
    std::uint64_t seed{0x4d5553485241ULL};
    bool include_mos{true};
    bool include_mushra{true};
};

void write_listening_study(const std::filesystem::path& wav_directory,
                           const std::filesystem::path& json_path,
                           const std::filesystem::path& csv_path,
                           StudyManifestOptions options = {});

}  // namespace vocal
