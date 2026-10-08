#pragma once

#include <filesystem>
#include <stdexcept>

namespace vocal::cli {
inline std::filesystem::path resolve_dictionary(const std::filesystem::path& assets) {
    const std::filesystem::path candidates[] = {
        assets / "cmudict.dict",
        assets.parent_path() / "shared" / "cmudict.dict",
        assets.parent_path() / "cmudict.dict",
        assets.parent_path() / "piper" / "cmudict.dict",
    };
    for (const auto& candidate : candidates)
        if (std::filesystem::is_regular_file(candidate)) return candidate;
    throw std::runtime_error("cannot find CMUdict in assets or adjacent shared/piper paths: " + assets.string());
}
} // namespace vocal::cli
