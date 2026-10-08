#pragma once

#include "vocal/control_params.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vocal {

struct PhonemizationResult {
    std::vector<std::int64_t> token_ids;
    std::string ipa;
    std::size_t words{};
    std::size_t dictionary_hits{};
    std::size_t fallback_words{};
    std::size_t missing_model_symbols{};
    std::vector<ProsodyTokenKind> token_kinds;
};

class CmuPhonemizer {
public:
    CmuPhonemizer(const std::filesystem::path& dictionary_path,
                  const std::filesystem::path& token_map_path);
    [[nodiscard]] PhonemizationResult phonemize(std::string_view text) const;

private:
    std::unordered_map<std::string, std::vector<std::string>> dictionary_;
    std::unordered_map<std::uint32_t, std::vector<std::int64_t>> token_map_;
    std::unordered_map<std::string, std::vector<std::int64_t>> arpa_token_map_;
    bool arpabet_frontend_{};
};

}  // namespace vocal
