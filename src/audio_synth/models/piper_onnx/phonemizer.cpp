#include "vocal/phonemizer.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vocal {
namespace {

std::vector<std::uint32_t> decode_utf8(std::string_view text) {
    std::vector<std::uint32_t> result;
    for (std::size_t i = 0; i < text.size();) {
        const auto first = static_cast<unsigned char>(text[i++]);
        std::uint32_t cp = first;
        std::size_t continuation = 0;
        if ((first & 0xe0U) == 0xc0U) { cp = first & 0x1fU; continuation = 1; }
        else if ((first & 0xf0U) == 0xe0U) { cp = first & 0x0fU; continuation = 2; }
        else if ((first & 0xf8U) == 0xf0U) { cp = first & 0x07U; continuation = 3; }
        for (std::size_t n = 0; n < continuation && i < text.size(); ++n)
            cp = (cp << 6) | (static_cast<unsigned char>(text[i++]) & 0x3fU);
        result.push_back(cp);
    }
    return result;
}

std::string encode_utf8(std::uint32_t cp) {
    std::string out;
    if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    return out;
}

std::string arpa_to_ipa(std::string phone) {
    char stress = 0;
    if (!phone.empty() && std::isdigit(static_cast<unsigned char>(phone.back()))) {
        stress = phone.back(); phone.pop_back();
    }
    if (phone == "AH") return (stress == '1' ? "ˈ" : stress == '2' ? "ˌ" : "") +
                               std::string(stress == '0' ? "ə" : "ʌ");
    if (phone == "ER") return (stress == '1' ? "ˈ" : stress == '2' ? "ˌ" : "") +
                               std::string(stress == '0' ? "ɚ" : "ɜː");
    static const std::unordered_map<std::string, std::string> map{
        {"AA", "ɑ"}, {"AE", "æ"},
        {"AO", "ɔ"}, {"AW", "aʊ"}, {"AY", "aɪ"}, {"EH", "ɛ"},
        {"EY", "eɪ"}, {"IH", "ɪ"},
        {"IY", "i"}, {"OW", "oʊ"}, {"OY", "ɔɪ"}, {"UH", "ʊ"}, {"UW", "u"},
        {"B", "b"}, {"CH", "tʃ"}, {"D", "d"}, {"DH", "ð"}, {"F", "f"},
        {"G", "ɡ"}, {"HH", "h"}, {"JH", "dʒ"}, {"K", "k"}, {"L", "l"},
        {"M", "m"}, {"N", "n"}, {"NG", "ŋ"}, {"P", "p"}, {"R", "ɹ"},
        {"S", "s"}, {"SH", "ʃ"}, {"T", "t"}, {"TH", "θ"}, {"V", "v"},
        {"W", "w"}, {"Y", "j"}, {"Z", "z"}, {"ZH", "ʒ"}};
    const auto found = map.find(phone);
    if (found == map.end()) return {};
    return (stress == '1' ? "ˈ" : stress == '2' ? "ˌ" : "") + found->second;
}

}  // namespace

CmuPhonemizer::CmuPhonemizer(const std::filesystem::path& dictionary_path,
                             const std::filesystem::path& token_map_path) {
    std::ifstream dictionary(dictionary_path);
    if (!dictionary) throw std::runtime_error("cannot open CMUdict: " + dictionary_path.string());
    for (std::string line; std::getline(dictionary, line);) {
        if (line.empty() || line[0] == ';') continue;
        std::istringstream row(line);
        std::string word; row >> word;
        const auto variant = word.find('(');
        if (variant != std::string::npos) word.resize(variant);
        if (dictionary_.contains(word)) continue;
        std::vector<std::string> phones;
        for (std::string phone; row >> phone;) phones.push_back(std::move(phone));
        if (!phones.empty()) dictionary_[std::move(word)] = std::move(phones);
    }
    std::ifstream tokens(token_map_path);
    if (!tokens) throw std::runtime_error("cannot open model token map: " + token_map_path.string());
    for (std::string line; std::getline(tokens, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "# frontend=arpabet") { arpabet_frontend_ = true; continue; }
        if (line.empty() || line[0] == '#') continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::vector<std::int64_t> ids;
        std::istringstream values(line.substr(tab + 1));
        for (std::string id; std::getline(values, id, ',');) ids.push_back(std::stoll(id));
        if (arpabet_frontend_) arpa_token_map_[line.substr(0, tab)] = std::move(ids);
        else {
            std::uint32_t cp{};
            const auto parsed = std::from_chars(line.data(), line.data() + tab, cp, 16);
            if (parsed.ec != std::errc{}) continue;
            token_map_[cp] = std::move(ids);
        }
    }
}

PhonemizationResult CmuPhonemizer::phonemize(std::string_view text) const {
    PhonemizationResult result;
    auto append_symbol = [&](std::uint32_t cp, bool pad, ProsodyTokenKind kind = ProsodyTokenKind::Unknown) {
        const auto found = token_map_.find(cp);
        if (found == token_map_.end()) { ++result.missing_model_symbols; return; }
        result.token_ids.insert(result.token_ids.end(), found->second.begin(), found->second.end());
        result.token_kinds.insert(result.token_kinds.end(), found->second.size(), kind);
        if (pad) {
            const auto padding = token_map_.find('_');
            if (padding != token_map_.end()) {
                result.token_ids.insert(result.token_ids.end(), padding->second.begin(), padding->second.end());
                result.token_kinds.insert(result.token_kinds.end(), padding->second.size(), ProsodyTokenKind::Unknown);
            }
        }
        result.ipa += encode_utf8(cp);
    };
    auto append_phone = [&](const std::string& phone, ProsodyTokenKind kind) {
        const auto found = arpa_token_map_.find(phone);
        if (found == arpa_token_map_.end()) { ++result.missing_model_symbols; return; }
        result.token_ids.insert(result.token_ids.end(), found->second.begin(), found->second.end());
        result.token_kinds.insert(result.token_kinds.end(), found->second.size(), kind);
        result.ipa += phone + " ";
    };
    if (!arpabet_frontend_) append_symbol('^', true);
    std::string word;
    auto flush_word = [&] {
        if (word.empty()) return;
        ++result.words;
        auto found = dictionary_.find(word);
        std::vector<std::string> fallback;
        const std::vector<std::string>* phones = nullptr;
        if (found != dictionary_.end()) { phones = &found->second; ++result.dictionary_hits; }
        else {
            // Spell OOV words using CMUdict's single-letter pronunciations.
            for (char letter : word) {
                const auto spelled = dictionary_.find(std::string(1, letter));
                if (spelled != dictionary_.end())
                    fallback.insert(fallback.end(), spelled->second.begin(), spelled->second.end());
            }
            phones = &fallback; ++result.fallback_words;
        }
        for (const auto& phone : *phones) {
            const auto ipa = arpa_to_ipa(phone);
            std::string arpa = phone;
            const bool stressed = !arpa.empty() && (arpa.back() == '1' || arpa.back() == '2');
            if (!arpa.empty() && std::isdigit(static_cast<unsigned char>(arpa.back()))) arpa.pop_back();
            const std::string vowels = " AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW ";
            const std::string unvoiced = " CH F HH K P S SH T TH ";
            auto kind = ProsodyTokenKind::Unknown;
            if (vowels.find(" " + arpa + " ") != std::string::npos)
                kind = stressed ? ProsodyTokenKind::StressedVowel : ProsodyTokenKind::Vowel;
            else if (unvoiced.find(" " + arpa + " ") != std::string::npos)
                kind = ProsodyTokenKind::UnvoicedConsonant;
            if (arpabet_frontend_) append_phone(phone, kind);
            else for (auto cp : decode_utf8(ipa))
                append_symbol(cp, true, cp == 0x02c8 || cp == 0x02cc || cp == 0x02d0 ? ProsodyTokenKind::Unknown : kind);
        }
        word.clear();
    };
    bool pending_space = false;
    for (unsigned char raw : text) {
        const char c = static_cast<char>(std::tolower(raw));
        if (std::isalnum(raw) || (c == '\'' && !word.empty())) { word += c; pending_space = false; }
        else {
            flush_word();
            if (std::string_view(".,!?;:").find(c) != std::string_view::npos) {
                if (arpabet_frontend_) append_phone("sp", ProsodyTokenKind::Boundary);
                else append_symbol(c, true, ProsodyTokenKind::Boundary);
            }
            if (std::isspace(raw)) pending_space = true;
            if (pending_space && result.words > 0) {
                if (!arpabet_frontend_) append_symbol(' ', true);
                pending_space = false;
            }
        }
    }
    flush_word();
    if (!arpabet_frontend_) append_symbol('$', false, ProsodyTokenKind::Boundary);
    else {
        // Publisher's merged English frontend omits trailing silence and BOS/EOS/padding.
        const auto silence = arpa_token_map_.find("sp");
        if (silence != arpa_token_map_.end() && !silence->second.empty() &&
            result.token_ids.size() >= silence->second.size() &&
            std::equal(silence->second.rbegin(), silence->second.rend(), result.token_ids.rbegin())) {
            result.token_ids.resize(result.token_ids.size() - silence->second.size());
            result.token_kinds.resize(result.token_ids.size());
        }
    }
    return result;
}

}  // namespace vocal
