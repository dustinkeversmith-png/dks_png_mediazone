#include <captions/hotwords.hpp>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace captions {
namespace {
const std::string kMarker = "\xE2\x96\x81";  // U+2581, SentencePiece word start

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}
HotwordPhrase parse_line(std::string line) {
    HotwordPhrase p;
    const auto colon = line.rfind(':');
    if (colon != std::string::npos) {
        const auto value = trim(line.substr(colon + 1));
        char* end = nullptr;
        const float boost = std::strtof(value.c_str(), &end);
        if (!value.empty() && end && *end == '\0') { p.boost = boost; line.resize(colon); }
    }
    p.text = trim(line);
    return p;
}

// Greedy longest-match spelling of `word` (with its word-start marker) in the
// vocabulary; empty if some character has no piece.
std::vector<int> spell(const std::string& word, const std::unordered_map<std::string, int>& vocab, size_t longest) {
    const std::string s = kMarker + word;
    std::vector<int> ids;
    for (size_t i = 0; i < s.size();) {
        bool found = false;
        for (size_t n = std::min(longest, s.size() - i); n > 0; --n) {
            const auto it = vocab.find(s.substr(i, n));
            if (it == vocab.end()) continue;
            ids.push_back(it->second);
            i += n;
            found = true;
            break;
        }
        if (!found) return {};
    }
    return ids;
}
} // namespace

std::vector<HotwordPhrase> read_hotwords(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("Cannot open hotwords file " + file.string());
    std::vector<HotwordPhrase> phrases;
    for (std::string line; std::getline(in, line);) {
        if (const auto hash = line.find('#'); hash != std::string::npos) line.resize(hash);
        auto p = parse_line(line);
        if (!p.text.empty()) phrases.push_back(p);
    }
    return phrases;
}

std::vector<HotwordPhrase> parse_hotwords(const std::string& list) {
    std::vector<HotwordPhrase> phrases;
    std::stringstream stream(list);
    for (std::string item; std::getline(stream, item, ',');) {
        auto p = parse_line(item);
        if (!p.text.empty()) phrases.push_back(p);
    }
    return phrases;
}

HotwordBiaser::HotwordBiaser(const std::vector<HotwordPhrase>& phrases, const std::vector<std::string>& pieces,
                             float default_boost, float start_scale) {
    nodes_.emplace_back();
    if (phrases.empty()) return;
    std::unordered_map<std::string, int> vocab;
    size_t longest = 1;
    bool cased = false;
    for (size_t id = 0; id < pieces.size(); ++id) {
        const auto& piece = pieces[id];
        if (piece.empty() || piece.front() == '<') continue;  // <blk>, <unk>, ...
        vocab.emplace(piece, static_cast<int>(id));
        longest = std::max(longest, piece.size());
        cased = cased || std::any_of(piece.begin(), piece.end(), [](unsigned char c) { return std::isupper(c); });
    }
    auto add = [&](const std::string& text, float boost) {
        std::vector<int> ids;
        std::stringstream words(text);
        for (std::string w; words >> w;) {
            const auto part = spell(w, vocab, longest);
            if (part.empty()) return false;
            ids.insert(ids.end(), part.begin(), part.end());
        }
        if (ids.empty()) return false;
        int node = 0, node_depth = 0;
        for (int id : ids) {
            auto it = nodes_[node].next.find(id);
            if (it == nodes_[node].next.end()) {
                nodes_.emplace_back();
                it = nodes_[node].next.emplace(id, static_cast<int>(nodes_.size() - 1)).first;
            }
            node = it->second;
            // The first token of a phrase is the riskiest to boost: it is often a
            // common piece, and boosting it everywhere invites insertions.
            const float scaled = node_depth == 0 ? boost * start_scale : boost;
            nodes_[node].boost = std::max(nodes_[node].boost, scaled);
            ++node_depth;
        }
        return true;
    };
    for (const auto& p : phrases) {
        const float boost = p.boost > 0 ? p.boost : default_boost;
        std::string text = p.text;
        if (!cased) std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (!add(text, boost)) { skipped_.push_back(p.text); continue; }
        ++phrases_;
        // A cased model writes a term capitalized at the start of a sentence.
        if (cased && !text.empty() && std::islower(static_cast<unsigned char>(text[0]))) {
            std::string capital = text;
            capital[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(capital[0])));
            add(capital, boost);
        }
    }
}

void HotwordBiaser::bias(int state, float* logits, size_t vocabulary) const {
    if (empty()) return;
    // Tokens that continue the partial match, then tokens that start a new
    // phrase; a token reachable both ways is boosted once.
    const auto& here = nodes_[static_cast<size_t>(state)].next;
    for (const auto& [token, child] : here)
        if (static_cast<size_t>(token) < vocabulary) logits[token] += nodes_[static_cast<size_t>(child)].boost;
    if (state == 0) return;
    for (const auto& [token, child] : nodes_[0].next)
        if (static_cast<size_t>(token) < vocabulary && !here.count(token))
            logits[token] += nodes_[static_cast<size_t>(child)].boost;
}

int HotwordBiaser::advance(int state, int token) const {
    if (empty()) return 0;
    const auto& here = nodes_[static_cast<size_t>(state)].next;
    if (const auto it = here.find(token); it != here.end()) return it->second;
    const auto& root = nodes_[0].next;
    if (const auto it = root.find(token); it != root.end()) return it->second;
    return 0;
}
} // namespace captions
