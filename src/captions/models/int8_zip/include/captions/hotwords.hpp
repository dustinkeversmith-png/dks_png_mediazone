#pragma once
// Domain-term biasing ("hotwords"): names, jargon, product terms a general
// model would otherwise spell like common words. Phrases are split into the
// model's own subword tokens and stored in a prefix tree; while decoding, a
// token that starts a phrase or continues a partial match gets a bonus added
// to its logit before the best token is chosen (shallow fusion). Emitting
// anything else falls back to the root, so a false start costs nothing.

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace captions {

struct HotwordPhrase {
    std::string text;
    float boost = 0;  // 0 = use the default boost
};

// One phrase per line; "#" starts a comment; an optional trailing ":<boost>"
// sets a per-phrase boost ("Kubernetes :3"). Blank lines are ignored.
std::vector<HotwordPhrase> read_hotwords(const std::filesystem::path& file);
// Comma-separated phrases from the command line ("Parakeet,ONNX Runtime").
std::vector<HotwordPhrase> parse_hotwords(const std::string& list);

class HotwordBiaser {
public:
    HotwordBiaser() = default;
    // `pieces[id]` is the SentencePiece string of token id ("▁word" starts a
    // word). Phrases that cannot be spelled with the vocabulary are skipped.
    HotwordBiaser(const std::vector<HotwordPhrase>& phrases, const std::vector<std::string>& pieces,
                  float default_boost, float start_scale = 0.25F);

    [[nodiscard]] bool empty() const { return nodes_.size() <= 1; }
    [[nodiscard]] size_t phrase_count() const { return phrases_; }
    [[nodiscard]] const std::vector<std::string>& skipped() const { return skipped_; }

    // Adds bonuses to `logits` (indexed by token id) for tokens that continue
    // a phrase from `state` or start a new one.
    void bias(int state, float* logits, size_t vocabulary) const;
    // State after emitting `token` from `state` (0 = no partial match).
    [[nodiscard]] int advance(int state, int token) const;

private:
    struct Node {
        std::unordered_map<int, int> next;
        float boost = 0;  // bonus for the token that leads into this node
    };
    std::vector<Node> nodes_;
    size_t phrases_ = 0;
    std::vector<std::string> skipped_;
};
} // namespace captions
