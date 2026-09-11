// Bigram language model with Katz-style backoff (Step 3 of models/README.md).
//
// Trained on LibriSpeech train-clean-100 transcripts, which are speaker- and
// book-disjoint from the test-clean set we evaluate on - training an LM on the
// evaluation transcripts would make the reported WER meaningless.
//
// Layout is built for the decoder's access pattern: bigrams are sorted by
// (history, word) and indexed by a per-history [begin, end) range, so
// enumerating "all successors of word w" - which the decoder does once per
// surviving word end per frame - is a contiguous scan.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace models {

class NgramLm {
public:
    struct Bigram {
        int32_t word;
        float log_prob;
    };

    // ---- training -------------------------------------------------------

    // One line = one utterance. `vocabulary_limit` keeps the most frequent
    // words; everything else maps to <unk>.
    bool train_from_text(const std::string& path, int vocabulary_limit = 20000,
                         double discount = 0.4) {
        std::ifstream file(path);
        if (!file) return false;

        std::unordered_map<std::string, int64_t> word_counts;
        std::vector<std::vector<std::string>> sentences;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream stream(line);
            std::vector<std::string> tokens;
            std::string token;
            while (stream >> token) {
                for (char& c : token) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                }
                tokens.push_back(token);
                ++word_counts[token];
            }
            if (!tokens.empty()) sentences.push_back(std::move(tokens));
        }
        if (sentences.empty()) return false;

        // Vocabulary: <s>, </s>, <unk> then the most frequent words.
        std::vector<std::pair<std::string, int64_t>> ranked(word_counts.begin(),
                                                           word_counts.end());
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first;
        });

        words_.clear();
        word_ids_.clear();
        add_word("<s>");
        add_word("</s>");
        add_word("<unk>");
        for (const auto& entry : ranked) {
            if (static_cast<int>(words_.size()) >= vocabulary_limit) break;
            if (word_ids_.count(entry.first)) continue;
            add_word(entry.first);
        }

        // Counts.
        std::vector<int64_t> unigram_counts(words_.size(), 0);
        std::unordered_map<int64_t, int64_t> bigram_counts;
        int64_t total_unigrams = 0;
        for (const std::vector<std::string>& sentence : sentences) {
            int previous = kBos;
            ++unigram_counts[kBos];
            ++total_unigrams;
            for (const std::string& token : sentence) {
                const int id = lookup_or_unk(token);
                ++unigram_counts[id];
                ++total_unigrams;
                ++bigram_counts[(static_cast<int64_t>(previous) << 32) | id];
                previous = id;
            }
            ++unigram_counts[kEos];
            ++total_unigrams;
            ++bigram_counts[(static_cast<int64_t>(previous) << 32) | kEos];
        }

        // Unigrams with add-one smoothing over the closed vocabulary.
        const double vocabulary = static_cast<double>(words_.size());
        log_unigram_.assign(words_.size(), 0.0f);
        for (size_t w = 0; w < words_.size(); ++w) {
            log_unigram_[w] = static_cast<float>(
                std::log((unigram_counts[w] + 1.0) / (total_unigrams + vocabulary)));
        }

        // Bigrams with absolute discounting; the discounted mass becomes the
        // per-history backoff weight.
        std::vector<std::vector<Bigram>> by_history(words_.size());
        std::vector<double> backoff_mass(words_.size(), 1.0);
        for (const auto& entry : bigram_counts) {
            const int history = static_cast<int>(entry.first >> 32);
            const int word = static_cast<int>(entry.first & 0xFFFFFFFF);
            const double count = static_cast<double>(entry.second);
            const double history_count = static_cast<double>(unigram_counts[history]);
            if (history_count <= 0.0 || count <= discount) continue;
            const double probability = (count - discount) / history_count;
            by_history[history].push_back({word, static_cast<float>(std::log(probability))});
            backoff_mass[history] -= probability;
        }

        bigrams_.clear();
        bigram_begin_.assign(words_.size() + 1, 0);
        log_backoff_.assign(words_.size(), 0.0f);
        for (size_t h = 0; h < words_.size(); ++h) {
            bigram_begin_[h] = static_cast<int32_t>(bigrams_.size());
            std::sort(by_history[h].begin(), by_history[h].end(),
                      [](const Bigram& a, const Bigram& b) { return a.word < b.word; });
            bigrams_.insert(bigrams_.end(), by_history[h].begin(), by_history[h].end());
            const double mass = backoff_mass[h] > 1e-6 ? backoff_mass[h] : 1e-6;
            log_backoff_[h] = static_cast<float>(std::log(mass));
        }
        bigram_begin_[words_.size()] = static_cast<int32_t>(bigrams_.size());
        return true;
    }

    // ---- query ----------------------------------------------------------

    int word_id(const std::string& word) const {
        const auto it = word_ids_.find(word);
        return it == word_ids_.end() ? -1 : it->second;
    }
    const std::string& word(int id) const { return words_[id]; }
    int vocabulary_size() const { return static_cast<int>(words_.size()); }
    int bos() const { return kBos; }
    int eos() const { return kEos; }
    int unk() const { return kUnk; }

    float log_unigram(int word) const { return log_unigram_[word]; }
    float log_backoff(int history) const { return log_backoff_[history]; }

    // Contiguous successor range for a history word.
    const Bigram* successors_begin(int history) const {
        return bigrams_.data() + bigram_begin_[history];
    }
    const Bigram* successors_end(int history) const {
        return bigrams_.data() + bigram_begin_[history + 1];
    }

    float log_probability(int history, int word) const {
        const Bigram* begin = successors_begin(history);
        const Bigram* end = successors_end(history);
        const Bigram* hit = std::lower_bound(
            begin, end, word, [](const Bigram& b, int w) { return b.word < w; });
        if (hit != end && hit->word == word) return hit->log_prob;
        return log_backoff_[history] + log_unigram_[word];
    }

    // Perplexity on held-out text, for sanity-checking the LM in isolation.
    double perplexity(const std::vector<std::vector<std::string>>& sentences) const {
        double log_sum = 0.0;
        int64_t tokens = 0;
        for (const auto& sentence : sentences) {
            int previous = kBos;
            for (const std::string& token : sentence) {
                const int id = lookup_or_unk_const(token);
                log_sum += log_probability(previous, id);
                previous = id;
                ++tokens;
            }
            log_sum += log_probability(previous, kEos);
            ++tokens;
        }
        if (tokens == 0) return 0.0;
        return std::exp(-log_sum / tokens);
    }

    // ---- persistence ----------------------------------------------------

    bool save(const std::string& path) const {
        FILE* file = std::fopen(path.c_str(), "wb");
        if (!file) return false;
        const int32_t magic = 0x4C4D4231;  // "LMB1"
        const int32_t vocabulary = static_cast<int32_t>(words_.size());
        const int32_t bigrams = static_cast<int32_t>(bigrams_.size());
        std::fwrite(&magic, sizeof(magic), 1, file);
        std::fwrite(&vocabulary, sizeof(vocabulary), 1, file);
        std::fwrite(&bigrams, sizeof(bigrams), 1, file);
        for (const std::string& word : words_) {
            const int32_t length = static_cast<int32_t>(word.size());
            std::fwrite(&length, sizeof(length), 1, file);
            std::fwrite(word.data(), 1, word.size(), file);
        }
        std::fwrite(log_unigram_.data(), sizeof(float), log_unigram_.size(), file);
        std::fwrite(log_backoff_.data(), sizeof(float), log_backoff_.size(), file);
        std::fwrite(bigram_begin_.data(), sizeof(int32_t), bigram_begin_.size(), file);
        std::fwrite(bigrams_.data(), sizeof(Bigram), bigrams_.size(), file);
        std::fclose(file);
        return true;
    }

    bool load(const std::string& path) {
        FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) return false;
        int32_t magic = 0, vocabulary = 0, bigrams = 0;
        if (std::fread(&magic, sizeof(magic), 1, file) != 1 ||
            std::fread(&vocabulary, sizeof(vocabulary), 1, file) != 1 ||
            std::fread(&bigrams, sizeof(bigrams), 1, file) != 1 || magic != 0x4C4D4231) {
            std::fclose(file);
            return false;
        }
        words_.clear();
        word_ids_.clear();
        words_.reserve(vocabulary);
        for (int i = 0; i < vocabulary; ++i) {
            int32_t length = 0;
            if (std::fread(&length, sizeof(length), 1, file) != 1 || length < 0) {
                std::fclose(file);
                return false;
            }
            std::string word(static_cast<size_t>(length), '\0');
            if (length > 0 && std::fread(&word[0], 1, length, file) != static_cast<size_t>(length)) {
                std::fclose(file);
                return false;
            }
            word_ids_[word] = static_cast<int>(words_.size());
            words_.push_back(std::move(word));
        }
        log_unigram_.resize(vocabulary);
        log_backoff_.resize(vocabulary);
        bigram_begin_.resize(static_cast<size_t>(vocabulary) + 1);
        bigrams_.resize(bigrams);
        const bool ok =
            std::fread(log_unigram_.data(), sizeof(float), log_unigram_.size(), file) ==
                log_unigram_.size() &&
            std::fread(log_backoff_.data(), sizeof(float), log_backoff_.size(), file) ==
                log_backoff_.size() &&
            std::fread(bigram_begin_.data(), sizeof(int32_t), bigram_begin_.size(), file) ==
                bigram_begin_.size() &&
            std::fread(bigrams_.data(), sizeof(Bigram), bigrams_.size(), file) == bigrams_.size();
        std::fclose(file);
        return ok;
    }

private:
    static constexpr int kBos = 0;
    static constexpr int kEos = 1;
    static constexpr int kUnk = 2;

    void add_word(const std::string& word) {
        word_ids_[word] = static_cast<int>(words_.size());
        words_.push_back(word);
    }
    int lookup_or_unk(const std::string& word) {
        const auto it = word_ids_.find(word);
        return it == word_ids_.end() ? kUnk : it->second;
    }
    int lookup_or_unk_const(const std::string& word) const {
        const auto it = word_ids_.find(word);
        return it == word_ids_.end() ? kUnk : it->second;
    }

    std::vector<std::string> words_;
    std::unordered_map<std::string, int> word_ids_;
    std::vector<float> log_unigram_;
    std::vector<float> log_backoff_;
    std::vector<int32_t> bigram_begin_;
    std::vector<Bigram> bigrams_;
};

}  // namespace models
