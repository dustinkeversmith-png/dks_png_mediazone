// Pronunciation lexicon (Step 2 of models/README.md).
//
// Loads the CMU Pronouncing Dictionary and turns every word into a sequence of
// phone ids from phone_set.hpp. The decoder never sees text until a word HMM
// finishes; this table is the only bridge between acoustics and orthography.
#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "phone_set.hpp"

namespace models {

struct Pronunciation {
    int word_id = -1;
    std::vector<int> phones;
};

class Lexicon {
public:
    // `vocabulary` restricts the lexicon to words we actually decode (empty =
    // keep everything). A 200k-word lexicon is mostly dead weight for a
    // read-speech test set and slows the search down for nothing.
    bool load_cmudict(const std::string& path,
                      const std::unordered_set<std::string>& vocabulary = {}) {
        std::ifstream file(path);
        if (!file) return false;

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == ';') continue;

            std::istringstream stream(line);
            std::string word;
            stream >> word;
            if (word.empty()) continue;

            // "word(2)" marks an alternate pronunciation of the same word.
            const size_t paren = word.find('(');
            if (paren != std::string::npos) word = word.substr(0, paren);
            for (char& c : word) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            if (!vocabulary.empty() && vocabulary.find(word) == vocabulary.end()) continue;

            std::vector<int> phones;
            std::string token;
            bool usable = true;
            while (stream >> token) {
                if (token[0] == '#') break;  // trailing comment
                const int phone = phone_id(strip_stress(token));
                if (phone < 0) {
                    usable = false;
                    break;
                }
                phones.push_back(phone);
            }
            if (!usable || phones.empty()) continue;

            int id;
            const auto it = word_ids_.find(word);
            if (it == word_ids_.end()) {
                id = static_cast<int>(words_.size());
                word_ids_[word] = id;
                words_.push_back(word);
            } else {
                id = it->second;
            }
            pronunciations_.push_back({id, std::move(phones)});
        }
        return !pronunciations_.empty();
    }

    // Programmatic entry, used to build non-word inventories - the Tier 2
    // phone loop is just a lexicon whose "words" are single phones.
    void add_entry(const std::string& word, const std::vector<int>& phones) {
        if (phones.empty()) return;
        int id;
        const auto it = word_ids_.find(word);
        if (it == word_ids_.end()) {
            id = static_cast<int>(words_.size());
            word_ids_[word] = id;
            words_.push_back(word);
        } else {
            id = it->second;
        }
        pronunciations_.push_back({id, phones});
    }

    // Words with no CMUDict entry are unrecoverable for the decoder; report
    // them so evaluation can separate lexicon gaps from search errors.
    int word_id(const std::string& word) const {
        const auto it = word_ids_.find(word);
        return it == word_ids_.end() ? -1 : it->second;
    }

    const std::string& word(int id) const { return words_[id]; }
    int num_words() const { return static_cast<int>(words_.size()); }
    const std::vector<Pronunciation>& pronunciations() const { return pronunciations_; }
    const std::vector<std::string>& words() const { return words_; }

private:
    std::vector<std::string> words_;
    std::unordered_map<std::string, int> word_ids_;
    std::vector<Pronunciation> pronunciations_;
};

}  // namespace models
