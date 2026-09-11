// Phone inventory shared by the acoustic model, the lexicon and the decoder.
//
// TIMIT labels its alignments with 61 symbols; CMUDict spells words with the
// 39 ARPAbet phones. Training on TIMIT and decoding with CMUDict only works if
// both collapse onto one inventory, so this header owns that folding (the
// standard Lee & Hon 61 -> 39 map) plus a silence unit.
#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace models {

// 39 ARPAbet phones + SIL. Index 0 is silence so an empty/zero state is safe.
inline const std::vector<std::string>& phone_names() {
    static const std::vector<std::string> names = {
        "SIL",
        "AA", "AE", "AH", "AO", "AW", "AY", "B",  "CH", "D",  "DH",
        "EH", "ER", "EY", "F",  "G",  "HH", "IH", "IY", "JH", "K",
        "L",  "M",  "N",  "NG", "OW", "OY", "P",  "R",  "S",  "SH",
        "T",  "TH", "UH", "UW", "V",  "W",  "Y",  "Z",  "ZH"};
    return names;
}

constexpr int kNumStatesPerPhone = 3;  // left-to-right HMM, no skips

inline int num_phones() { return static_cast<int>(phone_names().size()); }
inline int num_states() { return num_phones() * kNumStatesPerPhone; }

inline int phone_id(const std::string& name) {
    static const std::unordered_map<std::string, int> index = [] {
        std::unordered_map<std::string, int> map;
        const std::vector<std::string>& names = phone_names();
        for (int i = 0; i < static_cast<int>(names.size()); ++i) map[names[i]] = i;
        return map;
    }();
    const auto it = index.find(name);
    return it == index.end() ? -1 : it->second;
}

// TIMIT's 61 symbols folded onto the inventory above. Closures, pauses and the
// glottal stop become silence; allophones merge with their base phone.
inline int timit_phone_id(const std::string& label) {
    static const std::unordered_map<std::string, std::string> fold = {
        // silence, closures, pauses, glottal stop
        {"h#", "SIL"},   {"pau", "SIL"},  {"epi", "SIL"},  {"q", "SIL"},
        {"bcl", "SIL"},  {"dcl", "SIL"},  {"gcl", "SIL"},
        {"pcl", "SIL"},  {"tcl", "SIL"},  {"kcl", "SIL"},
        // vowels
        {"aa", "AA"},    {"ao", "AA"},    {"ae", "AE"},
        {"ah", "AH"},    {"ax", "AH"},    {"ax-h", "AH"},
        {"aw", "AW"},    {"ay", "AY"},    {"eh", "EH"},
        {"er", "ER"},    {"axr", "ER"},   {"ey", "EY"},
        {"ih", "IH"},    {"ix", "IH"},    {"iy", "IY"},
        {"ow", "OW"},    {"oy", "OY"},    {"uh", "UH"},
        {"uw", "UW"},    {"ux", "UW"},
        // consonants
        {"b", "B"},      {"ch", "CH"},    {"d", "D"},      {"dx", "D"},
        {"dh", "DH"},    {"f", "F"},      {"g", "G"},
        {"hh", "HH"},    {"hv", "HH"},    {"jh", "JH"},    {"k", "K"},
        {"l", "L"},      {"el", "L"},     {"m", "M"},      {"em", "M"},
        {"n", "N"},      {"en", "N"},     {"nx", "N"},
        {"ng", "NG"},    {"eng", "NG"},   {"p", "P"},      {"r", "R"},
        {"s", "S"},      {"sh", "SH"},    {"zh", "SH"},    {"t", "T"},
        {"th", "TH"},    {"v", "V"},      {"w", "W"},      {"y", "Y"},
        {"z", "Z"}};

    const auto it = fold.find(label);
    if (it == fold.end()) return -1;
    return phone_id(it->second);
}

// CMUDict spells phones with stress digits ("AH0", "EY1"); strip them.
inline std::string strip_stress(const std::string& phone) {
    std::string out;
    out.reserve(phone.size());
    for (char c : phone) {
        if (c < '0' || c > '9') out.push_back(c);
    }
    return out;
}

inline int state_index(int phone, int state) { return phone * kNumStatesPerPhone + state; }

}  // namespace models
