// Rule-based grapheme-to-phoneme fallback for words missing from CMUDict.
//
// Why this is worth having: in LibriSpeech train-clean-100, only ~1.8 % of word
// tokens are outside CMUDict - almost all of them proper nouns from 19th
// century novels (CHAUVELIN, PENCROFT, KENNICOTT). But utterances are ~34 words
// long, so that 1.8 % lands in 41 % of utterances. Dropping those utterances
// throws away 41 % of the acoustic training audio to avoid mis-spelling 1.8 %
// of the words; guessing a pronunciation is the better trade.
//
// This is deliberately a small letter-to-sound ruleset, not a trained G2P: it
// handles English digraphs, the silent-final-E rule and common suffixes. It is
// used only for forced alignment of training audio, never for decoding - the
// decoding lexicon stays CMUDict-only, so a bad guess here cannot invent a
// word in a transcript.
#pragma once

#include <string>
#include <vector>

#include "phone_set.hpp"

namespace models {

class GraphemeToPhoneme {
public:
    // Returns phone ids for an uppercase A-Z (+ apostrophe) word.
    std::vector<int> convert(const std::string& word) const {
        std::vector<std::string> phones;
        std::string text;
        for (char c : word) {
            if (c >= 'A' && c <= 'Z') text.push_back(c);
        }
        if (text.empty()) return {};

        const size_t n = text.size();
        for (size_t i = 0; i < n;) {
            const char c = text[i];
            const char next = i + 1 < n ? text[i + 1] : '\0';
            const char after = i + 2 < n ? text[i + 2] : '\0';
            const bool at_end = i + 1 == n;

            // --- two- and three-letter graphemes -------------------------
            if (c == 'C' && next == 'H') { phones.push_back("CH"); i += 2; continue; }
            if (c == 'S' && next == 'H') { phones.push_back("SH"); i += 2; continue; }
            if (c == 'T' && next == 'H') { phones.push_back("TH"); i += 2; continue; }
            if (c == 'P' && next == 'H') { phones.push_back("F");  i += 2; continue; }
            if (c == 'W' && next == 'H') { phones.push_back("W");  i += 2; continue; }
            if (c == 'C' && next == 'K') { phones.push_back("K");  i += 2; continue; }
            if (c == 'N' && next == 'G' && at_end_after(text, i + 1)) {
                phones.push_back("NG"); i += 2; continue;
            }
            if (c == 'Q' && next == 'U') { phones.push_back("K"); phones.push_back("W"); i += 2; continue; }
            if (c == 'G' && next == 'H') {
                // "GH" is silent word-finally (THROUGH) and /f/ in LAUGH-like
                // endings; treat the common cases and otherwise drop it.
                if (i + 2 == n && i > 0 && (text[i - 1] == 'U')) { i += 2; continue; }
                phones.push_back("F"); i += 2; continue;
            }
            if (c == 'P' && next == 'S' && i == 0) { phones.push_back("S"); i += 2; continue; }
            if (c == 'K' && next == 'N' && i == 0) { phones.push_back("N"); i += 2; continue; }
            if (c == 'W' && next == 'R' && i == 0) { phones.push_back("R"); i += 2; continue; }

            // --- vowels --------------------------------------------------
            if (is_vowel(c)) {
                // Common vowel digraphs.
                if (c == 'E' && next == 'E') { phones.push_back("IY"); i += 2; continue; }
                if (c == 'E' && next == 'A') { phones.push_back("IY"); i += 2; continue; }
                if (c == 'O' && next == 'O') { phones.push_back("UW"); i += 2; continue; }
                if (c == 'O' && next == 'U') { phones.push_back("AW"); i += 2; continue; }
                if (c == 'O' && next == 'W') { phones.push_back("OW"); i += 2; continue; }
                if (c == 'O' && next == 'I') { phones.push_back("OY"); i += 2; continue; }
                if (c == 'A' && next == 'I') { phones.push_back("EY"); i += 2; continue; }
                if (c == 'A' && next == 'Y') { phones.push_back("EY"); i += 2; continue; }
                if (c == 'A' && next == 'U') { phones.push_back("AA"); i += 2; continue; }
                if (c == 'E' && next == 'I') { phones.push_back("EY"); i += 2; continue; }
                if (c == 'I' && next == 'E') { phones.push_back("IY"); i += 2; continue; }

                // Silent final E (NAME, HOPE) - lengthens the previous vowel,
                // which the digraph rules above already approximate.
                if (c == 'E' && at_end && phones.size() > 0 && i > 0 && !is_vowel(text[i - 1])) {
                    ++i;
                    continue;
                }
                // R-coloured vowels.
                if (next == 'R' && (after == '\0' || !is_vowel(after))) {
                    phones.push_back("ER");
                    i += 2;
                    continue;
                }
                // Long vowel when a single consonant is followed by a final E.
                const bool magic_e = (i + 3 == n && !is_vowel(next) && after == 'E');
                phones.push_back(vowel_phone(c, magic_e));
                ++i;
                continue;
            }

            // --- consonants ----------------------------------------------
            switch (c) {
                case 'B': phones.push_back("B"); break;
                case 'C': phones.push_back(is_front_vowel(next) ? "S" : "K"); break;
                case 'D': phones.push_back("D"); break;
                case 'F': phones.push_back("F"); break;
                case 'G': phones.push_back(is_front_vowel(next) ? "JH" : "G"); break;
                case 'H': phones.push_back("HH"); break;
                case 'J': phones.push_back("JH"); break;
                case 'K': phones.push_back("K"); break;
                case 'L': phones.push_back("L"); break;
                case 'M': phones.push_back("M"); break;
                case 'N': phones.push_back("N"); break;
                case 'P': phones.push_back("P"); break;
                case 'R': phones.push_back("R"); break;
                case 'S': phones.push_back("S"); break;
                case 'T': phones.push_back("T"); break;
                case 'V': phones.push_back("V"); break;
                case 'W': phones.push_back("W"); break;
                case 'X': phones.push_back("K"); phones.push_back("S"); break;
                case 'Y': phones.push_back(i == 0 ? "Y" : "IY"); break;
                case 'Z': phones.push_back("Z"); break;
                default: break;
            }
            // Collapse doubled consonants (LL, TT, SS).
            if (next == c) ++i;
            ++i;
        }

        std::vector<int> ids;
        ids.reserve(phones.size());
        for (const std::string& phone : phones) {
            const int id = phone_id(phone);
            if (id >= 0) ids.push_back(id);
        }
        return ids;
    }

private:
    static bool is_vowel(char c) {
        return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
    }
    static bool is_front_vowel(char c) { return c == 'E' || c == 'I' || c == 'Y'; }
    static bool at_end_after(const std::string& text, size_t index) {
        return index + 1 >= text.size() || !is_vowel(text[index + 1]);
    }
    static const char* vowel_phone(char c, bool magic_e) {
        switch (c) {
            case 'A': return magic_e ? "EY" : "AE";
            case 'E': return magic_e ? "IY" : "EH";
            case 'I': return magic_e ? "AY" : "IH";
            case 'O': return magic_e ? "OW" : "AA";
            case 'U': return magic_e ? "UW" : "AH";
            default: return "AH";
        }
    }
};

}  // namespace models
