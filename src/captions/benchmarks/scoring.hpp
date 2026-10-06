// Evaluation utilities: Levenshtein alignment, word/phone error rate and a
// real-time-factor timer. Kept separate from the models so the benchmark can
// score the new decoder and the old frame-level baselines with one implementation.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace models {

struct EditCounts {
    int64_t substitutions = 0;
    int64_t deletions = 0;
    int64_t insertions = 0;
    int64_t reference_length = 0;

    int64_t errors() const { return substitutions + deletions + insertions; }
    double error_rate() const {
        return reference_length == 0 ? 0.0
                                     : static_cast<double>(errors()) / reference_length;
    }
    void operator+=(const EditCounts& other) {
        substitutions += other.substitutions;
        deletions += other.deletions;
        insertions += other.insertions;
        reference_length += other.reference_length;
    }
};

// Levenshtein with backtrace-free counting of S/D/I (two-row DP).
template <typename T>
EditCounts edit_distance(const std::vector<T>& reference, const std::vector<T>& hypothesis) {
    const size_t n = reference.size();
    const size_t m = hypothesis.size();

    struct Cell {
        int32_t cost = 0, sub = 0, del = 0, ins = 0;
    };
    std::vector<Cell> previous(m + 1), current(m + 1);
    for (size_t j = 0; j <= m; ++j) {
        previous[j] = {static_cast<int32_t>(j), 0, 0, static_cast<int32_t>(j)};
    }

    for (size_t i = 1; i <= n; ++i) {
        current[0] = {static_cast<int32_t>(i), 0, static_cast<int32_t>(i), 0};
        for (size_t j = 1; j <= m; ++j) {
            const bool match = reference[i - 1] == hypothesis[j - 1];
            Cell best = previous[j - 1];
            best.cost += match ? 0 : 1;
            if (!match) ++best.sub;

            Cell deletion = previous[j];
            deletion.cost += 1;
            ++deletion.del;
            if (deletion.cost < best.cost) best = deletion;

            Cell insertion = current[j - 1];
            insertion.cost += 1;
            ++insertion.ins;
            if (insertion.cost < best.cost) best = insertion;

            current[j] = best;
        }
        previous.swap(current);
    }

    EditCounts counts;
    counts.substitutions = previous[m].sub;
    counts.deletions = previous[m].del;
    counts.insertions = previous[m].ins;
    counts.reference_length = static_cast<int64_t>(n);
    return counts;
}

namespace detail {
inline const char* const kOnes[] = {"ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
                                    "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
                                    "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"};
inline const char* const kTens[] = {"", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"};
inline const char* const kOrdinals[] = {"ZEROTH", "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH",
                                        "EIGHTH", "NINTH", "TENTH", "ELEVENTH", "TWELFTH", "THIRTEENTH", "FOURTEENTH",
                                        "FIFTEENTH", "SIXTEENTH", "SEVENTEENTH", "EIGHTEENTH", "NINETEENTH"};

// Value of a spoken number word below one hundred, or -1.
inline int small_number(const std::string& w) {
    for (int i = 0; i < 20; ++i) if (w == kOnes[i]) return i;
    for (int i = 2; i < 10; ++i) if (w == kTens[i]) return i * 10;
    return -1;
}
inline long long scale_word(const std::string& w) {
    if (w == "HUNDRED") return 100;
    if (w == "THOUSAND") return 1000;
    if (w == "MILLION") return 1000000;
    if (w == "BILLION") return 1000000000;
    if (w == "TRILLION") return 1000000000000LL;
    return 0;
}
inline bool digit_word(const std::vector<std::string>& t, size_t i) {
    return i < t.size() && small_number(t[i]) >= 0 && small_number(t[i]) < 10;
}

// Spoken numbers -> digits, so "nineteen seventy five", "one thousand nine
// hundred and seventy five" and "1975" all compare equal. A word that cannot
// extend the running value starts a new digit group that is concatenated
// ("nineteen" + "seventy five" -> 1975, "twenty twenty" -> 2020); "oh"/"o"
// inside a number is a zero digit; "point" starts decimal digits.
inline void spoken_numbers_to_digits(std::vector<std::string>& tokens) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < tokens.size()) {
        // A run starts with a number word or a multiplier other than "hundred".
        if (small_number(tokens[i]) < 0 && (scale_word(tokens[i]) == 0 || tokens[i] == "HUNDRED")) {
            out.push_back(tokens[i++]);
            continue;
        }
        std::string digits;
        long long total = 0, current = 0;
        enum Kind { kNone, kUnit, kTeen, kTens, kScale } last = kNone;
        bool open = false;  // a value is pending in total/current
        auto flush = [&] {
            if (open) digits += std::to_string(total + current);
            total = current = 0;
            open = false;
        };
        for (; i < tokens.size(); ++i) {
            const auto& w = tokens[i];
            const int v = small_number(w);
            const long long scale = scale_word(w);
            if (v >= 0) {
                const bool joins = (last == kTens && v < 10) || last == kScale || last == kNone;
                if (!joins) flush();
                current += v;
                open = true;
                last = v < 10 ? kUnit : v < 20 ? kTeen : kTens;
            } else if (scale) {
                if (scale == 100) current = (current ? current : 1) * 100;
                else { total += (current ? current : 1) * scale; current = 0; }
                open = true;
                last = kScale;
            } else if (w == "AND" && open && last == kScale && i + 1 < tokens.size() &&
                       small_number(tokens[i + 1]) >= 0) {
                continue;  // "one hundred and five"
            } else if ((w == "O" || w == "OH") && open && digit_word(tokens, i + 1)) {
                flush();
                digits += '0';  // "nineteen o five"
                last = kNone;
            } else if (w == "POINT" && open && digit_word(tokens, i + 1)) {
                flush();
                digits += '.';
                while (digit_word(tokens, i + 1)) digits += std::to_string(small_number(tokens[++i]));
                ++i;
                break;
            } else {
                break;
            }
        }
        flush();
        if (!digits.empty()) out.push_back(digits);
    }
    tokens = std::move(out);
}

// "1,975" -> 1975; ordinals stay words: "3rd" -> THIRD, "21st" -> TWENTY FIRST.
inline bool digit_token(std::string token, std::vector<std::string>& out) {
    std::string suffix;
    for (const char* s : {"ST", "ND", "RD", "TH"})
        if (token.size() > 2 && token.ends_with(s) && std::isdigit(static_cast<unsigned char>(token[token.size() - 3]))) {
            suffix = s;
            token.resize(token.size() - 2);
            break;
        }
    std::string digits;
    for (char c : token) {
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') digits.push_back(c);
        else if (c != ',') return false;
    }
    if (digits.empty()) return false;
    if (suffix.empty()) { out.push_back(digits); return true; }
    if (digits.find('.') != std::string::npos || digits.size() > 2) return false;
    const int n = std::stoi(digits);
    if (n < 20) out.push_back(kOrdinals[n]);
    else if (n % 10) { out.push_back(kTens[n / 10]); out.push_back(kOrdinals[n % 10]); }
    else { std::string t = kTens[n / 10]; t.pop_back(); out.push_back(t + "IETH"); }
    return true;
}
} // namespace detail

// Word tokens for WER, normalized so that spelling conventions do not count
// as errors: uppercase, punctuation removed (apostrophes inside words kept),
// hyphens split, bracketed tags (<unk>, [noise]) and hesitations dropped,
// honorifics shortened (mister -> MR), and numbers canonicalized to digits
// ("one thousand nine hundred and seventy five" = "1975", "$5" -> 5 DOLLARS,
// "20%" -> 20 PERCENT). Applied to references and hypotheses alike, so it is
// neutral for transcripts that already agree on conventions.
inline std::vector<std::string> tokenize(const std::string& raw) {
    static const char* const fillers[] = {"UH", "UM", "UMM", "UHM", "HMM", "MM", "MMM", "MHM", "AH", "ER", "ERM"};
    std::string text;
    for (size_t i = 0; i < raw.size(); ++i) {
        // Typographic apostrophe (U+2019) -> ASCII.
        if (static_cast<unsigned char>(raw[i]) == 0xE2 && i + 2 < raw.size() &&
            static_cast<unsigned char>(raw[i + 1]) == 0x80 && static_cast<unsigned char>(raw[i + 2]) == 0x99) {
            text += '\'';
            i += 2;
            continue;
        }
        text += raw[i];
    }
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        if ((word.front() == '<' && word.back() == '>') || (word.front() == '[' && word.back() == ']')) continue;
        // Split on characters that cannot be part of a word or a number.
        std::string piece;
        std::vector<std::string> pieces;
        bool dollars = false, percent = false;
        for (size_t i = 0; i < word.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(word[i]);
            const bool digit_neighbours = i > 0 && i + 1 < word.size() &&
                std::isdigit(static_cast<unsigned char>(word[i - 1])) && std::isdigit(static_cast<unsigned char>(word[i + 1]));
            if (std::isalnum(c) || c == '\'' || ((c == '.' || c == ',') && digit_neighbours)) {
                piece.push_back(static_cast<char>(std::toupper(c)));
                continue;
            }
            if (c == '$') dollars = true;
            if (c == '%') percent = true;
            if (c == '&') { if (!piece.empty()) pieces.push_back(piece); piece = "AND"; }
            if (!piece.empty()) { pieces.push_back(piece); piece.clear(); }
        }
        if (!piece.empty()) pieces.push_back(piece);
        for (auto& p : pieces) {
            while (!p.empty() && p.front() == '\'') p.erase(p.begin());
            while (!p.empty() && p.back() == '\'') p.pop_back();
            if (p.empty()) continue;
            if (std::isdigit(static_cast<unsigned char>(p.front())) && detail::digit_token(p, tokens)) continue;
            if (std::find(std::begin(fillers), std::end(fillers), p) != std::end(fillers)) continue;
            // Honorifics are written both ways (Mr / mister); compare the short form.
            if (p == "MISTER") p = "MR";
            else if (p == "MISSUS" || p == "MISSES") p = "MRS";
            tokens.push_back(p);
        }
        if (percent) tokens.push_back("PERCENT");
        if (dollars) tokens.push_back("DOLLARS");
    }
    detail::spoken_numbers_to_digits(tokens);
    return tokens;
}

// Wall-clock timer that also reports the real-time factor of processed audio.
class RtfTimer {
public:
    void start() { start_ = std::chrono::steady_clock::now(); }
    double stop() {
        const auto end = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(end - start_).count();
        elapsed_ += seconds;
        return seconds;
    }
    void add_audio(double seconds) { audio_ += seconds; }

    double elapsed() const { return elapsed_; }
    double audio() const { return audio_; }
    double rtf() const { return audio_ > 0.0 ? elapsed_ / audio_ : 0.0; }
    double times_real_time() const { return elapsed_ > 0.0 ? audio_ / elapsed_ : 0.0; }

private:
    std::chrono::steady_clock::time_point start_;
    double elapsed_ = 0.0;
    double audio_ = 0.0;
};

}  // namespace models
