// Evaluation utilities: Levenshtein alignment, word/phone error rate and a
// real-time-factor timer. Kept separate from the models so the benchmark can
// score the new decoder and the old frame-level baselines with one implementation.
#pragma once

#include <algorithm>
#include <cctype>
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

// Uppercase whitespace tokenisation, stripping punctuation that LibriSpeech
// transcripts do not contain but CMUDict spellings would trip over.
inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string token;
    while (stream >> token) {
        std::string cleaned;
        for (char c : token) {
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '\'') {
                cleaned.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
            }
        }
        if (!cleaned.empty()) tokens.push_back(cleaned);
    }
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
