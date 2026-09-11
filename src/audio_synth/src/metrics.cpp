#include "vocal/metrics.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>

namespace vocal::metrics {
namespace {

// 10 / ln(10) * sqrt(2). A literal is used because MSVC 19.44 does not yet
// accept std::log/std::sqrt in constant evaluation in this language mode.
constexpr double kMcdScale = 6.141851463713754;

double cepstral_distance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) {
        throw std::invalid_argument("MCEP frames must have the same non-zero dimension");
    }
    // Conventionally c0 is energy and is excluded from MCD.
    const std::size_t first = a.size() > 1 ? 1 : 0;
    double squared = 0.0;
    for (std::size_t i = first; i < a.size(); ++i) {
        const double delta = a[i] - b[i];
        squared += delta * delta;
    }
    return kMcdScale * std::sqrt(squared);
}

std::vector<std::string> words(std::string_view text) {
    std::string normalized;
    normalized.reserve(text.size());
    for (unsigned char c : text) {
        normalized.push_back(std::isalnum(c) || c == '\'' ? static_cast<char>(std::tolower(c)) : ' ');
    }
    std::istringstream stream(normalized);
    std::vector<std::string> result;
    for (std::string word; stream >> word;) result.push_back(std::move(word));
    return result;
}

}  // namespace

double mcd_db(const Frames& reference, const Frames& synthesized, bool use_dtw) {
    if (reference.empty() || synthesized.empty()) {
        throw std::invalid_argument("MCD requires non-empty feature sequences");
    }
    if (!use_dtw && reference.size() != synthesized.size()) {
        throw std::invalid_argument("unaligned MCD inputs require DTW");
    }
    if (!use_dtw) {
        double sum = 0.0;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            sum += cepstral_distance(reference[i], synthesized[i]);
        }
        return sum / static_cast<double>(reference.size());
    }

    const std::size_t n = reference.size(), m = synthesized.size();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> previous(m + 1, inf), current(m + 1, inf);
    std::vector<std::size_t> previous_steps(m + 1), current_steps(m + 1);
    previous[0] = 0.0;
    for (std::size_t i = 1; i <= n; ++i) {
        current.assign(m + 1, inf);
        current_steps.assign(m + 1, 0);
        for (std::size_t j = 1; j <= m; ++j) {
            double best = previous[j - 1];
            std::size_t steps = previous_steps[j - 1];
            if (previous[j] < best) { best = previous[j]; steps = previous_steps[j]; }
            if (current[j - 1] < best) { best = current[j - 1]; steps = current_steps[j - 1]; }
            current[j] = best + cepstral_distance(reference[i - 1], synthesized[j - 1]);
            current_steps[j] = steps + 1;
        }
        std::swap(previous, current);
        std::swap(previous_steps, current_steps);
    }
    return previous[m] / static_cast<double>(previous_steps[m]);
}

PitchScore pitch(std::span<const double> reference_hz, std::span<const double> synthesized_hz) {
    if (reference_hz.empty() || reference_hz.size() != synthesized_hz.size()) {
        throw std::invalid_argument("pitch contours must have equal non-zero length");
    }
    double squared = 0.0;
    std::size_t vuv_errors = 0, voiced = 0;
    for (std::size_t i = 0; i < reference_hz.size(); ++i) {
        const bool ref_voiced = reference_hz[i] > 0.0;
        const bool syn_voiced = synthesized_hz[i] > 0.0;
        vuv_errors += ref_voiced != syn_voiced;
        if (ref_voiced && syn_voiced) {
            const auto delta = reference_hz[i] - synthesized_hz[i];
            squared += delta * delta;
            ++voiced;
        }
    }
    return {voiced ? std::sqrt(squared / static_cast<double>(voiced)) : 0.0,
            static_cast<double>(vuv_errors) / static_cast<double>(reference_hz.size()), voiced};
}

double word_error_rate(std::string_view reference, std::string_view hypothesis) {
    const auto ref = words(reference), hyp = words(hypothesis);
    if (ref.empty()) return hyp.empty() ? 0.0 : 1.0;
    std::vector<std::size_t> previous(hyp.size() + 1), current(hyp.size() + 1);
    for (std::size_t j = 0; j <= hyp.size(); ++j) previous[j] = j;
    for (std::size_t i = 1; i <= ref.size(); ++i) {
        current[0] = i;
        for (std::size_t j = 1; j <= hyp.size(); ++j) {
            current[j] = std::min({previous[j] + 1, current[j - 1] + 1,
                                   previous[j - 1] + (ref[i - 1] == hyp[j - 1] ? 0U : 1U)});
        }
        std::swap(previous, current);
    }
    return static_cast<double>(previous.back()) / static_cast<double>(ref.size());
}

}  // namespace vocal::metrics
