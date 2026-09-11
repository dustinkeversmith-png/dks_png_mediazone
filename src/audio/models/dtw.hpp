// Dynamic Time Warping over MFCC trajectories, plus a 1-NN template recogniser
// for isolated words (Tier 1: spoken digits).
//
// This is Step 1 of models/README.md: instead of collapsing repeated frames
// with `if (vowel == previous_vowel) continue;`, we align the whole feature
// trajectory against reference templates, which absorbs speaking-rate variation
// without any trained weights.
//
// Speed notes (the "optimality" half of this file):
//   * Sakoe-Chiba band: only |i*ratio - j| <= radius cells are visited, turning
//     the O(N*M) grid into a diagonal ribbon.
//   * Two-row rolling buffer: memory is O(min(N,M)) instead of O(N*M), so a
//     comparison stays in L1/L2 cache.
//   * Early abandoning: if the best cell in a row already exceeds the best
//     distance found so far, the comparison cannot win and is dropped.
//   * Envelope lower bound (LB_Keogh-style, on frame means): a cheap O(T) bound
//     that skips most templates before any DTW cell is touched.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "mfcc.hpp"

namespace models {

// Squared euclidean distance between two frames, unrolled in blocks of 4.
inline float frame_distance(const float* a, const float* b, int dim) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int d = 0;
    for (; d + 4 <= dim; d += 4) {
        const float d0 = a[d] - b[d];
        const float d1 = a[d + 1] - b[d + 1];
        const float d2 = a[d + 2] - b[d + 2];
        const float d3 = a[d + 3] - b[d + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    float sum = s0 + s1 + s2 + s3;
    for (; d < dim; ++d) {
        const float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

// Banded DTW with early abandoning. Returns the length-normalised path cost,
// or `abandon_above` if the alignment provably cannot beat that threshold.
class DtwAligner {
public:
    float distance(const FeatureMatrix& query, const FeatureMatrix& reference,
                   int radius = 12,
                   float abandon_above = std::numeric_limits<float>::infinity()) {
        const int n = query.num_frames;
        const int m = reference.num_frames;
        if (n == 0 || m == 0) return std::numeric_limits<float>::infinity();
        const int dim = query.dim;

        // Band follows the diagonal of a non-square grid.
        const float ratio = static_cast<float>(m) / static_cast<float>(n);
        const int band = std::max(radius, std::abs(n - m) / 2 + radius);

        constexpr float kInf = std::numeric_limits<float>::infinity();
        previous_.assign(m + 1, kInf);
        current_.assign(m + 1, kInf);
        previous_[0] = 0.0f;

        for (int i = 1; i <= n; ++i) {
            const int center = static_cast<int>((i - 1) * ratio) + 1;
            const int lo = std::max(1, center - band);
            const int hi = std::min(m, center + band);

            std::fill(current_.begin(), current_.end(), kInf);
            const float* query_frame = query.frame(i - 1);

            float row_best = kInf;
            for (int j = lo; j <= hi; ++j) {
                const float best_predecessor =
                    std::min(previous_[j], std::min(current_[j - 1], previous_[j - 1]));
                if (best_predecessor == kInf) continue;
                const float cost = frame_distance(query_frame, reference.frame(j - 1), dim);
                const float value = cost + best_predecessor;
                current_[j] = value;
                if (value < row_best) row_best = value;
            }

            // Every remaining path passes through this row, so a row minimum
            // above the incumbent best is a proof of loss.
            if (row_best >= abandon_above * (n + m)) return kInf;
            previous_.swap(current_);
        }

        const float total = previous_[m];
        if (total == kInf) return kInf;
        return total / static_cast<float>(n + m);
    }

private:
    std::vector<float> previous_;
    std::vector<float> current_;
};

// One stored reference utterance.
struct Template {
    std::string label;
    FeatureMatrix features;
};

// LB_Kim lower bound on the *normalised* DTW cost.
//
// Every warping path must contain (1,1) and (N,M), so the aligned cost is at
// least the cost of those two pairs. Dividing by (N+M) - the same denominator
// the aligner uses - keeps the bound admissible, so pruning on it can never
// discard the true nearest neighbour. Cheap: O(D), independent of length.
inline float lower_bound_kim(const FeatureMatrix& query, const FeatureMatrix& reference) {
    if (query.empty() || reference.empty()) return 0.0f;
    const int dim = query.dim;
    const float first = frame_distance(query.frame(0), reference.frame(0), dim);
    const float last = frame_distance(query.frame(query.num_frames - 1),
                                      reference.frame(reference.num_frames - 1), dim);
    return (first + last) / static_cast<float>(query.num_frames + reference.num_frames);
}

// Band-aware LB_Keogh over the first `dims` cepstral coefficients.
//
// Why it is admissible: any warping path maps query frame i to at least one
// reference frame inside the Sakoe-Chiba band [lo_i, hi_i], so the path cost is
// at least sum_i min_{j in band} d(q_i, r_j). Replacing that inner minimum with
// the distance to the band's per-dimension envelope, and summing over a subset
// of dimensions, can only lower the value further - so the result never exceeds
// the true DTW cost and pruning on it stays exact.
//
// Why it is cheap: lo_i and hi_i are non-decreasing, so the envelope is
// maintained with monotonic deques in O(n + m) per dimension - roughly two
// orders of magnitude less work than the full 39-dimensional alignment.
inline float lower_bound_keogh(const FeatureMatrix& query, const FeatureMatrix& reference,
                               int radius, int dims = 4) {
    const int n = query.num_frames;
    const int m = reference.num_frames;
    if (n == 0 || m == 0) return 0.0f;
    dims = std::min(dims, query.dim);

    const float ratio = static_cast<float>(m) / static_cast<float>(n);
    const int band = std::max(radius, std::abs(n - m) / 2 + radius);

    static thread_local std::vector<int> max_deque, min_deque;
    float total = 0.0f;

    for (int d = 0; d < dims; ++d) {
        max_deque.clear();
        min_deque.clear();
        // Head indices instead of erase(begin()): popping the front of a
        // vector is O(size), which would make this bound cost more than the
        // alignment it is meant to avoid.
        size_t max_head = 0, min_head = 0;
        int filled = 0;  // reference frames pushed so far

        for (int i = 0; i < n; ++i) {
            const int center = static_cast<int>(i * ratio);
            const int lo = std::max(0, center - band);
            const int hi = std::min(m - 1, center + band);

            while (filled <= hi) {
                const float value = reference.frame(filled)[d];
                while (max_deque.size() > max_head &&
                       reference.frame(max_deque.back())[d] <= value) {
                    max_deque.pop_back();
                }
                max_deque.push_back(filled);
                while (min_deque.size() > min_head &&
                       reference.frame(min_deque.back())[d] >= value) {
                    min_deque.pop_back();
                }
                min_deque.push_back(filled);
                ++filled;
            }
            while (max_deque.size() > max_head && max_deque[max_head] < lo) ++max_head;
            while (min_deque.size() > min_head && min_deque[min_head] < lo) ++min_head;
            if (max_deque.size() == max_head || min_deque.size() == min_head) continue;

            const float upper = reference.frame(max_deque[max_head])[d];
            const float lower = reference.frame(min_deque[min_head])[d];
            const float value = query.frame(i)[d];
            if (value > upper) {
                const float diff = value - upper;
                total += diff * diff;
            } else if (value < lower) {
                const float diff = lower - value;
                total += diff * diff;
            }
        }
    }
    return total / static_cast<float>(n + m);
}

// Energy-based endpointing. Isolated-digit clips are padded to a fixed length
// with leading/trailing silence; warping silence against silence is both slow
// and a source of false matches, so trim to the spoken region first.
// Operates on c0 (frame log-energy), which is dimension 0 of the MFCC vector.
inline FeatureMatrix endpoint(const FeatureMatrix& features, float threshold = 0.35f,
                              int margin = 3) {
    if (features.num_frames < 5) return features;

    float low = features.frame(0)[0];
    float high = low;
    for (int t = 1; t < features.num_frames; ++t) {
        low = std::min(low, features.frame(t)[0]);
        high = std::max(high, features.frame(t)[0]);
    }
    if (high - low < 1e-3f) return features;

    const float cutoff = low + threshold * (high - low);
    int start = 0, stop = features.num_frames - 1;
    while (start < features.num_frames && features.frame(start)[0] < cutoff) ++start;
    while (stop > start && features.frame(stop)[0] < cutoff) --stop;

    start = std::max(0, start - margin);
    stop = std::min(features.num_frames - 1, stop + margin);
    if (stop - start + 1 < 5) return features;

    FeatureMatrix trimmed;
    trimmed.resize(stop - start + 1, features.dim);
    std::copy(features.frame(start), features.frame(stop) + features.dim,
              trimmed.data.begin());
    return trimmed;
}

// 1-NN / k-NN isolated word recogniser over stored templates.
class DtwRecognizer {
public:
    struct Result {
        std::string label;
        float distance = std::numeric_limits<float>::infinity();
        int templates_scored = 0;   // how many survived the bound
        int templates_pruned = 0;   // skipped by the admissible lower bound
        int templates_abandoned = 0;  // DTW started but proven hopeless mid-way
    };

    void add_template(const std::string& label, FeatureMatrix features) {
        Template entry;
        entry.label = label;
        entry.features = std::move(features);
        templates_.push_back(std::move(entry));
    }

    size_t size() const { return templates_.size(); }
    const std::vector<Template>& templates() const { return templates_; }

    // k-NN with distance-weighted vote; k=1 is plain nearest neighbour.
    //
    // Templates are visited in increasing lower-bound order so a tight `best`
    // is found early, which is what makes both the bound test and the DTW
    // early-abandon fire often.
    Result classify(const FeatureMatrix& query, int k = 1, int radius = 12,
                    bool use_lower_bound = false) {
        Result result;
        if (templates_.empty() || query.empty()) return result;

        // LB_Kim is always worth its O(D) cost: it orders the templates so a
        // tight incumbent appears early. LB_Keogh is opt-in - measured on
        // 39-dim CMVN features with a +/-12 frame band it prunes essentially
        // nothing while costing O(dims * (n + m)) per template, because the
        // per-dimension envelope over that many frames is almost never
        // violated. It pays off only for narrow bands or low-dim features.
        order_.clear();
        order_.reserve(templates_.size());
        for (size_t i = 0; i < templates_.size(); ++i) {
            const FeatureMatrix& reference = templates_[i].features;
            float bound = lower_bound_kim(query, reference);
            if (use_lower_bound) {
                bound = std::max(bound, lower_bound_keogh(query, reference, radius, 13));
            }
            order_.push_back({bound, i});
        }
        std::sort(order_.begin(), order_.end(),
                  [](const Bounded& a, const Bounded& b) { return a.bound < b.bound; });

        scored_.clear();
        float best = std::numeric_limits<float>::infinity();
        for (const Bounded& candidate : order_) {
            const Template& entry = templates_[candidate.index];
            // Admissible bound: if it already exceeds the incumbent, no
            // alignment of this template can win.
            if (k == 1 && candidate.bound >= best) {
                ++result.templates_pruned;
                continue;
            }
            const float distance = aligner_.distance(
                query, entry.features, radius,
                k == 1 ? best : std::numeric_limits<float>::infinity());
            ++result.templates_scored;
            if (distance == std::numeric_limits<float>::infinity()) {
                ++result.templates_abandoned;
                continue;
            }
            if (distance < best) best = distance;
            scored_.push_back({distance, &entry});
        }

        if (scored_.empty()) return result;
        const int neighbours = std::min<int>(k, static_cast<int>(scored_.size()));
        std::partial_sort(scored_.begin(), scored_.begin() + neighbours, scored_.end(),
                          [](const Scored& a, const Scored& b) { return a.distance < b.distance; });

        // Distance-weighted vote across the k nearest templates.
        std::vector<std::pair<std::string, float>> votes;
        for (int i = 0; i < neighbours; ++i) {
            const std::string& label = scored_[i].entry->label;
            const float weight = 1.0f / (scored_[i].distance + 1e-6f);
            auto it = std::find_if(votes.begin(), votes.end(),
                                   [&](const auto& v) { return v.first == label; });
            if (it == votes.end()) {
                votes.emplace_back(label, weight);
            } else {
                it->second += weight;
            }
        }
        const auto winner = std::max_element(
            votes.begin(), votes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        result.label = winner->first;
        result.distance = scored_[0].distance;
        return result;
    }

private:
    struct Scored {
        float distance;
        const Template* entry;
    };
    struct Bounded {
        float bound;
        size_t index;
    };

    std::vector<Template> templates_;
    std::vector<Scored> scored_;
    std::vector<Bounded> order_;
    DtwAligner aligner_;
};

}  // namespace models
