#pragma once

// The shape half of the compiled knowledge base: a vocabulary of glyph
// archetypes plus a metric index for retrieving the nearest few.
//
// Glyphs are discovered, not enumerated by hand. Clustering the ingest corpus'
// shape descriptors yields the archetypes the corpus actually contains, and
// each cluster carries the statistics needed to explain a later match: how
// many regions supported it, how tight it is, and which semantic classes it
// was drawn from.
//
// Retrieval is a vantage-point tree. A VP-tree only needs a metric, not
// coordinates, so the same structure would serve if the descriptor were later
// swapped for one without a meaningful per-axis interpretation. Every step is
// deterministic: the clustering RNG is an explicit seeded LCG and all ties
// break toward the lower index.

#include "descriptors/shape_invariants/shape_invariants.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace structures {

// ---------------------------------------------------------------------------
// Vantage-point tree with bounded top-k search
// ---------------------------------------------------------------------------

class VectorIndex {
public:
    using Point = std::vector<float>;

    struct Neighbor {
        int id = -1;
        float distance = 0.0f;
    };

    static float l2(const Point& a, const Point& b) {
        float sum = 0.0f;
        const size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i) {
            const float d = a[i] - b[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    void build(std::vector<Point> points) {
        points_ = std::move(points);
        nodes_.clear();
        if (points_.empty()) {
            root_ = -1;
            return;
        }
        std::vector<int> order(points_.size());
        std::iota(order.begin(), order.end(), 0);
        uint32_t rng = 0x9e3779b9u;
        root_ = build_node(order, 0, static_cast<int>(order.size()), rng);
    }

    std::vector<Neighbor> query(const Point& target, int k) const {
        std::vector<Neighbor> heap;
        visited_ = 0;
        if (root_ >= 0 && k > 0) {
            float tau = std::numeric_limits<float>::max();
            search(root_, target, k, heap, tau);
        }
        std::sort(heap.begin(), heap.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.distance != b.distance ? a.distance < b.distance : a.id < b.id;
        });
        return heap;
    }

    size_t size() const { return points_.size(); }
    int visited_last_query() const { return visited_; }

private:
    struct Node {
        int point = -1;
        float threshold = 0.0f;
        int inside = -1;
        int outside = -1;
    };

    int build_node(std::vector<int>& order, int begin, int end, uint32_t& rng) {
        if (begin >= end) {
            return -1;
        }
        // Deterministic vantage-point choice from an explicit LCG.
        rng = rng * 1664525u + 1013904223u;
        const int pick = begin + static_cast<int>(rng % static_cast<uint32_t>(end - begin));
        std::swap(order[static_cast<size_t>(begin)], order[static_cast<size_t>(pick)]);

        const int node_index = static_cast<int>(nodes_.size());
        nodes_.push_back({});
        nodes_[static_cast<size_t>(node_index)].point = order[static_cast<size_t>(begin)];
        if (end - begin == 1) {
            return node_index;
        }

        const Point& vantage = points_[static_cast<size_t>(order[static_cast<size_t>(begin)])];
        const int median = (begin + 1 + end) / 2;
        std::nth_element(order.begin() + begin + 1, order.begin() + median, order.begin() + end,
                         [&](int a, int b) {
                             const float da = l2(vantage, points_[static_cast<size_t>(a)]);
                             const float db = l2(vantage, points_[static_cast<size_t>(b)]);
                             return da != db ? da < db : a < b;
                         });
        nodes_[static_cast<size_t>(node_index)].threshold =
            l2(vantage, points_[static_cast<size_t>(order[static_cast<size_t>(median)])]);
        nodes_[static_cast<size_t>(node_index)].inside = build_node(order, begin + 1, median, rng);
        nodes_[static_cast<size_t>(node_index)].outside = build_node(order, median, end, rng);
        return node_index;
    }

    void search(int node_index, const Point& target, int k, std::vector<Neighbor>& heap,
                float& tau) const {
        if (node_index < 0) {
            return;
        }
        const Node& node = nodes_[static_cast<size_t>(node_index)];
        ++visited_;
        const float d = l2(target, points_[static_cast<size_t>(node.point)]);
        if (static_cast<int>(heap.size()) < k || d < tau) {
            heap.push_back({node.point, d});
            std::sort(heap.begin(), heap.end(), [](const Neighbor& a, const Neighbor& b) {
                return a.distance != b.distance ? a.distance < b.distance : a.id < b.id;
            });
            if (static_cast<int>(heap.size()) > k) {
                heap.resize(static_cast<size_t>(k));
            }
            if (static_cast<int>(heap.size()) == k) {
                tau = heap.back().distance;
            }
        }
        if (node.inside < 0 && node.outside < 0) {
            return;
        }
        // Triangle inequality prunes whichever shell cannot hold anything
        // closer than the current k-th best.
        if (d < node.threshold) {
            search(node.inside, target, k, heap, tau);
            if (d + tau >= node.threshold) {
                search(node.outside, target, k, heap, tau);
            }
        } else {
            search(node.outside, target, k, heap, tau);
            if (d - tau <= node.threshold) {
                search(node.inside, target, k, heap, tau);
            }
        }
    }

    std::vector<Point> points_;
    std::vector<Node> nodes_;
    int root_ = -1;
    mutable int visited_ = 0;
};

// ---------------------------------------------------------------------------
// Glyph vocabulary
// ---------------------------------------------------------------------------

struct GlyphObservation {
    descriptors::ShapeDescriptor shape;
    int semantic_class = -1;
};

struct GlyphEntry {
    int id = 0;
    std::string archetype;
    std::vector<float> centroid;
    descriptors::GeometryTraits traits;
    int support = 0;
    float mean_radius = 0.0f;
    float max_radius = 0.0f;
    std::map<int, int> class_histogram;
    int dominant_class = -1;
    float purity = 0.0f;
};

class GlyphVocabulary {
public:
    int target_glyphs = 48;
    int iterations = 25;
    uint32_t seed = 0x5eed1234u;

    void fit(const std::vector<GlyphObservation>& observations) {
        glyphs_.clear();
        std::vector<std::vector<float>> points;
        points.reserve(observations.size());
        for (const GlyphObservation& observation : observations) {
            if (!observation.shape.valid) {
                continue;
            }
            const auto flat = observation.shape.flatten();
            points.emplace_back(flat.begin(), flat.end());
        }
        if (points.empty()) {
            return;
        }
        const int k = std::min(target_glyphs, static_cast<int>(points.size()));
        std::vector<std::vector<float>> centers = seed_centers(points, k);
        std::vector<int> assignment(points.size(), 0);

        for (int iteration = 0; iteration < iterations; ++iteration) {
            bool moved = false;
            for (size_t p = 0; p < points.size(); ++p) {
                int best = 0;
                float best_distance = std::numeric_limits<float>::max();
                for (int c = 0; c < k; ++c) {
                    const float d = VectorIndex::l2(points[p], centers[static_cast<size_t>(c)]);
                    if (d < best_distance) {
                        best_distance = d;
                        best = c;
                    }
                }
                if (assignment[p] != best) {
                    assignment[p] = best;
                    moved = true;
                }
            }
            std::vector<std::vector<double>> sums(
                static_cast<size_t>(k), std::vector<double>(points[0].size(), 0.0));
            std::vector<int> counts(static_cast<size_t>(k), 0);
            for (size_t p = 0; p < points.size(); ++p) {
                const size_t c = static_cast<size_t>(assignment[p]);
                for (size_t d = 0; d < points[p].size(); ++d) {
                    sums[c][d] += points[p][d];
                }
                ++counts[c];
            }
            for (int c = 0; c < k; ++c) {
                if (counts[static_cast<size_t>(c)] == 0) {
                    continue;
                }
                for (size_t d = 0; d < centers[static_cast<size_t>(c)].size(); ++d) {
                    centers[static_cast<size_t>(c)][d] = static_cast<float>(
                        sums[static_cast<size_t>(c)][d] / counts[static_cast<size_t>(c)]);
                }
            }
            if (!moved) {
                break;
            }
        }

        // Drop empty clusters so every surviving glyph id has real support.
        std::vector<int> remap(static_cast<size_t>(k), -1);
        for (int c = 0; c < k; ++c) {
            const int count = static_cast<int>(
                std::count(assignment.begin(), assignment.end(), c));
            if (count == 0) {
                continue;
            }
            GlyphEntry entry;
            entry.id = static_cast<int>(glyphs_.size());
            entry.centroid = centers[static_cast<size_t>(c)];
            entry.support = count;
            remap[static_cast<size_t>(c)] = entry.id;
            glyphs_.push_back(std::move(entry));
        }

        // Second pass over the members: radius statistics, the class histogram
        // and the traits of the member nearest the centroid, which is what the
        // archetype gets named from (a mean of traits can describe a shape that
        // no member actually has).
        std::vector<float> best_distance(glyphs_.size(), std::numeric_limits<float>::max());
        size_t valid_index = 0;
        for (const GlyphObservation& observation : observations) {
            if (!observation.shape.valid) {
                continue;
            }
            const int cluster = remap[static_cast<size_t>(assignment[valid_index])];
            const float d = VectorIndex::l2(points[valid_index],
                                            glyphs_[static_cast<size_t>(cluster)].centroid);
            GlyphEntry& entry = glyphs_[static_cast<size_t>(cluster)];
            entry.mean_radius += d;
            entry.max_radius = std::max(entry.max_radius, d);
            if (observation.semantic_class >= 0) {
                ++entry.class_histogram[observation.semantic_class];
            }
            if (d < best_distance[static_cast<size_t>(cluster)]) {
                best_distance[static_cast<size_t>(cluster)] = d;
                entry.traits = observation.shape.traits;
            }
            ++valid_index;
        }
        for (GlyphEntry& entry : glyphs_) {
            if (entry.support > 0) {
                entry.mean_radius /= static_cast<float>(entry.support);
            }
            entry.archetype = descriptors::name_archetype(entry.traits);
            int total = 0;
            for (const auto& bucket : entry.class_histogram) {
                total += bucket.second;
                if (entry.dominant_class < 0 ||
                    bucket.second > entry.class_histogram.at(entry.dominant_class)) {
                    entry.dominant_class = bucket.first;
                }
            }
            if (total > 0 && entry.dominant_class >= 0) {
                entry.purity = static_cast<float>(entry.class_histogram.at(entry.dominant_class)) /
                               static_cast<float>(total);
            }
        }

        std::vector<VectorIndex::Point> centroids;
        centroids.reserve(glyphs_.size());
        for (const GlyphEntry& entry : glyphs_) {
            centroids.push_back(entry.centroid);
        }
        index_.build(std::move(centroids));
    }

    std::vector<VectorIndex::Neighbor> query(const descriptors::ShapeDescriptor& shape,
                                             int k) const {
        if (!shape.valid) {
            return {};
        }
        const auto flat = shape.flatten();
        return index_.query(VectorIndex::Point(flat.begin(), flat.end()), k);
    }

    const std::vector<GlyphEntry>& glyphs() const { return glyphs_; }
    const VectorIndex& index() const { return index_; }

private:
    // k-means++ seeding: each new center is drawn with probability proportional
    // to its squared distance from the nearest existing one, which avoids the
    // degenerate initializations that make plain random seeding produce empty
    // or duplicated clusters.
    std::vector<std::vector<float>> seed_centers(const std::vector<std::vector<float>>& points,
                                                 int k) const {
        std::vector<std::vector<float>> centers;
        uint32_t rng = seed;
        auto next = [&rng]() {
            rng = rng * 1664525u + 1013904223u;
            return static_cast<double>(rng >> 8) / static_cast<double>(1u << 24);
        };
        centers.push_back(points[static_cast<size_t>(next() * (points.size() - 1))]);
        std::vector<double> nearest(points.size(), std::numeric_limits<double>::max());
        while (static_cast<int>(centers.size()) < k) {
            double total = 0.0;
            for (size_t p = 0; p < points.size(); ++p) {
                const double d = VectorIndex::l2(points[p], centers.back());
                nearest[p] = std::min(nearest[p], d * d);
                total += nearest[p];
            }
            if (total <= 0.0) {
                break;
            }
            double pick = next() * total;
            size_t chosen = points.size() - 1;
            for (size_t p = 0; p < points.size(); ++p) {
                pick -= nearest[p];
                if (pick <= 0.0) {
                    chosen = p;
                    break;
                }
            }
            centers.push_back(points[chosen]);
        }
        return centers;
    }

    std::vector<GlyphEntry> glyphs_;
    VectorIndex index_;
};

}  // namespace structures
