#pragma once

// The relational half of the compiled knowledge base: an Attributed Relational
// Graph over glyph archetypes.
//
// A node is an opaque integer identity supplied by the caller -- a semantic
// class for the whole, a shape archetype for the part, or whatever alphabet
// the task calls for. An edge (parent -> child) records a part-whole
// containment that was actually observed during ingest, together with running
// statistics for the attributes that distinguish a real part from a
// coincidental overlap: how much of the parent the child covers, how far their
// colours differ, whether they are the same material, where the child sits
// inside the parent, and how hard the boundary between them is.
//
// At runtime a candidate classification is scored against those statistics.
// Every constraint reports its own agreement term, so a decision can be read
// back as a sentence rather than a number. That is the whole point of paying
// for a symbolic layer instead of a second neural net.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace structures {

// One measured part-whole containment.
struct RelationObservation {
    float area_ratio = 0.0f;        // child area / parent area
    float delta_e = 0.0f;           // CIELAB distance between mean colours
    float material_distance = 0.0f; // 0 same material, 1 unrelated
    float radial_offset = 0.0f;     // |child centroid - parent centroid| / parent radius
    float boundary_contrast = 0.0f; // mean Lab step across the shared border
    float enclosure = 0.0f;         // fraction of the child's perimeter facing the parent
};

// Welford accumulator so the statistics stay stable and single-pass.
struct AttributeStats {
    int count = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;

    void observe(double value) {
        if (count == 0) {
            minimum = value;
            maximum = value;
        } else {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        ++count;
        const double delta = value - mean;
        mean += delta / count;
        m2 += delta * (value - mean);
    }

    double variance() const { return count > 1 ? m2 / (count - 1) : 0.0; }
    double deviation() const { return std::sqrt(variance()); }

    // Agreement in [0, 1]. A z-score is meaningless until a handful of samples
    // have been seen, and a constraint observed twice must not be allowed to
    // veto anything, so the floor on the spread is deliberately generous.
    float agreement(double value, double floor_deviation) const {
        if (count < 2) {
            return 0.5f;
        }
        const double spread = std::max(deviation(), floor_deviation);
        const double z = (value - mean) / spread;
        return static_cast<float>(std::exp(-0.5 * z * z));
    }
};

struct RelationEdge {
    int parent_node = 0;
    int child_node = 0;
    int support = 0;
    AttributeStats area_ratio;
    AttributeStats delta_e;
    AttributeStats material_distance;
    AttributeStats radial_offset;
    AttributeStats boundary_contrast;
    AttributeStats enclosure;
    std::map<int, int> child_class_histogram;
    int dominant_child_class = -1;
    // Share of this edge's children that carried the dominant class. An edge
    // whose parts were three different things describes a geometry, not a
    // meaning, and must not be allowed to name anything.
    float child_class_purity = 0.0f;
};

// Per-constraint breakdown, kept so a classification can explain itself.
struct ValidationTerm {
    const char* name = "";
    float agreement = 0.0f;
    float observed = 0.0f;
    float expected = 0.0f;
};

struct ValidationResult {
    bool edge_known = false;
    float support_weight = 0.0f;  // confidence that this edge is well attested
    float score = 0.0f;           // combined agreement in [0, 1]
    std::vector<ValidationTerm> terms;
    int predicted_child_class = -1;
    float class_purity = 0.0f;  // how consistently this edge names its child

    std::string explain() const {
        std::ostringstream out;
        if (!edge_known) {
            return "no attested part-whole relation between these glyphs";
        }
        out.setf(std::ios::fixed);
        out.precision(2);
        for (size_t i = 0; i < terms.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << terms[i].name << " " << terms[i].observed << " vs expected "
                << terms[i].expected << " (" << terms[i].agreement << ")";
        }
        return out.str();
    }
};

class RelationalDAG {
public:
    // A relation seen once or twice is an anecdote. Below this it still scores,
    // but its influence is scaled down rather than trusted outright.
    int confident_support = 6;

    void observe(int parent_node, int child_node, const RelationObservation& observation,
                 int child_class) {
        RelationEdge& edge = edges_[{parent_node, child_node}];
        edge.parent_node = parent_node;
        edge.child_node = child_node;
        ++edge.support;
        edge.area_ratio.observe(observation.area_ratio);
        edge.delta_e.observe(observation.delta_e);
        edge.material_distance.observe(observation.material_distance);
        edge.radial_offset.observe(observation.radial_offset);
        edge.boundary_contrast.observe(observation.boundary_contrast);
        edge.enclosure.observe(observation.enclosure);
        if (child_class >= 0) {
            ++edge.child_class_histogram[child_class];
        }
    }

    void finalize() {
        ordered_.clear();
        for (auto& entry : edges_) {
            RelationEdge& edge = entry.second;
            int total = 0;
            for (const auto& bucket : edge.child_class_histogram) {
                total += bucket.second;
                if (edge.dominant_child_class < 0 ||
                    bucket.second > edge.child_class_histogram.at(edge.dominant_child_class)) {
                    edge.dominant_child_class = bucket.first;
                }
            }
            if (total > 0 && edge.dominant_child_class >= 0) {
                edge.child_class_purity =
                    static_cast<float>(edge.child_class_histogram.at(edge.dominant_child_class)) /
                    static_cast<float>(total);
            }
            ordered_.push_back(&edge);
        }
        std::sort(ordered_.begin(), ordered_.end(), [](const RelationEdge* a, const RelationEdge* b) {
            if (a->support != b->support) {
                return a->support > b->support;
            }
            if (a->parent_node != b->parent_node) {
                return a->parent_node < b->parent_node;
            }
            return a->child_node < b->child_node;
        });
    }

    ValidationResult validate(int parent_node, int child_node,
                              const RelationObservation& observation) const {
        ValidationResult result;
        const auto it = edges_.find({parent_node, child_node});
        if (it == edges_.end()) {
            return result;
        }
        const RelationEdge& edge = it->second;
        result.edge_known = true;
        result.support_weight =
            std::min(1.0f, static_cast<float>(edge.support) /
                               static_cast<float>(std::max(1, confident_support)));
        result.predicted_child_class = edge.dominant_child_class;
        result.class_purity = edge.child_class_purity;

        // Floors reflect the natural spread of each attribute, so a constraint
        // is never sharper than the quantity it measures deserves.
        add_term(result, "area_ratio", edge.area_ratio, observation.area_ratio, 0.05);
        add_term(result, "delta_e", edge.delta_e, observation.delta_e, 6.0);
        add_term(result, "material", edge.material_distance, observation.material_distance, 0.10);
        add_term(result, "offset", edge.radial_offset, observation.radial_offset, 0.20);
        add_term(result, "contrast", edge.boundary_contrast, observation.boundary_contrast, 3.0);
        add_term(result, "enclosure", edge.enclosure, observation.enclosure, 0.15);

        // Geometric mean, not arithmetic: one badly violated constraint should
        // sink the hypothesis rather than be averaged away by five satisfied
        // ones. That is what makes the DAG a filter and not a vote.
        double product = 1.0;
        for (const ValidationTerm& term : result.terms) {
            product *= std::max(1e-4f, term.agreement);
        }
        result.score = static_cast<float>(std::pow(product, 1.0 / result.terms.size()));
        return result;
    }

    const std::vector<const RelationEdge*>& ordered_edges() const { return ordered_; }
    size_t size() const { return edges_.size(); }

    int support_for(int parent_node, int child_node) const {
        const auto it = edges_.find({parent_node, child_node});
        return it == edges_.end() ? 0 : it->second.support;
    }

    // Share of this relation's observed children that carried `semantic`.
    // Retrieval proposes a handful of plausible identities; this is what lets
    // context re-rank all of them rather than nominate a single winner, which
    // is the difference between the graph advising and the graph overruling.
    float class_share(int parent_node, int child_node, int semantic) const {
        const auto it = edges_.find({parent_node, child_node});
        if (it == edges_.end() || semantic < 0) {
            return 0.0f;
        }
        const auto bucket = it->second.child_class_histogram.find(semantic);
        if (bucket == it->second.child_class_histogram.end()) {
            return 0.0f;
        }
        int total = 0;
        for (const auto& entry : it->second.child_class_histogram) {
            total += entry.second;
        }
        return total > 0 ? static_cast<float>(bucket->second) / static_cast<float>(total) : 0.0f;
    }

private:
    static void add_term(ValidationResult& result, const char* name, const AttributeStats& stats,
                         float observed, double floor_deviation) {
        ValidationTerm term;
        term.name = name;
        term.observed = observed;
        term.expected = static_cast<float>(stats.mean);
        term.agreement = stats.agreement(observed, floor_deviation);
        result.terms.push_back(term);
    }

    std::map<std::pair<int, int>, RelationEdge> edges_;
    std::vector<const RelationEdge*> ordered_;
};

}  // namespace structures
