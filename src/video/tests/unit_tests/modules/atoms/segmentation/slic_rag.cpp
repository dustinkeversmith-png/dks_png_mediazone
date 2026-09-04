#include "test_harness.hpp"
#include "math/contour_metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace slic_rag {

struct Center {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
};

struct SlicResult {
    int width = 0;
    int height = 0;
    int requested_k = 0;
    int iterations_run = 0;
    int changed_last = 0;
    std::vector<Center> seeds;
    std::vector<Center> centers;
    std::vector<int> first_labels;
    std::vector<int> raw_labels;
    std::vector<int> labels;
    int raw_regions = 0;
    int connected_regions = 0;
};

class Slic {
public:
    int iterations = 10;
    float compactness = 15.0f;
    float orphan_fraction = 0.25f;

    SlicResult segment(const math::ImageBuffer& rgb, int requested_k) const {
        SlicResult result;
        result.width = rgb.width;
        result.height = rgb.height;
        result.requested_k = requested_k;
        const int w = rgb.width;
        const int h = rgb.height;
        const int n = w * h;
        if (rgb.empty() || n <= 0 || rgb.channels < 3) {
            return result;
        }

        const int k = std::clamp(requested_k, 1, n);
        const float spacing = std::sqrt(static_cast<float>(n) / static_cast<float>(k));
        const float inv_s2 = 1.0f / std::max(1.0f, spacing * spacing);
        const float m2 = compactness * compactness;

        std::vector<Center> centers;
        for (float y = spacing * 0.5f; y < static_cast<float>(h); y += spacing) {
            for (float x = spacing * 0.5f; x < static_cast<float>(w); x += spacing) {
                const int px = std::clamp(static_cast<int>(std::lround(x)), 0, w - 1);
                const int py = std::clamp(static_cast<int>(std::lround(y)), 0, h - 1);
                centers.push_back(sample_center(rgb, px, py));
            }
        }
        if (centers.empty()) {
            centers.push_back(sample_center(rgb, w / 2, h / 2));
        }
        result.seeds = centers;

        std::vector<int> labels(static_cast<size_t>(n), -1);
        std::vector<int> previous(static_cast<size_t>(n), -2);
        std::vector<float> distance(static_cast<size_t>(n));
        for (int iter = 0; iter < iterations; ++iter) {
            std::fill(distance.begin(), distance.end(), std::numeric_limits<float>::max());
            std::fill(labels.begin(), labels.end(), -1);

            for (int ci = 0; ci < static_cast<int>(centers.size()); ++ci) {
                const Center& c = centers[static_cast<size_t>(ci)];
                // A 2S x 2S local search window, centered on the cluster.
                const int x0 = std::max(0, static_cast<int>(std::floor(c.x - spacing)));
                const int x1 = std::min(w - 1, static_cast<int>(std::ceil(c.x + spacing)));
                const int y0 = std::max(0, static_cast<int>(std::floor(c.y - spacing)));
                const int y1 = std::min(h - 1, static_cast<int>(std::ceil(c.y + spacing)));
                for (int y = y0; y <= y1; ++y) {
                    for (int x = x0; x <= x1; ++x) {
                        const size_t p = static_cast<size_t>(y * w + x);
                        const float dr = static_cast<float>(rgb.at(x, y, 0)) - c.r;
                        const float dg = static_cast<float>(rgb.at(x, y, 1)) - c.g;
                        const float db = static_cast<float>(rgb.at(x, y, 2)) - c.b;
                        const float dc2 = dr * dr + dg * dg + db * db;
                        const float dx = static_cast<float>(x) - c.x;
                        const float dy = static_cast<float>(y) - c.y;
                        const float ds2 = dx * dx + dy * dy;
                        const float d = dc2 + m2 * inv_s2 * ds2;
                        if (d < distance[p]) {
                            distance[p] = d;
                            labels[p] = ci;
                        }
                    }
                }
            }

            // Defensive fallback for a pathological grid/window roundoff gap.
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t p = static_cast<size_t>(y * w + x);
                    if (labels[p] >= 0) {
                        continue;
                    }
                    int best = 0;
                    float best_d = std::numeric_limits<float>::max();
                    for (int ci = 0; ci < static_cast<int>(centers.size()); ++ci) {
                        const float dx = static_cast<float>(x) - centers[static_cast<size_t>(ci)].x;
                        const float dy = static_cast<float>(y) - centers[static_cast<size_t>(ci)].y;
                        const float d = dx * dx + dy * dy;
                        if (d < best_d) {
                            best_d = d;
                            best = ci;
                        }
                    }
                    labels[p] = best;
                }
            }

            if (iter == 0) {
                result.first_labels = labels;
            }
            int changed = 0;
            for (size_t p = 0; p < labels.size(); ++p) {
                changed += labels[p] != previous[p] ? 1 : 0;
            }

            std::vector<Center> sums(centers.size());
            std::vector<int> counts(centers.size(), 0);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int ci = labels[static_cast<size_t>(y * w + x)];
                    Center& a = sums[static_cast<size_t>(ci)];
                    a.r += rgb.at(x, y, 0);
                    a.g += rgb.at(x, y, 1);
                    a.b += rgb.at(x, y, 2);
                    a.x += static_cast<float>(x);
                    a.y += static_cast<float>(y);
                    ++counts[static_cast<size_t>(ci)];
                }
            }
            for (size_t ci = 0; ci < centers.size(); ++ci) {
                if (counts[ci] == 0) {
                    continue;
                }
                const float inv = 1.0f / static_cast<float>(counts[ci]);
                centers[ci] = {sums[ci].r * inv, sums[ci].g * inv, sums[ci].b * inv,
                               sums[ci].x * inv, sums[ci].y * inv};
            }

            previous = labels;
            result.iterations_run = iter + 1;
            result.changed_last = changed;
            if (iter >= 4 && changed == 0) {
                break;
            }
        }

        result.centers = centers;
        result.raw_labels = labels;
        result.raw_regions = static_cast<int>(centers.size());
        result.labels = enforce_connectivity(labels, w, h, spacing);
        result.connected_regions = dense_count(result.labels);
        return result;
    }

private:
    static Center sample_center(const math::ImageBuffer& rgb, int x, int y) {
        return {static_cast<float>(rgb.at(x, y, 0)), static_cast<float>(rgb.at(x, y, 1)),
                static_cast<float>(rgb.at(x, y, 2)), static_cast<float>(x),
                static_cast<float>(y)};
    }

    static int dense_count(const std::vector<int>& labels) {
        return labels.empty() ? 0 : *std::max_element(labels.begin(), labels.end()) + 1;
    }

    std::vector<int> enforce_connectivity(const std::vector<int>& input, int w, int h,
                                          float spacing) const {
        const int n = w * h;
        const int min_component =
            std::max(4, static_cast<int>(orphan_fraction * spacing * spacing));
        std::vector<int> output(static_cast<size_t>(n), -1);
        std::vector<uint8_t> visited(static_cast<size_t>(n), 0);
        std::vector<int> stack;
        std::vector<int> component;
        int next = 0;
        constexpr int dx[4] = {1, -1, 0, 0};
        constexpr int dy[4] = {0, 0, 1, -1};

        for (int seed = 0; seed < n; ++seed) {
            if (visited[static_cast<size_t>(seed)]) {
                continue;
            }
            const int source_label = input[static_cast<size_t>(seed)];
            stack.assign(1, seed);
            component.clear();
            visited[static_cast<size_t>(seed)] = 1;
            std::vector<std::pair<int, int>> adjacent;
            while (!stack.empty()) {
                const int p = stack.back();
                stack.pop_back();
                component.push_back(p);
                const int x = p % w;
                const int y = p / w;
                for (int d = 0; d < 4; ++d) {
                    const int xx = x + dx[d];
                    const int yy = y + dy[d];
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                        continue;
                    }
                    const int q = yy * w + xx;
                    if (input[static_cast<size_t>(q)] == source_label &&
                        !visited[static_cast<size_t>(q)]) {
                        visited[static_cast<size_t>(q)] = 1;
                        stack.push_back(q);
                    } else if (output[static_cast<size_t>(q)] >= 0) {
                        const int label = output[static_cast<size_t>(q)];
                        auto it = std::find_if(adjacent.begin(), adjacent.end(),
                                               [label](const auto& e) { return e.first == label; });
                        if (it == adjacent.end()) {
                            adjacent.push_back({label, 1});
                        } else {
                            ++it->second;
                        }
                    }
                }
            }

            int destination = next;
            if (static_cast<int>(component.size()) < min_component && !adjacent.empty()) {
                destination = std::max_element(
                                  adjacent.begin(), adjacent.end(),
                                  [](const auto& a, const auto& b) { return a.second < b.second; })
                                  ->first;
            } else {
                ++next;
            }
            for (int p : component) {
                output[static_cast<size_t>(p)] = destination;
            }
        }

        // The first scanned component can be an orphan without a processed
        // neighbor. Re-index after all absorptions to guarantee dense IDs.
        std::vector<int> remap(static_cast<size_t>(std::max(1, next)), -1);
        int dense = 0;
        for (int& label : output) {
            if (label >= static_cast<int>(remap.size())) {
                remap.resize(static_cast<size_t>(label + 1), -1);
            }
            if (remap[static_cast<size_t>(label)] < 0) {
                remap[static_cast<size_t>(label)] = dense++;
            }
            label = remap[static_cast<size_t>(label)];
        }
        return output;
    }
};

struct RegionNode {
    int area = 0;
    std::array<float, 3> mean{{0.0f, 0.0f, 0.0f}};
    std::array<float, 24> histogram{};
    std::set<int> neighbors;
    int version = 0;
    bool active = true;
};

struct RagEdge {
    int u = 0;
    int v = 0;
    float cost = 0.0f;
    int version_u = 0;
    int version_v = 0;

    bool operator>(const RagEdge& other) const { return cost > other.cost; }
};

struct RagResult {
    int initial_regions = 0;
    int final_regions = 0;
    int initial_edges = 0;
    int merges = 0;
    float last_merge_cost = 0.0f;
    std::vector<RegionNode> initial_nodes;
    std::vector<std::pair<int, int>> initial_edges_list;
    std::vector<float> initial_edge_costs;
    std::vector<int> middle_labels;
    std::vector<int> labels;
};

class RagHierarchicalMerger {
public:
    int target_regions = 20;
    float max_cost = 0.36f;
    float alpha_color = 0.80f;

    RagResult merge(const SlicResult& slic, const math::ImageBuffer& rgb) const {
        RagResult result;
        const int w = slic.width;
        const int h = slic.height;
        const int n = w * h;
        const int k = slic.connected_regions;
        if (n <= 0 || k <= 0) {
            return result;
        }

        std::vector<RegionNode> nodes(static_cast<size_t>(k));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int p = y * w + x;
                const int label = slic.labels[static_cast<size_t>(p)];
                RegionNode& node = nodes[static_cast<size_t>(label)];
                ++node.area;
                for (int c = 0; c < 3; ++c) {
                    const uint8_t value = rgb.at(x, y, c);
                    node.mean[static_cast<size_t>(c)] += value / 255.0f;
                    ++node.histogram[static_cast<size_t>(c * 8 + value / 32)];
                }
                if (x + 1 < w) {
                    connect(nodes, label, slic.labels[static_cast<size_t>(p + 1)]);
                }
                if (y + 1 < h) {
                    connect(nodes, label, slic.labels[static_cast<size_t>(p + w)]);
                }
            }
        }
        for (RegionNode& node : nodes) {
            normalize(node);
        }

        result.initial_regions = k;
        result.initial_nodes = nodes;
        for (int u = 0; u < k; ++u) {
            for (int v : nodes[static_cast<size_t>(u)].neighbors) {
                if (u < v) {
                    result.initial_edges_list.push_back({u, v});
                    result.initial_edge_costs.push_back(
                        cost(nodes[static_cast<size_t>(u)],
                             nodes[static_cast<size_t>(v)], n));
                }
            }
        }
        result.initial_edges = static_cast<int>(result.initial_edges_list.size());

        std::vector<int> parent(static_cast<size_t>(k));
        std::iota(parent.begin(), parent.end(), 0);
        auto root = [&parent](int x) {
            int r = x;
            while (parent[static_cast<size_t>(r)] != r) {
                r = parent[static_cast<size_t>(r)];
            }
            while (parent[static_cast<size_t>(x)] != x) {
                const int next = parent[static_cast<size_t>(x)];
                parent[static_cast<size_t>(x)] = r;
                x = next;
            }
            return r;
        };

        std::priority_queue<RagEdge, std::vector<RagEdge>, std::greater<RagEdge>> queue;
        auto push_edge = [&](int a, int b) {
            a = root(a);
            b = root(b);
            if (a == b || !nodes[static_cast<size_t>(a)].active ||
                !nodes[static_cast<size_t>(b)].active) {
                return;
            }
            if (a > b) {
                std::swap(a, b);
            }
            queue.push({a, b, cost(nodes[static_cast<size_t>(a)],
                                   nodes[static_cast<size_t>(b)], n),
                        nodes[static_cast<size_t>(a)].version,
                        nodes[static_cast<size_t>(b)].version});
        };
        for (const auto& edge : result.initial_edges_list) {
            push_edge(edge.first, edge.second);
        }

        int active = k;
        const int target = std::clamp(target_regions, 1, k);
        const int middle_target = target + (k - target) / 2;
        while (!queue.empty() && active > target) {
            const RagEdge candidate = queue.top();
            queue.pop();
            int u = root(candidate.u);
            int v = root(candidate.v);
            if (u == v || u != candidate.u || v != candidate.v) {
                continue;
            }
            RegionNode& a = nodes[static_cast<size_t>(u)];
            RegionNode& b = nodes[static_cast<size_t>(v)];
            // Reject stale costs. Every feature update increments the version.
            if (!a.active || !b.active || a.version != candidate.version_u ||
                b.version != candidate.version_v) {
                continue;
            }
            const float current_cost = cost(a, b, n);
            if (current_cost > max_cost) {
                break;
            }

            if (a.area < b.area) {
                std::swap(u, v);
            }
            RegionNode& keep = nodes[static_cast<size_t>(u)];
            RegionNode& drop = nodes[static_cast<size_t>(v)];
            const int combined_area = keep.area + drop.area;
            const float wk = static_cast<float>(keep.area) / combined_area;
            const float wd = static_cast<float>(drop.area) / combined_area;
            for (int c = 0; c < 3; ++c) {
                keep.mean[static_cast<size_t>(c)] =
                    wk * keep.mean[static_cast<size_t>(c)] +
                    wd * drop.mean[static_cast<size_t>(c)];
            }
            for (size_t bin = 0; bin < keep.histogram.size(); ++bin) {
                keep.histogram[bin] =
                    wk * keep.histogram[bin] + wd * drop.histogram[bin];
            }
            keep.area = combined_area;
            parent[static_cast<size_t>(v)] = u;
            drop.active = false;
            ++keep.version;
            ++drop.version;

            std::set<int> merged_neighbors;
            for (int neighbor : keep.neighbors) {
                const int r = root(neighbor);
                if (r != u && nodes[static_cast<size_t>(r)].active) {
                    merged_neighbors.insert(r);
                }
            }
            for (int neighbor : drop.neighbors) {
                const int r = root(neighbor);
                if (r != u && nodes[static_cast<size_t>(r)].active) {
                    merged_neighbors.insert(r);
                }
            }
            keep.neighbors = merged_neighbors;
            drop.neighbors.clear();
            for (int neighbor : keep.neighbors) {
                RegionNode& other = nodes[static_cast<size_t>(neighbor)];
                other.neighbors.erase(v);
                other.neighbors.erase(u);
                other.neighbors.insert(u);
                push_edge(u, neighbor);
            }

            --active;
            ++result.merges;
            result.last_merge_cost = current_cost;
            if (result.middle_labels.empty() && active <= middle_target) {
                result.middle_labels = relabel(slic.labels, parent, root);
            }
        }

        result.labels = relabel(slic.labels, parent, root);
        result.final_regions = dense_count(result.labels);
        if (result.middle_labels.empty()) {
            result.middle_labels = result.labels;
        }
        return result;
    }

private:
    static void connect(std::vector<RegionNode>& nodes, int a, int b) {
        if (a == b) {
            return;
        }
        nodes[static_cast<size_t>(a)].neighbors.insert(b);
        nodes[static_cast<size_t>(b)].neighbors.insert(a);
    }

    static void normalize(RegionNode& node) {
        if (node.area <= 0) {
            node.active = false;
            return;
        }
        const float inv_area = 1.0f / static_cast<float>(node.area);
        for (float& value : node.mean) {
            value *= inv_area;
        }
        const float inv_hist = inv_area / 3.0f;
        for (float& value : node.histogram) {
            value *= inv_hist;
        }
    }

    float cost(const RegionNode& a, const RegionNode& b, int total_pixels) const {
        float intersection = 0.0f;
        for (size_t bin = 0; bin < a.histogram.size(); ++bin) {
            intersection += std::min(a.histogram[bin], b.histogram[bin]);
        }
        const float size_similarity =
            std::clamp(1.0f - static_cast<float>(a.area + b.area) /
                                  static_cast<float>(total_pixels),
                       0.0f, 1.0f);
        const float similarity =
            alpha_color * intersection + (1.0f - alpha_color) * size_similarity;
        return 1.0f - std::clamp(similarity, 0.0f, 1.0f);
    }

    template <typename RootFn>
    static std::vector<int> relabel(const std::vector<int>& labels, std::vector<int>& parent,
                                    RootFn&& root) {
        std::vector<int> remap(parent.size(), -1);
        std::vector<int> output(labels.size(), 0);
        int next = 0;
        for (size_t p = 0; p < labels.size(); ++p) {
            const int r = root(labels[p]);
            if (remap[static_cast<size_t>(r)] < 0) {
                remap[static_cast<size_t>(r)] = next++;
            }
            output[p] = remap[static_cast<size_t>(r)];
        }
        return output;
    }

    static int dense_count(const std::vector<int>& labels) {
        return labels.empty() ? 0 : *std::max_element(labels.begin(), labels.end()) + 1;
    }
};

}  // namespace slic_rag

namespace {

struct AccuracyChecks {
    int total = 0;
    int passed = 0;
    std::vector<std::string> failures;

    void expect(bool condition, const std::string& message) {
        ++total;
        if (condition) {
            ++passed;
        } else {
            failures.push_back(message);
        }
    }
};

math::ImageBuffer rgb_for_sample(const ProviderLoadedSample* provider,
                                 const vision::GrayImage& luma, int max_side) {
    if (provider != nullptr && !provider->sample.rgb.empty()) {
        math::ImageBuffer rgb = downscale_max_side(provider->sample.rgb, max_side);
        if (rgb.width == luma.width && rgb.height == luma.height && rgb.channels >= 3) {
            return rgb;
        }
    }
    math::ImageBuffer rgb;
    rgb.width = luma.width;
    rgb.height = luma.height;
    rgb.channels = 3;
    rgb.data.resize(static_cast<size_t>(rgb.width * rgb.height * 3));
    for (int y = 0; y < rgb.height; ++y) {
        for (int x = 0; x < rgb.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                rgb.at(x, y, c) = luma.at(x, y);
            }
        }
    }
    return rgb;
}

vision::GrayImage overlay_boundaries(const vision::GrayImage& base,
                                     const std::vector<int>& labels) {
    return overlay_mask(base, label_boundaries(labels, base.width, base.height));
}

void draw_cross(vision::GrayImage& image, int x, int y, uint8_t value) {
    for (int d = -2; d <= 2; ++d) {
        if (x + d >= 0 && x + d < image.width && y >= 0 && y < image.height) {
            image.at(x + d, y) = value;
        }
        if (x >= 0 && x < image.width && y + d >= 0 && y + d < image.height) {
            image.at(x, y + d) = value;
        }
    }
}

vision::GrayImage center_overlay(const vision::GrayImage& base,
                                 const std::vector<slic_rag::Center>& centers) {
    vision::GrayImage output = base;
    for (uint8_t& value : output.data) {
        value = static_cast<uint8_t>(value / 2);
    }
    for (const auto& center : centers) {
        draw_cross(output, static_cast<int>(std::lround(center.x)),
                   static_cast<int>(std::lround(center.y)), 255);
    }
    return output;
}

std::vector<std::pair<float, float>> centroids(const std::vector<int>& labels, int n_labels,
                                               int w, int h) {
    std::vector<double> sx(static_cast<size_t>(n_labels), 0.0);
    std::vector<double> sy(static_cast<size_t>(n_labels), 0.0);
    std::vector<int> count(static_cast<size_t>(n_labels), 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int label = labels[static_cast<size_t>(y * w + x)];
            sx[static_cast<size_t>(label)] += x;
            sy[static_cast<size_t>(label)] += y;
            ++count[static_cast<size_t>(label)];
        }
    }
    std::vector<std::pair<float, float>> result(static_cast<size_t>(n_labels));
    for (int label = 0; label < n_labels; ++label) {
        if (count[static_cast<size_t>(label)] > 0) {
            result[static_cast<size_t>(label)] = {
                static_cast<float>(sx[static_cast<size_t>(label)] /
                                   count[static_cast<size_t>(label)]),
                static_cast<float>(sy[static_cast<size_t>(label)] /
                                   count[static_cast<size_t>(label)])};
        }
    }
    return result;
}

vision::GrayImage rag_overlay(const vision::GrayImage& base,
                              const slic_rag::RagResult& rag,
                              const std::vector<int>& labels) {
    vision::GrayImage output = base;
    for (uint8_t& value : output.data) {
        value = static_cast<uint8_t>(value / 3);
    }
    const auto centers = centroids(labels, rag.initial_regions, base.width, base.height);
    for (const auto& edge : rag.initial_edges_list) {
        const auto& a = centers[static_cast<size_t>(edge.first)];
        const auto& b = centers[static_cast<size_t>(edge.second)];
        plot_line(output, static_cast<int>(std::lround(a.first)),
                  static_cast<int>(std::lround(a.second)),
                  static_cast<int>(std::lround(b.first)),
                  static_cast<int>(std::lround(b.second)), 150);
    }
    for (const auto& center : centers) {
        draw_cross(output, static_cast<int>(std::lround(center.first)),
                   static_cast<int>(std::lround(center.second)), 255);
    }
    return output;
}

bool labels_are_dense(const std::vector<int>& labels, int expected) {
    if (labels.empty() || expected <= 0) {
        return false;
    }
    std::vector<uint8_t> seen(static_cast<size_t>(expected), 0);
    for (int label : labels) {
        if (label < 0 || label >= expected) {
            return false;
        }
        seen[static_cast<size_t>(label)] = 1;
    }
    return std::all_of(seen.begin(), seen.end(), [](uint8_t value) { return value != 0; });
}

bool each_label_is_connected(const std::vector<int>& labels, int n_labels, int w, int h) {
    std::vector<uint8_t> visited(labels.size(), 0);
    std::vector<int> components(static_cast<size_t>(n_labels), 0);
    std::vector<int> stack;
    constexpr int dx[4] = {1, -1, 0, 0};
    constexpr int dy[4] = {0, 0, 1, -1};
    for (int seed = 0; seed < w * h; ++seed) {
        if (visited[static_cast<size_t>(seed)]) {
            continue;
        }
        const int label = labels[static_cast<size_t>(seed)];
        if (++components[static_cast<size_t>(label)] > 1) {
            return false;
        }
        stack.assign(1, seed);
        visited[static_cast<size_t>(seed)] = 1;
        while (!stack.empty()) {
            const int p = stack.back();
            stack.pop_back();
            const int x = p % w;
            const int y = p / w;
            for (int d = 0; d < 4; ++d) {
                const int xx = x + dx[d];
                const int yy = y + dy[d];
                if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                    continue;
                }
                const int q = yy * w + xx;
                if (!visited[static_cast<size_t>(q)] &&
                    labels[static_cast<size_t>(q)] == label) {
                    visited[static_cast<size_t>(q)] = 1;
                    stack.push_back(q);
                }
            }
        }
    }
    return true;
}

vision::GrayImage ground_truth_boundary(const ProviderLoadedSample* provider, int w, int h) {
    if (provider == nullptr) {
        return {};
    }
    vision::GrayImage gt;
    if (!provider->sample.boundary.empty()) {
        gt = provider->sample.boundary;
    } else if (!provider->ground_truth.empty()) {
        gt = to_gray(contour::boundary_pixels(to_contour(binarize_mask(provider->ground_truth))));
    }
    if (!gt.empty() && (gt.width != w || gt.height != h)) {
        gt = resize_nearest(gt, w, h);
    }
    return binarize_mask(gt);
}

class SlicRagAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"slic_rag"};
    AccuracyChecks checks;
    std::ostringstream values;
    std::ostringstream nodes_tsv;
    std::ostringstream edges_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load full-image segmentation samples");
        const auto mission =
            load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, max_side);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        report.n_inputs = static_cast<int>(samples.size());
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name
                  << " (max side " << max_side << ")\n";
        return !samples.empty();
    }

    void run(const std::string& artifact_dir) {
        print_banner("run SLIC -> connectivity -> RAG contraction");
        ScopedTimer timer(&report.elapsed_ms);
        values << "file\tseed_centers\tslic_regions\tconnected_regions\trag_edges\tmerges"
                  "\tfinal_regions\tslic_iters\tchanged_last\tlast_cost\tboundary_f1\tms\n";
        nodes_tsv << "file\tnode\tarea\tmean_r\tmean_g\tmean_b\tdegree\n";
        edges_tsv << "file\tu\tv\tinitial_cost\n";
        double f1_sum = 0.0;
        int f1_count = 0;

        for (size_t index = 0; index < samples.size(); ++index) {
            const LoadedSample& sample = samples[index];
            const ProviderLoadedSample* provider =
                index < provider_samples.size() ? &provider_samples[index] : nullptr;
            const vision::GrayImage luma = mission_luma_image(provider, sample.image);
            const math::ImageBuffer rgb = rgb_for_sample(provider, luma, max_side);

            slic_rag::Slic slic;
            slic.iterations = slic_iterations;
            slic.compactness = compactness;
            slic_rag::RagHierarchicalMerger merger;
            merger.target_regions = target_regions;
            merger.max_cost = max_cost;

            double elapsed_ms = 0.0;
            slic_rag::SlicResult superpixels;
            slic_rag::RagResult rag;
            {
                ScopedTimer sample_timer(&elapsed_ms);
                superpixels = slic.segment(rgb, desired_superpixels);
                rag = merger.merge(superpixels, rgb);
            }

            const vision::GrayImage predicted_boundary =
                label_boundaries(rag.labels, luma.width, luma.height);
            const vision::GrayImage gt =
                ground_truth_boundary(provider, luma.width, luma.height);
            double boundary_f1 = 0.0;
            if (!gt.empty()) {
                boundary_f1 =
                    contour::boundary_f1(to_contour(predicted_boundary), to_contour(gt), 2);
                f1_sum += boundary_f1;
                ++f1_count;
            }

            const std::string stem = stem_of(sample.row.file);
            validate(stem, superpixels, rag);
            write_artifacts(artifact_dir, stem, luma, gt, superpixels, rag);
            write_graph_tables(stem, rag);

            values << sample.row.file << '\t' << superpixels.seeds.size() << '\t'
                   << superpixels.raw_regions << '\t' << superpixels.connected_regions << '\t'
                   << rag.initial_edges << '\t' << rag.merges << '\t' << rag.final_regions
                   << '\t' << superpixels.iterations_run << '\t' << superpixels.changed_last
                   << '\t' << rag.last_merge_cost << '\t' << boundary_f1 << '\t'
                   << elapsed_ms << '\n';
            std::cout << "  " << sample.row.file << "  SLIC="
                      << superpixels.connected_regions << " -> RAG=" << rag.final_regions
                      << "  merges=" << rag.merges << "  boundary_f1=" << std::fixed
                      << std::setprecision(3) << boundary_f1 << "  " << elapsed_ms << " ms\n";
            ++report.n_outputs;
        }

        std::ostringstream note;
        note << "SLIC K=" << desired_superpixels << ", m=" << compactness
             << ", target=" << target_regions << ", max_cost=" << max_cost;
        if (f1_count > 0) {
            note << ", mean boundary F1=" << std::fixed << std::setprecision(3)
                 << f1_sum / f1_count;
        }
        report.notes.push_back(note.str());
        if (f1_count >= 4) {
            checks.expect(f1_sum / f1_count >= 0.30,
                          "dataset: mean boundary F1 remains above 0.30");
        }
    }

    void write(const std::string& artifact_dir) {
        vision::write_text_file(vision::join_path(artifact_dir, "slic_rag.tsv"),
                                values.str());
        vision::write_text_file(vision::join_path(artifact_dir, "rag_nodes.tsv"),
                                nodes_tsv.str());
        vision::write_text_file(vision::join_path(artifact_dir, "rag_edges.tsv"),
                                edges_tsv.str());
        written.insert(written.begin(), "rag_edges.tsv");
        written.insert(written.begin(), "rag_nodes.tsv");
        written.insert(written.begin(), "slic_rag.tsv");
        write_atom_manifest(artifact_dir, report, written);
        report.print();
        std::cout << "invariants: " << checks.passed << " / " << checks.total << " passed\n";
        for (const std::string& failure : checks.failures) {
            std::cout << "  FAIL  " << failure << '\n';
        }
        std::cout << "artifacts -> " << artifact_dir << '\n';
    }

    int status() const { return checks.failures.empty() ? 0 : 1; }

private:
    int max_side = 192;
    int desired_superpixels = 180;
    int target_regions = 20;
    int slic_iterations = 10;
    float compactness = 15.0f;
    float max_cost = 0.36f;

    void validate(const std::string& stem, const slic_rag::SlicResult& slic,
                  const slic_rag::RagResult& rag) {
        const int n = slic.width * slic.height;
        checks.expect(static_cast<int>(slic.labels.size()) == n,
                      stem + ": SLIC labels cover every pixel");
        checks.expect(labels_are_dense(slic.labels, slic.connected_regions),
                      stem + ": connected SLIC labels are dense");
        checks.expect(each_label_is_connected(slic.labels, slic.connected_regions,
                                              slic.width, slic.height),
                      stem + ": every SLIC label is 4-connected");
        checks.expect(static_cast<int>(rag.labels.size()) == n,
                      stem + ": final labels cover every pixel");
        checks.expect(labels_are_dense(rag.labels, rag.final_regions),
                      stem + ": final labels are dense");
        checks.expect(each_label_is_connected(rag.labels, rag.final_regions,
                                              slic.width, slic.height),
                      stem + ": every final region is 4-connected");
        checks.expect(rag.final_regions <= rag.initial_regions,
                      stem + ": contraction never increases region count");
        checks.expect(rag.final_regions >= 1,
                      stem + ": at least one final region survives");
        checks.expect(rag.last_merge_cost <= max_cost + 1e-6f || rag.merges == 0,
                      stem + ": all accepted merges obey the cost threshold");
    }

    void save(const std::string& dir, const std::string& name,
              const vision::GrayImage& image) {
        vision::save_pgm(vision::join_path(dir, name), image);
        written.push_back(name);
    }

    void write_graph_tables(const std::string& stem, const slic_rag::RagResult& rag) {
        for (size_t node = 0; node < rag.initial_nodes.size(); ++node) {
            const auto& value = rag.initial_nodes[node];
            nodes_tsv << stem << '\t' << node << '\t' << value.area << '\t'
                      << value.mean[0] << '\t' << value.mean[1] << '\t'
                      << value.mean[2] << '\t' << value.neighbors.size() << '\n';
        }
        for (size_t edge = 0; edge < rag.initial_edges_list.size(); ++edge) {
            edges_tsv << stem << '\t' << rag.initial_edges_list[edge].first << '\t'
                      << rag.initial_edges_list[edge].second << '\t'
                      << rag.initial_edge_costs[edge] << '\n';
        }
    }

    void write_artifacts(const std::string& dir, const std::string& stem,
                         const vision::GrayImage& luma,
                         const vision::GrayImage& gt,
                         const slic_rag::SlicResult& slic,
                         const slic_rag::RagResult& rag) {
        save(dir, stem + "_00_input_base.pgm", luma);
        save(dir, stem + "_01_seed_grid_overlay.pgm",
             center_overlay(luma, slic.seeds));
        save(dir, stem + "_02_local_assignment_labels.pgm",
             colorize_labels(slic.first_labels, slic.width, slic.height));
        save(dir, stem + "_02_local_assignment_overlay.pgm",
             overlay_boundaries(luma, slic.first_labels));
        save(dir, stem + "_03_converged_slic_labels.pgm",
             colorize_labels(slic.raw_labels, slic.width, slic.height));
        save(dir, stem + "_03_centroid_overlay.pgm",
             center_overlay(overlay_boundaries(luma, slic.raw_labels), slic.centers));
        save(dir, stem + "_04_connectivity_labels.pgm",
             colorize_labels(slic.labels, slic.width, slic.height));
        save(dir, stem + "_04_connectivity_overlay.pgm",
             overlay_boundaries(luma, slic.labels));
        save(dir, stem + "_05_rag_overlay.pgm",
             rag_overlay(luma, rag, slic.labels));
        save(dir, stem + "_06_contraction_mid_labels.pgm",
             colorize_labels(rag.middle_labels, slic.width, slic.height));
        save(dir, stem + "_06_contraction_mid_overlay.pgm",
             overlay_boundaries(luma, rag.middle_labels));
        save(dir, stem + "_07_final_labels.pgm",
             colorize_labels(rag.labels, slic.width, slic.height));
        save(dir, stem + "_07_final_overlay.pgm",
             overlay_boundaries(luma, rag.labels));
        if (!gt.empty()) {
            save(dir, stem + "_gt_boundary_overlay.pgm", overlay_mask(luma, gt));
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "bsds500", [&](const AtomCli& cli) -> int {
        SlicRagAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for SLIC-RAG atom\n";
            return 1;
        }
        if (cli.list_only) {
            for (const auto& sample : atom.samples) {
                std::cout << "  " << sample.row.file << '\n';
            }
            return 0;
        }
        bool artifacts_overridden = false;
        for (int i = 1; i < argc; ++i) {
            artifacts_overridden = artifacts_overridden ||
                                   std::string(argv[i]) == "--artifacts";
        }
        const std::string artifact_request =
            artifacts_overridden ? cli.artifact_dir
                                 : vision::join_path(cli.artifact_dir, "slic_rag");
        const std::string artifacts = make_artifact_dir(artifact_request);
        atom.run(artifacts);
        atom.write(artifacts);
        return atom.status();
    });
}
