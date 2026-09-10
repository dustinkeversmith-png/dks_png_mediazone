#pragma once

// Compact superpixel segmentation followed by region-adjacency contraction.
//
// SLIC clusters in CIELAB so the color and spatial halves of the 5D distance
// sit on comparable scales; a lazy-versioned priority queue then contracts the
// region adjacency graph using color, boundary-contrast, shape and size
// evidence. Shared by the slic_rag atom and the integration pipeline.

#include "filters/lab_color/lab_color_space.hpp"
#include "math/contour_compat.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace segmentation::slic_rag {

// Cluster centers live in CIELAB, not gamma-encoded RGB. The SLIC distance
// trades color against space through m^2/S^2, so the two terms have to sit on
// comparable scales: a perceptually uniform axis whose useful range is ~100
// makes `compactness` behave like the literature's m. Feeding it raw 0-255 RGB
// inflates the color term by roughly two orders of magnitude and collapses the
// spatial term to noise, which produces sprawling, non-compact superpixels.
struct Center {
    float l = 0.0f;
    float a = 0.0f;
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
    std::vector<Center> lab;
    std::vector<float> gradient;
    int raw_regions = 0;
    int connected_regions = 0;
};

class Slic {
public:
    int iterations = 10;
    float compactness = 12.0f;
    float orphan_fraction = 0.25f;
    // Residual tolerance, as a fraction of the pixel count. Label churn decays
    // fast but rarely reaches exactly zero, because pixels equidistant from two
    // centers oscillate forever; waiting for that wasted half the iterations.
    float convergence_fraction = 0.002f;

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

        result.lab = lab_plane(rgb);
        result.gradient = lab_gradient(result.lab, w, h);
        const std::vector<Center>& lab = result.lab;

        const int k = std::clamp(requested_k, 1, n);
        const float nominal = std::sqrt(static_cast<float>(n) / static_cast<float>(k));
        // Tile the canvas by whole cells per axis instead of stepping by a
        // fixed stride from the top-left. Stepping left a remainder strip of
        // up to S along the right and bottom edges that belonged to no cell,
        // and the outermost clusters stretched across it.
        const int columns = std::max(1, static_cast<int>(std::lround(w / nominal)));
        const int rows = std::max(1, static_cast<int>(std::lround(h / nominal)));
        const float step_x = static_cast<float>(w) / static_cast<float>(columns);
        const float step_y = static_cast<float>(h) / static_cast<float>(rows);
        const float spacing = std::sqrt(step_x * step_y);
        const float inv_s2 = 1.0f / std::max(1.0f, spacing * spacing);
        const float inv_m2 = 1.0f / std::max(1e-3f, compactness * compactness);

        std::vector<Center> centers;
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < columns; ++column) {
                const int px = std::clamp(
                    static_cast<int>((static_cast<float>(column) + 0.5f) * step_x), 0, w - 1);
                const int py = std::clamp(
                    static_cast<int>((static_cast<float>(row) + 0.5f) * step_y), 0, h - 1);
                centers.push_back(lab[low_gradient_seed(result.gradient, w, h, px, py)]);
            }
        }
        if (centers.empty()) {
            centers.push_back(lab[static_cast<size_t>((h / 2) * w + w / 2)]);
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
                // A 2S x 2S local search window, centered on the cluster, sized
                // from the per-axis cell so neighbouring windows always overlap.
                const int x0 = std::max(0, static_cast<int>(std::floor(c.x - step_x)));
                const int x1 = std::min(w - 1, static_cast<int>(std::ceil(c.x + step_x)));
                const int y0 = std::max(0, static_cast<int>(std::floor(c.y - step_y)));
                const int y1 = std::min(h - 1, static_cast<int>(std::ceil(c.y + step_y)));
                for (int y = y0; y <= y1; ++y) {
                    for (int x = x0; x <= x1; ++x) {
                        const size_t p = static_cast<size_t>(y * w + x);
                        const Center& pixel = lab[p];
                        const float dl = pixel.l - c.l;
                        const float da = pixel.a - c.a;
                        const float db = pixel.b - c.b;
                        const float dc2 = dl * dl + da * da + db * db;
                        const float dx = static_cast<float>(x) - c.x;
                        const float dy = static_cast<float>(y) - c.y;
                        const float ds2 = dx * dx + dy * dy;
                        // Both terms are normalized before they are summed:
                        // color by m^2 and space by S^2. Either one alone can
                        // dominate, which is exactly what `compactness` tunes.
                        const float d = dc2 * inv_m2 + ds2 * inv_s2;
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
                    const size_t p = static_cast<size_t>(y * w + x);
                    const int ci = labels[p];
                    Center& acc = sums[static_cast<size_t>(ci)];
                    acc.l += lab[p].l;
                    acc.a += lab[p].a;
                    acc.b += lab[p].b;
                    acc.x += static_cast<float>(x);
                    acc.y += static_cast<float>(y);
                    ++counts[static_cast<size_t>(ci)];
                }
            }
            for (size_t ci = 0; ci < centers.size(); ++ci) {
                if (counts[ci] == 0) {
                    continue;
                }
                const float inv = 1.0f / static_cast<float>(counts[ci]);
                centers[ci] = {sums[ci].l * inv, sums[ci].a * inv, sums[ci].b * inv,
                               sums[ci].x * inv, sums[ci].y * inv};
            }

            previous = labels;
            result.iterations_run = iter + 1;
            result.changed_last = changed;
            if (iter >= 2 &&
                changed <= static_cast<int>(convergence_fraction * static_cast<float>(n))) {
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
    static std::vector<Center> lab_plane(const math::ImageBuffer& rgb) {
        std::vector<Center> lab(static_cast<size_t>(rgb.width * rgb.height));
        for (int y = 0; y < rgb.height; ++y) {
            for (int x = 0; x < rgb.width; ++x) {
                const contour::Lab value = contour::LabColor::at(rgb, x, y);
                lab[static_cast<size_t>(y * rgb.width + x)] = {
                    value.L, value.a, value.b, static_cast<float>(x), static_cast<float>(y)};
            }
        }
        return lab;
    }

    static float lab_delta(const Center& p, const Center& q) {
        const float dl = p.l - q.l;
        const float da = p.a - q.a;
        const float db = p.b - q.b;
        return std::sqrt(dl * dl + da * da + db * db);
    }

    static std::vector<float> lab_gradient(const std::vector<Center>& lab, int w, int h) {
        std::vector<float> gradient(lab.size(), 0.0f);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int xm = std::max(0, x - 1);
                const int xp = std::min(w - 1, x + 1);
                const int ym = std::max(0, y - 1);
                const int yp = std::min(h - 1, y + 1);
                const float gx = lab_delta(lab[static_cast<size_t>(y * w + xp)],
                                           lab[static_cast<size_t>(y * w + xm)]);
                const float gy = lab_delta(lab[static_cast<size_t>(yp * w + x)],
                                           lab[static_cast<size_t>(ym * w + x)]);
                gradient[static_cast<size_t>(y * w + x)] = std::sqrt(gx * gx + gy * gy);
            }
        }
        return gradient;
    }

    // Nudge a seed off any edge it happened to land on. A center straddling a
    // boundary is pulled apart by both sides and converges to a smeared
    // cluster that honors neither.
    static size_t low_gradient_seed(const std::vector<float>& gradient, int w, int h, int x,
                                    int y) {
        size_t best = static_cast<size_t>(y * w + x);
        float best_g = gradient[best];
        for (int yy = std::max(0, y - 1); yy <= std::min(h - 1, y + 1); ++yy) {
            for (int xx = std::max(0, x - 1); xx <= std::min(w - 1, x + 1); ++xx) {
                const size_t p = static_cast<size_t>(yy * w + xx);
                if (gradient[p] < best_g) {
                    best_g = gradient[p];
                    best = p;
                }
            }
        }
        return best;
    }

    static int dense_count(const std::vector<int>& labels) {
        return labels.empty() ? 0 : *std::max_element(labels.begin(), labels.end()) + 1;
    }

    // Splits every label into its 4-connected runs, then folds the undersized
    // runs into whichever surviving neighbor they share the most border with.
    //
    // The previous version absorbed during a single raster-order flood, so a
    // component could only merge into a neighbor that had already been
    // scanned. Components near the top-left had almost no candidates and were
    // promoted to standalone regions, while later ones over-absorbed; the
    // superpixel count drifted far below K and the survivors grew into
    // sprawling shapes whose centroids fell outside their own pixels.
    std::vector<int> enforce_connectivity(const std::vector<int>& input, int w, int h,
                                          float spacing) const {
        const int n = w * h;
        const int min_component =
            std::max(4, static_cast<int>(orphan_fraction * spacing * spacing));
        constexpr int dx[4] = {1, -1, 0, 0};
        constexpr int dy[4] = {0, 0, 1, -1};

        std::vector<int> component(static_cast<size_t>(n), -1);
        std::vector<int> sizes;
        std::vector<int> stack;
        for (int seed = 0; seed < n; ++seed) {
            if (component[static_cast<size_t>(seed)] >= 0) {
                continue;
            }
            const int source_label = input[static_cast<size_t>(seed)];
            const int id = static_cast<int>(sizes.size());
            sizes.push_back(0);
            component[static_cast<size_t>(seed)] = id;
            stack.assign(1, seed);
            while (!stack.empty()) {
                const int p = stack.back();
                stack.pop_back();
                ++sizes[static_cast<size_t>(id)];
                const int x = p % w;
                const int y = p / w;
                for (int d = 0; d < 4; ++d) {
                    const int xx = x + dx[d];
                    const int yy = y + dy[d];
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                        continue;
                    }
                    const int q = yy * w + xx;
                    if (component[static_cast<size_t>(q)] < 0 &&
                        input[static_cast<size_t>(q)] == source_label) {
                        component[static_cast<size_t>(q)] = id;
                        stack.push_back(q);
                    }
                }
            }
        }

        const int m = static_cast<int>(sizes.size());
        std::vector<std::map<int, int>> border(static_cast<size_t>(m));
        auto touch = [&border](int a, int b) {
            if (a == b) {
                return;
            }
            ++border[static_cast<size_t>(a)][b];
            ++border[static_cast<size_t>(b)][a];
        };
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int p = y * w + x;
                if (x + 1 < w) {
                    touch(component[static_cast<size_t>(p)],
                          component[static_cast<size_t>(p + 1)]);
                }
                if (y + 1 < h) {
                    touch(component[static_cast<size_t>(p)],
                          component[static_cast<size_t>(p + w)]);
                }
            }
        }

        std::vector<int> parent(static_cast<size_t>(m));
        std::iota(parent.begin(), parent.end(), 0);
        auto root = [&parent](int x) {
            while (parent[static_cast<size_t>(x)] != x) {
                parent[static_cast<size_t>(x)] =
                    parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
                x = parent[static_cast<size_t>(x)];
            }
            return x;
        };

        std::vector<uint8_t> survives(static_cast<size_t>(m), 0);
        std::vector<uint8_t> pending(static_cast<size_t>(m), 0);
        std::vector<int> orphans;
        for (int c = 0; c < m; ++c) {
            if (sizes[static_cast<size_t>(c)] >= min_component) {
                survives[static_cast<size_t>(c)] = 1;
            } else {
                pending[static_cast<size_t>(c)] = 1;
                orphans.push_back(c);
            }
        }
        // Smallest first, so specks disappear into real superpixels rather
        // than glueing themselves into a chain of specks.
        std::sort(orphans.begin(), orphans.end(), [&sizes](int a, int b) {
            return sizes[static_cast<size_t>(a)] < sizes[static_cast<size_t>(b)];
        });

        int remaining = static_cast<int>(orphans.size());
        while (remaining > 0) {
            bool progress = false;
            for (int c : orphans) {
                if (!pending[static_cast<size_t>(c)]) {
                    continue;
                }
                int best = -1;
                int best_border = 0;
                for (const auto& entry : border[static_cast<size_t>(c)]) {
                    const int r = root(entry.first);
                    if (r == c || !survives[static_cast<size_t>(r)]) {
                        continue;
                    }
                    if (entry.second > best_border) {
                        best_border = entry.second;
                        best = r;
                    }
                }
                if (best >= 0) {
                    parent[static_cast<size_t>(c)] = best;
                    sizes[static_cast<size_t>(best)] += sizes[static_cast<size_t>(c)];
                    pending[static_cast<size_t>(c)] = 0;
                    --remaining;
                    progress = true;
                }
            }
            if (progress) {
                continue;
            }
            // A pocket of orphans that only border each other. Promote the
            // largest so the rest have somewhere to go on the next sweep.
            int promote = -1;
            for (int c : orphans) {
                if (pending[static_cast<size_t>(c)] &&
                    (promote < 0 ||
                     sizes[static_cast<size_t>(c)] > sizes[static_cast<size_t>(promote)])) {
                    promote = c;
                }
            }
            if (promote < 0) {
                break;
            }
            survives[static_cast<size_t>(promote)] = 1;
            pending[static_cast<size_t>(promote)] = 0;
            --remaining;
        }

        std::vector<int> remap(static_cast<size_t>(std::max(1, m)), -1);
        std::vector<int> output(static_cast<size_t>(n), 0);
        int dense = 0;
        for (int p = 0; p < n; ++p) {
            const int r = root(component[static_cast<size_t>(p)]);
            if (remap[static_cast<size_t>(r)] < 0) {
                remap[static_cast<size_t>(r)] = dense++;
            }
            output[static_cast<size_t>(p)] = remap[static_cast<size_t>(r)];
        }
        return output;
    }
};

// Per-adjacency evidence. `contrast_sum` accumulates the Lab step measured
// across every pixel pair straddling the shared border, so a long, faint seam
// and a short, hard one are told apart by their mean rather than their length.
struct Boundary {
    int length = 0;
    float contrast_sum = 0.0f;

    float contrast() const {
        return length > 0 ? contrast_sum / static_cast<float>(length) : 0.0f;
    }
};

struct RegionNode {
    int area = 0;
    std::array<float, 3> mean{{0.0f, 0.0f, 0.0f}};
    std::array<float, 3> lab_mean{{0.0f, 0.0f, 0.0f}};
    std::array<float, 24> histogram{};
    std::map<int, Boundary> neighbors;
    // Internal variation: the same cross-pixel Lab step used for borders, but
    // measured on pairs that fall inside the region. Gives the contraction a
    // yardstick for how much contrast this region already tolerates.
    float internal_sum = 0.0f;
    int internal_count = 0;
    // Full outline length, counting the image frame, so a seam can be weighed
    // against how much of the region's own boundary it represents.
    int perimeter = 0;
    int version = 0;
    bool active = true;

    float internal() const {
        return internal_count > 0 ? internal_sum / static_cast<float>(internal_count) : 0.0f;
    }
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
    int cleanup_merges = 0;
    float last_merge_cost = 0.0f;
    float last_cleanup_cost = 0.0f;
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
    // Relative pull of each similarity term; normalized by their sum.
    float w_color = 0.50f;
    float w_edge = 0.28f;
    float w_share = 0.16f;
    float w_size = 0.06f;
    // Lab units at which two regions count as fully different / fully split,
    // before the texture-adaptive widening below is applied.
    float sigma_color = 14.0f;
    float sigma_edge = 9.0f;
    float texture_gain = 1.0f;
    // Smallest area, as a fraction of the image, that still counts as a region.
    float min_region_fraction = 0.005f;

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
                const Center& lab = slic.lab[static_cast<size_t>(p)];
                node.lab_mean[0] += lab.l;
                node.lab_mean[1] += lab.a;
                node.lab_mean[2] += lab.b;
                // Frame sides are outline too; the shared-border lengths get
                // folded in once the adjacency map is complete.
                node.perimeter += static_cast<int>(x == 0) + static_cast<int>(x == w - 1) +
                                  static_cast<int>(y == 0) + static_cast<int>(y == h - 1);
                if (x + 1 < w) {
                    connect(nodes, label, slic.labels[static_cast<size_t>(p + 1)],
                            step(slic.lab, p, p + 1));
                }
                if (y + 1 < h) {
                    connect(nodes, label, slic.labels[static_cast<size_t>(p + w)],
                            step(slic.lab, p, p + w));
                }
                // Same pixel-pair walk, opposite case: pairs that stay inside
                // the region feed its internal-variation baseline.
                if (x + 1 < w && slic.labels[static_cast<size_t>(p + 1)] == label) {
                    node.internal_sum += step(slic.lab, p, p + 1);
                    ++node.internal_count;
                }
                if (y + 1 < h && slic.labels[static_cast<size_t>(p + w)] == label) {
                    node.internal_sum += step(slic.lab, p, p + w);
                    ++node.internal_count;
                }
            }
        }
        for (RegionNode& node : nodes) {
            normalize(node);
        }

        result.initial_regions = k;
        result.initial_nodes = nodes;
        for (int u = 0; u < k; ++u) {
            for (const auto& entry : nodes[static_cast<size_t>(u)].neighbors) {
                const int v = entry.first;
                if (u < v) {
                    result.initial_edges_list.push_back({u, v});
                    result.initial_edge_costs.push_back(
                        cost(nodes[static_cast<size_t>(u)],
                             nodes[static_cast<size_t>(v)], entry.second, n));
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
            const auto it = nodes[static_cast<size_t>(a)].neighbors.find(b);
            if (it == nodes[static_cast<size_t>(a)].neighbors.end()) {
                return;
            }
            queue.push({a, b, cost(nodes[static_cast<size_t>(a)],
                                   nodes[static_cast<size_t>(b)], it->second, n),
                        nodes[static_cast<size_t>(a)].version,
                        nodes[static_cast<size_t>(b)].version});
        };
        for (const auto& edge : result.initial_edges_list) {
            push_edge(edge.first, edge.second);
        }

        // Contracts v into u. Both must be distinct, active, adjacent roots.
        // The larger region always survives, so the union-find tree stays
        // shallow and the dominant feature vector is the one that is kept.
        auto contract = [&](int u, int v) {
            if (nodes[static_cast<size_t>(u)].area < nodes[static_cast<size_t>(v)].area) {
                std::swap(u, v);
            }
            RegionNode& keep = nodes[static_cast<size_t>(u)];
            RegionNode& drop = nodes[static_cast<size_t>(v)];
            const Boundary seam = keep.neighbors.at(v);
            const int combined_area = keep.area + drop.area;
            const float wk = static_cast<float>(keep.area) / combined_area;
            const float wd = static_cast<float>(drop.area) / combined_area;
            for (int c = 0; c < 3; ++c) {
                keep.mean[static_cast<size_t>(c)] =
                    wk * keep.mean[static_cast<size_t>(c)] +
                    wd * drop.mean[static_cast<size_t>(c)];
                keep.lab_mean[static_cast<size_t>(c)] =
                    wk * keep.lab_mean[static_cast<size_t>(c)] +
                    wd * drop.lab_mean[static_cast<size_t>(c)];
            }
            for (size_t bin = 0; bin < keep.histogram.size(); ++bin) {
                keep.histogram[bin] =
                    wk * keep.histogram[bin] + wd * drop.histogram[bin];
            }
            keep.area = combined_area;
            // The seam just stopped being a border, so it joins the interior
            // and raises the bar for the next merge along this front.
            keep.internal_sum += drop.internal_sum + seam.contrast_sum;
            keep.internal_count += drop.internal_count + seam.length;
            keep.perimeter = keep.perimeter + drop.perimeter - 2 * seam.length;
            parent[static_cast<size_t>(v)] = u;
            drop.active = false;
            ++keep.version;
            ++drop.version;

            // Fuse both adjacency lists onto canonical roots, summing the
            // border evidence so a neighbor that touched u and v separately
            // ends up with one edge carrying the combined seam.
            std::map<int, Boundary> merged_neighbors;
            auto absorb = [&](const std::map<int, Boundary>& source) {
                for (const auto& entry : source) {
                    const int r = root(entry.first);
                    if (r == u || !nodes[static_cast<size_t>(r)].active) {
                        continue;
                    }
                    Boundary& slot = merged_neighbors[r];
                    slot.length += entry.second.length;
                    slot.contrast_sum += entry.second.contrast_sum;
                }
            };
            absorb(keep.neighbors);
            absorb(drop.neighbors);
            keep.neighbors = merged_neighbors;
            drop.neighbors.clear();
            for (const auto& entry : keep.neighbors) {
                RegionNode& other = nodes[static_cast<size_t>(entry.first)];
                other.neighbors.erase(v);
                other.neighbors.erase(u);
                other.neighbors[u] = entry.second;
                push_edge(u, entry.first);
            }
            ++result.merges;
        };

        int active = k;
        const int target = std::clamp(target_regions, 1, k);
        const int middle_target = target + (k - target) / 2;
        while (!queue.empty() && active > target) {
            const RagEdge candidate = queue.top();
            queue.pop();
            const int u = root(candidate.u);
            const int v = root(candidate.v);
            if (u == v || u != candidate.u || v != candidate.v) {
                continue;
            }
            const RegionNode& a = nodes[static_cast<size_t>(u)];
            const RegionNode& b = nodes[static_cast<size_t>(v)];
            // Reject stale costs. Every feature update increments the version.
            if (!a.active || !b.active || a.version != candidate.version_u ||
                b.version != candidate.version_v) {
                continue;
            }
            const auto edge_it = a.neighbors.find(v);
            if (edge_it == a.neighbors.end()) {
                continue;
            }
            const float current_cost = cost(a, b, edge_it->second, n);
            if (current_cost > max_cost) {
                break;
            }

            contract(u, v);
            --active;
            result.last_merge_cost = current_cost;
            if (result.middle_labels.empty() && active <= middle_target) {
                result.middle_labels = relabel(slic.labels, parent, root);
            }
        }

        // Closure pass. The threshold above answers "are these two regions the
        // same thing?", which correctly refuses to merge a small but distinct
        // patch. That still leaves the map littered with 30-pixel confetti
        // that no consumer would call a segment, and every one of them
        // contributes a false boundary. Anything under the minimum area is
        // folded into its cheapest neighbor regardless of the threshold: the
        // question here is only "where does this fragment belong?".
        const int min_region = static_cast<int>(min_region_fraction * static_cast<float>(n));
        while (active > 1) {
            int smallest = -1;
            for (int i = 0; i < k; ++i) {
                if (!nodes[static_cast<size_t>(i)].active ||
                    nodes[static_cast<size_t>(i)].area >= min_region) {
                    continue;
                }
                if (smallest < 0 ||
                    nodes[static_cast<size_t>(i)].area < nodes[static_cast<size_t>(smallest)].area) {
                    smallest = i;
                }
            }
            if (smallest < 0) {
                break;
            }
            int best = -1;
            float best_cost = std::numeric_limits<float>::max();
            for (const auto& entry : nodes[static_cast<size_t>(smallest)].neighbors) {
                const int r = root(entry.first);
                if (r == smallest || !nodes[static_cast<size_t>(r)].active) {
                    continue;
                }
                const float candidate_cost = cost(nodes[static_cast<size_t>(smallest)],
                                                  nodes[static_cast<size_t>(r)], entry.second, n);
                if (candidate_cost < best_cost) {
                    best_cost = candidate_cost;
                    best = r;
                }
            }
            if (best < 0) {
                break;
            }
            contract(smallest, best);
            --active;
            ++result.cleanup_merges;
            result.last_cleanup_cost = best_cost;
        }

        result.labels = relabel(slic.labels, parent, root);
        result.final_regions = dense_count(result.labels);
        if (result.middle_labels.empty()) {
            result.middle_labels = result.labels;
        }
        return result;
    }

private:
    static float step(const std::vector<Center>& lab, int p, int q) {
        const float dl = lab[static_cast<size_t>(p)].l - lab[static_cast<size_t>(q)].l;
        const float da = lab[static_cast<size_t>(p)].a - lab[static_cast<size_t>(q)].a;
        const float db = lab[static_cast<size_t>(p)].b - lab[static_cast<size_t>(q)].b;
        return std::sqrt(dl * dl + da * da + db * db);
    }

    static void connect(std::vector<RegionNode>& nodes, int a, int b, float contrast) {
        if (a == b) {
            return;
        }
        for (const auto pair : {std::make_pair(a, b), std::make_pair(b, a)}) {
            Boundary& edge = nodes[static_cast<size_t>(pair.first)].neighbors[pair.second];
            ++edge.length;
            edge.contrast_sum += contrast;
        }
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
        for (float& value : node.lab_mean) {
            value *= inv_area;
        }
        const float inv_hist = inv_area / 3.0f;
        for (float& value : node.histogram) {
            value *= inv_hist;
        }
        for (const auto& entry : node.neighbors) {
            node.perimeter += entry.second.length;
        }
    }

    // Three independent pieces of evidence, all mapped into [0, 1]:
    //   - appearance: coarse histogram overlap plus mean Lab agreement,
    //   - separation: how hard the image edge along the shared border is,
    //   - size: a mild bias toward absorbing small regions first.
    //
    // The old cost used histogram overlap alone. With 32-level bins two
    // regions of genuinely different color routinely overlapped enough to
    // merge, and nothing in the score knew whether a real image boundary sat
    // between them, so contractions walked straight across object silhouettes.
    //
    // Both tolerances widen with the pair's internal variation. Absolute
    // thresholds only suit smooth regions: inside foliage or gravel the
    // contrast between neighbouring superpixels is as strong as a genuine
    // silhouette, so a fixed sigma priced every merge out of reach and those
    // images stalled far above the target region count.
    float cost(const RegionNode& a, const RegionNode& b, const Boundary& edge,
               int total_pixels) const {
        float intersection = 0.0f;
        for (size_t bin = 0; bin < a.histogram.size(); ++bin) {
            intersection += std::min(a.histogram[bin], b.histogram[bin]);
        }
        float delta2 = 0.0f;
        for (size_t c = 0; c < a.lab_mean.size(); ++c) {
            const float d = a.lab_mean[c] - b.lab_mean[c];
            delta2 += d * d;
        }
        const float texture = texture_gain * 0.5f * (a.internal() + b.internal());
        const float mean_similarity =
            std::exp(-std::sqrt(delta2) / std::max(1e-3f, sigma_color + texture));
        const float color_similarity = 0.5f * intersection + 0.5f * mean_similarity;
        const float edge_similarity =
            std::exp(-edge.contrast() / std::max(1e-3f, sigma_edge + texture));
        // Fraction of the smaller region's outline that this seam accounts
        // for. A sliver wrapped by its neighbour scores near 1 and gets
        // absorbed early; two large regions brushing along a short seam score
        // near 0, which is what stops the contraction from stitching
        // unrelated areas into ragged dumbbell shapes.
        const float smaller_perimeter =
            static_cast<float>(std::max(1, std::min(a.perimeter, b.perimeter)));
        const float share_similarity =
            std::clamp(static_cast<float>(edge.length) / smaller_perimeter, 0.0f, 1.0f);
        const float size_similarity =
            std::clamp(1.0f - static_cast<float>(a.area + b.area) /
                                  static_cast<float>(total_pixels),
                       0.0f, 1.0f);

        const float total = std::max(1e-3f, w_color + w_edge + w_share + w_size);
        const float similarity =
            (w_color * color_similarity + w_edge * edge_similarity +
             w_share * share_similarity + w_size * size_similarity) /
            total;
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

}  // namespace segmentation::slic_rag
