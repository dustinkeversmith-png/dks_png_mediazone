#include "test_harness.hpp"
#include "math/contour_metrics.hpp"
#include "math/polygon.hpp"
#include "contour/rdp/rdp.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"
#include "filters/edge/canny/canny.hpp"

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

namespace edge_geom {

using math::ImageBuffer;
using math::Vec2;

struct EdgePoint {
    Vec2 p;
    Vec2 normal;
    float strength = 0.0f;
    int pixel = -1;
};

struct Chain {
    std::vector<EdgePoint> points;
};

struct Profile {
    std::array<float, 3> left{};
    std::array<float, 3> right{};
    float contrast = 0.0f;
    float along_variance = 0.0f;
};

struct Vertex {
    Vec2 p;
    std::vector<int> outgoing;
};

struct HalfEdge {
    int origin = -1;
    int destination = -1;
    int twin = -1;
    int next = -1;
    Profile profile;
    float strength = 0.0f;
    bool synthetic = false;
};

struct Face {
    std::vector<int> half_edges;
    std::vector<Vec2> polygon;
    int area = 0;
    int region_label = 0;
    std::array<float, 3> mean{};
    float variance = 0.0f;
    std::array<float, 3> affine{};  // luma = ax + by + c
    float affine_residual = 0.0f;
    math::Rect bbox;
};

struct Result {
    int width = 0;
    int height = 0;
    ImageBuffer gradient;
    ImageBuffer coarse_edges;
    ImageBuffer detail_edges;
    ImageBuffer thin_edges;
    std::vector<Chain> chains;
    std::vector<Vertex> vertices;
    std::vector<HalfEdge> half_edges;
    std::vector<std::pair<int, int>> candidate_gaps;
    std::vector<std::pair<int, int>> accepted_gaps;
    std::vector<Face> faces;
    std::vector<int> labels;
    int raw_edge_pixels = 0;
    int simplified_segments = 0;
    int dangling_before = 0;
    int dangling_after = 0;
    int t_junctions = 0;
};

class Pipeline {
public:
    float canny_low = 0.06f;
    float canny_high = 0.18f;
    float rdp_epsilon = 1.25f;
    float profile_offset = 3.0f;
    float gap_radius = 34.0f;
    float max_bridge_energy = 2.15f;
    float color_weight = 0.925f;
    float gradient_weight = 0.60f;
    float min_face_area_frac = 0.00115f;
    float max_face_variance = 0.120f;
    float face_merge_color = 0.070f;

    Result run(const ImageBuffer& rgb, const ImageBuffer& luma) const {
        Result r;
        r.width = rgb.width;
        r.height = rgb.height;
        const ImageBuffer smooth = gaussian5(luma);
        const ImageBuffer coarse =
            gaussian5(gaussian5(gaussian5(gaussian5(smooth))));
        Derivatives deriv = derivatives(smooth);
        r.gradient = gradient_image(deriv);

        contour::Canny coarse_canny;
        coarse_canny.low = canny_low * 0.725f;
        coarse_canny.high = canny_high * 0.725f;
        r.coarse_edges = coarse_canny.detect(coarse);
        contour::Canny detail_canny;
        detail_canny.low = canny_low;
        detail_canny.high = canny_high;
        r.detail_edges = detail_canny.detect(smooth);

        // Structural edges are always retained. Fine edges are admitted only
        // where local color variance says that the containing surface is not
        // already smooth; decals cannot subdivide a flat fuselage indefinitely.
        r.thin_edges = r.coarse_edges;
        for (int y = 1; y < r.height - 1; ++y) {
            for (int x = 1; x < r.width - 1; ++x) {
                bool near_structure = false;
                for (int dy = -6; dy <= 6 && !near_structure; ++dy) {
                    for (int dx = -6; dx <= 6; ++dx) {
                        const int xx = std::clamp(x + dx, 0, r.width - 1);
                        const int yy = std::clamp(y + dy, 0, r.height - 1);
                        if (r.coarse_edges.at(xx, yy)) {
                            near_structure = true;
                            break;
                        }
                    }
                }
                const bool independently_strong =
                    deriv.mag[static_cast<size_t>(y * r.width + x)] >
                    0.25f * deriv.max_mag;
                if (r.detail_edges.at(x, y) &&
                    local_variance(rgb, x, y, 4) > 0.0030f &&
                    (near_structure || independently_strong)) {
                    r.thin_edges.at(x, y) = 255;
                }
            }
        }
        std::vector<EdgePoint> points = localize(r.thin_edges, deriv);
        r.raw_edge_pixels = static_cast<int>(points.size());
        r.chains = link_and_simplify(points, r.thin_edges, rdp_epsilon);

        build_dcel(r, rgb);
        close_gaps(r, rgb, deriv);
        link_half_edges(r);
        extract_faces(r, rgb);
        classify_and_merge_faces(r);
        return r;
    }

private:
    struct Derivatives {
        int w = 0;
        int h = 0;
        std::vector<float> ix, iy, ixx, ixy, iyy, mag;
        float max_mag = 1.0f;
    };

    static ImageBuffer gaussian5(const ImageBuffer& src) {
        const int kernel[5] = {1, 4, 6, 4, 1};
        std::vector<float> temp(static_cast<size_t>(src.width * src.height), 0.0f);
        ImageBuffer out = math::make_gray(src.width, src.height, 0);
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                float sum = 0.0f;
                for (int k = -2; k <= 2; ++k) {
                    sum += kernel[k + 2] * src.gray(std::clamp(x + k, 0, src.width - 1), y);
                }
                temp[static_cast<size_t>(y * src.width + x)] = sum / 16.0f;
            }
        }
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                float sum = 0.0f;
                for (int k = -2; k <= 2; ++k) {
                    sum += kernel[k + 2] *
                           temp[static_cast<size_t>(std::clamp(y + k, 0, src.height - 1) *
                                                    src.width + x)];
                }
                out.at(x, y) =
                    static_cast<uint8_t>(std::clamp(sum / 16.0f, 0.0f, 255.0f));
            }
        }
        return out;
    }

    static Derivatives derivatives(const ImageBuffer& image) {
        Derivatives d;
        d.w = image.width;
        d.h = image.height;
        const size_t n = static_cast<size_t>(d.w * d.h);
        d.ix.resize(n);
        d.iy.resize(n);
        d.ixx.resize(n);
        d.ixy.resize(n);
        d.iyy.resize(n);
        d.mag.resize(n);
        auto value = [&](int x, int y) {
            return image.gray(std::clamp(x, 0, d.w - 1), std::clamp(y, 0, d.h - 1)) /
                   255.0f;
        };
        for (int y = 0; y < d.h; ++y) {
            for (int x = 0; x < d.w; ++x) {
                const size_t i = static_cast<size_t>(y * d.w + x);
                d.ix[i] = 0.5f * (value(x + 1, y) - value(x - 1, y));
                d.iy[i] = 0.5f * (value(x, y + 1) - value(x, y - 1));
                d.ixx[i] = value(x + 1, y) - 2.0f * value(x, y) + value(x - 1, y);
                d.iyy[i] = value(x, y + 1) - 2.0f * value(x, y) + value(x, y - 1);
                d.ixy[i] = 0.25f * (value(x + 1, y + 1) - value(x + 1, y - 1) -
                                     value(x - 1, y + 1) + value(x - 1, y - 1));
                d.mag[i] = std::hypot(d.ix[i], d.iy[i]);
                d.max_mag = std::max(d.max_mag, d.mag[i]);
            }
        }
        return d;
    }

    static ImageBuffer gradient_image(const Derivatives& d) {
        ImageBuffer out = math::make_gray(d.w, d.h, 0);
        const float scale = 255.0f / std::max(1e-6f, d.max_mag);
        for (size_t i = 0; i < out.data.size(); ++i) {
            out.data[i] =
                static_cast<uint8_t>(std::clamp(d.mag[i] * scale, 0.0f, 255.0f));
        }
        return out;
    }

    static float local_variance(const ImageBuffer& rgb, int x, int y, int radius) {
        std::array<double, 3> sum{};
        std::array<double, 3> sum2{};
        int count = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const int xx = std::clamp(x + dx, 0, rgb.width - 1);
                const int yy = std::clamp(y + dy, 0, rgb.height - 1);
                for (int c = 0; c < 3; ++c) {
                    const double v = rgb.at(xx, yy, c) / 255.0;
                    sum[static_cast<size_t>(c)] += v;
                    sum2[static_cast<size_t>(c)] += v * v;
                }
                ++count;
            }
        }
        float variance = 0.0f;
        for (int c = 0; c < 3; ++c) {
            const double mean = sum[static_cast<size_t>(c)] / count;
            variance += static_cast<float>(sum2[static_cast<size_t>(c)] / count - mean * mean);
        }
        return variance / 3.0f;
    }

    static std::vector<EdgePoint> localize(const ImageBuffer& edges, const Derivatives& d) {
        std::vector<EdgePoint> out;
        for (int y = 1; y < d.h - 1; ++y) {
            for (int x = 1; x < d.w - 1; ++x) {
                if (!edges.at(x, y)) {
                    continue;
                }
                const size_t i = static_cast<size_t>(y * d.w + x);
                const float a = d.ixx[i];
                const float b = d.ixy[i];
                const float c = d.iyy[i];
                const float disc = std::sqrt(std::max(0.0f, (a - c) * (a - c) + 4.0f * b * b));
                const float l1 = 0.5f * (a + c + disc);
                const float l2 = 0.5f * (a + c - disc);
                const float lambda = std::fabs(l1) >= std::fabs(l2) ? l1 : l2;
                Vec2 normal{b, lambda - a};
                float length = std::hypot(normal.x, normal.y);
                if (length < 1e-6f) {
                    normal = {d.ix[i], d.iy[i]};
                    length = std::hypot(normal.x, normal.y);
                }
                if (length < 1e-6f) {
                    normal = {1.0f, 0.0f};
                    length = 1.0f;
                }
                normal.x /= length;
                normal.y /= length;
                const float denominator = a * normal.x * normal.x +
                                          2.0f * b * normal.x * normal.y +
                                          c * normal.y * normal.y;
                float t = 0.0f;
                if (std::fabs(denominator) > 1e-6f) {
                    t = -(d.ix[i] * normal.x + d.iy[i] * normal.y) / denominator;
                }
                // Canny provides robust chain topology. Steger localization
                // refines it only when the Hessian extremum lies in this pixel.
                if (std::fabs(t) > 0.5f || !std::isfinite(t)) {
                    t = 0.0f;
                }
                out.push_back({{static_cast<float>(x) + t * normal.x,
                                static_cast<float>(y) + t * normal.y},
                               normal, d.mag[i] / d.max_mag, y * d.w + x});
            }
        }
        return out;
    }

    static std::vector<Chain> link_and_simplify(const std::vector<EdgePoint>& points,
                                                const ImageBuffer& edge_map, float epsilon) {
        const int w = edge_map.width;
        const int h = edge_map.height;
        std::vector<int> point_at(static_cast<size_t>(w * h), -1);
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            point_at[static_cast<size_t>(points[static_cast<size_t>(i)].pixel)] = i;
        }
        std::vector<std::vector<int>> neighbors(points.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            const int p = points[static_cast<size_t>(i)].pixel;
            const int x = p % w;
            const int y = p / w;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                        continue;
                    }
                    // Do not add a diagonal across an occupied orthogonal
                    // corner. It creates triangular junctions and crossing
                    // straight-line edges from an otherwise one-pixel chain.
                    if (dx != 0 && dy != 0 &&
                        (point_at[static_cast<size_t>(y * w + xx)] >= 0 ||
                         point_at[static_cast<size_t>(yy * w + x)] >= 0)) {
                        continue;
                    }
                    const int q = point_at[static_cast<size_t>(yy * w + xx)];
                    if (q >= 0) {
                        neighbors[static_cast<size_t>(i)].push_back(q);
                    }
                }
            }
        }

        std::set<std::pair<int, int>> used;
        std::vector<Chain> chains;
        auto trace = [&](int start, int next) {
            std::vector<int> ids{start};
            int previous = start;
            int current = next;
            used.insert(std::minmax(start, next));
            while (true) {
                ids.push_back(current);
                if (neighbors[static_cast<size_t>(current)].size() != 2) {
                    break;
                }
                const int candidate =
                    neighbors[static_cast<size_t>(current)][0] == previous
                        ? neighbors[static_cast<size_t>(current)][1]
                        : neighbors[static_cast<size_t>(current)][0];
                const auto key = std::minmax(current, candidate);
                if (used.count(key)) {
                    break;
                }
                used.insert(key);
                previous = current;
                current = candidate;
            }
            if (ids.size() < 2) {
                return;
            }
            std::vector<Vec2> original;
            original.reserve(ids.size());
            for (int id : ids) {
                original.push_back(points[static_cast<size_t>(id)].p);
            }
            const std::vector<Vec2> simple =
                vision::RamerDouglasPeucker::simplify(original, epsilon);
            if (simple.size() < 2) {
                return;
            }
            Chain chain;
            for (const Vec2& p : simple) {
                size_t best = 0;
                float best_d = std::numeric_limits<float>::max();
                for (size_t j = 0; j < original.size(); ++j) {
                    const float distance = math::dist2(p, original[j]);
                    if (distance < best_d) {
                        best_d = distance;
                        best = j;
                    }
                }
                chain.points.push_back(points[static_cast<size_t>(ids[best])]);
                chain.points.back().p = p;
            }
            chains.push_back(std::move(chain));
        };

        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            if (neighbors[static_cast<size_t>(i)].size() == 2) {
                continue;
            }
            for (int neighbor : neighbors[static_cast<size_t>(i)]) {
                if (!used.count(std::minmax(i, neighbor))) {
                    trace(i, neighbor);
                }
            }
        }
        // Remaining degree-2 components are closed edge chains.
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            for (int neighbor : neighbors[static_cast<size_t>(i)]) {
                if (!used.count(std::minmax(i, neighbor))) {
                    trace(i, neighbor);
                }
            }
        }
        return chains;
    }

    static float bilinear(const ImageBuffer& image, float x, float y, int channel) {
        x = std::clamp(x, 0.0f, static_cast<float>(image.width - 1));
        y = std::clamp(y, 0.0f, static_cast<float>(image.height - 1));
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = std::min(image.width - 1, x0 + 1);
        const int y1 = std::min(image.height - 1, y0 + 1);
        const float tx = x - x0;
        const float ty = y - y0;
        const float a = image.at(x0, y0, channel) * (1.0f - tx) +
                        image.at(x1, y0, channel) * tx;
        const float b = image.at(x0, y1, channel) * (1.0f - tx) +
                        image.at(x1, y1, channel) * tx;
        return (a * (1.0f - ty) + b * ty) / 255.0f;
    }

    Profile sample_profile(const ImageBuffer& rgb, Vec2 a, Vec2 b) const {
        Profile p;
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float length = std::max(1e-6f, std::hypot(dx, dy));
        const Vec2 normal{-dy / length, dx / length};
        constexpr int samples = 7;
        std::array<float, 3> sum2{};
        for (int s = 0; s < samples; ++s) {
            const float t = (s + 0.5f) / samples;
            const float x = a.x + t * dx;
            const float y = a.y + t * dy;
            for (int c = 0; c < 3; ++c) {
                const float left =
                    bilinear(rgb, x + profile_offset * normal.x,
                             y + profile_offset * normal.y, c);
                const float right =
                    bilinear(rgb, x - profile_offset * normal.x,
                             y - profile_offset * normal.y, c);
                p.left[static_cast<size_t>(c)] += left;
                p.right[static_cast<size_t>(c)] += right;
                sum2[static_cast<size_t>(c)] += left * left + right * right;
            }
        }
        for (int c = 0; c < 3; ++c) {
            p.left[static_cast<size_t>(c)] /= samples;
            p.right[static_cast<size_t>(c)] /= samples;
            const float mean =
                0.5f * (p.left[static_cast<size_t>(c)] + p.right[static_cast<size_t>(c)]);
            p.along_variance +=
                sum2[static_cast<size_t>(c)] / (2.0f * samples) - mean * mean;
            const float delta =
                p.left[static_cast<size_t>(c)] - p.right[static_cast<size_t>(c)];
            p.contrast += delta * delta;
        }
        p.contrast = std::sqrt(p.contrast);
        p.along_variance /= 3.0f;
        return p;
    }

    static int find_vertex(std::vector<Vertex>& vertices, Vec2 p) {
        for (int i = 0; i < static_cast<int>(vertices.size()); ++i) {
            if (math::dist2(vertices[static_cast<size_t>(i)].p, p) <= 0.80f * 0.80f) {
                return i;
            }
        }
        vertices.push_back({p, {}});
        return static_cast<int>(vertices.size()) - 1;
    }

    void add_segment(Result& r, const ImageBuffer& rgb, Vec2 a, Vec2 b, float strength,
                     bool synthetic) const {
        if (math::dist2(a, b) < 0.25f) {
            return;
        }
        const int va = find_vertex(r.vertices, a);
        const int vb = find_vertex(r.vertices, b);
        if (va == vb) {
            return;
        }
        for (int edge : r.vertices[static_cast<size_t>(va)].outgoing) {
            if (r.half_edges[static_cast<size_t>(edge)].destination == vb) {
                return;
            }
        }
        const Profile forward = sample_profile(rgb, a, b);
        Profile reverse = forward;
        std::swap(reverse.left, reverse.right);
        const int e = static_cast<int>(r.half_edges.size());
        r.half_edges.push_back({va, vb, e + 1, -1, forward, strength, synthetic});
        r.half_edges.push_back({vb, va, e, -1, reverse, strength, synthetic});
        r.vertices[static_cast<size_t>(va)].outgoing.push_back(e);
        r.vertices[static_cast<size_t>(vb)].outgoing.push_back(e + 1);
        if (!synthetic) {
            ++r.simplified_segments;
        }
    }

    void build_dcel(Result& r, const ImageBuffer& rgb) const {
        for (const Chain& chain : r.chains) {
            for (size_t i = 1; i < chain.points.size(); ++i) {
                const float strength =
                    0.5f * (chain.points[i - 1].strength + chain.points[i].strength);
                add_segment(r, rgb, chain.points[i - 1].p, chain.points[i].p, strength, false);
            }
        }
    }

    struct KdNode {
        int point = -1;
        int left = -1;
        int right = -1;
        int axis = 0;
    };

    class KdTree {
    public:
        KdTree(const std::vector<Vec2>& points) : points_(points) {
            std::vector<int> ids(points.size());
            std::iota(ids.begin(), ids.end(), 0);
            root_ = build(ids, 0);
        }
        std::vector<int> radius(Vec2 query, float r) const {
            std::vector<int> out;
            search(root_, query, r * r, out);
            return out;
        }

    private:
        const std::vector<Vec2>& points_;
        std::vector<KdNode> nodes_;
        int root_ = -1;

        int build(std::vector<int>& ids, int depth) {
            if (ids.empty()) {
                return -1;
            }
            const int axis = depth & 1;
            const size_t middle = ids.size() / 2;
            std::nth_element(ids.begin(), ids.begin() + static_cast<ptrdiff_t>(middle), ids.end(),
                             [&](int a, int b) {
                                 return axis == 0 ? points_[static_cast<size_t>(a)].x <
                                                        points_[static_cast<size_t>(b)].x
                                                  : points_[static_cast<size_t>(a)].y <
                                                        points_[static_cast<size_t>(b)].y;
                             });
            const int point = ids[middle];
            std::vector<int> left(ids.begin(), ids.begin() + static_cast<ptrdiff_t>(middle));
            std::vector<int> right(ids.begin() + static_cast<ptrdiff_t>(middle + 1), ids.end());
            const int node = static_cast<int>(nodes_.size());
            nodes_.push_back({point, -1, -1, axis});
            nodes_[static_cast<size_t>(node)].left = build(left, depth + 1);
            nodes_[static_cast<size_t>(node)].right = build(right, depth + 1);
            return node;
        }

        void search(int node, Vec2 q, float r2, std::vector<int>& out) const {
            if (node < 0) {
                return;
            }
            const KdNode& n = nodes_[static_cast<size_t>(node)];
            const Vec2 p = points_[static_cast<size_t>(n.point)];
            if (math::dist2(p, q) <= r2) {
                out.push_back(n.point);
            }
            const float delta = n.axis == 0 ? q.x - p.x : q.y - p.y;
            search(delta <= 0.0f ? n.left : n.right, q, r2, out);
            if (delta * delta <= r2) {
                search(delta <= 0.0f ? n.right : n.left, q, r2, out);
            }
        }
    };

    static bool segments_intersect(Vec2 a, Vec2 b, Vec2 c, Vec2 d) {
        auto cross = [](Vec2 p, Vec2 q, Vec2 r) {
            return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
        };
        const float ab_c = cross(a, b, c);
        const float ab_d = cross(a, b, d);
        const float cd_a = cross(c, d, a);
        const float cd_b = cross(c, d, b);
        return ab_c * ab_d < -1e-5f && cd_a * cd_b < -1e-5f;
    }

    static float color_distance(const std::array<float, 3>& a,
                                const std::array<float, 3>& b) {
        float sum = 0.0f;
        for (int c = 0; c < 3; ++c) {
            const float delta = a[static_cast<size_t>(c)] - b[static_cast<size_t>(c)];
            sum += delta * delta;
        }
        return std::sqrt(sum / 3.0f);
    }

    float bridge_energy(const Result& r, int vi, int vj, const Derivatives& d) const {
        const Vec2 a = r.vertices[static_cast<size_t>(vi)].p;
        const Vec2 b = r.vertices[static_cast<size_t>(vj)].p;
        const Vec2 chord{b.x - a.x, b.y - a.y};
        const float distance = std::hypot(chord.x, chord.y);
        if (distance < 1e-5f) {
            return std::numeric_limits<float>::max();
        }
        const Vec2 direction{chord.x / distance, chord.y / distance};
        auto endpoint_data = [&](int vertex) {
            const int edge = r.vertices[static_cast<size_t>(vertex)].outgoing.front();
            const HalfEdge& e = r.half_edges[static_cast<size_t>(edge)];
            const Vec2 other = r.vertices[static_cast<size_t>(e.destination)].p;
            const Vec2 here = r.vertices[static_cast<size_t>(vertex)].p;
            const float len = std::max(1e-6f, math::dist(here, other));
            const Vec2 tangent{(here.x - other.x) / len, (here.y - other.y) / len};
            return std::make_pair(tangent, e.profile);
        };
        const auto left = endpoint_data(vi);
        const auto right = endpoint_data(vj);
        const float bend_i = 1.0f - (left.first.x * direction.x + left.first.y * direction.y);
        const float bend_j =
            1.0f - (right.first.x * -direction.x + right.first.y * -direction.y);
        const float geometric = distance / gap_radius + 0.65f * (bend_i + bend_j);
        const float same = color_distance(left.second.left, right.second.right) +
                           color_distance(left.second.right, right.second.left);
        const float swapped = color_distance(left.second.left, right.second.left) +
                              color_distance(left.second.right, right.second.right);
        const float color = std::min(same, swapped);

        const int samples = std::max(2, static_cast<int>(std::ceil(distance)));
        float support = 0.0f;
        for (int s = 0; s <= samples; ++s) {
            const float t = static_cast<float>(s) / samples;
            const int x = std::clamp(static_cast<int>(std::lround(a.x + t * chord.x)), 0, d.w - 1);
            const int y = std::clamp(static_cast<int>(std::lround(a.y + t * chord.y)), 0, d.h - 1);
            support += d.mag[static_cast<size_t>(y * d.w + x)] / d.max_mag;
        }
        support /= samples + 1;
        return geometric + color_weight * color + gradient_weight * (1.0f - support);
    }

    static std::vector<Vec2> geodesic_bridge(Vec2 start, Vec2 goal, Vec2 tangent,
                                             const Derivatives& d, float radius) {
        constexpr int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        constexpr int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        const int sx = std::clamp(static_cast<int>(std::lround(start.x)), 0, d.w - 1);
        const int sy = std::clamp(static_cast<int>(std::lround(start.y)), 0, d.h - 1);
        const int gx = std::clamp(static_cast<int>(std::lround(goal.x)), 0, d.w - 1);
        const int gy = std::clamp(static_cast<int>(std::lround(goal.y)), 0, d.h - 1);
        int initial_dir = 0;
        float best_dot = -2.0f;
        for (int dir = 0; dir < 8; ++dir) {
            const float len = std::hypot(static_cast<float>(dx[dir]), static_cast<float>(dy[dir]));
            const float dot = tangent.x * dx[dir] / len + tangent.y * dy[dir] / len;
            if (dot > best_dot) {
                best_dot = dot;
                initial_dir = dir;
            }
        }
        const int states = d.w * d.h * 8;
        std::vector<float> distance(static_cast<size_t>(states),
                                    std::numeric_limits<float>::max());
        std::vector<int> previous(static_cast<size_t>(states), -1);
        using Entry = std::pair<float, int>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
        const int start_state = (sy * d.w + sx) * 8 + initial_dir;
        distance[static_cast<size_t>(start_state)] = 0.0f;
        queue.push({0.0f, start_state});
        int goal_state = -1;
        while (!queue.empty()) {
            const auto [priority, state] = queue.top();
            queue.pop();
            const float known = distance[static_cast<size_t>(state)];
            if (priority > known + std::hypot(static_cast<float>((state / 8) % d.w - gx),
                                              static_cast<float>((state / 8) / d.w - gy)) *
                                       0.03f + 1e-5f) {
                continue;
            }
            const int pixel = state / 8;
            const int old_dir = state % 8;
            const int x = pixel % d.w;
            const int y = pixel / d.w;
            if (std::abs(x - gx) <= 1 && std::abs(y - gy) <= 1) {
                goal_state = state;
                break;
            }
            if (std::hypot(static_cast<float>(x - sx), static_cast<float>(y - sy)) > radius) {
                continue;
            }
            for (int dir = 0; dir < 8; ++dir) {
                const int xx = x + dx[dir];
                const int yy = y + dy[dir];
                if (xx < 0 || yy < 0 || xx >= d.w || yy >= d.h) {
                    continue;
                }
                const float old_angle = std::atan2(static_cast<float>(dy[old_dir]),
                                                   static_cast<float>(dx[old_dir]));
                const float new_angle =
                    std::atan2(static_cast<float>(dy[dir]), static_cast<float>(dx[dir]));
                const float curve = 1.0f - std::cos(new_angle - old_angle);
                const float support =
                    d.mag[static_cast<size_t>(yy * d.w + xx)] / d.max_mag;
                const float step = (dir & 1 ? 1.4142f : 1.0f) *
                                   (0.08f + 1.0f - support + 0.40f * curve);
                const int next = (yy * d.w + xx) * 8 + dir;
                const float candidate = known + step;
                if (candidate < distance[static_cast<size_t>(next)]) {
                    distance[static_cast<size_t>(next)] = candidate;
                    previous[static_cast<size_t>(next)] = state;
                    const float heuristic =
                        0.03f * std::hypot(static_cast<float>(xx - gx),
                                           static_cast<float>(yy - gy));
                    queue.push({candidate + heuristic, next});
                }
            }
        }
        if (goal_state < 0) {
            return {start, goal};
        }
        std::vector<Vec2> reverse;
        for (int state = goal_state; state >= 0; state = previous[static_cast<size_t>(state)]) {
            const int pixel = state / 8;
            reverse.push_back({static_cast<float>(pixel % d.w),
                               static_cast<float>(pixel / d.w)});
            if (state == start_state) {
                break;
            }
        }
        std::reverse(reverse.begin(), reverse.end());
        if (!reverse.empty()) {
            reverse.front() = start;
            reverse.back() = goal;
        }
        return vision::RamerDouglasPeucker::simplify(reverse, 0.75f);
    }

    int split_segment(Result& r, const ImageBuffer& rgb, int edge_index, Vec2 point) const {
        const int even = edge_index & ~1;
        HalfEdge old = r.half_edges[static_cast<size_t>(even)];
        const int twin = old.twin;
        const int a = old.origin;
        const int b = old.destination;
        const int junction = find_vertex(r.vertices, point);
        if (junction == a || junction == b) {
            return junction;
        }
        auto& out_b = r.vertices[static_cast<size_t>(b)].outgoing;
        out_b.erase(std::remove(out_b.begin(), out_b.end(), twin), out_b.end());
        r.half_edges[static_cast<size_t>(even)] =
            {a, junction, twin, -1,
             sample_profile(rgb, r.vertices[static_cast<size_t>(a)].p, point),
             old.strength, old.synthetic};
        Profile reverse = r.half_edges[static_cast<size_t>(even)].profile;
        std::swap(reverse.left, reverse.right);
        r.half_edges[static_cast<size_t>(twin)] =
            {junction, a, even, -1, reverse, old.strength, old.synthetic};
        r.vertices[static_cast<size_t>(junction)].outgoing.push_back(twin);
        add_segment(r, rgb, point, r.vertices[static_cast<size_t>(b)].p,
                    old.strength, old.synthetic);
        return junction;
    }

    void close_gaps(Result& r, const ImageBuffer& rgb, const Derivatives& d) const {
        std::vector<int> dangling;
        std::vector<Vec2> points;
        for (int v = 0; v < static_cast<int>(r.vertices.size()); ++v) {
            if (r.vertices[static_cast<size_t>(v)].outgoing.size() < 2 &&
                !r.vertices[static_cast<size_t>(v)].outgoing.empty()) {
                dangling.push_back(v);
                points.push_back(r.vertices[static_cast<size_t>(v)].p);
            }
        }
        r.dangling_before = static_cast<int>(dangling.size());
        KdTree tree(points);
        struct Candidate {
            int a, b;
            float energy;
        };
        std::vector<Candidate> candidates;
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            const int vi = dangling[static_cast<size_t>(i)];
            const int incident =
                r.vertices[static_cast<size_t>(vi)].outgoing.front();
            const Vec2 other =
                r.vertices[static_cast<size_t>(
                               r.half_edges[static_cast<size_t>(incident)].destination)]
                    .p;
            const Vec2 p = points[static_cast<size_t>(i)];
            const float tangent_length = std::max(1e-6f, math::dist(p, other));
            const Vec2 tangent{(p.x - other.x) / tangent_length,
                               (p.y - other.y) / tangent_length};
            for (int j : tree.radius(p, gap_radius)) {
                if (j <= i) {
                    continue;
                }
                const Vec2 q = points[static_cast<size_t>(j)];
                const Vec2 chord{q.x - p.x, q.y - p.y};
                if (tangent.x * chord.x + tangent.y * chord.y <= 0.0f) {
                    continue;
                }
                const int vj = dangling[static_cast<size_t>(j)];
                const int other_edge =
                    r.vertices[static_cast<size_t>(vj)].outgoing.front();
                const Vec2 q_other =
                    r.vertices[static_cast<size_t>(
                                   r.half_edges[static_cast<size_t>(other_edge)].destination)]
                        .p;
                const Vec2 tangent_j{(q.x - q_other.x) /
                                         std::max(1e-6f, math::dist(q, q_other)),
                                     (q.y - q_other.y) /
                                         std::max(1e-6f, math::dist(q, q_other))};
                if (tangent_j.x * -chord.x + tangent_j.y * -chord.y <= 0.0f) {
                    continue;
                }
                r.candidate_gaps.push_back({vi, vj});
                candidates.push_back({vi, vj, bridge_energy(r, vi, vj, d)});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.energy < b.energy; });
        std::set<int> used;
        for (const Candidate& c : candidates) {
            if (c.energy > max_bridge_energy || used.count(c.a) || used.count(c.b)) {
                continue;
            }
            const Vec2 a = r.vertices[static_cast<size_t>(c.a)].p;
            const Vec2 b = r.vertices[static_cast<size_t>(c.b)].p;
            bool crosses = false;
            for (size_t e = 0; e < r.half_edges.size(); e += 2) {
                const HalfEdge& edge = r.half_edges[e];
                if (edge.origin == c.a || edge.destination == c.a || edge.origin == c.b ||
                    edge.destination == c.b) {
                    continue;
                }
                if (segments_intersect(a, b, r.vertices[static_cast<size_t>(edge.origin)].p,
                                       r.vertices[static_cast<size_t>(edge.destination)].p)) {
                    crosses = true;
                    break;
                }
            }
            if (!crosses) {
                const int incident =
                    r.vertices[static_cast<size_t>(c.a)].outgoing.front();
                const Vec2 other =
                    r.vertices[static_cast<size_t>(
                                   r.half_edges[static_cast<size_t>(incident)].destination)]
                        .p;
                const float length = std::max(1e-6f, math::dist(a, other));
                const Vec2 tangent{(a.x - other.x) / length, (a.y - other.y) / length};
                const auto path = geodesic_bridge(a, b, tangent, d, gap_radius * 1.4f);
                for (size_t p = 1; p < path.size(); ++p) {
                    add_segment(r, rgb, path[p - 1], path[p], 0.0f, true);
                }
                r.accepted_gaps.push_back({c.a, c.b});
                used.insert(c.a);
                used.insert(c.b);
            }
        }

        // Remaining endpoints may terminate into the middle of a sound edge.
        // Project onto nearby segments in the forward cone, split the segment,
        // and route a gradient-following path into the new T-junction.
        const int original_half_edges = static_cast<int>(r.half_edges.size());
        for (int vi : dangling) {
            if (used.count(vi) || r.vertices[static_cast<size_t>(vi)].outgoing.size() != 1) {
                continue;
            }
            const Vec2 p = r.vertices[static_cast<size_t>(vi)].p;
            const int incident = r.vertices[static_cast<size_t>(vi)].outgoing.front();
            const Vec2 other =
                r.vertices[static_cast<size_t>(
                               r.half_edges[static_cast<size_t>(incident)].destination)]
                    .p;
            const float tangent_length = std::max(1e-6f, math::dist(p, other));
            const Vec2 tangent{(p.x - other.x) / tangent_length,
                               (p.y - other.y) / tangent_length};
            int best_edge = -1;
            Vec2 best_point{};
            float best_energy = std::min(max_bridge_energy, 1.55f);
            for (int edge = 0; edge < original_half_edges; edge += 2) {
                const HalfEdge& target = r.half_edges[static_cast<size_t>(edge)];
                if (target.origin == vi || target.destination == vi || target.synthetic) {
                    continue;
                }
                const Vec2 a = r.vertices[static_cast<size_t>(target.origin)].p;
                const Vec2 b = r.vertices[static_cast<size_t>(target.destination)].p;
                const Vec2 ab{b.x - a.x, b.y - a.y};
                const float ab2 = math::dist2(a, b);
                if (ab2 < 1e-6f) {
                    continue;
                }
                const float t = std::clamp(((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / ab2,
                                           0.0f, 1.0f);
                if (t < 0.12f || t > 0.88f) {
                    continue;
                }
                const Vec2 q{a.x + t * ab.x, a.y + t * ab.y};
                const Vec2 chord{q.x - p.x, q.y - p.y};
                const float distance = std::hypot(chord.x, chord.y);
                if (distance > gap_radius || distance < 1.5f ||
                    tangent.x * chord.x + tangent.y * chord.y <= 0.0f) {
                    continue;
                }
                const Profile& source =
                    r.half_edges[static_cast<size_t>(incident)].profile;
                if (source.contrast < 0.035f || target.profile.contrast < 0.035f) {
                    continue;
                }
                const float profile_cost =
                    std::min(color_distance(source.left, target.profile.left) +
                                 color_distance(source.right, target.profile.right),
                             color_distance(source.left, target.profile.right) +
                                 color_distance(source.right, target.profile.left));
                const Vec2 direction{chord.x / distance, chord.y / distance};
                const float bend =
                    1.0f - tangent.x * direction.x - tangent.y * direction.y;
                float support = 0.0f;
                const int samples = std::max(2, static_cast<int>(std::ceil(distance)));
                for (int sample = 0; sample <= samples; ++sample) {
                    const float u = static_cast<float>(sample) / samples;
                    const int x = std::clamp(
                        static_cast<int>(std::lround(p.x + u * chord.x)), 0, d.w - 1);
                    const int y = std::clamp(
                        static_cast<int>(std::lround(p.y + u * chord.y)), 0, d.h - 1);
                    support += d.mag[static_cast<size_t>(y * d.w + x)] / d.max_mag;
                }
                support /= samples + 1;
                const float energy = distance / gap_radius + 0.65f * bend +
                                     color_weight * profile_cost +
                                     gradient_weight * (1.0f - support);
                if (energy < best_energy) {
                    best_energy = energy;
                    best_edge = edge;
                    best_point = q;
                }
            }
            if (best_edge >= 0) {
                const int junction = split_segment(r, rgb, best_edge, best_point);
                const auto path =
                    geodesic_bridge(p, r.vertices[static_cast<size_t>(junction)].p,
                                    tangent, d, gap_radius * 1.4f);
                for (size_t k = 1; k < path.size(); ++k) {
                    add_segment(r, rgb, path[k - 1], path[k], 0.0f, true);
                }
                r.candidate_gaps.push_back({vi, junction});
                r.accepted_gaps.push_back({vi, junction});
                used.insert(vi);
                ++r.t_junctions;
            }
        }
        r.dangling_after = 0;
        for (const Vertex& vertex : r.vertices) {
            r.dangling_after += vertex.outgoing.size() == 1 ? 1 : 0;
        }
    }

    static void link_half_edges(Result& r) {
        for (Vertex& vertex : r.vertices) {
            std::sort(vertex.outgoing.begin(), vertex.outgoing.end(), [&](int ea, int eb) {
                const HalfEdge& a = r.half_edges[static_cast<size_t>(ea)];
                const HalfEdge& b = r.half_edges[static_cast<size_t>(eb)];
                const Vec2 pa = r.vertices[static_cast<size_t>(a.destination)].p;
                const Vec2 pb = r.vertices[static_cast<size_t>(b.destination)].p;
                return std::atan2(pa.y - vertex.p.y, pa.x - vertex.p.x) <
                       std::atan2(pb.y - vertex.p.y, pb.x - vertex.p.x);
            });
        }
        for (int e = 0; e < static_cast<int>(r.half_edges.size()); ++e) {
            HalfEdge& edge = r.half_edges[static_cast<size_t>(e)];
            const auto& outgoing =
                r.vertices[static_cast<size_t>(edge.destination)].outgoing;
            const auto twin_it = std::find(outgoing.begin(), outgoing.end(), edge.twin);
            if (twin_it == outgoing.end() || outgoing.empty()) {
                continue;
            }
            const size_t index = static_cast<size_t>(twin_it - outgoing.begin());
            // Immediately counterclockwise from twin, as specified.
            edge.next = outgoing[(index + 1) % outgoing.size()];
        }
    }

    void extract_faces(Result& r, const ImageBuffer& rgb) const {
        r.faces.clear();
        std::vector<uint8_t> visited(r.half_edges.size(), 0);
        const int min_area =
            std::max(12, static_cast<int>(min_face_area_frac * r.width * r.height));
        for (int start = 0; start < static_cast<int>(r.half_edges.size()); ++start) {
            if (visited[static_cast<size_t>(start)]) {
                continue;
            }
            Face face;
            std::vector<int> local;
            int current = start;
            bool valid = true;
            while (true) {
                if (current < 0 || current >= static_cast<int>(r.half_edges.size())) {
                    valid = false;
                    break;
                }
                if (std::find(local.begin(), local.end(), current) != local.end()) {
                    valid = current == start;
                    break;
                }
                if (visited[static_cast<size_t>(current)]) {
                    valid = false;
                    break;
                }
                local.push_back(current);
                const HalfEdge& edge = r.half_edges[static_cast<size_t>(current)];
                face.polygon.push_back(
                    r.vertices[static_cast<size_t>(edge.origin)].p);
                current = edge.next;
            }
            for (int edge : local) {
                visited[static_cast<size_t>(edge)] = 1;
            }
            if (!valid || face.polygon.size() < 3) {
                continue;
            }
            // Convert image y-down coordinates to Cartesian y-up for winding.
            float signed_area = 0.0f;
            for (size_t i = 0; i < face.polygon.size(); ++i) {
                const Vec2& a = face.polygon[i];
                const Vec2& b = face.polygon[(i + 1) % face.polygon.size()];
                signed_area += b.x * a.y - a.x * b.y;
            }
            signed_area *= 0.5f;
            if (signed_area <= 10.0f) {
                continue;  // clockwise/unbounded face
            }
            face.half_edges = std::move(local);
            measure_face(face, rgb);
            if (face.area >= min_area && face.variance <= max_face_variance) {
                r.faces.push_back(std::move(face));
            }
        }
    }

    static void measure_face(Face& face, const ImageBuffer& rgb) {
        const ImageBuffer mask =
            math::rasterize_polygon(face.polygon, rgb.width, rgb.height);
        int min_x = rgb.width, min_y = rgb.height, max_x = -1, max_y = -1;
        std::array<double, 3> sum{};
        std::array<double, 3> sum2{};
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0;
        double sz = 0.0, sxz = 0.0, syz = 0.0, sz2 = 0.0;
        for (int y = 0; y < rgb.height; ++y) {
            for (int x = 0; x < rgb.width; ++x) {
                if (!mask.at(x, y)) {
                    continue;
                }
                ++face.area;
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
                const double xn = static_cast<double>(x) / std::max(1, rgb.width - 1);
                const double yn = static_cast<double>(y) / std::max(1, rgb.height - 1);
                const double z = rgb.gray(x, y) / 255.0;
                sx += xn; sy += yn; sxx += xn * xn; sxy += xn * yn; syy += yn * yn;
                sz += z; sxz += xn * z; syz += yn * z; sz2 += z * z;
                for (int c = 0; c < 3; ++c) {
                    const double value = rgb.at(x, y, c) / 255.0;
                    sum[static_cast<size_t>(c)] += value;
                    sum2[static_cast<size_t>(c)] += value * value;
                }
            }
        }
        if (face.area <= 0) {
            return;
        }
        for (int c = 0; c < 3; ++c) {
            face.mean[static_cast<size_t>(c)] =
                static_cast<float>(sum[static_cast<size_t>(c)] / face.area);
            face.variance += static_cast<float>(
                sum2[static_cast<size_t>(c)] / face.area -
                face.mean[static_cast<size_t>(c)] * face.mean[static_cast<size_t>(c)]);
        }
        face.variance /= 3.0f;
        double matrix[3][4] = {
            {sxx, sxy, sx, sxz},
            {sxy, syy, sy, syz},
            {sx, sy, static_cast<double>(face.area), sz}};
        for (int col = 0; col < 3; ++col) {
            int pivot = col;
            for (int row = col + 1; row < 3; ++row) {
                if (std::fabs(matrix[row][col]) > std::fabs(matrix[pivot][col])) {
                    pivot = row;
                }
            }
            for (int k = col; k < 4; ++k) {
                std::swap(matrix[col][k], matrix[pivot][k]);
            }
            const double divisor = std::fabs(matrix[col][col]) > 1e-9
                                       ? matrix[col][col] : 1e-9;
            for (int k = col; k < 4; ++k) {
                matrix[col][k] /= divisor;
            }
            for (int row = 0; row < 3; ++row) {
                if (row == col) continue;
                const double factor = matrix[row][col];
                for (int k = col; k < 4; ++k) {
                    matrix[row][k] -= factor * matrix[col][k];
                }
            }
        }
        for (int i = 0; i < 3; ++i) {
            face.affine[static_cast<size_t>(i)] = static_cast<float>(matrix[i][3]);
        }
        const double a = face.affine[0], b = face.affine[1], c = face.affine[2];
        const double sse = sz2 + a * a * sxx + b * b * syy +
                           c * c * face.area + 2.0 * a * b * sxy +
                           2.0 * a * c * sx + 2.0 * b * c * sy -
                           2.0 * a * sxz - 2.0 * b * syz - 2.0 * c * sz;
        face.affine_residual =
            static_cast<float>(std::max(0.0, sse / face.area));
        face.bbox = {static_cast<float>(min_x), static_cast<float>(min_y),
                     static_cast<float>(max_x - min_x + 1),
                     static_cast<float>(max_y - min_y + 1)};
    }

    void classify_and_merge_faces(Result& r) const {
        const int n = static_cast<int>(r.faces.size());
        std::vector<int> parent(static_cast<size_t>(n));
        std::iota(parent.begin(), parent.end(), 0);
        auto root = [&parent](int x) {
            while (parent[static_cast<size_t>(x)] != x) {
                parent[static_cast<size_t>(x)] =
                    parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
                x = parent[static_cast<size_t>(x)];
            }
            return x;
        };
        std::vector<ImageBuffer> masks;
        masks.reserve(r.faces.size());
        for (const Face& face : r.faces) {
            masks.push_back(math::rasterize_polygon(face.polygon, r.width, r.height));
        }
        std::vector<int> edge_face(r.half_edges.size(), -1);
        for (int face = 0; face < n; ++face) {
            for (int edge : r.faces[static_cast<size_t>(face)].half_edges) {
                edge_face[static_cast<size_t>(edge)] = face;
            }
        }
        for (int edge = 0; edge < static_cast<int>(r.half_edges.size()); ++edge) {
            const int twin = r.half_edges[static_cast<size_t>(edge)].twin;
            if (twin < 0 || edge > twin) {
                continue;
            }
            const int a = edge_face[static_cast<size_t>(edge)];
            const int b = edge_face[static_cast<size_t>(twin)];
            if (a < 0 || b < 0 || a == b) {
                continue;
            }
            const Face& fa = r.faces[static_cast<size_t>(a)];
            const Face& fb = r.faces[static_cast<size_t>(b)];
            const float ax = (fb.bbox.x + 0.5f * fb.bbox.w) / std::max(1, r.width - 1);
            const float ay = (fb.bbox.y + 0.5f * fb.bbox.h) / std::max(1, r.height - 1);
            const float bx = (fa.bbox.x + 0.5f * fa.bbox.w) / std::max(1, r.width - 1);
            const float by = (fa.bbox.y + 0.5f * fa.bbox.h) / std::max(1, r.height - 1);
            const float luma_a = 0.299f * fa.mean[0] + 0.587f * fa.mean[1] + 0.114f * fa.mean[2];
            const float luma_b = 0.299f * fb.mean[0] + 0.587f * fb.mean[1] + 0.114f * fb.mean[2];
            const float cross_fit =
                std::fabs((fa.affine[0] * ax + fa.affine[1] * ay + fa.affine[2]) - luma_b) +
                std::fabs((fb.affine[0] * bx + fb.affine[1] * by + fb.affine[2]) - luma_a);
            const bool weak_physical_edge =
                r.half_edges[static_cast<size_t>(edge)].profile.contrast < 0.09f;
            const bool compatible_planes =
                cross_fit < 0.12f && fa.affine_residual < 0.025f &&
                fb.affine_residual < 0.025f;
            if (weak_physical_edge || compatible_planes ||
                color_distance(fa.mean, fb.mean) < face_merge_color) {
                parent[static_cast<size_t>(root(b))] = root(a);
            }
        }
        r.labels.assign(static_cast<size_t>(r.width * r.height), 0);
        std::vector<int> remap(static_cast<size_t>(n), -1);
        int next = 1;  // zero remains the unbounded/background face
        // Larger faces first; smaller nested faces then overwrite their interior.
        std::vector<int> order(static_cast<size_t>(n));
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return r.faces[static_cast<size_t>(a)].area >
                   r.faces[static_cast<size_t>(b)].area;
        });
        for (int id : order) {
            const int rt = root(id);
            if (remap[static_cast<size_t>(rt)] < 0) {
                remap[static_cast<size_t>(rt)] = next++;
            }
            const int label = remap[static_cast<size_t>(rt)];
            r.faces[static_cast<size_t>(id)].region_label = label;
            const ImageBuffer& mask = masks[static_cast<size_t>(id)];
            for (size_t p = 0; p < r.labels.size(); ++p) {
                if (mask.data[p]) {
                    r.labels[p] = label;
                }
            }
        }
    }
};

}  // namespace edge_geom

namespace {

struct Checks {
    int passed = 0;
    int total = 0;
    std::vector<std::string> failures;
    void expect(bool ok, const std::string& message) {
        ++total;
        if (ok) {
            ++passed;
        } else {
            failures.push_back(message);
        }
    }
};

math::ImageBuffer sample_rgb(const ProviderLoadedSample* provider,
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

vision::GrayImage gt_boundary(const ProviderLoadedSample* provider, int w, int h) {
    if (provider == nullptr) {
        return {};
    }
    vision::GrayImage gt = provider->sample.boundary;
    if (gt.empty() && !provider->ground_truth.empty()) {
        gt = to_gray(contour::boundary_pixels(to_contour(binarize_mask(provider->ground_truth))));
    }
    if (!gt.empty() && (gt.width != w || gt.height != h)) {
        gt = resize_nearest(gt, w, h);
    }
    return binarize_mask(gt);
}

vision::GrayImage chains_overlay(const vision::GrayImage& base,
                                 const std::vector<edge_geom::Chain>& chains) {
    vision::GrayImage out = base;
    for (uint8_t& value : out.data) {
        value = static_cast<uint8_t>(value / 2);
    }
    for (const auto& chain : chains) {
        std::vector<math::Vec2> points;
        for (const auto& point : chain.points) {
            points.push_back(point.p);
        }
        out = overlay_polyline(out, points, false, 255, false);
    }
    return out;
}

vision::GrayImage dcel_overlay(const vision::GrayImage& base,
                               const edge_geom::Result& result, bool bridges) {
    vision::GrayImage out = base;
    for (uint8_t& value : out.data) {
        value = static_cast<uint8_t>(value / 2);
    }
    for (size_t e = 0; e < result.half_edges.size(); e += 2) {
        const auto& edge = result.half_edges[e];
        if (edge.synthetic != bridges) {
            continue;
        }
        const auto a = result.vertices[static_cast<size_t>(edge.origin)].p;
        const auto b = result.vertices[static_cast<size_t>(edge.destination)].p;
        plot_line(out, static_cast<int>(std::lround(a.x)), static_cast<int>(std::lround(a.y)),
                  static_cast<int>(std::lround(b.x)), static_cast<int>(std::lround(b.y)),
                  bridges ? 190 : 255);
    }
    return out;
}

vision::GrayImage closure_overlay(const vision::GrayImage& base,
                                  const edge_geom::Result& result) {
    vision::GrayImage out = dcel_overlay(base, result, false);
    for (const auto& gap : result.accepted_gaps) {
        const auto a = result.vertices[static_cast<size_t>(gap.first)].p;
        const auto b = result.vertices[static_cast<size_t>(gap.second)].p;
        plot_line(out, static_cast<int>(std::lround(a.x)), static_cast<int>(std::lround(a.y)),
                  static_cast<int>(std::lround(b.x)), static_cast<int>(std::lround(b.y)), 128);
    }
    return out;
}

vision::GrayImage gap_candidate_overlay(const vision::GrayImage& base,
                                        const edge_geom::Result& result) {
    vision::GrayImage out = base;
    for (uint8_t& value : out.data) {
        value = static_cast<uint8_t>(value / 3);
    }
    for (const auto& gap : result.candidate_gaps) {
        const auto a = result.vertices[static_cast<size_t>(gap.first)].p;
        const auto b = result.vertices[static_cast<size_t>(gap.second)].p;
        plot_line(out, static_cast<int>(std::lround(a.x)), static_cast<int>(std::lround(a.y)),
                  static_cast<int>(std::lround(b.x)), static_cast<int>(std::lround(b.y)), 110);
    }
    for (const auto& gap : result.accepted_gaps) {
        const auto a = result.vertices[static_cast<size_t>(gap.first)].p;
        const auto b = result.vertices[static_cast<size_t>(gap.second)].p;
        plot_line(out, static_cast<int>(std::lround(a.x)), static_cast<int>(std::lround(a.y)),
                  static_cast<int>(std::lround(b.x)), static_cast<int>(std::lround(b.y)), 255);
    }
    return out;
}

vision::GrayImage face_overlay(const vision::GrayImage& base,
                               const std::vector<edge_geom::Face>& faces) {
    vision::GrayImage out = base;
    for (uint8_t& value : out.data) {
        value = static_cast<uint8_t>(value / 2);
    }
    for (const auto& face : faces) {
        out = overlay_polyline(out, face.polygon, true, 255, false);
    }
    return out;
}

vision::GrayImage classified_boundary_overlay(const vision::GrayImage& base,
                                              const edge_geom::Result& result) {
    vision::GrayImage out = base;
    for (uint8_t& value : out.data) {
        value = static_cast<uint8_t>(value / 2);
    }
    std::set<int> labels;
    for (int label : result.labels) {
        if (label > 0) {
            labels.insert(label);
        }
    }
    for (int label : labels) {
        math::ImageBuffer region = math::make_gray(result.width, result.height, 0);
        for (size_t p = 0; p < result.labels.size(); ++p) {
            region.data[p] = result.labels[p] == label ? 255 : 0;
        }
        const auto components = vision::ConnectedComponentLabeler::label(region);
        for (const auto& component : components.components) {
            if (component.area < 8) {
                continue;
            }
            math::ImageBuffer component_mask =
                math::make_gray(result.width, result.height, 0);
            for (size_t p = 0; p < components.labels.size(); ++p) {
                component_mask.data[p] =
                    components.labels[p] == component.label ? 255 : 0;
            }
            const auto contour = vision::MooreNeighborTracer::trace(component_mask);
            if (!contour.closed || contour.points.size() < 3) {
                continue;
            }
            const auto polyline =
                vision::RamerDouglasPeucker::simplify(contour.points, 0.75f);
            out = overlay_polyline(out, polyline, true, 255, false);
        }
    }
    return out;
}

class EdgeGeomAtom {
public:
    std::vector<ProviderLoadedSample> providers;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"edge_geom"};
    Checks checks;
    std::ostringstream summary;
    std::ostringstream edges;
    std::ostringstream faces;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load boundary-annotated samples");
        const auto mission =
            load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 16, max_side, true);
        providers = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        report.n_inputs = static_cast<int>(samples.size());
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name
                  << '\n';
        return !samples.empty();
    }

    void run(const std::string& dir) {
        print_banner("run sub-pixel edges -> DCEL -> elastica closure -> faces");
        ScopedTimer timer(&report.elapsed_ms);
        summary << "file\tedge_pixels\tchains\tsegments\thalf_edges\tdangling_before"
                   "\tcandidates\tbridges\tt_junctions\tdangling_after\tfaces\tboundary_f1\tms\n";
        edges << "file\tedge\torigin\tdestination\ttwin\tnext\tsynthetic\tstrength"
                 "\tcontrast\talong_variance\n";
        faces << "file\tface\tarea\tx\ty\tw\th\tmean_r\tmean_g\tmean_b\tvariance"
                 "\taffine_a\taffine_b\taffine_c\taffine_residual\n";
        double f1_sum = 0.0;
        int scored = 0;

        for (size_t i = 0; i < samples.size(); ++i) {
            const ProviderLoadedSample* provider = i < providers.size() ? &providers[i] : nullptr;
            const vision::GrayImage luma = mission_luma_image(provider, samples[i].image);
            const math::ImageBuffer rgb = sample_rgb(provider, luma, max_side);
            edge_geom::Pipeline pipeline;
            edge_geom::Result result;
            double elapsed = 0.0;
            {
                ScopedTimer sample_timer(&elapsed);
                result = pipeline.run(rgb, luma);
            }
            const vision::GrayImage gt = gt_boundary(provider, luma.width, luma.height);
            const vision::GrayImage predicted =
                label_boundaries(result.labels, luma.width, luma.height);
            double f1 = 0.0;
            if (!gt.empty()) {
                f1 = contour::boundary_f1(to_contour(predicted), to_contour(gt), 2);
                f1_sum += f1;
                ++scored;
            }
            const std::string stem = stem_of(samples[i].row.file);
            validate(stem, result);
            write_artifacts(dir, stem, luma, gt, result);
            write_tables(stem, result);
            summary << samples[i].row.file << '\t' << result.raw_edge_pixels << '\t'
                    << result.chains.size() << '\t' << result.simplified_segments << '\t'
                    << result.half_edges.size() << '\t' << result.dangling_before << '\t'
                    << result.candidate_gaps.size() << '\t' << result.accepted_gaps.size()
                    << '\t' << result.t_junctions << '\t' << result.dangling_after << '\t'
                    << result.faces.size() << '\t' << f1 << '\t' << elapsed << '\n';
            std::cout << "  " << samples[i].row.file << "  chains=" << result.chains.size()
                      << " bridges=" << result.accepted_gaps.size()
                      << " faces=" << result.faces.size() << " f1=" << std::fixed
                      << std::setprecision(3) << f1 << "  " << elapsed << " ms\n";
            ++report.n_outputs;
        }
        std::ostringstream note;
        note << "edge-first geometry; mean boundary F1="
             << (scored ? f1_sum / scored : 0.0);
        report.notes.push_back(note.str());
        if (scored >= 4) {
            checks.expect(f1_sum / scored >= 0.15,
                          "dataset: mean face-boundary F1 remains above 0.15");
        }
    }

    void write(const std::string& dir) {
        write_text(dir, "edge_geom.tsv", summary.str());
        write_text(dir, "half_edges.tsv", edges.str());
        write_text(dir, "faces.tsv", faces.str());
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "invariants: " << checks.passed << " / " << checks.total << " passed\n";
        for (const auto& failure : checks.failures) {
            std::cout << "  FAIL  " << failure << '\n';
        }
        std::cout << "artifacts -> " << dir << '\n';
    }

    int status() const { return checks.failures.empty() ? 0 : 1; }

private:
    int max_side = 192;

    void validate(const std::string& stem, const edge_geom::Result& r) {
        checks.expect(r.raw_edge_pixels > 0, stem + ": vector edge extraction is non-empty");
        checks.expect(!r.chains.empty(), stem + ": edge pixels link into chains");
        checks.expect(r.half_edges.size() % 2 == 0, stem + ": every half-edge has a twin");
        bool twins_ok = true;
        bool finite_profiles = true;
        for (size_t e = 0; e < r.half_edges.size(); ++e) {
            const auto& edge = r.half_edges[e];
            twins_ok = twins_ok && edge.twin >= 0 &&
                        edge.twin < static_cast<int>(r.half_edges.size()) &&
                        r.half_edges[static_cast<size_t>(edge.twin)].twin ==
                            static_cast<int>(e);
            finite_profiles = finite_profiles && std::isfinite(edge.profile.contrast) &&
                              std::isfinite(edge.profile.along_variance);
        }
        checks.expect(twins_ok, stem + ": DCEL twin relation is involutive");
        checks.expect(finite_profiles, stem + ": bilateral profiles are finite");
        checks.expect(r.dangling_after <= r.dangling_before,
                      stem + ": closure never creates dangling endpoints");
        checks.expect(r.labels.size() == static_cast<size_t>(r.width * r.height),
                      stem + ": final label map covers the image");
        bool closed = true;
        for (const auto& face : r.faces) {
            closed = closed && face.polygon.size() >= 3 && face.area > 0 &&
                     std::isfinite(face.variance);
        }
        checks.expect(closed, stem + ": accepted faces are closed and measurable");
    }

    void save(const std::string& dir, const std::string& name,
              const vision::GrayImage& image) {
        vision::save_pgm(vision::join_path(dir, name), image);
        written.push_back(name);
    }

    void write_text(const std::string& dir, const std::string& name,
                    const std::string& text) {
        vision::write_text_file(vision::join_path(dir, name), text);
        written.push_back(name);
    }

    void write_artifacts(const std::string& dir, const std::string& stem,
                         const vision::GrayImage& luma, const vision::GrayImage& gt,
                         const edge_geom::Result& r) {
        save(dir, stem + "_00_input_base.pgm", luma);
        save(dir, stem + "_01_gradient_magnitude.pgm", to_gray(r.gradient));
        save(dir, stem + "_01_coarse_structural_edges.pgm", to_gray(r.coarse_edges));
        save(dir, stem + "_01_fine_detail_edges.pgm", to_gray(r.detail_edges));
        save(dir, stem + "_01_thin_edges.pgm", to_gray(r.thin_edges));
        save(dir, stem + "_01_vector_chains_overlay.pgm", chains_overlay(luma, r.chains));
        save(dir, stem + "_02_dcel_profile_edges.pgm", dcel_overlay(luma, r, false));
        save(dir, stem + "_03_gap_candidates.pgm", gap_candidate_overlay(luma, r));
        save(dir, stem + "_04_elastica_closure.pgm", closure_overlay(luma, r));
        save(dir, stem + "_05_planar_faces.pgm", face_overlay(luma, r.faces));
        save(dir, stem + "_06_region_labels.pgm",
             colorize_labels(r.labels, r.width, r.height));
        // Draw the explicit closed vectors. Re-deriving boundaries from the
        // sparse face label map is what produced the dotted/staggered artifact.
        save(dir, stem + "_06_final_overlay.pgm",
             classified_boundary_overlay(luma, r));
        if (!gt.empty()) {
            save(dir, stem + "_gt_boundary_overlay.pgm", overlay_mask(luma, gt));
        }
    }

    void write_tables(const std::string& stem, const edge_geom::Result& r) {
        for (size_t e = 0; e < r.half_edges.size(); ++e) {
            const auto& edge = r.half_edges[e];
            edges << stem << '\t' << e << '\t' << edge.origin << '\t'
                  << edge.destination << '\t' << edge.twin << '\t' << edge.next << '\t'
                  << (edge.synthetic ? 1 : 0) << '\t' << edge.strength << '\t'
                  << edge.profile.contrast << '\t' << edge.profile.along_variance << '\n';
        }
        for (size_t f = 0; f < r.faces.size(); ++f) {
            const auto& face = r.faces[f];
            faces << stem << '\t' << f + 1 << '\t' << face.area << '\t' << face.bbox.x
                  << '\t' << face.bbox.y << '\t' << face.bbox.w << '\t' << face.bbox.h
                  << '\t' << face.mean[0] << '\t' << face.mean[1] << '\t'
                  << face.mean[2] << '\t' << face.variance << '\t'
                  << face.affine[0] << '\t' << face.affine[1] << '\t'
                  << face.affine[2] << '\t' << face.affine_residual << '\n';
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "bsds500", [&](const AtomCli& cli) -> int {
        EdgeGeomAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for edge_geom atom\n";
            return 1;
        }
        if (cli.list_only) {
            for (const auto& sample : atom.samples) {
                std::cout << "  " << sample.row.file << '\n';
            }
            return 0;
        }
        bool override = false;
        for (int i = 1; i < argc; ++i) {
            override = override || std::string(argv[i]) == "--artifacts";
        }
        const std::string request =
            override ? cli.artifact_dir : vision::join_path(cli.artifact_dir, "edge_geom");
        const std::string artifacts = make_artifact_dir(request);
        atom.run(artifacts);
        atom.write(artifacts);
        return atom.status();
    });
}
