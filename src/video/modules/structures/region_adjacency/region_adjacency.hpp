#pragma once

#include "contour_kit/types.hpp"
#include "color/lab_color_space.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

namespace contour {

// Region adjacency graph + DCEL half-edges from a raster label map.
class RegionAdjacency {
public:
    struct Face {
        int id = 0;
        int area = 0;
        Vec2 centroid;
        Lab mean_lab;
        float mean_intensity = 0;
        int half_edge = -1;
    };

    struct Vertex {
        float x = 0;
        float y = 0;
        int half_edge = -1;
    };

    struct HalfEdge {
        int origin = -1;
        int twin = -1;
        int next = -1;
        int incident_face = -1;
        float hue_delta = 0;
        float intensity_step = 0;
    };

    struct AdjEdge {
        int a = 0;
        int b = 0;
        int length = 0;
        float mean_hue_delta = 0;
        float mean_intensity_step = 0;
    };

    struct Result {
        int width = 0;
        int height = 0;
        std::vector<int> labels;
        std::vector<Face> faces;
        std::vector<Vertex> vertices;
        std::vector<HalfEdge> half_edges;
        std::vector<AdjEdge> adj;
    };

    static Result from_labels(const std::vector<int>& labels, int w, int h, const ImageBuffer& image) {
        Result r;
        r.width = w;
        r.height = h;
        r.labels = labels;
        int max_id = -1;
        for (int v : labels) {
            max_id = std::max(max_id, v);
        }
        r.faces.assign(static_cast<size_t>(max_id + 1), Face{});
        for (int i = 0; i <= max_id; ++i) {
            r.faces[static_cast<size_t>(i)].id = i;
        }
        std::vector<Lab> acc_lab(static_cast<size_t>(max_id + 1));
        std::vector<double> acc_x(static_cast<size_t>(max_id + 1), 0);
        std::vector<double> acc_y(static_cast<size_t>(max_id + 1), 0);
        std::vector<double> acc_i(static_cast<size_t>(max_id + 1), 0);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int lab = labels[static_cast<size_t>(y * w + x)];
                if (lab < 0) {
                    continue;
                }
                auto& f = r.faces[static_cast<size_t>(lab)];
                ++f.area;
                const Lab p = LabColor::at(image, x, y);
                acc_lab[static_cast<size_t>(lab)].L += p.L;
                acc_lab[static_cast<size_t>(lab)].a += p.a;
                acc_lab[static_cast<size_t>(lab)].b += p.b;
                acc_x[static_cast<size_t>(lab)] += x + 0.5;
                acc_y[static_cast<size_t>(lab)] += y + 0.5;
                acc_i[static_cast<size_t>(lab)] += image.gray(x, y);
            }
        }
        for (int i = 0; i <= max_id; ++i) {
            auto& f = r.faces[static_cast<size_t>(i)];
            if (f.area <= 0) {
                continue;
            }
            const float inv = 1.0f / static_cast<float>(f.area);
            f.mean_lab = {acc_lab[static_cast<size_t>(i)].L * inv, acc_lab[static_cast<size_t>(i)].a * inv,
                          acc_lab[static_cast<size_t>(i)].b * inv};
            f.centroid = {static_cast<float>(acc_x[static_cast<size_t>(i)] * inv),
                          static_cast<float>(acc_y[static_cast<size_t>(i)] * inv)};
            f.mean_intensity = static_cast<float>(acc_i[static_cast<size_t>(i)] * inv);
        }

        struct Key {
            int a, b;
            bool operator<(const Key& o) const { return a < o.a || (a == o.a && b < o.b); }
        };
        std::map<Key, AdjEdge> adj;
        auto touch = [&](int x0, int y0, int x1, int y1) {
            const int a = labels[static_cast<size_t>(y0 * w + x0)];
            const int b = labels[static_cast<size_t>(y1 * w + x1)];
            if (a == b || a < 0 || b < 0) {
                return;
            }
            Key k{std::min(a, b), std::max(a, b)};
            auto& e = adj[k];
            e.a = k.a;
            e.b = k.b;
            ++e.length;
            e.mean_hue_delta += std::sqrt(LabColor::delta2(LabColor::at(image, x0, y0), LabColor::at(image, x1, y1)));
            e.mean_intensity_step += std::fabs(image.gray(x0, y0) - image.gray(x1, y1));
        };
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (x + 1 < w) {
                    touch(x, y, x + 1, y);
                }
                if (y + 1 < h) {
                    touch(x, y, x, y + 1);
                }
            }
        }
        r.adj.reserve(adj.size());
        for (auto& kv : adj) {
            auto e = kv.second;
            if (e.length > 0) {
                e.mean_hue_delta /= static_cast<float>(e.length);
                e.mean_intensity_step /= static_cast<float>(e.length);
            }
            r.adj.push_back(e);
        }

        build_dcel(r, image);
        return r;
    }

private:
    static int vertex_id(Result& r, std::map<std::pair<int, int>, int>& vm, int gx, int gy) {
        const auto key = std::make_pair(gx, gy);
        const auto it = vm.find(key);
        if (it != vm.end()) {
            return it->second;
        }
        Vertex v;
        v.x = static_cast<float>(gx);
        v.y = static_cast<float>(gy);
        const int id = static_cast<int>(r.vertices.size());
        r.vertices.push_back(v);
        vm[key] = id;
        return id;
    }

    static void build_dcel(Result& r, const ImageBuffer& image) {
        const int w = r.width;
        const int h = r.height;
        std::map<std::pair<int, int>, int> vm;
        auto add_pair = [&](int ox, int oy, int dx, int dy, int face, int twin_face, float hue, float step) {
            const int a = vertex_id(r, vm, ox, oy);
            const int b = vertex_id(r, vm, ox + dx, oy + dy);
            HalfEdge e0, e1;
            e0.origin = a;
            e0.incident_face = face;
            e0.hue_delta = hue;
            e0.intensity_step = step;
            e1.origin = b;
            e1.incident_face = twin_face;
            e1.hue_delta = hue;
            e1.intensity_step = step;
            const int i0 = static_cast<int>(r.half_edges.size());
            const int i1 = i0 + 1;
            e0.twin = i1;
            e1.twin = i0;
            r.half_edges.push_back(e0);
            r.half_edges.push_back(e1);
            if (r.vertices[static_cast<size_t>(a)].half_edge < 0) {
                r.vertices[static_cast<size_t>(a)].half_edge = i0;
            }
            if (r.vertices[static_cast<size_t>(b)].half_edge < 0) {
                r.vertices[static_cast<size_t>(b)].half_edge = i1;
            }
            if (face >= 0 && face < static_cast<int>(r.faces.size()) && r.faces[static_cast<size_t>(face)].half_edge < 0) {
                r.faces[static_cast<size_t>(face)].half_edge = i0;
            }
        };

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w - 1; ++x) {
                const int a = r.labels[static_cast<size_t>(y * w + x)];
                const int b = r.labels[static_cast<size_t>(y * w + x + 1)];
                if (a == b) {
                    continue;
                }
                const float hue =
                    std::sqrt(LabColor::delta2(LabColor::at(image, x, y), LabColor::at(image, x + 1, y)));
                const float step = std::fabs(image.gray(x, y) - image.gray(x + 1, y));
                add_pair(x + 1, y, 0, 1, a, b, hue, step);
            }
        }
        for (int y = 0; y < h - 1; ++y) {
            for (int x = 0; x < w; ++x) {
                const int a = r.labels[static_cast<size_t>(y * w + x)];
                const int b = r.labels[static_cast<size_t>((y + 1) * w + x)];
                if (a == b) {
                    continue;
                }
                const float hue =
                    std::sqrt(LabColor::delta2(LabColor::at(image, x, y), LabColor::at(image, x, y + 1)));
                const float step = std::fabs(image.gray(x, y) - image.gray(x, y + 1));
                add_pair(x, y + 1, 1, 0, a, b, hue, step);
            }
        }

        std::vector<std::vector<int>> out(r.vertices.size());
        for (int i = 0; i < static_cast<int>(r.half_edges.size()); ++i) {
            out[static_cast<size_t>(r.half_edges[static_cast<size_t>(i)].origin)].push_back(i);
        }
        auto ang = [&](int he) {
            const auto& e = r.half_edges[static_cast<size_t>(he)];
            const auto& a = r.vertices[static_cast<size_t>(e.origin)];
            const auto& b = r.vertices[static_cast<size_t>(r.half_edges[static_cast<size_t>(e.twin)].origin)];
            return std::atan2(b.y - a.y, b.x - a.x);
        };
        for (auto& bundle : out) {
            std::sort(bundle.begin(), bundle.end(), [&](int i, int j) { return ang(i) < ang(j); });
            const int m = static_cast<int>(bundle.size());
            for (int i = 0; i < m; ++i) {
                const int he = bundle[static_cast<size_t>(i)];
                const int nxt = bundle[static_cast<size_t>((i + 1) % m)];
                r.half_edges[static_cast<size_t>(r.half_edges[static_cast<size_t>(he)].twin)].next = nxt;
            }
        }
    }
};

}  // namespace contour
