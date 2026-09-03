#pragma once

#include "math/contour_compat.hpp"
#include "math/contour_metrics.hpp"
#include "sdf/8ssedt/8SSEDT.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <unordered_set>

namespace contour {

class DualContouring2D {
public:
    struct Hermite {
        Vec2 p;
        Vec2 n;
    };

    std::vector<Vec2> cell_vertices;
    std::vector<std::pair<int, int>> edges;
    std::vector<Polyline> last_loops;

    Polyline extract(const Field& sdf, float iso = 0.0f) {
        const int cw = sdf.width - 1;
        const int ch = sdf.height - 1;
        // Per cell, per side (0=top,1=right,2=bottom,3=left): vertex id used on that side.
        // Ambiguous 4-crossing cells get two QEF vertices so the dual graph stays manifold.
        std::vector<std::array<int, 4>> side_vert(
            static_cast<size_t>(std::max(0, cw * ch)), std::array<int, 4>{-1, -1, -1, -1});
        cell_vertices.clear();
        edges.clear();
        last_loops.clear();
        if (cw <= 0 || ch <= 0) {
            return {};
        }

        auto hermite_edge = [&](int x0, int y0, int x1, int y1) -> std::optional<Hermite> {
            const float v0 = sdf.at(x0, y0) - iso;
            const float v1 = sdf.at(x1, y1) - iso;
            if ((v0 >= 0) == (v1 >= 0)) {
                return std::nullopt;
            }
            const float t = std::fabs(v1 - v0) < 1e-12f ? 0.5f : v0 / (v0 - v1);
            const float tt = std::clamp(t, 0.0f, 1.0f);
            Hermite h;
            h.p = {static_cast<float>(x0) + (x1 - x0) * tt, static_cast<float>(y0) + (y1 - y0) * tt};
            const int ix = std::clamp(static_cast<int>(std::lround(h.p.x)), 0, sdf.width - 1);
            const int iy = std::clamp(static_cast<int>(std::lround(h.p.y)), 0, sdf.height - 1);
            h.n = normalize(ExactSDF::gradient_at(sdf, h.p.x, h.p.y));
            if (length(h.n) < 1e-6f) {
                h.n = normalize(ExactSDF::gradient(sdf, ix, iy));
            }
            return h;
        };

        auto add_clamped = [&](const std::vector<Hermite>& Hs, int x, int y) -> int {
            Vec2 v = solve_qef(Hs, {x + 0.5f, y + 0.5f});
            v.x = std::clamp(v.x, static_cast<float>(x), static_cast<float>(x + 1));
            v.y = std::clamp(v.y, static_cast<float>(y), static_cast<float>(y + 1));
            const int id = static_cast<int>(cell_vertices.size());
            cell_vertices.push_back(v);
            return id;
        };

        for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
                std::array<std::optional<Hermite>, 4> sides;
                sides[0] = hermite_edge(x, y, x + 1, y);
                sides[1] = hermite_edge(x + 1, y, x + 1, y + 1);
                sides[2] = hermite_edge(x, y + 1, x + 1, y + 1);
                sides[3] = hermite_edge(x, y, x, y + 1);
                std::vector<int> present;
                std::vector<Hermite> H;
                for (int s = 0; s < 4; ++s) {
                    if (sides[s]) {
                        present.push_back(s);
                        H.push_back(*sides[s]);
                    }
                }
                if (present.size() < 2) {
                    continue;
                }
                auto& sv = side_vert[static_cast<size_t>(y * cw + x)];
                if (present.size() == 4) {
                    // Asymptotic decider (same pairing idea as marching-squares 5/10).
                    const float v00 = sdf.at(x, y);
                    const float v10 = sdf.at(x + 1, y);
                    const float v11 = sdf.at(x + 1, y + 1);
                    const float v01 = sdf.at(x, y + 1);
                    // Bilinear saddle test (same idea as MS ambiguous cases).
                    const bool pair_tr_bl = (v00 * v11 >= v10 * v01);
                    if (pair_tr_bl) {
                        // top+right and bottom+left
                        const int a = add_clamped({*sides[0], *sides[1]}, x, y);
                        const int b = add_clamped({*sides[2], *sides[3]}, x, y);
                        sv[0] = sv[1] = a;
                        sv[2] = sv[3] = b;
                    } else {
                        // top+left and right+bottom
                        const int a = add_clamped({*sides[0], *sides[3]}, x, y);
                        const int b = add_clamped({*sides[1], *sides[2]}, x, y);
                        sv[0] = sv[3] = a;
                        sv[1] = sv[2] = b;
                    }
                } else {
                    const int id = add_clamped(H, x, y);
                    for (int s : present) {
                        sv[s] = id;
                    }
                }
            }
        }

        auto link = [&](int ia, int ib) {
            if (ia < 0 || ib < 0 || ia == ib) {
                return;
            }
            edges.push_back({std::min(ia, ib), std::max(ia, ib)});
        };
        // Horizontal primal crossings: bottom side of cell above <-> top side of cell below.
        for (int y = 0; y < sdf.height; ++y) {
            for (int x = 0; x < sdf.width - 1; ++x) {
                const float v0 = sdf.at(x, y) - iso;
                const float v1 = sdf.at(x + 1, y) - iso;
                if ((v0 >= 0) == (v1 >= 0)) {
                    continue;
                }
                const int above = (y > 0) ? side_vert[static_cast<size_t>((y - 1) * cw + x)][2] : -1;
                const int below = (y < ch) ? side_vert[static_cast<size_t>(y * cw + x)][0] : -1;
                link(above, below);
            }
        }
        // Vertical primal crossings: right side of left cell <-> left side of right cell.
        for (int y = 0; y < sdf.height - 1; ++y) {
            for (int x = 0; x < sdf.width; ++x) {
                const float v0 = sdf.at(x, y) - iso;
                const float v1 = sdf.at(x, y + 1) - iso;
                if ((v0 >= 0) == (v1 >= 0)) {
                    continue;
                }
                const int left = (x > 0) ? side_vert[static_cast<size_t>(y * cw + (x - 1))][1] : -1;
                const int right = (x < cw) ? side_vert[static_cast<size_t>(y * cw + x)][3] : -1;
                link(left, right);
            }
        }

        // Deduplicate undirected edges.
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

        std::vector<std::vector<int>> adj(cell_vertices.size());
        for (auto [a, b] : edges) {
            adj[static_cast<size_t>(a)].push_back(b);
            adj[static_cast<size_t>(b)].push_back(a);
        }

        auto edge_key = [](int a, int b) -> long long {
            const int lo = std::min(a, b);
            const int hi = std::max(a, b);
            return (static_cast<long long>(lo) << 32) | static_cast<unsigned int>(hi);
        };
        std::unordered_set<long long> used;
        last_loops.clear();

        auto emit = [&](std::vector<int> ids, bool closed) {
            if (ids.size() < 3) {
                return;
            }
            std::vector<Vec2> pts;
            pts.reserve(ids.size());
            for (int id : ids) {
                pts.push_back(cell_vertices[static_cast<size_t>(id)]);
            }
            Polyline poly;
            poly.points = std::move(pts);
            poly.closed = closed;
            last_loops.push_back(std::move(poly));
        };

        for (size_t start = 0; start < cell_vertices.size(); ++start) {
            for (int nb0 : adj[start]) {
                const long long ek0 = edge_key(static_cast<int>(start), nb0);
                if (used.count(ek0)) {
                    continue;
                }
                std::vector<int> loop;
                int prev = -1;
                int cur = static_cast<int>(start);
                bool closed = false;
                for (int guard = 0; guard < static_cast<int>(cell_vertices.size()) + 4; ++guard) {
                    loop.push_back(cur);
                    int nxt = -1;
                    for (int nb : adj[static_cast<size_t>(cur)]) {
                        if (nb == prev) {
                            continue;
                        }
                        const long long ek = edge_key(cur, nb);
                        if (used.count(ek)) {
                            continue;
                        }
                        nxt = nb;
                        used.insert(ek);
                        break;
                    }
                    if (nxt < 0) {
                        break;
                    }
                    prev = cur;
                    cur = nxt;
                    if (cur == static_cast<int>(start)) {
                        closed = true;
                        break;
                    }
                }
                emit(std::move(loop), closed);
            }
        }

        // Pick the loop whose filled polygon best matches the SDF interior.
        const ImageBuffer gt = rasterize_mask_from_field(sdf, iso);
        Polyline best;
        double best_score = -1.0;
        for (const auto& p : last_loops) {
            if (p.points.size() < 3) {
                continue;
            }
            const ImageBuffer pred = rasterize_polygon(p.points, sdf.width, sdf.height);
            const double iou = mask_iou(pred, gt);
            const float area = std::fabs(shoelace(p.points));
            // IoU dominates; slight closed bonus; area as tiny tie-break.
            const double score =
                iou * 1000.0 + (p.closed ? 0.5 : 0.0) + static_cast<double>(area) * 1e-6;
            if (score > best_score) {
                best_score = score;
                best = p;
            }
        }
        if (best.points.empty()) {
            float best_n = -1.0f;
            for (const auto& p : last_loops) {
                const float a = static_cast<float>(p.points.size());
                if (a > best_n) {
                    best_n = a;
                    best = p;
                }
            }
        }
        return best;
    }

    static Vec2 solve_qef(const std::vector<Hermite>& H, const Vec2& cell_center) {
        Vec2 mass{0, 0};
        for (const auto& h : H) {
            mass = mass + h.p;
        }
        mass = mass * (1.0f / static_cast<float>(H.size()));
        (void)cell_center;
        if (H.size() == 1) {
            return H[0].p;
        }
        double ata00 = 0, ata01 = 0, ata11 = 0;
        double atb0 = 0, atb1 = 0;
        for (const auto& h : H) {
            const double nx = h.n.x;
            const double ny = h.n.y;
            const double nb = nx * h.p.x + ny * h.p.y;
            ata00 += nx * nx;
            ata01 += nx * ny;
            ata11 += ny * ny;
            atb0 += nx * nb;
            atb1 += ny * nb;
        }
        auto solve2 = [](double a00, double a01, double a11, double b0, double b1) {
            const double det = a00 * a11 - a01 * a01;
            return Vec2{static_cast<float>((a11 * b0 - a01 * b1) / det),
                        static_cast<float>((-a01 * b0 + a00 * b1) / det)};
        };
        const double det = ata00 * ata11 - ata01 * ata01;
        const double tr = ata00 + ata11;
        const double disc = std::max(0.0, (tr * 0.5) * (tr * 0.5) - det);
        const double l1 = tr * 0.5 + std::sqrt(disc);
        const double l2 = tr * 0.5 - std::sqrt(disc);
        const double cond = (l2 > 1e-12) ? (l1 / l2) : 1.0e9;
        if (l2 < 1e-2 || cond > 100.0) {
            return mass;
        }
        const double scale = 1.0 + ata00 * ata00 + ata11 * ata11;
        if (std::fabs(det) > 1e-6 * scale) {
            return solve2(ata00, ata01, ata11, atb0, atb1);
        }
        const double eps = 1e-3;
        return solve2(ata00 + eps, ata01, ata11 + eps, atb0 + eps * mass.x, atb1 + eps * mass.y);
    }
};

}  // namespace contour
