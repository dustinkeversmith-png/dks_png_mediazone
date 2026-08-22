#pragma once

#include "math/contour_compat.hpp"
#include "math/contour_metrics.hpp"
#include "sdf/8ssedt/8SSEDT.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>

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
        std::vector<int> cell_id(static_cast<size_t>(cw * ch), -1);
        cell_vertices.clear();
        edges.clear();
        last_loops.clear();

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

        for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
                std::vector<Hermite> H;
                if (auto h = hermite_edge(x, y, x + 1, y)) H.push_back(*h);
                if (auto h = hermite_edge(x + 1, y, x + 1, y + 1)) H.push_back(*h);
                if (auto h = hermite_edge(x, y + 1, x + 1, y + 1)) H.push_back(*h);
                if (auto h = hermite_edge(x, y, x, y + 1)) H.push_back(*h);
                if (H.empty()) {
                    continue;
                }
                Vec2 v = solve_qef(H, {x + 0.5f, y + 0.5f});
                v.x = std::clamp(v.x, static_cast<float>(x), static_cast<float>(x + 1));
                v.y = std::clamp(v.y, static_cast<float>(y), static_cast<float>(y + 1));
                cell_id[static_cast<size_t>(y * cw + x)] = static_cast<int>(cell_vertices.size());
                cell_vertices.push_back(v);
            }
        }

        auto connect = [&](int ax, int ay, int bx, int by) {
            if (ax < 0 || ay < 0 || bx < 0 || by < 0 || ax >= cw || bx >= cw || ay >= ch || by >= ch) {
                return;
            }
            const int ia = cell_id[static_cast<size_t>(ay * cw + ax)];
            const int ib = cell_id[static_cast<size_t>(by * cw + bx)];
            if (ia >= 0 && ib >= 0 && ia != ib) {
                edges.push_back({ia, ib});
            }
        };
        for (int y = 0; y < sdf.height; ++y) {
            for (int x = 0; x < sdf.width - 1; ++x) {
                const float v0 = sdf.at(x, y) - iso;
                const float v1 = sdf.at(x + 1, y) - iso;
                if ((v0 >= 0) == (v1 >= 0)) {
                    continue;
                }
                connect(x, y - 1, x, y);
            }
        }
        for (int y = 0; y < sdf.height - 1; ++y) {
            for (int x = 0; x < sdf.width; ++x) {
                const float v0 = sdf.at(x, y) - iso;
                const float v1 = sdf.at(x, y + 1) - iso;
                if ((v0 >= 0) == (v1 >= 0)) {
                    continue;
                }
                connect(x - 1, y, x, y);
            }
        }

        std::vector<std::vector<int>> adj(cell_vertices.size());
        for (auto [a, b] : edges) {
            if (a == b) {
                continue;
            }
            adj[static_cast<size_t>(a)].push_back(b);
            adj[static_cast<size_t>(b)].push_back(a);
        }
        last_loops.clear();
        std::vector<char> seen(cell_vertices.size(), 0);
        float best_a = -1.0f;
        Polyline best;
        for (size_t start = 0; start < cell_vertices.size(); ++start) {
            if (seen[start] || adj[start].empty()) {
                continue;
            }
            std::vector<int> loop;
            int prev = -1;
            int cur = static_cast<int>(start);
            for (int guard = 0; guard < static_cast<int>(cell_vertices.size()) + 2; ++guard) {
                loop.push_back(cur);
                seen[static_cast<size_t>(cur)] = 1;
                int nxt = -1;
                for (int nb : adj[static_cast<size_t>(cur)]) {
                    if (nb != prev) {
                        nxt = nb;
                        break;
                    }
                }
                if (nxt < 0) {
                    break;
                }
                prev = cur;
                cur = nxt;
                if (cur == static_cast<int>(start)) {
                    break;
                }
            }
            std::vector<Vec2> pts;
            pts.reserve(loop.size());
            for (int id : loop) {
                pts.push_back(cell_vertices[static_cast<size_t>(id)]);
            }
            Polyline poly;
            poly.points = pts;
            poly.closed = true;
            last_loops.push_back(poly);
            const float a = shoelace(pts);
            if (a > best_a) {
                best_a = a;
                best = std::move(poly);
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
