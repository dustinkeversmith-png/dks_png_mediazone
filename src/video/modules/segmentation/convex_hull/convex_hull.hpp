#pragma once

#include "../../math/vision_types.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"

#include <vector>
#include <algorithm>
#include <cmath>

namespace vision {

class ConvexHull {
public:
    struct Defect {
        Vec2 hull_a, hull_b, farthest;
        float depth = 0.0f;
    };

    struct Result {
        std::vector<Vec2> hull;
        std::vector<Defect> defects;
        float hull_area = 0.0f;
    };

    static int cross(const Vec2& o, const Vec2& a, const Vec2& b) {
        const float v = (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        if (v > 1e-5f) {
            return 1;
        }
        if (v < -1e-5f) {
            return -1;
        }
        return 0;
    }

    static std::vector<Vec2> monotone_chain(std::vector<Vec2> pts) {
        if (pts.size() < 2) {
            return pts;
        }
        std::sort(pts.begin(), pts.end(), [](const Vec2& a, const Vec2& b) {
            if (a.x == b.x) {
                return a.y < b.y;
            }
            return a.x < b.x;
        });
        pts.erase(std::unique(pts.begin(), pts.end(),
                              [](const Vec2& a, const Vec2& b) {
                                  return dist2(a, b) < 1e-8f;
                              }),
                  pts.end());

        std::vector<Vec2> lower, upper;
        for (const auto& p : pts) {
            while (lower.size() >= 2 &&
                   cross(lower[lower.size() - 2], lower.back(), p) <= 0) {
                lower.pop_back();
            }
            lower.push_back(p);
        }
        for (int i = static_cast<int>(pts.size()) - 1; i >= 0; --i) {
            const auto& p = pts[static_cast<size_t>(i)];
            while (upper.size() >= 2 &&
                   cross(upper[upper.size() - 2], upper.back(), p) <= 0) {
                upper.pop_back();
            }
            upper.push_back(p);
        }
        lower.pop_back();
        upper.pop_back();
        lower.insert(lower.end(), upper.begin(), upper.end());
        return lower;
    }

    static float point_line_dist(const Vec2& p, const Vec2& a, const Vec2& b) {
        const float vx = b.x - a.x;
        const float vy = b.y - a.y;
        const float mag = std::hypot(vx, vy);
        if (mag < 1e-6f) {
            return dist(p, a);
        }
        return std::fabs(vx * (a.y - p.y) - vy * (a.x - p.x)) / mag;
    }

    static Result analyze(const GrayImage& image) {
        Result r;
        auto contour = MooreNeighborTracer::trace(image);
        r.hull = monotone_chain(contour.points);
        r.hull_area = MooreNeighborTracer::shoelace(r.hull);
        if (r.hull.size() < 3 || contour.points.size() < 3) {
            return r;
        }
        for (size_t i = 0; i < r.hull.size(); ++i) {
            const Vec2 a = r.hull[i];
            const Vec2 b = r.hull[(i + 1) % r.hull.size()];
            Defect best;
            best.hull_a = a;
            best.hull_b = b;
            for (const auto& p : contour.points) {
                const float d = point_line_dist(p, a, b);
                const float side = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
                if (side > 0.0f && d > best.depth) {
                    best.depth = d;
                    best.farthest = p;
                }
            }
            if (best.depth > 2.0f) {
                r.defects.push_back(best);
            }
        }
        return r;
    }
};

}  // namespace vision
