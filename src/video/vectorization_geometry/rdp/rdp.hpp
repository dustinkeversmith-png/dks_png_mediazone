#pragma once

#include "../../math/vision_types.hpp"

#include <vector>
#include <cmath>
#include <algorithm>

namespace vision {

class RamerDouglasPeucker {
public:
    static float perp_dist(const Vec2& p, const Vec2& a, const Vec2& b) {
        const float vx = b.x - a.x;
        const float vy = b.y - a.y;
        const float mag = std::hypot(vx, vy);
        if (mag < 1e-8f) {
            return dist(p, a);
        }
        return std::fabs(vx * (a.y - p.y) - vy * (a.x - p.x)) / mag;
    }

    static std::vector<Vec2> simplify(const std::vector<Vec2>& pts, float epsilon) {
        if (pts.size() < 3) {
            return pts;
        }
        std::vector<Vec2> out;
        recurse(pts, 0, static_cast<int>(pts.size()) - 1, epsilon, out);
        out.push_back(pts.back());
        return out;
    }

    static float max_error(const std::vector<Vec2>& original, const std::vector<Vec2>& simplified) {
        if (simplified.size() < 2) {
            return 0.0f;
        }
        float m = 0.0f;
        for (const auto& p : original) {
            float best = 1.0e12f;
            for (size_t i = 1; i < simplified.size(); ++i) {
                best = std::min(best, perp_dist(p, simplified[i - 1], simplified[i]));
            }
            m = std::max(m, best);
        }
        return m;
    }

private:
    static void recurse(const std::vector<Vec2>& pts, int start, int end, float eps, std::vector<Vec2>& out) {
        float max_d = 0.0f;
        int idx = start;
        for (int i = start + 1; i < end; ++i) {
            const float d = perp_dist(pts[static_cast<size_t>(i)], pts[static_cast<size_t>(start)],
                                      pts[static_cast<size_t>(end)]);
            if (d > max_d) {
                max_d = d;
                idx = i;
            }
        }
        if (max_d > eps) {
            recurse(pts, start, idx, eps, out);
            recurse(pts, idx, end, eps, out);
        } else {
            out.push_back(pts[static_cast<size_t>(start)]);
        }
    }
};

}  // namespace vision
