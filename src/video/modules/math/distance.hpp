#pragma once

#include "vision_types.hpp"
#include "geometry.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace math {

inline std::vector<Vec2> thin_points(std::vector<Vec2> pts, size_t cap = 400) {
    if (pts.size() <= cap) {
        return pts;
    }
    std::vector<Vec2> out;
    out.reserve(cap);
    const float step = static_cast<float>(pts.size()) / static_cast<float>(cap);
    for (size_t i = 0; i < cap; ++i) {
        out.push_back(pts[static_cast<size_t>(i * step)]);
    }
    return out;
}

inline double mean_min_distance(const std::vector<Vec2>& a, const std::vector<Vec2>& b) {
    const auto aa = thin_points(a);
    const auto bb = thin_points(b);
    if (aa.empty() || bb.empty()) {
        return 0.0;
    }
    double acc = 0;
    for (const auto& p : aa) {
        float best = 1e9f;
        for (const auto& q : bb) {
            best = std::min(best, dist(p, q));
        }
        acc += best;
    }
    return acc / static_cast<double>(aa.size());
}

inline double hausdorff(const std::vector<Vec2>& a, const std::vector<Vec2>& b) {
    const auto aa = thin_points(a);
    const auto bb = thin_points(b);
    if (aa.empty() || bb.empty()) {
        return 0.0;
    }
    auto directed = [](const std::vector<Vec2>& u, const std::vector<Vec2>& v) {
        float h = 0;
        for (const auto& p : u) {
            float best = 1e9f;
            for (const auto& q : v) {
                best = std::min(best, dist(p, q));
            }
            h = std::max(h, best);
        }
        return h;
    };
    return std::max(directed(aa, bb), directed(bb, aa));
}

} // namespace math
