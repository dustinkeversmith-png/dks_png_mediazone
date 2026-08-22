#pragma once

#include "vision_types.hpp"
#include "geometry.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace math {

inline ImageBuffer rasterize_polygon(const std::vector<Vec2>& poly, int w, int h) {
    ImageBuffer mask = make_gray(w, h, 0);
    if (poly.size() < 3) {
        return mask;
    }
    for (int y = 0; y < h; ++y) {
        std::vector<float> xs;
        const float yy = static_cast<float>(y) + 0.5f;
        for (size_t i = 0; i < poly.size(); ++i) {
            const Vec2 a = poly[i];
            const Vec2 b = poly[(i + 1) % poly.size()];
            if ((a.y <= yy && b.y > yy) || (b.y <= yy && a.y > yy)) {
                const float t = (yy - a.y) / (b.y - a.y);
                xs.push_back(a.x + t * (b.x - a.x));
            }
        }
        std::sort(xs.begin(), xs.end());
        for (size_t k = 0; k + 1 < xs.size(); k += 2) {
            const int x0 = std::clamp(static_cast<int>(std::ceil(xs[k])), 0, w);
            const int x1 = std::clamp(static_cast<int>(std::floor(xs[k + 1])), 0, w - 1);
            for (int x = x0; x <= x1; ++x) {
                mask.at(x, y) = 255;
            }
        }
    }
    return mask;
}

inline ImageBuffer polyline_to_boundary(const std::vector<Vec2>& pts, int w, int h, bool closed) {
    ImageBuffer b = make_gray(w, h, 0);
    auto plot = [&](int x, int y) {
        if (x >= 0 && y >= 0 && x < w && y < h) {
            b.at(x, y) = 255;
        }
    };
    auto line = [&](Vec2 a, Vec2 bpt) {
        const int n = std::max(1, static_cast<int>(dist(a, bpt) * 2.0f));
        for (int i = 0; i <= n; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(n);
            plot(static_cast<int>(std::lround(a.x + (bpt.x - a.x) * t)),
                 static_cast<int>(std::lround(a.y + (bpt.y - a.y) * t)));
        }
    };
    if (pts.size() < 2) {
        return b;
    }
    for (size_t i = 1; i < pts.size(); ++i) {
        line(pts[i - 1], pts[i]);
    }
    if (closed) {
        line(pts.back(), pts.front());
    }
    return b;
}

inline ImageBuffer xor_fill(const std::vector<Polyline>& loops, int w, int h) {
    ImageBuffer pred = make_gray(w, h, 0);
    for (const auto& p : loops) {
        if (p.points.size() < 3) {
            continue;
        }
        const ImageBuffer m = rasterize_polygon(p.points, w, h);
        for (size_t i = 0; i < pred.data.size(); ++i) {
            pred.data[i] = static_cast<uint8_t>(pred.data[i] ^ m.data[i]);
        }
    }
    return pred;
}

} // namespace math
