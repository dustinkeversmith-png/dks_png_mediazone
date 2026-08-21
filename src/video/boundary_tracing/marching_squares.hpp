#pragma once

#include "../contour_kit/types.hpp"
#include "../contour_kit/metrics.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

namespace contour {

class MarchingSquares {
public:
    std::vector<Polyline> extract(const Field& sdf, float iso = 0.0f) const {
        std::vector<std::pair<Vec2, Vec2>> segs;
        const int w = sdf.width;
        const int h = sdf.height;
        auto lerp = [&](float x0, float y0, float v0, float x1, float y1, float v1) {
            const float den = v1 - v0;
            const float t = std::fabs(den) < 1e-12f ? 0.5f : (iso - v0) / den;
            const float tt = std::clamp(t, 0.0f, 1.0f);
            return Vec2{x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt};
        };
        for (int y = 0; y < h - 1; ++y) {
            for (int x = 0; x < w - 1; ++x) {
                const float v00 = sdf.at(x, y);
                const float v10 = sdf.at(x + 1, y);
                const float v11 = sdf.at(x + 1, y + 1);
                const float v01 = sdf.at(x, y + 1);
                const int idx = (v00 < iso ? 1 : 0) | (v10 < iso ? 2 : 0) | (v11 < iso ? 4 : 0) |
                                (v01 < iso ? 8 : 0);
                if (idx == 0 || idx == 15) {
                    continue;
                }
                const Vec2 e0 = lerp(static_cast<float>(x), static_cast<float>(y), v00,
                                     static_cast<float>(x + 1), static_cast<float>(y), v10);
                const Vec2 e1 = lerp(static_cast<float>(x + 1), static_cast<float>(y), v10,
                                     static_cast<float>(x + 1), static_cast<float>(y + 1), v11);
                const Vec2 e2 = lerp(static_cast<float>(x), static_cast<float>(y + 1), v01,
                                     static_cast<float>(x + 1), static_cast<float>(y + 1), v11);
                const Vec2 e3 = lerp(static_cast<float>(x), static_cast<float>(y), v00,
                                     static_cast<float>(x), static_cast<float>(y + 1), v01);
                auto add = [&](const Vec2& a, const Vec2& b) { segs.push_back({a, b}); };
                switch (idx) {
                    case 1: case 14: add(e0, e3); break;
                    case 2: case 13: add(e0, e1); break;
                    case 3: case 12: add(e3, e1); break;
                    case 4: case 11: add(e1, e2); break;
                    case 6: case 9:  add(e0, e2); break;
                    case 7: case 8:  add(e3, e2); break;
                    case 5:
                        add(e0, e1);
                        add(e3, e2);
                        break;
                    case 10:
                        add(e0, e3);
                        add(e1, e2);
                        break;
                    default: break;
                }
            }
        }
        return stitch(segs);
    }

    Polyline largest_closed(const Field& sdf, float iso = 0.0f) const {
        auto loops = extract(sdf, iso);
        Polyline best;
        float best_a = -1.0f;
        for (auto& p : loops) {
            const float a = shoelace(p.points);
            if (a > best_a) {
                best_a = a;
                best = std::move(p);
            }
        }
        return best;
    }

private:
    static std::vector<Polyline> stitch(const std::vector<std::pair<Vec2, Vec2>>& segs) {
        std::vector<char> used(segs.size(), 0);
        std::vector<Polyline> out;
        auto close = [](const Vec2& a, const Vec2& b) { return dist2(a, b) < 1e-6f; };
        for (size_t s = 0; s < segs.size(); ++s) {
            if (used[s]) {
                continue;
            }
            used[s] = 1;
            Polyline p;
            p.points.push_back(segs[s].first);
            p.points.push_back(segs[s].second);
            bool grew = true;
            while (grew) {
                grew = false;
                for (size_t i = 0; i < segs.size(); ++i) {
                    if (used[i]) {
                        continue;
                    }
                    if (close(p.points.back(), segs[i].first)) {
                        p.points.push_back(segs[i].second);
                        used[i] = 1;
                        grew = true;
                    } else if (close(p.points.back(), segs[i].second)) {
                        p.points.push_back(segs[i].first);
                        used[i] = 1;
                        grew = true;
                    } else if (close(p.points.front(), segs[i].first)) {
                        p.points.insert(p.points.begin(), segs[i].second);
                        used[i] = 1;
                        grew = true;
                    } else if (close(p.points.front(), segs[i].second)) {
                        p.points.insert(p.points.begin(), segs[i].first);
                        used[i] = 1;
                        grew = true;
                    }
                }
            }
            if (p.points.size() >= 3 && close(p.points.front(), p.points.back())) {
                p.points.pop_back();
                p.closed = true;
            }
            if (p.points.size() >= 3) {
                out.push_back(std::move(p));
            }
        }
        return out;
    }
};

}  // namespace contour
