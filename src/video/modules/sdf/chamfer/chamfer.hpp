#pragma once

#include "math/contour_compat.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace contour {

// Chamfer 3-4 / 8-point sequential EDT approximating Euclidean distance.
class ChamferSDF {
public:
    static Field distance_to_zero(const Field& seeds) {
        Field d = seeds;
        const float inf = 1.0e8f;
        const float ax = 1.0f, ad = 1.41421356f;
        auto at = [&](int x, int y) -> float& { return d.at(x, y); };
        auto get = [&](int x, int y) -> float {
            if (x < 0 || y < 0 || x >= d.width || y >= d.height) {
                return inf;
            }
            return d.at(x, y);
        };
        for (int y = 0; y < d.height; ++y) {
            for (int x = 0; x < d.width; ++x) {
                float v = at(x, y);
                v = std::min(v, get(x - 1, y) + ax);
                v = std::min(v, get(x, y - 1) + ax);
                v = std::min(v, get(x - 1, y - 1) + ad);
                v = std::min(v, get(x + 1, y - 1) + ad);
                at(x, y) = v;
            }
        }
        for (int y = d.height - 1; y >= 0; --y) {
            for (int x = d.width - 1; x >= 0; --x) {
                float v = at(x, y);
                v = std::min(v, get(x + 1, y) + ax);
                v = std::min(v, get(x, y + 1) + ax);
                v = std::min(v, get(x + 1, y + 1) + ad);
                v = std::min(v, get(x - 1, y + 1) + ad);
                at(x, y) = v;
            }
        }
        return d;
    }

    static Field from_mask(const ImageBuffer& mask, uint8_t thr = 127) {
        Field fg = make_field(mask.width, mask.height, 1.0e8f);
        Field bg = make_field(mask.width, mask.height, 1.0e8f);
        for (int y = 0; y < mask.height; ++y) {
            for (int x = 0; x < mask.width; ++x) {
                if (mask.at(x, y) > thr) {
                    fg.at(x, y) = 0.0f;
                } else {
                    bg.at(x, y) = 0.0f;
                }
            }
        }
        fg = distance_to_zero(fg);
        bg = distance_to_zero(bg);
        Field sdf = make_field(mask.width, mask.height, 0.0f);
        for (int y = 0; y < mask.height; ++y) {
            for (int x = 0; x < mask.width; ++x) {
                sdf.at(x, y) = bg.at(x, y) - fg.at(x, y);  // negative inside
            }
        }
        return sdf;
    }

    static Vec2 gradient(const Field& f, int x, int y) {
        const float dx = f.at(std::min(f.width - 1, x + 1), y) - f.at(std::max(0, x - 1), y);
        const float dy = f.at(x, std::min(f.height - 1, y + 1)) - f.at(x, std::max(0, y - 1));
        return {0.5f * dx, 0.5f * dy};
    }

    static Field analytic_box_sdf(int w, int h, float cx, float cy, float hx, float hy) {
        Field f = make_field(w, h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float px = std::fabs(static_cast<float>(x) - cx) - hx;
                const float py = std::fabs(static_cast<float>(y) - cy) - hy;
                const float ax = std::max(px, 0.0f);
                const float ay = std::max(py, 0.0f);
                const float outside = std::sqrt(ax * ax + ay * ay);
                const float inside = std::min(std::max(px, py), 0.0f);
                f.at(x, y) = outside + inside;
            }
        }
        return f;
    }

    static Field analytic_circle_sdf(int w, int h, float cx, float cy, float r) {
        Field f = make_field(w, h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float dx = static_cast<float>(x) - cx;
                const float dy = static_cast<float>(y) - cy;
                f.at(x, y) = std::sqrt(dx * dx + dy * dy) - r;
            }
        }
        return f;
    }
};

}  // namespace contour
