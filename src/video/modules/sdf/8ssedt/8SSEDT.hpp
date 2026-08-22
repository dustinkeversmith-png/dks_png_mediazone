#pragma once

#include "contour_kit/types.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace contour {

// Exact Euclidean SDF via Felzenszwalb–Huttenlocher 1D parabolic envelopes
// (linear-time 8-connected sequential equivalent of 8SSEDT).
class ExactSDF {
public:
    static constexpr float kInf = 1.0e12f;

    static Field squared_edt(const Field& f) {
        Field tmp = make_field(f.width, f.height, kInf);
        std::vector<float> row(static_cast<size_t>(std::max(f.width, f.height)));
        std::vector<float> dt(row.size());
        for (int y = 0; y < f.height; ++y) {
            for (int x = 0; x < f.width; ++x) {
                row[static_cast<size_t>(x)] = f.at(x, y);
            }
            distance_1d(row.data(), dt.data(), f.width);
            for (int x = 0; x < f.width; ++x) {
                tmp.at(x, y) = dt[static_cast<size_t>(x)];
            }
        }
        Field out = make_field(f.width, f.height, 0);
        for (int x = 0; x < f.width; ++x) {
            for (int y = 0; y < f.height; ++y) {
                row[static_cast<size_t>(y)] = tmp.at(x, y);
            }
            distance_1d(row.data(), dt.data(), f.height);
            for (int y = 0; y < f.height; ++y) {
                out.at(x, y) = dt[static_cast<size_t>(y)];
            }
        }
        return out;
    }

    static Field from_mask(const ImageBuffer& mask, uint8_t thr = 127) {
        Field fg = make_field(mask.width, mask.height, kInf);
        Field bg = make_field(mask.width, mask.height, kInf);
        for (int y = 0; y < mask.height; ++y) {
            for (int x = 0; x < mask.width; ++x) {
                if (mask.at(x, y) > thr) {
                    fg.at(x, y) = 0.0f;
                } else {
                    bg.at(x, y) = 0.0f;
                }
            }
        }
        fg = squared_edt(fg);
        bg = squared_edt(bg);
        Field sdf = make_field(mask.width, mask.height, 0);
        for (int y = 0; y < mask.height; ++y) {
            for (int x = 0; x < mask.width; ++x) {
                const float dout = std::sqrt(std::max(0.0f, bg.at(x, y)));
                const float din = std::sqrt(std::max(0.0f, fg.at(x, y)));
                sdf.at(x, y) = dout - din;  // negative inside
            }
        }
        return sdf;
    }

    static Vec2 gradient(const Field& f, int x, int y) {
        const float dx = f.at(std::min(f.width - 1, x + 1), y) - f.at(std::max(0, x - 1), y);
        const float dy = f.at(x, std::min(f.height - 1, y + 1)) - f.at(x, std::max(0, y - 1));
        return {0.5f * dx, 0.5f * dy};
    }

    static Vec2 gradient_at(const Field& f, float x, float y) {
        const float dx = f.sample(x + 1.0f, y) - f.sample(x - 1.0f, y);
        const float dy = f.sample(x, y + 1.0f) - f.sample(x, y - 1.0f);
        return {0.5f * dx, 0.5f * dy};
    }

private:
    static void distance_1d(const float* f, float* d, int n) {
        std::vector<int> v(static_cast<size_t>(n));
        std::vector<float> z(static_cast<size_t>(n) + 1);
        int k = 0;
        v[0] = 0;
        z[0] = -std::numeric_limits<float>::infinity();
        z[1] = std::numeric_limits<float>::infinity();
        auto square = [](int q) { return static_cast<float>(q) * static_cast<float>(q); };
        for (int q = 1; q < n; ++q) {
            auto intersect = [&](int qv, int vk) {
                return ((f[qv] + square(qv)) - (f[vk] + square(vk))) /
                       (2.0f * static_cast<float>(qv - vk));
            };
            float s = intersect(q, v[static_cast<size_t>(k)]);
            while (s <= z[static_cast<size_t>(k)]) {
                --k;
                s = intersect(q, v[static_cast<size_t>(k)]);
            }
            ++k;
            v[static_cast<size_t>(k)] = q;
            z[static_cast<size_t>(k)] = s;
            z[static_cast<size_t>(k) + 1] = std::numeric_limits<float>::infinity();
        }
        k = 0;
        for (int q = 0; q < n; ++q) {
            while (z[static_cast<size_t>(k) + 1] < static_cast<float>(q)) {
                ++k;
            }
            const int vk = v[static_cast<size_t>(k)];
            d[q] = square(q - vk) + f[vk];
        }
    }
};

}  // namespace contour
