#pragma once

// Felzenszwalb–Huttenlocher O(N) Euclidean distance transform
// (Distance Transforms of Sampled Functions, Theory of Computing 2012).
// Used for signed distance fields and raster medial-axis ridges.

#include "vision_types.hpp"

#include <limits>
#include <vector>
#include <cmath>
#include <algorithm>

namespace vision {

class FelzenszwalbDistanceTransform {
public:
    static constexpr float kInf = 1.0e20f;

    static void dt_1d(const float* f, float* d, int n) {
        std::vector<int> v(static_cast<size_t>(n));
        std::vector<float> z(static_cast<size_t>(n) + 1);
        int k = 0;
        v[0] = 0;
        z[0] = -kInf;
        z[1] = kInf;
        for (int q = 1; q < n; ++q) {
            float s = 0.0f;
            while (true) {
                const int vk = v[static_cast<size_t>(k)];
                s = ((f[q] + static_cast<float>(q) * q) - (f[vk] + static_cast<float>(vk) * vk)) /
                    (2.0f * static_cast<float>(q - vk));
                if (s > z[static_cast<size_t>(k)]) {
                    break;
                }
                --k;
                if (k < 0) {
                    k = 0;
                    s = -kInf;
                    break;
                }
            }
            ++k;
            v[static_cast<size_t>(k)] = q;
            z[static_cast<size_t>(k)] = s;
            z[static_cast<size_t>(k) + 1] = kInf;
        }
        k = 0;
        for (int q = 0; q < n; ++q) {
            while (z[static_cast<size_t>(k) + 1] < static_cast<float>(q)) {
                ++k;
            }
            const int vk = v[static_cast<size_t>(k)];
            const float diff = static_cast<float>(q - vk);
            d[q] = diff * diff + f[vk];
        }
    }

    // Squared Euclidean DT. Foreground (value > threshold) has distance 0.
    static std::vector<float> squared_edt(const GrayImage& image, uint8_t thr = 128) {
        const int w = image.width;
        const int h = image.height;
        std::vector<float> f(static_cast<size_t>(w * h));
        std::vector<float> d(static_cast<size_t>(w * h));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                f[static_cast<size_t>(y * w + x)] = image.fg(x, y, thr) ? 0.0f : kInf;
            }
        }
        std::vector<float> col(static_cast<size_t>(h));
        std::vector<float> col_out(static_cast<size_t>(h));
        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) {
                col[static_cast<size_t>(y)] = f[static_cast<size_t>(y * w + x)];
            }
            dt_1d(col.data(), col_out.data(), h);
            for (int y = 0; y < h; ++y) {
                f[static_cast<size_t>(y * w + x)] = col_out[static_cast<size_t>(y)];
            }
        }
        std::vector<float> row(static_cast<size_t>(w));
        std::vector<float> row_out(static_cast<size_t>(w));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                row[static_cast<size_t>(x)] = f[static_cast<size_t>(y * w + x)];
            }
            dt_1d(row.data(), row_out.data(), w);
            for (int x = 0; x < w; ++x) {
                d[static_cast<size_t>(y * w + x)] = row_out[static_cast<size_t>(x)];
            }
        }
        return d;
    }

    static std::vector<float> edt(const GrayImage& image, uint8_t thr = 128) {
        auto sq = squared_edt(image, thr);
        for (float& v : sq) {
            v = (v >= kInf * 0.5f) ? kInf : std::sqrt(std::max(0.0f, v));
        }
        return sq;
    }

    // Positive outside the silhouette, negative inside.
    static std::vector<float> signed_distance(const GrayImage& image, uint8_t thr = 128) {
        GrayImage inv = image;
        for (uint8_t& p : inv.pixels) {
            p = p > thr ? 0 : 255;
        }
        auto outside = edt(image, thr);
        auto inside = edt(inv, thr);
        std::vector<float> sdf(outside.size());
        for (size_t i = 0; i < sdf.size(); ++i) {
            const float o = outside[i] >= kInf * 0.5f ? 0.0f : outside[i];
            const float inn = inside[i] >= kInf * 0.5f ? 0.0f : inside[i];
            sdf[i] = o - inn;
        }
        return sdf;
    }
};

}  // namespace vision
