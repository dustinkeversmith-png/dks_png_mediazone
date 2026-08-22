#pragma once

#include "../../math/vision_types.hpp"

#include <cmath>
#include <algorithm>
#include <vector>

namespace vision {

class TemplateNCC {
public:
    static float ncc(const GrayImage& image, const GrayImage& templ, int ox, int oy) {
        if (templ.empty() || image.empty()) {
            return -1.0f;
        }
        if (ox < 0 || oy < 0 || ox + templ.width > image.width || oy + templ.height > image.height) {
            return -1.0f;
        }
        double sum_i = 0, sum_t = 0, sum_ii = 0, sum_tt = 0, sum_it = 0;
        const int n = templ.width * templ.height;
        for (int y = 0; y < templ.height; ++y) {
            for (int x = 0; x < templ.width; ++x) {
                const double iv = image.at(ox + x, oy + y);
                const double tv = templ.at(x, y);
                sum_i += iv;
                sum_t += tv;
                sum_ii += iv * iv;
                sum_tt += tv * tv;
                sum_it += iv * tv;
            }
        }
        const double mean_i = sum_i / n;
        const double mean_t = sum_t / n;
        const double num = sum_it - n * mean_i * mean_t;
        const double den = std::sqrt((sum_ii - n * mean_i * mean_i) * (sum_tt - n * mean_t * mean_t));
        if (den < 1e-8) {
            return (std::fabs(mean_i - mean_t) < 1e-6) ? 1.0f : 0.0f;
        }
        return static_cast<float>(num / den);
    }

    static float ncc_at_box(const GrayImage& image, const Rect& box) {
        const int x = std::max(0, static_cast<int>(box.x));
        const int y = std::max(0, static_cast<int>(box.y));
        const int w = std::max(1, static_cast<int>(box.w));
        const int h = std::max(1, static_cast<int>(box.h));
        if (x + w > image.width || y + h > image.height) {
            return -1.0f;
        }
        GrayImage templ;
        templ.width = w;
        templ.height = h;
        templ.pixels.resize(static_cast<size_t>(w * h));
        for (int yy = 0; yy < h; ++yy) {
            for (int xx = 0; xx < w; ++xx) {
                templ.at(xx, yy) = image.at(x + xx, y + yy);
            }
        }
        return ncc(image, templ, x, y);
    }
};

}  // namespace vision
