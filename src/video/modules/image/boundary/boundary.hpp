#pragma once

#include "../../math/vision_types.hpp"
#include "../../filters/morph_clean/helpers.hpp"
#include <algorithm>
#include <vector>

namespace image {

inline math::ImageBuffer boundary_pixels(const math::ImageBuffer& mask, uint8_t thr = 127) {
    math::ImageBuffer b = math::make_gray(mask.width, mask.height, 0);
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            if (mask.at(x, y) <= thr) {
                continue;
            }
            bool edge = x == 0 || y == 0 || x == mask.width - 1 || y == mask.height - 1;
            if (!edge) {
                for (int k = 0; k < 4 && !edge; ++k) {
                    static const int dx[4] = {1, -1, 0, 0};
                    static const int dy[4] = {0, 0, 1, -1};
                    if (mask.at(x + dx[k], y + dy[k]) <= thr) {
                        edge = true;
                    }
                }
            }
            if (edge) {
                b.at(x, y) = 255;
            }
        }
    }
    return b;
}

inline double boundary_f1(const math::ImageBuffer& pred_boundary, const math::ImageBuffer& gt_boundary, int tol = 2) {
    const auto gt_d = contour::dilate_binary(gt_boundary, tol);
    const auto pr_d = contour::dilate_binary(pred_boundary, tol);
    int pred_n = 0, gt_n = 0, hit_p = 0, hit_g = 0;
    const int w = std::min(pred_boundary.width, gt_boundary.width);
    const int h = std::min(pred_boundary.height, gt_boundary.height);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (pred_boundary.at(x, y) > 0) {
                ++pred_n;
                if (gt_d.at(x, y) > 0) {
                    ++hit_p;
                }
            }
            if (gt_boundary.at(x, y) > 0) {
                ++gt_n;
                if (pr_d.at(x, y) > 0) {
                    ++hit_g;
                }
            }
        }
    }
    const double prec = pred_n > 0 ? static_cast<double>(hit_p) / pred_n : 0.0;
    const double rec = gt_n > 0 ? static_cast<double>(hit_g) / gt_n : 0.0;
    return (prec + rec) > 1e-12 ? 2.0 * prec * rec / (prec + rec) : 0.0;
}

} // namespace image
