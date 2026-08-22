#pragma once

#include "../../math/vision_types.hpp"
#include "../../math/geometry.hpp"
#include "../../segmentation/convex_hull/helpers.hpp"
#include <algorithm>
#include <vector>

namespace image {

inline double mask_iou(const math::ImageBuffer& a, const math::ImageBuffer& b, uint8_t thr = 127) {
    const int w = std::min(a.width, b.width);
    const int h = std::min(a.height, b.height);
    int inter = 0, uni = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const bool pa = a.at(x, y) > thr;
            const bool pb = b.at(x, y) > thr;
            inter += (pa && pb);
            uni += (pa || pb);
        }
    }
    return uni > 0 ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0;
}

inline double tightness_ratio(const math::ImageBuffer& mask, uint8_t thr = 127) {
    std::vector<math::Vec2> pts;
    int area = 0;
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            if (mask.at(x, y) > thr) {
                ++area;
                pts.push_back({static_cast<float>(x), static_cast<float>(y)});
            }
        }
    }
    if (area == 0) {
        return 0.0;
    }
    return static_cast<double>(math::shoelace(contour::convex_hull(std::move(pts)))) / static_cast<double>(area);
}

} // namespace image
