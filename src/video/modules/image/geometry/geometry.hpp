#pragma once

#include "../../math/vision_types.hpp"
#include <vector>

namespace image {

inline std::vector<math::Vec2> collect_on(const math::ImageBuffer& b) {
    std::vector<math::Vec2> pts;
    for (int y = 0; y < b.height; ++y) {
        for (int x = 0; x < b.width; ++x) {
            if (b.at(x, y) > 0) {
                pts.push_back({static_cast<float>(x), static_cast<float>(y)});
            }
        }
    }
    return pts;
}

} // namespace image
