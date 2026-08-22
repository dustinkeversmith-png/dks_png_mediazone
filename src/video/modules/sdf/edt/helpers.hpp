#pragma once

#include "types.hpp"

namespace contour {

inline ImageBuffer rasterize_mask_from_field(const Field& sdf, float iso = 0.0f) {
    ImageBuffer m = make_gray(sdf.width, sdf.height, 0);
    for (int y = 0; y < sdf.height; ++y) {
        for (int x = 0; x < sdf.width; ++x) {
            if (sdf.at(x, y) <= iso) {
                m.at(x, y) = 255;
            }
        }
    }
    return m;
}

} // namespace contour
